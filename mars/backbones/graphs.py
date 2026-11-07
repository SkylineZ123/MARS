# Adapted from RL-Token-Pi05-open/routerlt (Apache-2.0). See NOTICE.
"""CUDA-graph replay of SmolVLA's flow-matching denoise loop.

Captures the denoising loop per batch/prefix shape to reuse CUDA launches.
The suffix attention masks and Euler time tensors are precomputed; model
operations use the frozen SmolVLA modules. The eager implementation remains
available for numerical comparisons with the captured path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    keys: list[Tensor]
    values: list[Tensor]
    prefix_pad_masks: Tensor
    noise: Tensor
    out: Tensor


class GraphedDenoiser:
    def __init__(self, model, max_graphs: int = 16):
        self.model = model
        self.max_graphs = max_graphs
        self.entries: dict[tuple[int, int], _GraphEntry] = {}
        self.pool = torch.cuda.graph_pool_handle()

    # ------------------------------------------------------------------ eager re-implementation
    def _denoise_step(self, x_t: Tensor, timestep: Tensor, prefix_pad_masks: Tensor, cache, suffix_att: Tensor):
        from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding, make_att_2d_masks

        m = self.model
        # embed_suffix
        action_emb = m.action_in_proj(x_t)
        bsize = action_emb.shape[0]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, m.vlm_with_expert.expert_hidden_size, m.config.min_period, m.config.max_period,
            device=action_emb.device,
        ).type(dtype=action_emb.dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = m.action_time_mlp_out(F.silu(m.action_time_mlp_in(torch.cat([action_emb, time_emb], dim=2))))
        suffix_pad_masks = torch.ones(bsize, action_time_emb.shape[1], dtype=torch.bool, device=action_emb.device)
        suffix_att_masks = suffix_att.expand(bsize, -1)
        # denoise_step
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        outputs_embeds, _ = m.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=cache,
            inputs_embeds=[None, action_time_emb],
            use_cache=m.config.use_cache,
        )
        cache.crop(prefix_len)
        suffix_out = outputs_embeds[1][:, -m.config.chunk_size:].to(dtype=torch.float32)
        return m.action_out_proj(suffix_out)

    def _fresh_cache(self, keys, values):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for i, (k, v) in enumerate(zip(keys, values, strict=True)):
            cache.update(k, v, i)
        return cache

    def _euler(self, noise: Tensor, prefix_pad_masks: Tensor, cache, times: list[Tensor], suffix_att: Tensor):
        num_steps = len(times)
        dt = -1.0 / num_steps
        x_t = noise
        bsize = noise.shape[0]
        for step in range(num_steps):
            v_t = self._denoise_step(x_t, times[step].expand(bsize), prefix_pad_masks, cache, suffix_att)
            x_t = x_t + dt * v_t
        return x_t

    # ------------------------------------------------------------------ capture / replay
    def _capture(self, kv, prefix_pad_masks: Tensor, noise: Tensor) -> _GraphEntry:
        m = self.model
        device = noise.device
        num_steps = m.config.num_steps
        dt = -1.0 / num_steps
        times = [torch.tensor(1.0 + k * dt, dtype=torch.float32, device=device) for k in range(num_steps)]
        suffix_att = torch.ones(1, m.config.chunk_size, dtype=m.action_time_mlp_out.weight.dtype, device=device)
        init_keys = [layer.keys.clone() for layer in kv.layers]
        init_values = [layer.values.clone() for layer in kv.layers]
        static_ppm = prefix_pad_masks.clone()
        static_noise = noise.clone()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._euler(static_noise, static_ppm, self._fresh_cache(init_keys, init_values), times, suffix_att)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        cache = self._fresh_cache(init_keys, init_values)
        # DynamicCache may store copies on first update: the graph reads whatever tensors the
        # cache holds when capture starts, so those (kept alive in the entry) are the static inputs.
        keys = [layer.keys for layer in cache.layers]
        values = [layer.values for layer in cache.layers]
        with torch.cuda.graph(graph, pool=self.pool):
            out = self._euler(static_noise, static_ppm, cache, times, suffix_att)
        entry = _GraphEntry(graph, keys, values, static_ppm, static_noise, out)
        entry.times = times  # keep alive
        entry.suffix_att = suffix_att
        return entry

    @torch.no_grad()
    def __call__(self, kv, prefix_pad_masks: Tensor, noise: Tensor) -> Tensor:
        key = (noise.shape[0], prefix_pad_masks.shape[1])
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_graphs:
                self.entries.pop(next(iter(self.entries)))
            entry = self._capture(kv, prefix_pad_masks, noise)
            self.entries[key] = entry
        for i, layer in enumerate(kv.layers):
            entry.keys[i].copy_(layer.keys)
            entry.values[i].copy_(layer.values)
        entry.prefix_pad_masks.copy_(prefix_pad_masks)
        entry.noise.copy_(noise)
        entry.graph.replay()
        return entry.out.clone()

    @torch.no_grad()
    def eager(self, kv, prefix_pad_masks: Tensor, noise: Tensor) -> Tensor:
        """The re-implemented loop without graphs (for parity checks)."""
        m = self.model
        dt = -1.0 / m.config.num_steps
        times = [torch.tensor(1.0 + k * dt, dtype=torch.float32, device=noise.device)
                 for k in range(m.config.num_steps)]
        suffix_att = torch.ones(1, m.config.chunk_size, dtype=m.action_time_mlp_out.weight.dtype,
                                device=noise.device)
        keys = [layer.keys.clone() for layer in kv.layers]
        values = [layer.values.clone() for layer in kv.layers]
        return self._euler(noise, prefix_pad_masks, self._fresh_cache(keys, values), times, suffix_att)


@dataclass
class _PrefillEntry:
    graph: torch.cuda.CUDAGraph
    embs: Tensor
    att_2d: Tensor
    position_ids: Tensor
    hidden: Tensor
    cache: object  # DynamicCache whose layer tensors are graph outputs


class GraphedPrefill:
    """CUDA-graph replay of the 32-layer prefix forward that fills the KV cache.

    `embed_prefix` (vision tower + token embeddings) stays eager because it builds
    tensors from host lists; only `vlm_with_expert.forward([prefix, None], use_cache=True)`
    is captured. Returned caches are fresh copies, so callers may mutate them.
    """

    def __init__(self, model, max_graphs: int = 16):
        self.model = model
        self.max_graphs = max_graphs
        self.entries: dict[tuple, _PrefillEntry] = {}
        self.pool = torch.cuda.graph_pool_handle()

    def _forward(self, embs, att_2d, position_ids):
        return self.model.vlm_with_expert.forward(
            attention_mask=att_2d, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=True,
        )

    def _capture(self, embs, att_2d, position_ids) -> _PrefillEntry:
        s_embs, s_att, s_pos = embs.clone(), att_2d.clone(), position_ids.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._forward(s_embs, s_att, s_pos)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            outputs, cache = self._forward(s_embs, s_att, s_pos)
        return _PrefillEntry(graph, s_embs, s_att, s_pos, outputs[0], cache)

    @torch.no_grad()
    def __call__(self, embs: Tensor, att_2d: Tensor, position_ids: Tensor):
        from transformers.cache_utils import DynamicCache

        key = (tuple(embs.shape), embs.dtype)
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_graphs:
                self.entries.pop(next(iter(self.entries)))
            entry = self._capture(embs, att_2d, position_ids)
            self.entries[key] = entry
        entry.embs.copy_(embs)
        entry.att_2d.copy_(att_2d)
        entry.position_ids.copy_(position_ids)
        entry.graph.replay()
        cache = DynamicCache()
        for i, layer in enumerate(entry.cache.layers):
            cache.update(layer.keys.clone(), layer.values.clone(), i)
        return entry.hidden.clone(), cache
