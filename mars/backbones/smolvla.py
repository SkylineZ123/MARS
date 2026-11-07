# Adapted from RL-Token-Pi05-open/routerlt (Apache-2.0). See NOTICE.
"""Frozen SmolVLA backbone: one prefix prefill per observation, reusable for both the
RL-token readout (image + language hidden states) and the reference action chunk.

The observation path is exactly the one `lerobot_eval.rollout` uses
(`preprocess_observation` -> `LiberoProcessorStep` -> checkpoint preprocessor), and
`prefill` + `reference_chunk` reproduce `VLAFlowMatching.sample_actions` step by step,
so `reference_chunk(prefill(batch), noise)` equals `policy.predict_action_chunk(batch, noise)`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import Tensor

from .base import PrefixFeatures, _AffineNormalizer

logger = logging.getLogger(__name__)

OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
OBS_STATE = "observation.state"


def resolve_pretrained_path(path_or_repo: str) -> Path:
    path = Path(path_or_repo).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_repo, local_files_only=True))


def batchify_raw_obs(raw_obs: dict) -> dict:
    """Give a single (non-vector) LiberoEnv observation the leading batch dim a
    SyncVectorEnv(n=1) would add. Already-batched observations pass through."""
    images = raw_obs["pixels"]
    batched = next(iter(images.values())).ndim == 4
    if batched:
        return raw_obs

    def _add(x):
        if isinstance(x, dict):
            return {k: _add(v) for k, v in x.items()}
        if x is None:
            return None
        return np.asarray(x)[None]

    return {k: _add(v) for k, v in raw_obs.items()}


def _slice_tree(tree, n):
    if isinstance(tree, dict):
        return {k: _slice_tree(v, n) for k, v in tree.items()}
    return None if tree is None else np.asarray(tree)[:n]


class SmolVLABackbone:
    d_q = 8
    use_graphed_prefill = True

    def __init__(self, pretrained_path: str, device: str = "cuda", c_vla: int = 10, fast_preprocess: bool = True,
                 cuda_graphs: bool = True, max_graphs: int = 16, observation_format: str = "libero"):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        if observation_format not in ("libero", "assembly"):
            raise ValueError(f"unknown observation format {observation_format}")
        self.observation_format = observation_format

        self.path = resolve_pretrained_path(pretrained_path)
        self.device = torch.device(device)
        # Resolve bundled VLM metadata/tokenizer relative to the release, never cwd.
        config = None
        if (self.path / "vlm").is_dir():
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            config = SmolVLAConfig.from_pretrained(str(self.path))
            config.vlm_model_name = str(self.path / "vlm")
        policy = SmolVLAPolicy.from_pretrained(str(self.path), config=config)
        policy.config.device = str(self.device)
        self.policy = policy.to(self.device).eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        cfg = self.policy.config
        if cfg.rtc_config is not None and getattr(cfg.rtc_config, "enabled", False):
            raise ValueError("RTC must be disabled for the frozen generalist")
        if not cfg.use_cache:
            raise ValueError("prefix KV-cache reuse requires use_cache=True")
        if cfg.add_image_special_tokens or cfg.prefix_length > 0:
            raise ValueError("token-group slicing assumes add_image_special_tokens=False and prefix_length=0")

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(self.path),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        if observation_format == "libero":
            from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline
            self.env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
            self._libero_step = LiberoProcessorStep()
        self.fast_preprocess = fast_preprocess
        self.graphed = None
        self.graphed_prefill = None
        if cuda_graphs and self.device.type == "cuda":
            from .graphs import GraphedDenoiser, GraphedPrefill

            self.graphed = GraphedDenoiser(self.policy.model, max_graphs=max_graphs)
            self.graphed_prefill = GraphedPrefill(self.policy.model, max_graphs=max_graphs)
        self._token_cache: dict[str, tuple[Tensor, Tensor]] = {}

        stats_files = sorted(self.path.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
        if not stats_files:
            raise FileNotFoundError(f"no normalizer stats in {self.path}")
        stats = load_file(str(stats_files[0]))
        self.action_norm = _AffineNormalizer(stats["action.mean"].numpy(), stats["action.std"].numpy())
        self.state_norm = _AffineNormalizer(
            stats["observation.state.mean"].numpy(), stats["observation.state.std"].numpy()
        )

        self.H = cfg.chunk_size
        self.d_a = cfg.action_feature.shape[0]
        self.d_b = self.policy.model.vlm_with_expert.config.text_config.hidden_size
        self.c_vla = c_vla
        self.max_action_dim = cfg.max_action_dim
        self.image_keys = list(cfg.image_features)
        self.image_groups = [k.rsplit(".", 1)[-1] for k in self.image_keys]
        logger.info(
            "SmolVLA backbone: %s d_b=%d H=%d d_a=%d images=%s vlm_layers=%d",
            self.path,
            self.d_b,
            self.H,
            self.d_a,
            self.image_groups,
            self.policy.model.vlm_with_expert.num_vlm_layers,
        )

    # ------------------------------------------------------------------ observation path
    def preprocess(self, raw_obs: dict, task: str | list[str]) -> dict:
        if self.observation_format == "assembly":
            expected = set(self.image_groups)
            observed = set(raw_obs["pixels"])
            if expected != observed:
                raise ValueError(f"checkpoint cameras {sorted(expected)} do not match environment {sorted(observed)}; retrain with the matching camera schema")
        tasks = [task] if isinstance(task, str) else list(task)
        if self.fast_preprocess and len(set(tasks)) == 1:
            return self._preprocess_fast(raw_obs, tasks[0])
        return self.preprocess_reference(raw_obs, task)

    def preprocess_reference(self, raw_obs: dict, task: str | list[str]) -> dict:
        """The exact lerobot_eval path: preprocess_observation -> LiberoProcessorStep -> policy preprocessor."""
        if self.observation_format == "assembly":
            raw_obs = batchify_raw_obs(raw_obs)
            batch = {f"observation.images.{key}": torch.from_numpy(np.ascontiguousarray(img))
                     .permute(0, 3, 1, 2).float() / 255 for key, img in raw_obs["pixels"].items()}
            n = next(iter(batch.values())).shape[0]
            batch[OBS_STATE] = torch.from_numpy(self.state_from_raw(raw_obs).reshape(n, -1))
            batch["task"] = [task]*n if isinstance(task, str) else task
            return self.preprocessor(batch)
        from lerobot.envs.utils import preprocess_observation

        obs = preprocess_observation(batchify_raw_obs(raw_obs))
        batch_size = next(v for k, v in obs.items() if k.startswith("observation.images.")).shape[0]
        obs["task"] = [task] * batch_size if isinstance(task, str) else list(task)
        obs = self.env_preprocessor(obs)
        return self.preprocessor(obs)

    def _preprocess_fast(self, raw_obs: dict, task: str) -> dict:
        """Same tensors as `preprocess_reference` (checked bitwise in s0_parity) without the processor
        pipeline overhead and per-step tokenizer call: language tokens are cached per instruction."""
        raw_obs = batchify_raw_obs(raw_obs)
        if task not in self._token_cache:
            ref = self.preprocess_reference(
                {"pixels": {k: v[:1] for k, v in raw_obs["pixels"].items()},
                 "robot_state": _slice_tree(raw_obs["robot_state"], 1)}, task)
            self._token_cache[task] = (ref[OBS_LANGUAGE_TOKENS], ref[OBS_LANGUAGE_ATTENTION_MASK])
        tokens, mask = self._token_cache[task]
        batch: dict = {}
        batch_size = 1
        for name, img in raw_obs["pixels"].items():
            # Same CPU ops as preprocess_observation (GPU division differs by 1 ulp); the flip is exact.
            x = torch.from_numpy(np.ascontiguousarray(img)).permute(0, 3, 1, 2).contiguous()
            x = x.type(torch.float32)
            x /= 255
            x = x.to(self.device)
            batch[f"observation.images.{name}"] = (torch.flip(x, dims=[2, 3])
                                                   if self.observation_format == "libero" else x)
            batch_size = x.shape[0]
        state = torch.from_numpy(self.state_from_raw(raw_obs).reshape(batch_size, -1)).to(self.device)
        batch[OBS_STATE] = self.state_norm.normalize(state)
        batch[OBS_LANGUAGE_TOKENS] = tokens.expand(batch_size, -1)
        batch[OBS_LANGUAGE_ATTENTION_MASK] = mask.expand(batch_size, -1)
        return batch

    def state_from_raw(self, raw_obs: dict) -> np.ndarray:
        """Unnormalized 8-d state [eef_pos, axis-angle, gripper_qpos], as LiberoProcessorStep builds it."""
        rs = raw_obs["robot_state"]
        quat = torch.as_tensor(np.asarray(rs["eef"]["quat"], dtype=np.float32).reshape(-1, 4))
        if self.observation_format == "assembly":
            from scipy.spatial.transform import Rotation
            axis_angle = Rotation.from_quat(quat.numpy()).as_rotvec().astype(np.float32)
        else:
            axis_angle = self._libero_step._quat2axisangle(quat).numpy()
        pos = np.asarray(rs["eef"]["pos"], dtype=np.float32).reshape(-1, 3)
        grip = np.asarray(rs["gripper"]["qpos"], dtype=np.float32).reshape(-1, 2)
        state = np.concatenate([pos, axis_angle, grip], axis=-1).astype(np.float32)
        return state[0] if state.shape[0] == 1 else state

    # ------------------------------------------------------------------ frozen forward
    @torch.no_grad()
    def prefill(self, batch: dict) -> PrefixFeatures:
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        policy = self.policy
        model = policy.model
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Graphs only pay off for the per-step batch-1 calls; offline batches (adapter training,
        # z precompute) have varying shapes and are GPU-bound anyway.
        if self.graphed_prefill is not None and self.use_graphed_prefill and prefix_embs.shape[0] == 1:
            hidden, past_key_values = self.graphed_prefill(prefix_embs, prefix_att_2d_masks, prefix_position_ids)
        else:
            outputs, past_key_values = model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            hidden = outputs[0]

        batch_size = hidden.shape[0]
        n_lang = lang_tokens.shape[1]
        n_cam = len(images)
        n_img_total = prefix_embs.shape[1] - n_lang - 1  # one state token closes the prefix
        if n_img_total <= 0 or n_img_total % n_cam:
            raise RuntimeError(f"cannot split prefix of length {prefix_embs.shape[1]} into {n_cam} cameras")
        n_img = n_img_total // n_cam

        slices: dict[str, slice] = {}
        group_masks: dict[str, Tensor] = {}
        start = 0
        groups = self.image_groups[:n_cam] + [f"empty_camera_{i}" for i in range(n_cam - len(self.image_groups))]
        for name, mask in zip(groups, img_masks, strict=True):
            slices[name] = slice(start, start + n_img)
            group_masks[name] = mask.bool()[:, None].expand(batch_size, n_img)
            start += n_img
        slices["lang"] = slice(start, start + n_lang)
        group_masks["lang"] = lang_masks.bool()
        slices["state"] = slice(start + n_lang, start + n_lang + 1)
        group_masks["state"] = torch.ones(batch_size, 1, dtype=torch.bool, device=hidden.device)

        return PrefixFeatures(
            hidden=hidden,
            slices=slices,
            group_masks=group_masks,
            state_norm=batch[OBS_STATE].float().reshape(batch_size, -1),
            cache=(past_key_values, prefix_pad_masks),
        )

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> Tensor:
        """(K, d_b) frozen-VLM text embeddings for PATR controller queries: a language-only prefix
        forward (same scaling and bidirectional mask as the language part of `prefill`), final-norm
        hidden states mean-pooled over the valid tokens."""
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        vlm = self.policy.model.vlm_with_expert
        tok = vlm.processor.tokenizer(list(texts), padding=True, return_tensors="pt")
        ids = tok["input_ids"].to(self.device)
        mask = tok["attention_mask"].bool().to(self.device)
        emb = vlm.embed_language_tokens(ids)
        emb = emb * emb.shape[-1] ** 0.5
        att_2d = make_att_2d_masks(mask, torch.zeros_like(mask))
        outputs, _ = vlm.forward(attention_mask=att_2d, position_ids=torch.cumsum(mask, dim=1) - 1,
                                 past_key_values=None, inputs_embeds=[emb, None], use_cache=True)
        hidden = outputs[0].float()
        weight = mask[..., None].float()
        return ((hidden * weight).sum(1) / weight.sum(1)).cpu()

    @torch.no_grad()
    def reference_chunk(self, feats: PrefixFeatures, noise: Tensor) -> Tensor:
        """Normalized (B, H, d_a) proposal; reuses the prefill KV cache (cropped back after each step)."""
        past_key_values, prefix_pad_masks = feats.cache
        if self.graphed is not None and noise.shape[0] == 1:
            return self.graphed(past_key_values, prefix_pad_masks, noise.to(self.device))[:, :, : self.d_a].float()
        return self.reference_chunk_eager(feats, noise)

    @torch.no_grad()
    def reference_chunk_eager(self, feats: PrefixFeatures, noise: Tensor) -> Tensor:
        from lerobot.policies.common.flow_matching import euler_integrate

        model = self.policy.model
        past_key_values, prefix_pad_masks = feats.cache
        actions = euler_integrate(
            lambda x_t, timestep: model.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=timestep,
            ),
            noise.to(self.device),
            model.config.num_steps,
        )
        return actions[:, :, : self.d_a].float()

    @torch.no_grad()
    def policy_chunk(self, batch: dict, noise: Tensor) -> Tensor:
        """The unmodified lerobot path, for parity checks."""
        return self.policy.predict_action_chunk(dict(batch), noise=noise.to(self.device)).float()

    def make_noise(self, key: int, batch_size: int = 1) -> Tensor:
        generator = torch.Generator().manual_seed(int(key) % (2**63 - 1))
        noise = torch.randn((batch_size, self.H, self.max_action_dim), generator=generator, dtype=torch.float32)
        return noise.to(self.device)

    # ------------------------------------------------------------------ normalization
    def normalize_action(self, a):
        return self.action_norm.normalize(a)

    def unnormalize_action(self, a):
        return self.action_norm.unnormalize(a)

    def normalize_state(self, q):
        return self.state_norm.normalize(q)
