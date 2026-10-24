"""Frozen candidate descriptions, encoded once and embedded in router checkpoints.

Each candidate controller is described in plain language (`MARSConfig.skill_text`). The frozen
SigLIP2 text encoder turns the descriptions into normalized vectors once, before training; the
router stores them as a buffer, so neither training nor deployment loads the encoder again.

The encoder needs `open_clip` (pip install "mars-router[siglip2]"). Training and inference only
need the resulting small, validated tensor cache:

    python -m mars.text --config configs/mars.yaml --classes vla pickup place --output siglip2_text.pt
    python -m mars.text --spec skill_text_spec.json --output siglip2_text.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def text_spec(cfg, classes):
    """Cache metadata: encoder identity, candidate order and the exact descriptions."""
    descriptions = cfg.skill_text
    missing = [c for c in classes if c not in descriptions or not descriptions[c].strip()]
    if missing:
        raise ValueError(f"missing controller descriptions: {missing}")
    return dict(version=1, encoder="siglip2", model=cfg.text_model, pretrained=cfg.text_pretrained,
                classes=list(classes), texts=[descriptions[c] for c in classes], normalize=True)


def cache_key(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def load_text_cache(path, spec):
    """Load (K, text_dim) embeddings, refusing a cache written for other candidates or texts."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("spec") != spec:
        raise ValueError(f"text cache metadata mismatch (encoder/classes/descriptions): {path}")
    emb = payload["embeddings"]
    if (emb.ndim != 2 or emb.shape[0] != len(spec["classes"]) or emb.shape[1] == 0
            or not torch.isfinite(emb).all() or (emb.norm(dim=-1) == 0).any()):
        raise ValueError(f"invalid text embeddings: {path}")
    return emb.float()


@torch.no_grad()
def encode_siglip2(spec, device="cpu"):
    import open_clip

    if spec["encoder"] != "siglip2" or not spec["normalize"]:
        raise ValueError("expected a normalized SigLIP2 text specification")
    model, _, _ = open_clip.create_model_and_transforms(
        spec["model"], pretrained=spec["pretrained"], device=device)
    model.eval().requires_grad_(False)
    # Resolve the tokenizer snapshot explicitly. Some Transformers versions probe
    # a missing model config when a tokenizer-only Hub repo is used in offline mode.
    # Keep OpenCLIP's canonicalization, context length, padding and EOS handling.
    from huggingface_hub import hf_hub_download
    from open_clip.tokenizer import HFTokenizer
    text_cfg = open_clip.get_model_config(spec["model"])["text_cfg"]
    tokenizer_dir = Path(hf_hub_download(text_cfg["hf_tokenizer_name"], "tokenizer_config.json")).parent
    tokenizer = HFTokenizer(str(tokenizer_dir), context_length=text_cfg["context_length"],
                            **text_cfg.get("tokenizer_kwargs", {}))
    emb = model.encode_text(tokenizer(spec["texts"]).to(device), normalize=True)
    return emb.float().cpu()


def candidate_embeddings(cfg, classes, out_dir):
    """(embeddings, spec): the explicit `cfg.text_embeddings_path`, a matching cache in `out_dir`,
    or a fresh SigLIP2 encoding written to `out_dir/skill_text_{key}.pt`."""
    spec = text_spec(cfg, classes)
    explicit = cfg.text_embeddings_path
    path = Path(explicit) if explicit else Path(out_dir) / f"skill_text_{cache_key(spec)}.pt"
    if path.exists() or explicit:
        return load_text_cache(path, spec), spec
    emb = encode_siglip2(spec)  # only a few descriptions; no GPU needed
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": spec, "embeddings": emb}, path)
    return load_text_cache(path, spec), spec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", type=Path, help="JSON written by text_spec()")
    source.add_argument("--config", type=Path, help="MARS YAML config with skill_text")
    parser.add_argument("--classes", nargs="+", help="candidate order (with --config)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.spec:
        spec = json.loads(args.spec.read_text())
    else:
        from .config import MARSConfig

        cfg = MARSConfig.from_yaml(args.config)
        spec = text_spec(cfg, args.classes or list(cfg.skill_text))
    if args.output.exists():
        emb = load_text_cache(args.output, spec)
    else:
        emb = encode_siglip2(spec)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"spec": spec, "embeddings": emb}, args.output)
        load_text_cache(args.output, spec)
    print(json.dumps({"path": str(args.output), "shape": list(emb.shape), "spec": spec}))


if __name__ == "__main__":
    main()
