"""Download a pinned HF release and verify its complete SHA256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError(f"unsafe manifest path: {name!r}")
    target = Path(root).joinpath(*p.parts)
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"path escapes release: {name}")
    return target


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "mars-assets-v1" or not manifest.get("files"):
        raise ValueError("invalid release manifest")
    for name, expected in manifest["files"].items():
        path = safe_path(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"asset hash mismatch: {name}")
    return manifest


def download(repo_id, revision, output):
    if not revision or revision in ("main", "master", "HEAD"):
        raise ValueError("an immutable HF commit revision is required")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("revision must be a full 40-character commit SHA")
    from huggingface_hub import hf_hub_download
    root = Path(output)
    hf_hub_download(repo_id, "manifest.json", revision=revision, local_dir=root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "mars-assets-v1" or not manifest.get("files"):
        raise ValueError("invalid release manifest")
    for name in manifest["files"]:
        safe_path(root, name)
        hf_hub_download(repo_id, name, revision=revision, local_dir=root)
    return verify(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("--repo-id", required=True)
    dl.add_argument("--revision", required=True)
    dl.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (download(args.repo_id, args.revision, args.output) if args.command == "download"
              else verify(args.assets))
    print(json.dumps({"verified_files": len(result["files"])}))


if __name__ == "__main__":
    main()
