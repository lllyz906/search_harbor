#!/usr/bin/env python3
"""Offline host checks for the search task; does not start training or scoring."""
import argparse
import json
import shutil
from pathlib import Path

from environment.data_tools import require, verify_bundle

ROOT = Path(__file__).resolve().parent


def check_model(path: Path) -> None:
    config = json.loads((path / "config.json").read_text())
    require("Qwen3_5ForConditionalGeneration" in config.get("architectures", []),
            "Expected Qwen3_5ForConditionalGeneration architecture")
    require(bool(config.get("vision_config")), "Model config is missing the vision component")
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        require((path / name).is_file(), f"Missing {path / name}")
    index = path / "model.safetensors.index.json"
    if index.exists():
        weights = set(json.loads(index.read_text())["weight_map"].values())
    else:
        weights = {"model.safetensors"}
    require(bool(weights), "Model weight index is empty")
    for name in weights:
        weight = path / name
        require(weight.resolve().is_relative_to(path.resolve()), f"Invalid weight path: {name}")
        require(weight.is_file() and weight.stat().st_size > 0, f"Missing or empty weights: {weight}")
    print(f"Model files: OK ({len(weights)} weight files; GPU loading not tested)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Optional host path of the supplied base checkpoint")
    args = parser.parse_args()
    for name in ["task.toml", "instruction.md", "environment/Dockerfile",
                 "environment/requirements.txt", "environment/.dockerignore",
                 "environment/data/README.md", "environment/data/source_manifest.json",
                 "tests/test.sh", "tests/evaluate.py"]:
        require((ROOT / name).is_file(), f"Missing task file: {name}")
    for entry in verify_bundle(ROOT / "environment" / "data"):
        print(entry["path"], entry["rows"], "OK")
    source = ROOT / "examples" / "public.jsonl"
    staged = ROOT / "environment" / "public_examples" / "public.jsonl"
    require(source.read_bytes() == staged.read_bytes(),
            "Staged public examples differ; run python prepare_data.py")
    print("Public examples: OK")
    if args.model:
        check_model(args.model)
    else:
        print("Model files: not checked (supply --model PATH)")
    for command in ["docker", "harbor"]:
        print(f"{command}: {shutil.which(command) or 'not installed'}")
    print("Offline checks passed. Container build, GPU allocation, and model loading still require runtime validation.")


if __name__ == "__main__":
    main()
