#!/usr/bin/env python3
"""Download public auxiliary datasets and prepare the Harbor image build context.

Host dependency: pip install pyarrow==17.0.0
Usage: python prepare_data.py
"""
import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path

from environment.data_tools import (
    HOTPOT_SPLITS, MUSIQUE_SPLITS, describe, normalize_hotpot,
    require, sha256, verify_bundle, write_json,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "environment" / "data"
HOT = "https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/main/"
MUS = "https://hf-mirror.com/datasets/dgslibisey/MuSiQue/resolve/main/"


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return
    partial = path.with_suffix(path.suffix + ".partial")
    for attempt in range(3):
        try:
            print("Downloading", url, flush=True)
            with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
                expected = response.headers.get("Content-Length")
            if expected and partial.stat().st_size != int(expected):
                raise ValueError(f"Incomplete download: {path}")
            partial.replace(path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def stage_examples():
    examples = ROOT / "environment" / "public_examples"
    examples.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "examples" / "public.jsonl", examples / "public.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rebuild", action="store_true",
                      help="Reconvert and validate cached sources; download missing sources")
    mode.add_argument("--verify-only", action="store_true",
                      help="Validate the existing bundle offline without writing files")
    args = parser.parse_args()
    if args.verify_only or ((DATA / "manifest.json").exists() and not args.rebuild):
        for entry in verify_bundle(DATA):
            print(entry["path"], entry["rows"], "OK")
        if not args.verify_only:
            stage_examples()
        return

    import pyarrow.parquet as pq

    # Reject altered cached inputs before blessing them with fresh checksums.
    source_manifest = DATA / "source_manifest.json"
    known_sources = {}
    if source_manifest.exists():
        for entry in json.loads(source_manifest.read_text())["files"]:
            path = DATA / entry["path"]
            require(path.resolve().is_relative_to(DATA.resolve()), f"Invalid source path: {path}")
            known_sources[entry["path"]] = entry["sha256"]
            if path.exists():
                require(sha256(path) == entry["sha256"], f"Cached source changed: {path}")

    def get_source(url, path):
        download(url, path)
        expected_hash = known_sources.get(str(path.relative_to(DATA)))
        if expected_hash:
            require(sha256(path) == expected_hash, f"Source checksum mismatch: {path}")

    manifest, sources = [], []
    for split, (names, expected) in HOTPOT_SPLITS.items():
        output = DATA / "hotpot_qa" / (split + ".jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(".jsonl.partial")
        seen = set()
        with partial.open("w", encoding="utf-8") as handle:
            for name in names:
                path = DATA / "hotpot_qa" / name
                get_source(HOT + name, path)
                parquet = pq.ParquetFile(path)
                sources.append(describe(path, DATA, url=HOT + name, rows=parquet.metadata.num_rows))
                for batch in parquet.iter_batches(batch_size=256):
                    for row in batch.to_pylist():
                        row = normalize_hotpot(row, split)
                        if row["_id"] in seen:
                            raise ValueError(f"Duplicate ID in {split}")
                        seen.add(row["_id"])
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if len(seen) != expected:
            raise ValueError(f"{split}: expected {expected}, got {len(seen)}")
        partial.replace(output)
        manifest.append(describe(output, DATA, rows=len(seen)))
        print(output.name, len(seen), flush=True)

    for split, expected in MUSIQUE_SPLITS.items():
        name = f"musique_ans_v1.0_{split}.jsonl"
        path = DATA / "musique" / name
        get_source(MUS + name, path)
        seen = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                require(bool(row["id"] and row["question"] and row["answer"] and row["paragraphs"]),
                        f"Missing required MuSiQue fields: {path}")
                require(row["answerable"] is True, f"Unanswerable MuSiQue-Ans record: {path}")
                require(row["id"] not in seen, f"Duplicate MuSiQue ID: {row['id']}")
                seen.add(row["id"])
        if len(seen) != expected:
            raise ValueError(f"{name}: expected {expected}, got {len(seen)}")
        manifest.append(describe(path, DATA, rows=len(seen)))
        sources.append(describe(path, DATA, url=MUS + name, rows=len(seen)))
        print(name, len(seen), flush=True)

    for name, content in [("manifest.json", {"files": manifest}),
                          ("source_manifest.json", {"files": sources})]:
        write_json(DATA / name, content)
    verify_bundle(DATA)
    stage_examples()


if __name__ == "__main__":
    main()
