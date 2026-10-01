# Harbor task: joint HotpotQA + MuSiQue post-training

Post-train the supplied Qwen3.5-4B checkpoint in 5 hours on two RTX A6000
GPUs, using HotpotQA and MuSiQue-Ans evidence-conditioned questions. The held
out evaluation uses both datasets and reports only joint F1.
The agent-facing contract is `instruction.md`; execution timeouts are in `task.toml`.

## Directory layout

```text
search/
├── instruction.md              # Agent-facing rules
├── task.toml                   # Harbor metadata and timeouts
├── prepare_data.py             # Download, validate, and stage public data
├── preflight.py                # Offline host checks
├── requirements-prepare.txt    # Host conversion dependency
├── checks/                     # Host tooling regression tests
├── examples/public.jsonl       # Authoritative public example
├── environment/                # Docker build context
│   ├── Dockerfile
│   ├── requirements.txt        # Pinned training dependencies
│   ├── data_tools.py           # Shared data specs and validation
│   ├── verify_data.py          # Build-time verification entrypoint
│   ├── data/                  # JSONL, source cache, and manifests
│   └── public_examples/       # Staged public example
└── tests/                      # Verifier; never copy into the agent image
```

## Prepare and verify data

Run these commands from this task directory with Python 3.9 or newer.
For the already-downloaded bundle, no additional package installation or network
access is needed:

```bash
python prepare_data.py
python preflight.py
```

Default preparation verifies the existing bundle and stages public examples.
It fails if an existing manifest or bundle is invalid. To only verify without
writing files, use `python prepare_data.py --verify-only`.

For a fresh download or an explicit reconstruction of the JSONL files:

```bash
python -m pip install --only-binary=:all: -r requirements-prepare.txt
python prepare_data.py --rebuild
```

Missing sources are downloaded. Cached sources are checked against the existing
source manifest before conversion; altered sources cause a failure. To recover
from a damaged source file, remove that specific file and rerun `--rebuild`.
Downloads, converted outputs, and manifests use `.partial` files followed by
renaming. A failed preparation may leave completed files available for reuse;
run verification before building the image.

## Public datasets

| Dataset/split | Records | Container path under `/task/data/` |
| --- | ---: | --- |
| HotpotQA train | 90,447 | `hotpot_qa/train.jsonl` |
| HotpotQA held-out | 7,405 | `hotpot_qa/test.jsonl` |
| MuSiQue-Ans train | 19,938 | `musique/musique_ans_v1.0_train.jsonl` |
| MuSiQue-Ans held-out | 2,417 | `musique/musique_ans_v1.0_test.jsonl` |

Both datasets are English. Reserve development/test splits for validation.
MuSiQue-Full and MuSiQue test data are not included. HotpotQA is converted from
Parquet to its original record structure in JSONL; MuSiQue JSONL is unchanged.
The container can read both using Python's standard library. Raw Parquet cache
files stay on the host and are excluded from the Docker context.

See `environment/data/README.md` for formats and licenses. `manifest.json`
records prepared-file SHA-256 hashes, byte sizes, and row counts;
`source_manifest.json` records source URLs and checksums. Counts are also checked
against the dataset specification, independently of the manifest.

## Build and runtime readiness

The Docker build context must be `environment/`. Data is copied into
`/task/data/`, and public examples into `/task/examples/public.jsonl`. Building
verifies all six data files before succeeding. Rebuild the image after changing
data; existing containers do not receive local changes automatically.

When Docker is available, a direct image build is:

```bash
docker build -t bilingual-search-posttrain:local environment/
```

The image build does not mount the base model or allocate GPUs. Configure those
through the Harbor execution backend before starting the agent. Optionally
check host model files before launching:

```bash
python preflight.py --model /absolute/host/path/to/Qwen3.5-4B
```

Preflight checks data integrity, staged examples, task files, and optional model
configuration/tokenizer/weight-file presence. It reports Docker and Harbor CLI
availability but does not invoke them. Passing these checks does not establish
GPU availability or successful model loading; those require runtime validation.

## Host tooling checks

```bash
python -m unittest discover -s checks -v
python -O -m unittest discover -s checks -v
python environment/verify_data.py environment/data
```

These checks do not read private evaluation records or run training. The
preparation and integrity checks use explicit errors and remain active under
Python optimization.

## Execution environment

Directory follows the Harbor task layout: `task.toml`, `instruction.md`, `environment/Dockerfile`, `tests/test.sh`, and `tests/evaluate.py`. Harbor mounts `tests/` for the verifier, not the agent; keep it inaccessible during the run. Run with a Harbor version supporting this layout and expose exactly two NVIDIA RTX A6000 GPUs (48 GB VRAM each) to the task container through the execution backend. The agent timeout in `task.toml` is 18000 seconds (5 hours), covering preparation, training, experiments, and artifact saving, but not container setup or final verification. This timeout does not itself allocate GPUs; the organizer must provision the two devices. The Docker build downloads pinned Python dependencies and needs access to the PyTorch wheel index. The submitted model must fit in GPU memory alongside evaluation.

## Base model mount

The designated starting checkpoint is the host directory `/mnt/f70f6709-366e-49a0-861f-497645d98975/lyz/QWen3.5vl_R3SGG/models/Qwen3.5-4B`. Bind-mount this directory read-only at `/models/Qwen3.5-4B` for both agent execution and verification, using the execution backend's volume configuration. A Docker-compatible mount specification is `--mount type=bind,source=/mnt/f70f6709-366e-49a0-861f-497645d98975/lyz/QWen3.5vl_R3SGG/models/Qwen3.5-4B,target=/models/Qwen3.5-4B,readonly`. The Dockerfile does not copy this external host directory; provisioning the mount is mandatory before starting the clock. This checkpoint is `Qwen3_5ForConditionalGeneration`, not the previous Qwen3 causal-LM base. The scorer loads the full multimodal architecture but evaluates text only, using the supplied tokenizer/chat template with thinking disabled. Transformers and PEFT pins were updated for this architecture; the container build and GPU loading still require end-to-end validation on the deployment host. LoRA submissions must record the container-local base path, not the host path. Do not download or substitute a different checkpoint.

## Evaluation

The verifier constructs a held-out mixed set from HotpotQA and MuSiQue. Models
must return JSON containing `answer` plus `supporting_facts` for HotpotQA or
`supporting_paragraphs` for MuSiQue. The only reported metric is joint F1 for
each dataset. The reward is the mean of the two joint-F1 values; `grade.json`
contains the per-dataset joint F1 values and their aggregate.

## Task provenance

Provenance: the task **adapts the research workflow** of [PostTrainBench](https://github.com/aisa-group/PostTrainBench) (fixed base, time budget, trainable final artifact, held-out model evaluation), and takes the bilingual difficult fact-finding theme from [BrowseComp](https://openai.com/index/browsecomp/) and [BrowseComp-ZH](https://arxiv.org/search/?query=BrowseComp-ZH&searchtype=all). None of their evaluation questions or answers are copied. All example and smoke-test names, facts, and evidence were invented for this task. The new fixed evidence-input protocol isolates post-training ability from search-engine availability and makes scoring deterministic. This is intentionally a PostTrainBench-like **new task**, not a faithful measurement of either BrowseComp benchmark.

## Scoring outputs

Harbor rewards are in `[0,1]` in `/logs/verifier/reward.txt`; `/logs/verifier/grade.json` contains only aggregate score/counts or a failure class. To smoke-test the data and scoring helpers without a GPU, run `python3 -m unittest discover -s tests -p 'test_unit.py'` inside this directory. A full end-to-end run needs the base-model cache, GPU, and submitted checkpoint.
