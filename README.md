# Harbor task: joint HotpotQA + MuSiQue post-training

Post-train the supplied Qwen3.5-4B checkpoint in 5 hours on two RTX A6000
GPUs, using HotpotQA and MuSiQue-Ans evidence-conditioned questions. The held
out evaluation uses both datasets and reports only joint F1.
The agent-facing contract is `instruction.md`; execution timeouts are in `task.toml`.

## GitHub sample bundle

This repository includes only the first **10 records per dataset split** (40 dataset records in total). The local full datasets are not uploaded. 

## Public datasets

| Dataset/split | Records | Container path under `/task/data/` |
| --- | ---: | --- |
| HotpotQA train | 90,447 | `hotpot_qa/train.jsonl` |
| HotpotQA held-out | 7,405 | `hotpot_qa/test.jsonl` |
| MuSiQue-Ans train | 19,938 | `musique/musique_ans_v1.0_train.jsonl` |
| MuSiQue-Ans held-out | 2,417 | `musique/musique_ans_v1.0_test.jsonl` |


## Build and runtime readiness

The Docker build context must be `environment/`. Data is copied into
`/task/data/`.

When Docker is available, a direct image build is:

```bash
docker build -t bilingual-search-posttrain:local environment/
```

## Execution environment

Directory follows the Harbor task layout: `task.toml`, `instruction.md`, `environment/Dockerfile`, `tests/test.sh`, and `tests/evaluate.py`. Harbor mounts `tests/` for the verifier, not the agent; keep it inaccessible during the run. Run with a Harbor version supporting this layout and expose exactly two NVIDIA RTX A6000 GPUs (48 GB VRAM each) to the task container through the execution backend. The agent timeout in `task.toml` is 18000 seconds (5 hours), covering preparation, training, experiments, and artifact saving, but not container setup or final verification. This timeout does not itself allocate GPUs; the organizer must provision the two devices. The Docker build downloads pinned Python dependencies and needs access to the PyTorch wheel index. The submitted model must fit in GPU memory alongside evaluation.

