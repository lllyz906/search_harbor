# Joint HotpotQA + MuSiQue post-training

## Objective and resources

You have 5 hours (18000 seconds) and two NVIDIA RTX A6000 GPUs (48 GB VRAM each). Post-train the supplied **Qwen3.5-4B** checkpoint at `/models/Qwen3.5-4B` jointly on HotpotQA and MuSiQue-Ans. Both are English multi-hop evidence QA tasks. Improve the model's ability to combine evidence, reject distractors, answer briefly, and identify the evidence used. This read-only checkpoint is the only permitted training starting point.

## Time budget

The 5-hour budget covers the entire agent execution, including data preparation, training, experiments, and saving the final artifact. Container setup and final verifier execution are outside this budget. You may use both GPUs, but multi-GPU training is optional; their memory is not a single unified 96 GB allocation.

## Input format and data rules

The input includes a question and numbered excerpts; each excerpt is already retrieved. No browser or external search is used at scoring time. Train the model to return only JSON in this form:

```json
{"answer": "short answer", "supporting_facts": [["title", 0]]}
```

For MuSiQue, use `supporting_paragraphs` instead, with integer paragraph indices. Do not include explanations or citations. Use only the supplied HotpotQA and MuSiQue training splits for supervised training; reserve held-out splits for validation. Do not modify the verifier or evaluation data.

## Submission and scoring

Submit a full Hugging Face `Qwen3_5ForConditionalGeneration` checkpoint at `/task/final_model`. A PEFT LoRA adapter is also accepted if it has `adapter_config.json` and weights and its `base_model_name_or_path` is `/models/Qwen3.5-4B`; the scorer loads it over that same checkpoint. The folder must contain model weights and configuration, not just a download script. Keep the complete model architecture, including the vision component, even though this task uses text only. The tokenizer is loaded from the supplied checkpoint. Scoring uses greedy decoding with `enable_thinking=False`. The only evaluation metric is joint F1, computed separately for HotpotQA and MuSiQue. The reward is `100 * (HotpotQA joint_f1 + MuSiQue joint_f1) / 2`; `grade.json` contains only these joint-F1 results and the aggregate score. Answers use lowercase/article/punctuation normalization, and MuSiQue aliases are accepted. Failed loading or missing artifacts score zero.

## Preloaded auxiliary datasets

Public datasets are preloaded under `/task/data/`: HotpotQA in `hotpot_qa/train.jsonl` and `hotpot_qa/test.jsonl`, and MuSiQue-Ans v1.0 in `musique/musique_ans_v1.0_train.jsonl` and `musique/musique_ans_v1.0_test.jsonl`. Use the training splits for post-training and reserve the held-out splits for validation. See `/task/data/README.md` for formats, provenance, and licenses. Loading JSONL requires only Python's standard library.

## Model and evaluation boundaries

Only train the stated base model; do not substitute an instruction-tuned model. The `/task` directory persists for submission. The evaluation set and aliases are private, and no evaluation output containing questions or answers is returned to the agent.
