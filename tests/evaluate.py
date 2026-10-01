"""Offline HotpotQA + MuSiQue evaluator.

The model returns JSON containing an answer and evidence identifiers. The
reward is the mean joint F1 across HotpotQA and MuSiQue.
"""
import argparse
import json
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path

BASE = "/models/Qwen3.5-4B"


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).lower()
    value = "".join(c for c in value if c not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_scores(prediction: str, gold: str):
    pred, truth = normalize_answer(prediction).split(), normalize_answer(gold).split()
    em = float(pred == truth)
    if not pred and not truth:
        return em, 1.0, 1.0, 1.0
    if not pred or not truth:
        return em, 0.0, 0.0, 0.0
    overlap = sum((Counter(pred) & Counter(truth)).values())
    precision, recall = overlap / len(pred), overlap / len(truth)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return em, precision, recall, f1


def support_scores(predicted: set, gold: set):
    if not predicted and not gold:
        return 1.0, 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0, 0.0
    overlap = len(predicted & gold)
    precision, recall = overlap / len(predicted), overlap / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(predicted == gold), precision, recall, f1


def joint_metrics(answer, support):
    answer_em, answer_p, answer_r, answer_f1 = answer
    support_em, support_p, support_r, support_f1 = support
    joint_p, joint_r = answer_p * support_p, answer_r * support_r
    joint_f1 = 2 * joint_p * joint_r / (joint_p + joint_r) if joint_p + joint_r else 0.0
    return {"joint_f1": joint_f1}


def parse_prediction(text: str):
    """Accept the required JSON and fall back to a plain answer."""
    text = text.strip()
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        value = json.loads(match.group(0) if match else text)
        return str(value.get("answer", "")), value.get(
            "supporting_facts", value.get("supporting_paragraphs", []))
    except (TypeError, ValueError, json.JSONDecodeError):
        return (text.splitlines()[0] if text else ""), []


def prompt(question, evidence, support_key):
    return (
        "Answer using the numbered evidence. Return only valid JSON with an "
        f"answer string and a {support_key} list. Use [title, sentence_index] "
        "pairs for supporting_facts and integer paragraph indices for "
        f"supporting_paragraphs.\nEvidence:\n{chr(10).join(evidence)}\n"
        f"Question: {question}\nJSON:"
    )


def hotpot_prompt(row):
    evidence = [f"[{title}#{i}] {sentence}"
                for title, sentences in row["context"]
                for i, sentence in enumerate(sentences)]
    return prompt(row["question"], evidence, "supporting_facts")


def musique_prompt(row):
    evidence = [f"[{p['idx']}] {p['title']}: {p['paragraph_text']}"
                for p in row["paragraphs"]]
    return prompt(row["question"], evidence, "supporting_paragraphs")


def normalize_support(value, dataset):
    result = set()
    if not isinstance(value, list):
        return result
    for item in value:
        if dataset == "hotpotqa" and isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                result.add((str(item[0]), int(item[1])))
            except (TypeError, ValueError):
                pass
        elif dataset == "musique":
            try:
                result.add(int(item))
            except (TypeError, ValueError):
                pass
    return result


def gold_support(row, dataset):
    if dataset == "hotpotqa":
        return {(str(title), int(index)) for title, index in row.get("supporting_facts", [])}
    return {int(p["idx"]) for p in row["paragraphs"] if p.get("is_supporting")}


def validate(rows):
    if not rows or {r.get("dataset") for r in rows} != {"hotpotqa", "musique"}:
        raise ValueError("Evaluation requires nonempty HotpotQA and MuSiQue subsets")
    ids = [r.get("id") for r in rows]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("Missing or duplicate item IDs")


def evaluate(data: Path, model_path: Path) -> dict:
    import torch
    from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer

    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate(rows)
    if not model_path.is_dir():
        raise ValueError("Missing final_model directory")
    if (model_path / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel
        config = PeftConfig.from_pretrained(str(model_path), local_files_only=True)
        if config.base_model_name_or_path != BASE:
            raise ValueError("Incorrect adapter base model")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            BASE, dtype=torch.bfloat16, device_map="auto", local_files_only=True)
        model = PeftModel.from_pretrained(model, str(model_path), local_files_only=True)
    else:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_path), dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model.eval()
    sums, counts = {d: Counter() for d in ("hotpotqa", "musique")}, Counter()
    for row in rows:
        dataset = row["dataset"]
        content = hotpot_prompt(row) if dataset == "hotpotqa" else musique_prompt(row)
        text = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                           truncation=True, max_length=3072).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**tokens, max_new_tokens=128, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(generated[0, tokens.input_ids.shape[1]:], skip_special_tokens=True)
        prediction, predicted_support = parse_prediction(decoded)
        gold_answers = [row["answer"], *row.get("answer_aliases", [])]
        candidates = [answer_scores(prediction, gold) for gold in gold_answers]
        answer = max(candidates, key=lambda value: value[3])
        support = support_scores(normalize_support(predicted_support, dataset), gold_support(row, dataset))
        sums[dataset].update(joint_metrics(answer, support))
        counts[dataset] += 1
    report = {"counts": dict(counts)}
    for dataset in ("hotpotqa", "musique"):
        report[dataset] = {key: value / counts[dataset] for key, value in sums[dataset].items()}
    report["score"] = 100 * (report["hotpotqa"]["joint_f1"] + report["musique"]["joint_f1"]) / 2
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.data, args.model)
    except Exception as exc:
        result = {"score": 0.0, "error": type(exc).__name__}
    args.output.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
