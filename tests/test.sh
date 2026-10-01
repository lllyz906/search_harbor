#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
python3 - <<'PY'
import json
from pathlib import Path

out = Path('/tmp/search_eval.jsonl')
hot = Path('/task/data/hotpot_qa/test_fullwiki.jsonl')
if not hot.exists():
    hot = Path('/task/data/hotpot_qa/test.jsonl')
mus = Path('/task/data/musique/musique_ans_v1.0_dev.jsonl')
if not mus.exists():
    mus = Path('/task/data/musique/musique_ans_v1.0_test.jsonl')
with out.open('w', encoding='utf-8') as dst:
    for dataset, path in [('hotpotqa', hot), ('musique', mus)]:
        with path.open(encoding='utf-8') as src:
            for line in src:
                row = json.loads(line)
                row['dataset'] = dataset
                row['id'] = f'{dataset}:{row.get("id", row.get("_id"))}'
                dst.write(json.dumps(row, ensure_ascii=False) + '\n')
PY
python3 /tests/evaluate.py --data /tmp/search_eval.jsonl --model /task/final_model --output /logs/verifier/grade.json
python3 -c 'import json; x=json.load(open("/logs/verifier/grade.json")); open("/logs/verifier/reward.txt","w").write(str(x["score"] / 100) + "\n")'
