"""Submit all four CoT conditions as a single OpenAI batch job.

Builds one JSONL request file covering all 800 (condition × problem) pairs,
uploads it, and prints the batch ID to use with collect_batch.py.

Usage:
    python3 experiments/cot_mechanism/submit_batch.py --model gpt-4o-mini
"""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

CONDITIONS = ["direct", "scribe", "usurer", "broken", "permissive_direct"]

DIRECT_PROMPT = """Solve the following math problem. Give only the final numerical answer, nothing else.

Problem: {question}

Answer:"""

PERMISSIVE_DIRECT_PROMPT = """Solve the following math problem. Show your work if helpful.
End your response with the final numerical answer on its own line.

Problem: {question}"""

TRACE_PROMPT = """You are shown a math problem and some prior reasoning steps. Using the
reasoning as context, give the final numerical answer only — do not restate the reasoning.

Problem: {question}

Reasoning:
{trace}

Final answer (number only):"""

_FINAL_ANSWER_RE = re.compile(r"\n?\s*[Tt]he answer is\s*[\$]?[\d,\.\-]+\.?\s*$")


def strip_final_answer(trace: str) -> str:
    return _FINAL_ANSWER_RE.sub("", trace).rstrip()


def load_condition(traces_dir: Path, condition: str) -> list[dict]:
    path = traces_dir / f"{condition}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_request(custom_id: str, model: str, prompt: str, max_tokens: int = 50) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def build_batch_file(traces_dir: Path, model: str, conditions: list[str] | None = None) -> tuple[list[dict], Path]:
    requests = []

    for condition in (conditions or CONDITIONS):
        records = load_condition(traces_dir, condition)
        for rec in records:
            trace = rec["trace"]
            if condition == "broken" and trace:
                trace = strip_final_answer(trace)

            if condition == "permissive_direct":
                prompt = PERMISSIVE_DIRECT_PROMPT.format(question=rec["question"])
                max_tokens = 500
            elif trace is None:
                prompt = DIRECT_PROMPT.format(question=rec["question"])
                max_tokens = 50
            else:
                prompt = TRACE_PROMPT.format(question=rec["question"], trace=trace)
                max_tokens = 50

            # custom_id encodes condition + problem ID so collect_batch can reconstruct
            custom_id = f"{condition}_{rec['id']}"
            requests.append(build_request(custom_id, model, prompt, max_tokens))

    # Write to a temp file for upload
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix="cot_batch_"
    )
    for req in requests:
        tmp.write(json.dumps(req) + "\n")
    tmp.close()

    return requests, Path(tmp.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="experiments/cot_mechanism/traces")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--conditions", nargs="+", default=None, choices=CONDITIONS,
                        help="Which conditions to include (default: all)")
    args = parser.parse_args()

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=key)
    traces_dir = Path(args.traces)
    selected = args.conditions or CONDITIONS

    print(f"Building batch requests ({args.model}, {len(selected)} condition(s) × 200 problems)...")
    requests, batch_path = build_batch_file(traces_dir, args.model, selected)
    print(f"  {len(requests)} requests written to {batch_path}")

    print("Uploading batch file...")
    with open(batch_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    print(f"  File uploaded: {batch_file.id}")

    print("Submitting batch...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(f"\n  Batch ID: {batch.id}")
    print(f"  Status:   {batch.status}")
    print(f"\nSave this batch ID. When ready, collect results with:")
    print(f"  python3 experiments/cot_mechanism/collect_batch.py --batch-id {batch.id}")

    # Save batch ID to a file so you don't have to copy it manually
    label = "_".join(sorted(selected)) if selected != CONDITIONS else "all"
    id_path = Path(f"experiments/cot_mechanism/results/batch_id_{label}.txt")
    id_path.parent.mkdir(parents=True, exist_ok=True)
    id_path.write_text(batch.id)
    print(f"\n  (Also saved to {id_path})")
    print(f"\nCollect with:")
    print(f"  python3 experiments/cot_mechanism/collect_batch.py --batch-id {batch.id}")


if __name__ == "__main__":
    main()
