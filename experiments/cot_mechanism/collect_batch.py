"""Poll for and collect results from an OpenAI batch job.

Reads the batch ID (from --batch-id or results/batch_id.txt), waits until
complete, downloads results, and writes per-condition JSONL files in the
same format that analyze.py expects.

Usage:
    python3 experiments/cot_mechanism/collect_batch.py
    python3 experiments/cot_mechanism/collect_batch.py --batch-id batch_abc123
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

CONDITIONS = ["direct", "scribe", "usurer", "broken", "permissive_direct"]


def extract_number(text: str) -> str | None:
    text = text.strip()
    # Prefer explicit answer markers; take the LAST match to handle CoT with many intermediate = signs
    markers = re.findall(
        r"(?:final answer|the answer is|answer is|answer:)\s*[:\s]*\$?([\d,]+(?:\.\d+)?)",
        text, re.IGNORECASE,
    )
    if markers:
        return markers[-1].replace(",", "").strip()
    # Fall back to last number in text (works for responses that end with the answer)
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else None


def numbers_match(pred: str | None, gt: str | None) -> bool:
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-3
    except ValueError:
        return pred.strip() == gt.strip()


def load_traces(traces_dir: Path) -> dict[str, dict]:
    """Return {custom_id: record} for ground-truth lookup."""
    lookup = {}
    for condition in CONDITIONS:
        path = traces_dir / f"{condition}.jsonl"
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                lookup[f"{condition}_{rec['id']}"] = {
                    "condition": condition,
                    "id": rec["id"],
                    "question": rec["question"],
                    "gt_answer": rec["gt_answer"],
                    "trace_length": len(rec["trace"]) if rec["trace"] else 0,
                }
    return lookup


def poll_until_done(client: OpenAI, batch_id: str, poll_interval: int = 60) -> object:
    print(f"Polling batch {batch_id}...")
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = counts.total or 0
        completed = counts.completed or 0
        failed = counts.failed or 0

        print(f"  Status: {batch.status}  "
              f"({completed}/{total} done, {failed} failed)")

        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--traces", default="experiments/cot_mechanism/traces")
    parser.add_argument("--out", default="experiments/cot_mechanism/results")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between status checks (default 60)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="Model name to embed in result filenames")
    args = parser.parse_args()

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=key)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve batch ID
    batch_id = args.batch_id
    if not batch_id:
        id_path = out_dir / "batch_id.txt"
        if id_path.exists():
            batch_id = id_path.read_text().strip()
            print(f"Using saved batch ID: {batch_id}")
        else:
            raise SystemExit(
                "No --batch-id given and results/batch_id.txt not found. "
                "Run submit_batch.py first."
            )

    batch = poll_until_done(client, batch_id, args.poll_interval)

    if batch.status != "completed":
        raise SystemExit(f"Batch ended with status: {batch.status}")

    print("\nDownloading results...")
    result_content = client.files.content(batch.output_file_id).text
    raw_results = [json.loads(line) for line in result_content.strip().splitlines()]
    print(f"  {len(raw_results)} results received.")

    lookup = load_traces(Path(args.traces))
    slug = args.model.replace("/", "-").replace(".", "-")

    # Group by condition and write per-condition JSONL
    by_condition: dict[str, list] = {c: [] for c in CONDITIONS}

    for raw in raw_results:
        custom_id = raw["custom_id"]
        rec = lookup.get(custom_id)
        if rec is None:
            print(f"  Warning: unknown custom_id {custom_id}, skipping.")
            continue

        error = raw.get("error")
        if error:
            print(f"  Warning: {custom_id} failed — {error}")
            raw_text = ""
        else:
            raw_text = raw["response"]["body"]["choices"][0]["message"]["content"] or ""

        predicted = extract_number(raw_text)
        is_correct = numbers_match(predicted, rec["gt_answer"])

        by_condition[rec["condition"]].append({
            "id": rec["id"],
            "model": args.model,
            "condition": rec["condition"],
            "question": rec["question"],
            "gt_answer": rec["gt_answer"],
            "trace_length": rec["trace_length"],
            "raw_response": raw_text,
            "predicted": predicted,
            "correct": is_correct,
        })

    for condition, results in by_condition.items():
        if not results:
            continue
        results.sort(key=lambda r: r["id"])
        out_path = out_dir / f"{condition}_{slug}_results.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        correct = sum(1 for r in results if r["correct"])
        print(f"  {condition}: {correct}/{len(results)} correct "
              f"({correct/len(results):.1%}) → {out_path.name}")

    print("\nDone. Run analyze.py to see the summary table:")
    print(f"  python3 experiments/cot_mechanism/analyze.py --results {out_dir} --model {args.model}")


if __name__ == "__main__":
    main()
