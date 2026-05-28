"""Generate CoT traces for the four experimental conditions.

Downloads GSM8K test split, samples N problems, generates:
  - direct:  no trace (placeholder so all conditions have same schema)
  - scribe:  correct step-by-step solution
  - usurer:  injects extraneous framing before correct answer
  - broken:  correct format/length, deliberately wrong intermediate arithmetic

Saves one JSONL per condition to --out directory.
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

TRACE_MODEL = os.getenv("TRACE_MODEL", "grok-3-mini")
XAI_BASE = "https://api.x.ai/v1"
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math"
    "/master/grade_school_math/data/test.jsonl"
)


# ── Prompts ────────────────────────────────────────────────────────────────

SCRIBE_SYSTEM = """You are a careful math tutor. Solve the problem step by step.
Each step must be correct and follow directly from the previous one.
End with: "The answer is <number>."
Be concise — no more than 8 steps."""

USURER_SYSTEM = """You are a math tutor who over-complicates problems.
Before solving, reframe the problem in terms of optimization or cost minimization
(even if that framing is unnecessary). Then solve correctly.
End with: "The answer is <number>."
Keep total length similar to a direct solution."""

BROKEN_SYSTEM = """You are a math tutor. Write a step-by-step solution that LOOKS correct
but contains wrong intermediate arithmetic. The steps should be plausible and fluent.
You MUST get the intermediate calculations wrong (use wrong numbers in at least 2 steps).
Despite the wrong intermediate steps, end with: "The answer is <number>."
where <number> is the CORRECT answer to the original problem.
Keep total length similar to a direct solution."""


def load_gsm8k(n: int, seed: int = 42) -> list[dict]:
    print(f"Downloading GSM8K test split...")
    r = requests.get(GSM8K_URL, timeout=30)
    r.raise_for_status()
    problems = [json.loads(line) for line in r.text.strip().splitlines()]
    random.seed(seed)
    sample = random.sample(problems, min(n, len(problems)))
    print(f"  Sampled {len(sample)} problems from {len(problems)} total.")
    return sample


def extract_answer(text: str) -> str | None:
    """Extract ground truth answer from GSM8K '#### <number>' format."""
    m = re.search(r"####\s*([\d,\.\-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


def call_model(client: OpenAI, system: str, user: str, max_tokens: int = 400) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=TRACE_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt == 2:
                print(f"    Failed after 3 attempts: {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def generate_all(problems: list[dict], client: OpenAI) -> list[dict]:
    records = []
    for i, prob in enumerate(problems):
        question = prob["question"]
        gt_raw = prob["answer"]
        gt_answer = extract_answer(gt_raw)

        print(f"  [{i+1}/{len(problems)}] {question[:60]}...")

        scribe = call_model(client, SCRIBE_SYSTEM, question)
        time.sleep(0.3)
        usurer = call_model(client, USURER_SYSTEM, question)
        time.sleep(0.3)
        broken = call_model(client, BROKEN_SYSTEM, question)
        time.sleep(0.3)

        records.append({
            "id": i,
            "question": question,
            "gt_answer": gt_answer,
            "traces": {
                "direct": None,
                "scribe": scribe,
                "usurer": usurer,
                "broken": broken,
            },
        })

    return records


def save_by_condition(records: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = ["direct", "scribe", "usurer", "broken"]
    for cond in conditions:
        path = out_dir / f"{cond}.jsonl"
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps({
                    "id": rec["id"],
                    "question": rec["question"],
                    "gt_answer": rec["gt_answer"],
                    "trace": rec["traces"][cond],
                }) + "\n")
        print(f"  Saved {len(records)} records → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of problems to sample")
    parser.add_argument("--out", type=str, default="experiments/cot_mechanism/traces")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise SystemExit("XAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key, base_url=XAI_BASE)
    problems = load_gsm8k(args.n, seed=args.seed)

    print(f"\nGenerating traces with {TRACE_MODEL}...")
    records = generate_all(problems, client)

    print(f"\nSaving...")
    save_by_condition(records, Path(args.out))
    print("Done.")


if __name__ == "__main__":
    main()
