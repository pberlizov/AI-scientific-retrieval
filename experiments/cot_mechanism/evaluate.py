"""Evaluate all four CoT conditions on GSM8K problems.

For each (problem, trace) pair, prompts the evaluator model to produce
a final answer. Extracts the number and compares to ground truth.

Key design note — broken condition:
  The broken traces were generated with a correct final "The answer is X" line.
  We strip that line before evaluation so the evaluator must derive its answer
  from the (wrong) intermediate steps alone. Without this, the evaluator would
  just copy the correct final line, making broken ≈ scribe trivially.

Supported providers (set via --provider):
  xai      — xAI Grok models (requires XAI_API_KEY)
  openai   — OpenAI models (requires OPENAI_API_KEY)
  anthropic — Anthropic Claude models (requires ANTHROPIC_API_KEY)
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CONDITIONS = ["direct", "scribe", "usurer", "broken", "permissive_direct"]

DIRECT_PROMPT = """Solve the following math problem. Give only the final numerical answer, nothing else.

Problem: {question}

Answer:"""

# Permissive variant: model may reason freely; we extract the final number.
# Used to check whether the strict direct prompt artificially suppresses accuracy.
PERMISSIVE_DIRECT_PROMPT = """Solve the following math problem. Show your work if helpful.
End your response with the final numerical answer on its own line.

Problem: {question}"""

TRACE_PROMPT = """You are shown a math problem and some prior reasoning steps. Using the
reasoning as context, give the final numerical answer only — do not restate the reasoning.

Problem: {question}

Reasoning:
{trace}

Final answer (number only):"""

_FINAL_ANSWER_RE = re.compile(
    r"\n?\s*[Tt]he answer is\s*[\$]?[\d,\.\-]+\.?\s*$"
)


def strip_final_answer(trace: str) -> str:
    """Remove trailing 'The answer is X' line from broken traces."""
    return _FINAL_ANSWER_RE.sub("", trace).rstrip()


def load_condition(traces_dir: Path, condition: str) -> list[dict]:
    path = traces_dir / f"{condition}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


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
    if nums:
        return nums[-1].replace(",", "").strip()
    return None


def numbers_match(pred: str | None, gt: str | None) -> bool:
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-3
    except ValueError:
        return pred.strip() == gt.strip()


def make_client(provider: str):
    if provider == "xai":
        from openai import OpenAI
        key = os.getenv("XAI_API_KEY")
        if not key:
            raise SystemExit("XAI_API_KEY not set.")
        return OpenAI(api_key=key, base_url="https://api.x.ai/v1")

    elif provider == "openai":
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY not set.")
        return OpenAI(api_key=key)

    elif provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise SystemExit("pip install anthropic to use the Anthropic provider.")
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY not set.")
        return anthropic.Anthropic(api_key=key)

    else:
        raise SystemExit(f"Unknown provider: {provider}. Use xai, openai, or anthropic.")


def evaluate_one(client, provider: str, model: str, question: str, trace: str | None) -> str:
    if trace is None:
        prompt = DIRECT_PROMPT.format(question=question)
    elif trace == "__permissive__":
        prompt = PERMISSIVE_DIRECT_PROMPT.format(question=question)
    else:
        prompt = TRACE_PROMPT.format(question=question, trace=trace)

    # Reasoning models and permissive_direct need token budget for output CoT
    is_reasoning = "reasoning" in model.lower()
    is_permissive = trace == "__permissive__"
    max_tokens = 1500 if (is_reasoning or is_permissive) else 50

    for attempt in range(3):
        try:
            if provider == "anthropic":
                resp = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text or ""
            else:
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt == 2:
                print(f"      API error: {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def evaluate_condition(
    client, provider: str, model: str,
    records: list[dict], condition: str, out_dir: Path
) -> list[dict]:
    slug = model.replace("/", "-").replace(".", "-")
    out_path = out_dir / f"{condition}_{slug}_results.jsonl"

    # Load already-completed IDs so we can resume without redoing work.
    done_ids: set[int] = set()
    existing: list[dict] = []
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r["id"])
                existing.append(r)
        if done_ids:
            print(f"  Resuming {condition}: {len(done_ids)} already done, "
                  f"{len(records) - len(done_ids)} remaining.")

    correct = sum(1 for r in existing if r["correct"])
    results = list(existing)

    pending = [r for r in records if r["id"] not in done_ids]
    total = len(records)

    print(f"\n  [{model}] condition={condition} ({total} problems, {len(pending)} to run)")

    # Open in append mode — completed records are never rewritten.
    with open(out_path, "a") as out_f:
        for rec in pending:
            trace = rec["trace"]
            if condition == "permissive_direct":
                trace = "__permissive__"
            elif condition == "broken" and trace:
                trace = strip_final_answer(trace)

            raw_response = evaluate_one(client, provider, model, rec["question"], trace)
            predicted = extract_number(raw_response)
            is_correct = numbers_match(predicted, rec["gt_answer"])
            if is_correct:
                correct += 1

            result = {
                "id": rec["id"],
                "model": model,
                "condition": condition,
                "question": rec["question"],
                "gt_answer": rec["gt_answer"],
                "trace_length": len(trace) if trace else 0,
                "raw_response": raw_response,
                "predicted": predicted,
                "correct": is_correct,
            }
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            done_so_far = len(done_ids) + len(results) - len(existing)
            if done_so_far % 20 == 0:
                print(f"    [{done_so_far}/{total}] running accuracy: {correct/done_so_far:.1%}")

            time.sleep(0.2)

    acc = correct / total if total else 0
    print(f"  → {acc:.1%} ({correct}/{total})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="experiments/cot_mechanism/traces")
    parser.add_argument("--out", default="experiments/cot_mechanism/results")
    parser.add_argument("--model", default=os.getenv("EVAL_MODEL", "grok-3-mini"),
                        help="Model to evaluate with")
    parser.add_argument("--provider", default="xai",
                        choices=["xai", "openai", "anthropic"],
                        help="API provider")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS,
                        choices=CONDITIONS, help="Which conditions to run")
    args = parser.parse_args()

    client = make_client(args.provider)
    traces_dir = Path(args.traces)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating with {args.provider}/{args.model}")
    for condition in args.conditions:
        records = load_condition(traces_dir, condition)
        evaluate_condition(client, args.provider, args.model, records, condition, out_dir)

    print(f"\nDone. Results in {out_dir}/")
    print("Run analyze.py to see the summary table.")


if __name__ == "__main__":
    main()
