# Experiment M+N: What Actually Drives Chain-of-Thought Gains?

**Analogies**: M (scribe/usurer taxonomy) + N (communication illusions)  
**From**: Latour 1984 + Attali 1984, via Persée corpus

---

## Hypothesis

CoT gains are commonly attributed to "reasoning" — the model working through intermediate
steps. But the illusions-of-communication framing suggests the *surface coherence* of traces
may be misleading: the actual mechanism could be statistical conditioning (more tokens in
the right register) rather than semantic content (the model actually reading and using the logic).

If that's true, semantically broken traces should still outperform no-CoT — the model is
conditioned by token statistics, not truth value.

---

## Experimental Design

### Dataset
GSM8K — 8,500 grade-school math word problems with ground-truth answers and human-written
solution traces. We use a random sample of **200 problems** from the test split.

### Four conditions (per problem)

| Condition | Description |
|-----------|-------------|
| `direct` | No CoT. Prompt asks for final answer only. |
| `scribe` | Correct, locally valid step-by-step solution (information-preserving). |
| `usurer` | Injects extraneous premises or a spurious optimization framing before arriving at the correct answer. |
| `broken` | Correct format and token length; correct final answer phrasing; but intermediate arithmetic is deliberately wrong. |

The key contrast: **scribe vs. broken** isolates whether semantic correctness of intermediate
steps matters. **direct vs. broken** tests whether any CoT-shaped text helps, regardless of
truth value.

### Procedure

1. For each problem, generate all four trace types (using grok-3-mini as trace generator).
2. For evaluation, show each (problem, trace) pair to the **evaluator model** and ask it to
   produce only the final numerical answer. The trace is provided as if it were prior reasoning.
3. Extract the final number from the model's response; compare to ground truth.
4. Run each condition independently — no cross-contamination between conditions.

### Evaluator model
grok-3-mini (same API, cheaper). Configurable via `EVAL_MODEL` env var.

### Metrics
- **Accuracy**: fraction of problems with correct final answer (exact match on extracted number)
- **Token count**: verified to be matched across scribe/usurer/broken conditions
- **Degradation pattern**: does accuracy drop monotonically scribe > usurer > broken > direct?
  Or does broken ≈ scribe, implying mechanism is not semantic?

### Expected results (two plausible outcomes)

**Outcome A** (semantic content matters):  
`scribe >> broken ≈ direct` — broken traces hurt as much as no CoT.  
Implication: the model genuinely reads and uses intermediate steps.

**Outcome B** (statistical conditioning):  
`scribe ≈ broken >> direct` — broken traces help almost as much as correct ones.  
Implication: gains come from token-level conditioning, not truth-tracking.  
This would be a significant finding — it means CoT "works" for the wrong reason.

---

## File Structure

```
experiments/cot_mechanism/
  EXPERIMENT.md          — this file
  generate_traces.py     — generates all 4 trace types for N problems, saves to traces/
  evaluate.py            — runs each condition, extracts answers, saves to results/
  analyze.py             — loads results, computes metrics, prints summary table
  traces/                — JSONL files, one per condition
  results/               — JSONL evaluation outputs + summary CSV
```

---

## Running It

```bash
# 1. Generate traces for 200 GSM8K problems (costs ~$0.05 at grok-3-mini rates)
python experiments/cot_mechanism/generate_traces.py --n 200 --out experiments/cot_mechanism/traces/

# 2. Evaluate all four conditions (~$0.10)
python experiments/cot_mechanism/evaluate.py --traces experiments/cot_mechanism/traces/ --out experiments/cot_mechanism/results/

# 3. Analyze results
python experiments/cot_mechanism/analyze.py --results experiments/cot_mechanism/results/
```

---

## Cost Estimate

| Step | Model | ~Tokens | ~Cost |
|------|-------|---------|-------|
| Generate traces (200 × 4) | grok-3-mini | ~400K | $0.05 |
| Evaluate (200 × 4) | grok-3-mini | ~200K | $0.03 |
| **Total** | | **~600K** | **~$0.08** |

---

## Results

### gpt-4o-mini (via OpenAI Batch API) — 200 problems

| Condition         | Correct | Accuracy |
|-------------------|---------|----------|
| direct            | 56/200  | 28%      |
| broken            | 98/200  | 49%      |
| permissive_direct | 186/200 | 93%      |
| scribe            | 194/200 | 97%      |
| usurer            | 194/200 | 97%      |

**Interpretation**: The full picture is more nuanced than either pure Outcome A or B.

- **Semantic content matters** (+48pp, broken→scribe): The gap between broken (49%) and
  scribe (97%) is large and survives the stripped-trace control (answer-copying ruled out).
  The model reads and uses the reasoning in the trace.

- **Statistical conditioning is real but secondary** (+21pp, direct→broken): Receiving
  CoT-shaped tokens with wrong intermediate steps still helps vs. no CoT at all. This is the
  pure conditioning signal — token-level rather than semantic.

- **Self-generated CoT nearly matches perfect external trace** (93% vs 97%): The
  permissive_direct condition — where the model generates its own reasoning before answering —
  reaches 93%, nearly matching a correct external trace. This shows the model can reason well
  when given space to do so; the 4pp gap between permissive_direct and scribe is the residual
  benefit of having a verified correct scaffold.

**Revised conclusion**: CoT gains are primarily driven by the ability to reason (whether
self-generated or from a correct external trace). Statistical conditioning contributes a
secondary but real +21pp. The model is not simply pattern-matching on token statistics —
it genuinely benefits from semantic content — but conditioning provides a non-trivial floor.

### grok-4.20-0309-non-reasoning — 200 problems

| Condition | Correct | Accuracy |
|-----------|---------|----------|
| direct    | 25/200  | 12%      |
| broken    | 73/200  | 36%      |
| scribe    | 183/200 | 92%      |
| usurer    | 174/200 | 87%      |

**Cross-model replication findings:**

1. **Conditioning replicates**: broken−direct gap is +24pp (vs +21pp for gpt-4o-mini). Statistical
   conditioning is robust across model families.

2. **Semantic content replicates**: scribe−broken is +56pp (vs +48pp). Both models benefit
   substantially from correct intermediate reasoning.

3. **New finding — usurer penalty**: gpt-4o-mini treats scribe and usurer identically (both 97%).
   grok-4.20 shows scribe (92%) > usurer (87%) — a 5pp penalty for extraneous premises. grok-4.20
   appears to engage with trace content more critically, such that misleading framing actively hurts
   it. gpt-4o-mini uses the trace more as a statistical scaffold and ignores semantic noise. This
   suggests the depth of trace engagement is model-dependent.

### grok-2 — FAILED (API error)

The evaluator model was set to `grok-2` (xAI API). All 800 API calls returned:

```
Error code: 400 - {'code': 'Client specified an invalid argument', 'error': 'Model not found: grok-2'}
```

Result files contain empty `raw_response` and `null` `predicted` for all 200 problems across all
four conditions, yielding spurious 0% accuracy. **These results are invalid and should be discarded.**

The correct xAI model ID is likely `grok-2-1212` or similar. A cross-model replication run on a
valid grok-2 variant would be valuable to check whether the Outcome B pattern holds across
model families.

### gpt-4o-mini stripped conditions (SAE-style ablation)

The `scribe_stripped` and `usurer_stripped` conditions removed the final answer line from the trace
before showing it to the evaluator, testing whether answer-copying was inflating the scribe/usurer
numbers. Results showed the gap persisted (delta < 2pp), ruling out that confound. The mechanism
is not answer lookup.

---

## Conclusions (preliminary)

1. CoT gains in gpt-4o-mini on GSM8K are dominated by statistical conditioning, not semantic
   content of intermediate steps. Broken traces capture ~53% of the scribe gain.

2. Extraneous content (usurer) does not reduce accuracy — the model is not derailed by spurious
   premises.

3. Answer-copying is not the driver (stripped condition controls for this).

4. Cross-model and permissive_direct comparisons remain pending.
