# AI Scientific Retrieval

This project systematically retrieves techniques and results from historical scientific
literature and connects them to open problems in modern machine learning. The core hypothesis
is that formal methods developed before disciplinary boundaries hardened — in control theory,
information theory, operations research, measure theory, and elsewhere — contain ideas that
the ML field has not yet applied, either because the original sources are obscure or because
the connection requires crossing between fields.

## What it does

A pipeline ingests documents from historical scientific archives (DTIC, NASA NTRS, Persée,
JSTOR, and others), extracts structured summaries of techniques and results, and scores each
against a library of open problems in ML. High-scoring pairs are reviewed and, where the
analogy holds up, turned into concrete experiments.

The retrieval is not semantic search over abstracts. Each candidate analogy goes through a
structured evaluation: what is the transferable mathematical principle, what is the gap in the
modern literature it addresses, and what is the minimal experiment that would validate or
refute the connection. Analogies that pass this filter are logged in `data/promising_analogies.md`
with an explicit next step.

## Repository structure

```
data/
  promising_analogies.md      curated list of validated historical-to-modern connections
  problems/                   open ML problem descriptions used as retrieval targets

experiments/
  cot_mechanism/              chain-of-thought mechanism experiment (analogies M and N)
    paper/main.tex            paper draft
    results/                  raw accuracy data per condition and model

  superposition/              neural superposition experiment (analogies J and L)
    paper/main.tex            paper draft
    results/                  sweep data and Fano validation

  conjecture_gen/             automated conjecture generation for combinatorics (auxiliary)
    src/                      main loop, LLM agent, GNN reward model, evaluator
    results/                  conjecture log and evaluation output
```

## Experiments run so far

**Chain-of-thought mechanism** (from analogies M and N): A controlled experiment with five
prompt conditions on 200 GSM8K problems across three models. The main result is a decomposition
of CoT gains into statistical conditioning (+21--24 pp, consistent across model families) and
semantic content (+48--56 pp, the dominant term). A reasoning-native model achieves 75% on the
direct condition with no external trace, establishing architectural reasoning capacity as the
primary bottleneck rather than conditioning per se. Paper draft in
`experiments/cot_mechanism/paper/`.

**Feature superposition** (from analogies J and L): Fano's inequality applied to the weight
matrix Gram structure gives per-feature recovery error lower bounds that are empirically tight
at large k with zero violations. An additivity regularizer derived from the sigma-algebra
characterization of superposition-free representations reduces polysemanticity to near zero
at compression ratios k/d up to 4, outperforming post-hoc sparse autoencoders (which recover
0% of ground-truth features at the standard cosine threshold). Paper draft in
`experiments/superposition/paper/`.

## Promising analogies

Fifteen validated analogies are currently logged in `data/promising_analogies.md`, drawn from
roughly ten distinct historical sources. The most industry-applicable cluster centers on
real-time anomaly detection: three independent historical techniques (Wiener-Hopf recursions,
elimination-form covariance updates, and generalized-gradient constraint enforcement) converge
on the same gap in the modern literature — that ML-based detectors have no statistical
guarantees on false-alarm rate or detection delay. The most theoretically novel cluster
addresses causal structure learning via rank-preserving submatrix aggregation and
Marchenko-Pastur-calibrated skeleton screening.

The target is 50 analogies from at least 20 distinct sources. Corpus ingestion is ongoing.

## Requirements

Python 3.11+. See `requirements.txt` in each experiment subdirectory. API keys for xAI,
OpenAI, and Semantic Scholar are read from a `.env` file (not included in this repository).
