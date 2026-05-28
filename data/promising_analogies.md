# Promising Analogies — For Follow-up

Curated from digests as genuinely novel and actionable.
Return to these after corpus grows (target: 1000+ documents, 20+ distinct sources).

---

## A. Wiener-Hopf recursion → streaming anomaly detection with guarantees

**From digest**: 2026-05-25, proposal #3  
**Score**: 2.565 (similarity: 0.916)  
**Historical source**: [DTIC AD0007204: Optimum Controllers for Linear Closed-Loop Systems](https://archive.org/details/DTIC_AD0007204) (1953)  
**Modern problem**: `problem_029` — Real-time anomaly detection in high-dimensional multivariate time series

**Why it's interesting**: Modern ML anomaly detectors (autoencoders, VAEs, transformers) have no
sequential decision-theoretic guarantees — no controlled false-alarm rate, no minimax detection
delay. Classical CUSUM/SPRT have guarantees but don't scale to high dimensions. The 1953 paper's
core move — converting a stability-constrained quadratic optimization into a Wiener-Hopf filtering
problem — suggests a concrete bridge: use a Wiener-Hopf recursion on sliding covariance estimates
to build a sequential detector that inherits average-run-length control from the filtering theory.
This is practically deployable (sensor monitoring, network intrusion, grid fault detection) and the
theoretical gap it addresses is real and open.

**Concrete next step**: Implement a sliding-window Wiener-Hopf covariance estimator feeding a
CUSUM statistic; compare ARL and detection delay against autoencoder baseline on a multivariate
industrial dataset (e.g., SKAB, SMAP).

---

## B. Closed-loop → open-loop transformation → stationary MARL surrogate

**From digest**: 2026-05-25, proposal #7  
**Score**: 2.529 (similarity: 0.903)  
**Historical source**: [DTIC AD0007204: Optimum Controllers for Linear Closed-Loop Systems](https://archive.org/details/DTIC_AD0007204) (1953)  
**Modern problem**: `problem_056` — Convergence of multi-agent reinforcement learning

**Why it's interesting**: The core MARL non-convergence problem is that each agent's environment
is non-stationary because other agents are learning simultaneously — single-agent convergence
theorems require a fixed environment. The 1953 paper solves an analogous problem: a multiloop
closed-loop system (where each loop's behavior depends on all others) is algebraically converted
into an equivalent open-loop operator, making the optimization stationary. Applying this to MARL:
linearize the joint Bellman operator around the current joint policy to produce a stationary
surrogate that each agent optimizes locally, with a spectral-radius regularizer enforcing that
the joint update remains contractive. This is structurally distinct from mean-field approximations
(which assume exchangeability) and from opponent modeling (which requires predicting others'
policies explicitly). Not standard in the literature.

**Concrete next step**: Derive the linearized joint Bellman operator for 2-agent matrix games;
test whether optimizing each agent against the stationary surrogate + spectral regularizer
converges to Nash in cases where independent PPO diverges.

---

## C. LP cutting planes → neural network verification at scale

**From digest**: 2026-05-25, proposal #1  
**Score**: 2.616 (similarity: 0.934)  
**Historical source**: [DTIC AD0604648: Recent Advances in Linear Programming](https://archive.org/details/DTIC_AD0604648) (1955)  
**Modern problem**: `problem_033` — Neural network verification / robustness certification

**Why it's interesting**: The 1955 survey frames combinatorial selection as a target for LP
relaxation + cutting-plane refinement — discrete constraints are encoded as linear inequalities
whose successive tightening yields both feasibility and dual bounds without exhaustive search.
Neural network verification faces exactly the same structure: exponentially many ReLU activation
patterns generate an exponential number of linear regions, and current LP/MIP solvers time out
on million-neuron networks. The cutting-plane oracle angle — deriving valid inequalities from
joint activation patterns of nearby neurons — is not the current approach (existing tools use
fixed relaxations like Planet or α-CROWN). The 1955 paper's large-scale LP decomposition ideas
suggest a sliding-window oracle that generates activation-pattern cuts dynamically inside a
branch-and-cut loop.

**Concrete next step**: Implement a joint-activation cutting-plane oracle for 2-3 adjacent ReLU
neurons; embed inside α-CROWN or a branch-and-bound verifier; measure optimality gap reduction
on VNN-COMP benchmarks at 500K–2M neuron scale.

---

## D. Gaussian elimination form → streaming sparse covariance with chi-squared control

**From digest**: 2026-05-25, proposal #2  
**Score**: 2.615 (similarity: 0.934)  
**Historical source**: [DTIC AD0604711: The Elimination Form of the Inverse](https://archive.org/details/DTIC_AD0604711) (1955)  
**Modern problem**: `problem_029` — Real-time anomaly detection in high-dimensional multivariate time series

**Why it's interesting**: The 1955 paper's key move is maintaining an inverse as a product of
elementary elimination matrices rather than a dense object, so update cost scales with sparsity
not dimension. Applied to streaming covariance: maintain a product-form Cholesky factor of the
precision matrix, updating only the rows/columns affected by each new observation. The anomaly
score is the squared residual, which remains exactly chi-squared under the null — giving
distribution-free false-alarm control without retraining. This addresses a genuine gap:
existing high-dimensional detectors either destroy sparsity (losing statistical optimality) or
become cubic per update (losing real-time feasibility). The chi-squared control limit is what
neither autoencoders nor classical CUSUM provide simultaneously.

**Concrete next step**: Implement product-form Cholesky updater for sparse precision matrices;
measure ARL and detection delay vs. autoencoder baseline on SKAB/SMAP benchmarks.
*(Note: shares the same target problem as entry A — the two approaches could be compared or combined.)*

---

## E. mean-square error minimization → SDP lifting via constraint-preserving equivalence

**From digest**: 2026-05-25, proposal #4  
**Score**: 2.606 (similarity: 0.931)  
**Historical source**: [DTIC AD0007204: Optimum Controllers for Linear Closed-Loop Systems](https://archive.org/details/DTIC_AD0007204) (1953)  
**Modern problem**: `problem_052` — Scalable semidefinite programming / sum-of-squares relaxations

**Why it's interesting**: The 1953 paper converts a stability-constrained multiloop design into
an equivalent open-loop quadratic program whose stationary solution is analytically tractable.
The transferable principle is a constraint-preserving equivalence transformation that relocates
an optimization into a domain where a tractable criterion applies. For SDPs: construct an
algebraic lifting that maps a given SDP (or Burer-Monteiro factorization) into an auxiliary
quadratic program whose objective is exactly the trace inner-product, while encoding PSD and
rank constraints as linear side constraints. The step-size schedule for a first-order method
can then be derived from the spectral properties of the lifted operator. This is structurally
different from standard Burer-Monteiro: the lifted quadratic program has a flat landscape by
construction, eliminating spurious local minima rather than just hoping for them to be absent.

**Concrete next step**: Derive the algebraic lifting for 3×3 SDPs; verify that optimizing the
auxiliary QP + spectral regularizer recovers feasible SDP solutions on instances where
Burer-Monteiro gets stuck.

---

## F. Gaussian elimination ordering → sparse causal discovery

**From digest**: 2026-05-25, proposal #5  
**Score**: 2.605 (similarity: 0.930)  
**Historical source**: [DTIC AD0604711: The Elimination Form of the Inverse](https://archive.org/details/DTIC_AD0604711) (1955)  
**Modern problem**: `problem_008` — Causal structure learning from observational data

**Why it's interesting**: Causal discovery algorithms either optimize over dense adjacency
matrices (NOTEARS-style, costly) or do exhaustive conditional independence tests (exponential).
The elimination form idea suggests a third path: maintain the current partial ancestral graph
as a product of local elimination operators, where each operator records only the parents and
children involved in removing one variable. At each step solve a low-dimensional sub-problem
bounded by the current frontier width. This converts structure learning into an ordered sparse
factorization — similar in spirit to junction tree methods, but with cost scaling from the
elimination sequence rather than treewidth, making it explicitly controllable via variable
ordering heuristics.

**Concrete next step**: Implement elimination-ordered causal discovery on synthetic sparse DAGs
(p=500, average degree 3); compare runtime and SHD against NOTEARS and PC algorithm.

---

## G. Principal submatrices → causal discovery via rank-preserving subsampling

**From digest**: 2026-05-25, proposal #7  
**Score**: 2.595 (similarity: 0.927)  
**Historical source**: [Principal Submatrices of a Full-Rowed Non-Negative Matrix](https://archive.org/details/jresv63Bn1p19) (1959)  
**Modern problem**: `problem_008` — Causal structure learning from observational data

**Why it's interesting**: The 1959 result shows that rank and linear dependence are preserved
under simultaneous deletion of matching rows and columns — global matrix properties are exactly
recoverable from an ensemble of small, symmetrically induced submatrices. Applied to causal
discovery: draw random index sets S of size k ≪ p, run an exact causal learner on the
principal submatrix of the empirical covariance on S, and aggregate local DAGs via consistency
vote. The rank-preservation property provides a principled justification (rather than a
heuristic) that the local subproblems retain the global Markov properties needed for consistent
stitching. This is distinct from existing ensemble causal methods which aggregate without
algebraic guarantees.

**Concrete next step**: Implement the subsampling aggregator; derive the acceptance threshold
from the rank-preservation property; test on 1000-node sparse DAGs and compare to NOTEARS and
PC algorithm at the same computational budget.

---

## H. Gaussian elimination form → sparse SDP solver via product-of-factors updates

**From digest**: 2026-05-25, proposal #9  
**Score**: 2.592 (similarity: 0.926)  
**Historical source**: [DTIC AD0604711: The Elimination Form of the Inverse](https://archive.org/details/DTIC_AD0604711) (1955)  
**Modern problem**: `problem_052` — Scalable semidefinite programming / sum-of-squares relaxations

**Why it's interesting**: Interior-point SDP solvers must repeatedly form and invert dense Schur
complements, destroying the aggregate sparsity of SOS/robustness constraints and producing cubic
or worse scaling. The elimination form idea: replace the dense Newton step with an
"elimination-form" operator maintained as a product of sparse elementary matrices whose sparsity
pattern is inherited from the chordal completion or term-sparsity graph of the SOS instance.
At each iteration only the nonzero entries are updated via rank-one corrections — per-iteration
cost becomes linear in nonzeros rather than cubic in dimension. Related to chordal SDP
decompositions (e.g., CHOMPACK) but the elimination product form gives a finer-grained update
rule rather than a block decomposition.

**Concrete next step**: Implement elimination-form Schur complement updates for chordally sparse
SDPs; benchmark against MOSEK and SCS on SOS certification instances from polynomial
optimization benchmarks.
*(Note: same historical source as entries D and F — the 1955 elimination paper has unusually broad reach.)*

---

## I. Correlation templates + Marchenko-Pastur → RMT-calibrated causal skeleton screening

**From digest**: 2026-05-25, proposal #10  
**Score**: 2.584 (similarity: 0.923)  
**Historical source**: [DTIC AD0419094: Pattern Recognition with Self-Organizing Machines](https://archive.org/details/DTIC_AD0419094) (1963)  
**Modern problem**: `problem_008` — Causal structure learning from observational data

**Why it's interesting**: The 1963 paper achieves pattern recognition via a single non-iterative
correlation pass — no search, no iteration, explicit noise tolerance. Applied to causal
discovery: form the empirical covariance once, treat each column as a "template," and retain
only entries whose absolute correlation exceeds a threshold derived from the Marchenko-Pastur
law (the null distribution of correlations under no true dependence). Feed this sparse support
as a differentiable mask or warm-start for subsequent continuous optimization. The PC algorithm
already does something like this in phase 1, but uses a fixed p-value threshold rather than an
RMT-derived one calibrated to dimensionality. The RMT calibration is the actionable novelty —
it makes the screening provably conservative in the high-dimensional regime where p ≈ n.

**Concrete next step**: Replace the PC skeleton phase's independence test with a
Marchenko-Pastur-thresholded correlation screen; measure FPR/TPR on high-dimensional synthetic
DAGs (p=1000, n=500) where p ≈ n makes standard tests unreliable.

---

## J. Fano bounds → information-theoretic theory of neural superposition

**From digest**: 2026-05-25, proposal #11  
**Score**: 2.584 (similarity: 0.923)  
**Historical source**: [The Transmission of Information](https://archive.org/details/fano-tr65.7z) (1949)  
**Modern problem**: `problem_015` — Superposition and polysemanticity in neural networks

**Why it's interesting**: Mechanistic interpretability has identified superposition empirically
(one neuron represents multiple features) but has no quantitative theory predicting *when* it
should occur or *how many* features can be packed before decoding fails. Fano's 1949 formulation
predicts exactly this kind of phase transition: given a channel (weight matrix) and a source
distribution (feature frequencies), Fano's inequality bounds the probability of recovery error
as a function of mutual information. Applying this: formalize each feature as a "message,"
estimate its prior probability from the data, and derive a Fano-style bound on the probability
that a linear probe recovers it from the representation vector. This gives a first-principles
prediction of the superposition threshold — a number of features beyond which error must exceed
a given level — which could then be used as a training regularizer. This is a genuine gap: no
existing interpretability paper derives a capacity bound for superposition from first principles.

**Concrete next step**: On a toy superposition model (Elhage et al. 2022 setup), compute the
Fano bound on feature recovery error as a function of representation dimension and feature
frequency; verify that the bound predicts the empirical phase transition where probing accuracy
drops.

---

## K. Measure theory → additive regret bounds for adversarial concept drift

**From digest**: 2026-05-26, proposal #1  
**Score**: 3.018 (similarity: 0.898)  
**Historical source**: [Introduction à la théorie de la mesure](https://www.persee.fr/doc/hism_0982-1783_1988_num_3_1_1286) (1988, FR — Persée)  
**Modern problem**: `problem_019` — Online learning under concept drift

**Why it's interesting**: Existing regret analyses for concept drift assume either that changes
are statistically detectable or that the process between changes is stationary enough for standard
adaptation to apply. Once an adversary can reshape both the timing and the geometry of the shift,
both assumptions collapse. The 1988 paper's core move — treating statistical modeling as the
construction of an axiomatic sigma-algebra that licenses consistent numerical inference regardless
of the empirical status of the measured quantities — suggests a direct fix: equip the hypothesis
class with a sigma-algebra whose atoms are "drift events" and define an additive measure on it.
Any online algorithm whose updates respect additivity then accumulates regret linear in the
measure of the realized drift set, not in the number or magnitude of changes. This gives
distribution-free guarantees without detection or stationarity.

**Concrete next step**: Formalize the drift sigma-algebra for a 1D concept-drift toy problem;
derive whether additivity-respecting online gradient descent achieves sublinear regret against
an adversary who picks drift events from a fixed measure-zero set.

---

## L. Feature sigma-algebra → axiomatic test for superposition in neural networks

**From digest**: 2026-05-26, proposal #2  
**Score**: 2.997 (similarity: 0.892)  
**Historical source**: [Introduction à la théorie de la mesure](https://www.persee.fr/doc/hism_0982-1783_1988_num_3_1_1286) (1988, FR — Persée)  
**Modern problem**: `problem_015` — Superposition and polysemanticity in neural networks

**Why it's interesting**: The strongest new entry. Mechanistic interpretability studies
superposition empirically (cosine similarity, activation statistics) without an axiomatic
criterion for when a set of features can be treated as additively measurable. The 1988 paper
constructs exactly that: a sigma-algebra that licenses additive measurement of otherwise
non-classical sets. Applied to networks: define a "feature sigma-algebra" on the activation
manifold as the smallest sigma-algebra that renders every decoder direction measurable, then
construct an additive measure whose value on a union of features equals the sum of individual
measures *precisely when their supports are superposition-free*. An auxiliary training loss that
penalizes deviation from additivity on randomly sampled disjoint feature unions would be
architecture-agnostic and provides the first principled (not heuristic) criterion for
superposition. Nothing like this exists in the interpretability literature.

**Concrete next step**: On the Elhage et al. (2022) toy superposition model, define the feature
sigma-algebra explicitly and measure whether the additivity penalty reduces polysemanticity while
preserving model performance; compare against orthogonality and sparsity baselines.

---

## M. Scribe vs. usurer → taxonomy of chain-of-thought intermediaries

**From digest**: 2026-05-26, proposal #5  
**Score**: 2.757 (similarity: 0.821)  
**Historical source**: [Le scribe et l'usurier (Pour une sociologie des médiations)](https://www.persee.fr/doc/reso_0751-7971_1984_num_2_8_1134) (1984, FR — Persée)  
**Modern problem**: `problem_046` — Chain-of-thought reasoning in language models

**Why it's interesting**: The 1984 sociology paper distinguishes "scribes" (mediators that
transmit without imposing external cost — information-preserving rewrites) from "usurers"
(mediators that insert an obligatory rent that reshapes the terms of the exchange). CoT research
currently treats all intermediate tokens as interchangeable and cannot predict when a step
enlarges the solution space vs. silently biases it. The scribe/usurer taxonomy applied to CoT
traces gives a concrete typology: scribe steps are local, truth-preserving rewrites; usurer steps
introduce an auxiliary objective or extraneous premise. This is testable — construct matched
trace families for the same arithmetic/planning problems, one purely scribe-style and one
usurer-style, and measure differential effects on accuracy, calibration, and attention patterns.

**Concrete next step**: Generate matched scribe vs. usurer trace pairs for GSM8K problems;
measure final accuracy and attention-head entropy differences; check whether usurer-style traces
systematically hurt calibration even when they improve surface fluency.

---

## N. Communication illusions → controlled experiment on CoT mechanism

**From digest**: 2026-05-26, proposal #7  
**Score**: 2.748 (similarity: 0.818)  
**Historical source**: [Les illusions de la communication](https://www.persee.fr/doc/reso_0751-7971_1984_num_2_8_1136) (1984, FR — Persée)  
**Modern problem**: `problem_046` — Chain-of-thought reasoning in language models

**Why it's interesting**: The 1984 paper argues that apparent communicative coherence is sustained
by systematic illusions — the surface of messages misleads observers about the actual causal
mechanisms producing effects. Applied to CoT: current accounts can't isolate whether gains come
from semantic content, statistical conditioning irrespective of content, or sheer forward-pass
expansion. The "illusion" framing suggests a decisive experiment: construct CoT traces that
preserve token length, syntactic structure, and attention statistics but deliberately violate
local semantic coherence (correct intermediate steps followed by logically inconsistent
conclusions, or fluent nonsense). If semantically broken traces still improve performance, the
mechanism is statistical conditioning, not genuine self-correction — a major result with direct
implications for interpretability and for when CoT can be trusted.

**Concrete next step**: Generate semantically broken CoT traces for multi-step arithmetic (correct
format, wrong intermediate logic); test on GPT-4 and Llama-3 with token-count control; report
accuracy delta vs. no-CoT baseline.
*(Closely related to entry M — could be run as a single experiment comparing scribe/usurer/broken traces.)*

---

## O. Generalized gradients → constraint-enforcing streaming covariance updates

**From digest**: 2026-05-26, proposal #11  
**Score**: 2.632 (similarity: 0.940)  
**Historical source**: [NASA NTRS 19730016924: An Innovative Approach to Compensator Design](https://archive.org/details/NASA_NTRS_Archive_19730016924) (1973, EN)  
**Modern problem**: `problem_029` — Real-time anomaly detection in high-dimensional multivariate time series

**Why it's interesting**: The 1973 NASA paper turns an infinite-dimensional stability criterion
(open-loop frequency response) into a finite collection of scalar constraint functions, then uses
generalized gradients of active constraint violations to adjust only the minimal parameters needed
to restore feasibility. The analogy: parameterize a streaming precision-matrix estimator so that
its implied quadratic forms at a small set of "probe lags" are explicit scalar functions; when an
empirical false-alarm threshold is crossed, compute generalized gradients of those functions and
perform a low-rank update that restores the constraint while preserving positive-definiteness.
The result is a streaming covariance estimator with active false-alarm control — not a post-hoc
tuning knob but a constraint enforced in the update step itself. This closes a real gap: existing
shrinkage/factor estimators optimize statistical risk but have no mechanism for sequentially
controlling alarm rates as the covariance drifts.
*(Complements entries A and D, which address the same problem via different mechanisms.)*

**Concrete next step**: Implement the probe-lag constraint formulation for a 50-dimensional
precision matrix; compare false-alarm rate control against shrinkage baselines (Ledoit-Wolf,
factor model) on a synthetic non-stationary sensor stream.

---

*Last updated: 2026-05-26*

*Source concentration note: Entries D, F, H trace to the same 1955 elimination paper (AD0604711);
A, B, E to the same 1953 controls paper (AD0007204); K, L to a 1988 French measure theory paper;
M, N to 1984 French communication sociology papers. Diversity will improve past 1000 docs.*

*Progress toward goal: 15 analogies from ~10 distinct historical sources. Target: 50 from 20+.*
