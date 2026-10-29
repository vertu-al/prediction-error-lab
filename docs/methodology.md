# Methodology

## Question and design

The benchmark asks whether a learner that represents uncertainty over recent run length can track an abruptly changing Bernoulli contingency better than a constant-rate learner. Each seed generates one sequence of 800 binary outcomes from four contiguous segments:

| Trials | Latent probability of outcome 1 |
| ---: | ---: |
| 0–199 | 0.80 |
| 200–399 | 0.20 |
| 400–599 | 0.75 |
| 600–799 | 0.30 |

Both learners receive precisely the same observation sequence within a seed. The simulation runs 64 fixed seeds (`0` through `63`), and the full parameter set is versioned in `configs/reversal_benchmark.json`.

## Learners

### Fixed-rate delta rule

The baseline predicts a Bernoulli probability \(v_t\) and updates it according to

\[
v_{t+1}=v_t+\alpha(o_t-v_t),
\]

where \(o_t\in\{0,1\}\) is the observed outcome. The learning rate is \(\alpha=0.08\); the initial prediction is 0.5.

### Hazard-aware Bayesian online change-point detection

The BOCPD learner maintains a posterior distribution over run lengths. For each possible run length \(r\), a Beta posterior \(\operatorname{Beta}(a_r,b_r)\) provides the next-trial prediction:

\[
p(o_{t+1}=1\mid o_{1:t}) = \sum_r p(r_t=r\mid o_{1:t}) \frac{a_r}{a_r+b_r}.
\]

A constant hazard \(H=0.005\) assigns a prior chance of a new segment at each trial. This corresponds to an expected run length of 200 trials, matching the *generative design* of this first controlled benchmark. The initial Beta prior is uniform, \(\operatorname{Beta}(1,1)\), and run-length support is truncated at 250 to keep computation bounded.

This parameter choice is deliberately visible rather than treated as an implicit implementation detail. It is not a deployment-ready calibration strategy; see [limitations](limitations.md).

## Evaluation protocol

For every learner and seed, the program stores the prediction made *before* each observation and computes:

| Metric | Definition | Interpretation |
| --- | --- | --- |
| Brier score | Mean \((p_t-o_t)^2\) | Probabilistic accuracy; lower is better. |
| Negative log-likelihood | Mean binary cross-entropy | Penalises confident wrong predictions; lower is better. |
| Latent tracking MAE | Mean \(|p_t-q_t|\) for simulated latent \(q_t\) | Diagnostic tracking error; unavailable in unlabelled real data. |
| Recovery time | First post-change trial whose following ten predictions remain within 0.10 of new \(q_t\) | Stability after reversal; lower is better. |

Each sequence has three changes, yielding 192 learner-specific recovery observations (64 seeds × 3 reversals). Summary files contain seed-level outcomes, individual recovery observations, and descriptive 95% intervals. Those intervals are summaries, not a basis for p-values or strong inferential claims.

## Reproduction contract

The committed result is defined by:

1. package version `0.2.0`;
2. `configs/reversal_benchmark.json`;
3. deterministic NumPy local random generators;
4. the command in the root README.

`results/experiment_log.csv` records the SHA-256 hash of the exact config used to generate committed tables and the figure.

## Update-rate sensitivity analysis (v0.2)

The primary benchmark fixes \(\alpha=0.08\) and \(H=0.005\). Version 0.2 adds a robustness grid, configured in `configs/sensitivity_benchmark.json`:

| Learner | Evaluated values |
| --- | --- |
| Fixed-rate delta rule | \(\alpha \in \{0.03, 0.05, 0.08, 0.10, 0.15\}\) |
| Hazard-aware BOCPD | \(H \in \{0.001, 0.0025, 0.005, 0.01, 0.02\}\) |

Every grid condition receives the same outcome stream for a given seed. The sweep varies one model-family-specific update-rate assumption at a time, so it is a robustness analysis rather than a globally fair hyperparameter search between families. The result files retain the parameter name and value beside every seed-level and aggregate metric.

The grid was added after the initial v0.1 benchmark, so it is explicitly not described as preregistration. Its role is to make the dependency of conclusions on the update-rate assumption visible and reproducible.
