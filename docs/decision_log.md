# Decision log

## D001 — Start with a synthetic reversal benchmark

**Decision:** Build a minimal Bernoulli reversal benchmark before connecting the project to EEG, language, or behavioural data.

**Reason:** A synthetic environment makes the true contingency, change points, and expected failure mode observable. That lets the repository test the plumbing of prediction, update, evaluation, and reproducibility before making domain-specific claims.

**Consequence:** The initial repository has a narrow empirical scope. It earns credibility through transparency and a clear extension path rather than through prematurely broad claims.

## D002 — Keep notebooks out of the execution path

**Decision:** Put all executable benchmark logic in `src/`; notebooks may only import it.

**Reason:** A reviewer should be able to reproduce the committed result without hidden cell order, undeclared globals, or manual notebook intervention.

## D003 — Commit derived results, not every trial row

**Decision:** Commit compact summary tables, a representative trace, and the figure. Exclude optional full trial-level output by default.

**Reason:** The repository remains inspectable and light while preserving a one-command path to the complete generated data.

## D004 — Add an explicit update-rate sensitivity analysis

**Decision:** Add a versioned sweep over five fixed learning rates and five BOCPD hazards after the initial benchmark.

**Reason:** The first comparison depended on one documented value for each update-rate assumption. A sensitivity surface makes the robustness and the metric trade-offs visible without pretending that the resulting grid is a preregistered or globally optimal selection procedure.

**Consequence:** Version 0.2 reports a clearer, more defensible conclusion: the preferred setting depends on the metric, and all conclusions remain scoped to the synthetic reversal environment.
