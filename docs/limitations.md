# Limitations and non-claims

This repository is intentionally explicit about where its evidence ends.

1. **Synthetic task only.** A piecewise Bernoulli sequence is useful for a transparent falsification surface, but it does not reproduce sensory noise, non-stationary context, actions, costs, or a real experimental participant.

2. **Design-informed hazard.** The BOCPD hazard corresponds to the expected 200-trial segment length in the generator. That is defensible for a controlled demonstration but favorable information that a real deployment would need to estimate, validate, or robustly marginalise over.

3. **One primary baseline.** The fixed-rate delta rule is interpretable, but not the strongest possible comparator. Adaptive-rate learners, hidden Markov models, and learned recurrent predictors are natural follow-ups.

4. **No hyperparameter search claim.** This first version uses a single, documented configuration. It does not claim either model is globally tuned or fairly optimised across every environment.

5. **Latent tracking is simulation-specific.** Recovery time and tracking MAE use the known latent probability. In real data, that quantity may be unknown, estimated with error, or conceptually ill-defined.

6. **No neurocognitive conclusion.** Prediction error is used here as a computational motif. The code neither estimates EEG signals nor provides evidence about biological predictive coding, consciousness, or active inference.

7. **Descriptive uncertainty only.** Repeated random seeds provide robustness checks, not independent human participants or a substitute for an inferential study design.

8. **Sensitivity is not optimisation.** The v0.2 update-rate sweep makes a selected range visible, but it is not a preregistered protocol, a nested model-selection procedure, or evidence that any setting generalises outside the current generative environment.
