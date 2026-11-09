---
name: rfr-gpr-model-builder
description: Train, save, compare (train/test RMSE) and reuse Random Forest (RFR), Gaussian Process (GPR), Kernel Ridge (KRR) and MLP regression models on tabular CSV data, with user-chosen targets and training variables; also serves the lab's two saved film-quality and EDS-ratio models, and searches independently for better models when given a task. Supports ONLY RFR, GPR, KRR and MLP.
---

# RFR, GPR, KRR and MLP model-builder skills

Scope: **only** `rfr` (scikit-learn `RandomForestRegressor`), `gpr`
(`GaussianProcessRegressor`), `krr` (`KernelRidge`) and `mlp` (`MLPRegressor`, one hidden layer). If the user asks for any other model family
(neural net, XGBoost, linear, SVM, ...), say it is not supported here and offer
RFR, GPR, KRR or MLP instead. Never fabricate results for unsupported models.

## 1. Required inputs -- never guess these
- **Target(s)**: the column(s) to predict.
- **Training variables (features)**: the columns used as inputs.

If either is missing or ambiguous, call `describe_training_data`, show the
numeric columns, and **ask the user** to choose. Do not infer them from column
names, and do not call `train_models` until the user has answered. Only numeric
columns are usable. A column cannot be both target and feature.

Everything else has a default; do not ask about it unless the user brings it up:
model types = RFR, GPR, KRR and MLP, test_size = 0.25, random_state = 42.

## 2. Workflow
1. Resolve target(s) and features (section 1).
2. `train_models` -- one model per (target, model type), all sharing the same
   train/test split so they are comparable. Models are saved automatically.
3. `compare_models` -- always follow training with a comparison and present a
   markdown table of model_id, type, train RMSE, test RMSE, test/train ratio.
   State which model has the lowest test RMSE for each target.
4. Training several variants (e.g. a shallow and a deep RFR, different GPR
   kernels, KRR with different kernels, or MLPs of different sizes) = several `train_models` calls with a different `name_prefix`, then
   one `compare_models`.
5. Use saved models only when asked: `predict_with_model`,
   `optimize_with_model`, `get_model_info`, `list_models`.

## 2b. The lab's two saved models (use these for lab predictions)
Trained on the 17 initial growths and tested on the 4 later ones:
- `lab_RHEED_Quality_Film` -- film quality (0-100, %). An MLP (3 nodes, alpha 1,
  batch size 4) with **10 inputs**: `growthtime` (s), `filamentpower` (W),
  `flux_ratio` (unitless), `Substrate_quality` (%), plus the raw substrate numbers
  `subcfwhm`, `sublfwhm`, `subrfwhm`, `submean_width`, `subwidth_variance`,
  `subcurve_variance`.
- `lab_EDS_ratio` -- Fe:Sn stoichiometry ratio (unitless, ideal 3.0). A Random
  Forest (100 trees, depth 3) with **4 inputs**: `growthtime`, `filamentpower`,
  `flux_ratio`, `Substrate_quality`.

A request such as "predict the film quality and EDS ratio for ..." is answered
with one `predict_with_model` call per lab model -- no training. The substrate
values come from a RHEED image via the main agent's `rheed_quality` tool, which
returns all 7 of them. If a film-quality request lacks any of the 10 inputs, ask
for the missing values or a substrate RHEED image path. Never invent or
approximate the raw substrate numbers; you can still give the EDS ratio from the
4 inputs. Report each prediction with its `uncertainty_std` and any
`extrapolation_warning`.

Be honest about accuracy. Film-quality errors are typically near 15 points and the
best films tend to be under-predicted. `uncertainty_std` is rough and far smaller
than the real error, so never present it as an error bar. EDS test values for 2 of
the 4 later growths are imputed, so its test result is weak evidence.

They are listed by `list_models` (`built_in: true`) and can be compared, inspected
and optimized like any other model, but not retrained here. Their test RMSE comes
from the 4 later growths, so it is not strictly comparable to a random split. To
try to beat them, train new models or run an autonomous search and compare.

## 3. Random Forest (rfr)
- Averages many decision trees fit on bootstrap samples. No feature scaling
  needed (inputs are standardized anyway for a uniform interface).
- Fully grown trees (`max_depth=None`, `min_samples_leaf=1`) overfit small
  data: train RMSE near 0 with a much larger test RMSE. Constrain with
  `max_depth` (e.g. 3-6) and/or raise `min_samples_leaf` (e.g. 2-3).
- `max_features` below 1.0 (e.g. 0.5) adds randomness and can reduce
  variance; default 1.0 is plain bagging.
- Cannot extrapolate: predictions stay within the range of training targets.
- Uncertainty = spread across trees (cheap, but often under-estimates error).
- `get_model_info` gives `feature_importances` -- use these to explain which
  variables matter.

## 4. Gaussian Process (gpr)
- A smooth probabilistic model; best for small, low-dimensional, smooth data.
  Gives a predictive mean and a standard deviation, so it is the natural choice
  for uncertainty-aware search.
- Setup used here: standardized inputs, `normalize_y=True`, kernel =
  `Constant * (Matern|RBF|RationalQuadratic) + WhiteKernel`. The WhiteKernel
  learns observation noise, so train RMSE is not forced to zero. Matern/RBF
  use one length scale per feature (ARD), so a long length scale means that
  feature barely matters.
- Kernels: `matern52` (default, smooth but not too rigid), `matern32` (rougher),
  `rbf` (very smooth), `rq` (mixture of length scales, single length scale).
- `n_restarts_optimizer` (default 5) re-runs hyperparameter fitting from
  random starts to avoid poor local optima; raise it if `warnings` mention
  non-convergence.
- Cost grows with the cube of the number of rows -- fine for hundreds of rows,
  slow for many thousands.
- The reported std includes the learned noise level.

## 4b. Kernel Ridge Regression (krr)
- A smooth kernel-based regressor: ridge regression in an implicit feature
  space. Cheap, deterministic, and often as good as a GPR on small smooth
  data. The lab's built-in EDS-ratio model is a KRR.
- Options: `krr_kernel` = `rbf` (default), `laplacian` (rougher), `linear`;
  `alpha` = regularization strength (larger = smoother, less overfit);
  `gamma` = RBF/Laplacian width (small = nearly linear, large = wiggly).
- KRR is **sensitive** to `alpha` and `gamma`: a poor pair can score far worse than a good one. Leave them unset (default) and they are tuned by
  inner CV on the training split only, so the test RMSE stays honest. The
  chosen values are in `get_model_info` (`tuned_by_inner_cv`).
- No native uncertainty: `uncertainty_std` is the spread across 5 CV fold
  models -- a rough indicator that usually under-estimates the error.
- A near-zero `gamma` means the relationship is close to linear.

## 4c. MLP (mlp)
- One hidden layer, tanh activation, Adam solver, standardized inputs and target.
- Options: `hidden_nodes` (default 3), `mlp_alpha` (L2 penalty, default 1.0),
  `batch_size` (default 4), `learning_rate` (default 0.01).
- On tiny datasets regularization matters far more than size: strong `mlp_alpha`
  keeps the fit simple, and weak values overfit badly. Keep it small.
- Results vary with the random seed, so check more than one if the result matters.
- Uncertainty is the spread across 5 CV fold models (rough; usually an
  under-estimate).

## 4d. Working independently (run_autonomous_search)
Use this for open-ended tasks: "find a better model", "improve film quality", "try
hyperparameters and more features". Do not ask for permission at each step.
1. Decide the target, the inputs and a time budget. Honour any budget the user
   gives; otherwise use 5 minutes. Inputs must be known before growth if the model
   is for planning: EDS_ratio and RHEED_Quality_Film are measured after growth.
2. Call `run_autonomous_search`. Pass new columns to try as `extra_features`.
3. If time remains, follow up on what the report suggests, for example a narrower
   search or other inputs, with `train_models` or another search.
4. Report: the winner and its settings, its score on each check, how the other
   model types compared, what was tried and failed, and the saved model id. If the
   user asked to save the results, call `save_results` last.

Judging results honestly:
- Compare models with each other on the same check, using `best_cv_by_family` and
  `top_trials`.
- Lead with `selection_on_training_rows_only`, the most conservative figure.
- The shuffled-target test shows whether the winner does better than chance:
  say so if `fraction_shuffled_as_good` is high.
- A high score from one configuration among many is often luck. With few rows,
  never declare a winner on a small difference, and say when models are tied.
- Scores are optimistic because the winner is picked on these same rows; recommend
  testing on new growths.
- Do not delete or alter data, and do not replace the lab's two saved models. New
  models are saved separately; let the user decide whether to adopt them.
- This tool does not read the internet, so do not claim literature results from it.

## 5. Interpreting train vs. test RMSE
- RMSE is in the same units as the target. Lower is better.
- Test RMSE much larger than train RMSE (ratio well above ~2) = overfitting.
- Both high = underfitting or uninformative features.
- With few test rows (< ~5) the test RMSE is very noisy: say so, and avoid
  declaring a confident winner on a small difference.
- `refit_on_all_data=true` refits the chosen configuration on every row after
  evaluating on the split; the reported RMSEs still come from the split. Offer
  it when data is scarce and the user wants a final model for use.

## 6. Using saved models
- `predict_with_model`: needs a value for **every** feature of that model. If
  the user omits one, ask. Report the prediction with its `uncertainty_std` and
  repeat any `extrapolation_warning`.
- `optimize_with_model`: random-search the model for the inputs that maximize,
  minimize, or hit a target value of its prediction. Use `bounds` / `fixed` for
  user constraints. Raise `kappa` (> 0) to favour uncertain regions
  (exploration); keep 0 for pure exploitation. These are surrogate
  suggestions, to be verified by a real experiment.
- Only models of the same target and the same split are strictly comparable.

## 7. Reporting
- Plain-text numbers, no LaTeX. Use markdown tables for comparisons.
- Always surface `notes` and `warnings` returned by the tools.
- Always give the model_id of anything saved so the user can refer to it later.

## 8. Saving results
- Trained models are saved automatically to `model_builder_results/models/`, because
  they must exist to be reused. This is the only automatic saving.
- Everything else (tool outputs, comparison tables, search reports, predictions) stays
  in the conversation unless the user explicitly asks to save, write out or export it.
  Then call `save_results` once the work is done, and report the `saved_to` path.
- Never call `save_results` on your own initiative, and never say a result was saved
  unless the tool returned its path. `save_results` writes the whole session, so it
  can't save only part of it; say so if the user asks for just one item.
