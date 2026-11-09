---
name: d-optimal-doe
description: Run a D-optimal Design of Experiments over growthtime, filamentpower and flux_ratio for the Fe3Sn MBE growths. Selects the design by coordinate exchange, scores each run with the lab's film-quality MLP and EDS-ratio RFR, fits a quadratic response surface to Score_Holistic, and reports the design table, fit statistics and the optimum. Use only when the user explicitly asks for a DoE / design of experiments / D-optimal design / the optimal set of experiments to run next.
---

# D-optimal DoE skills (`agent/doe.py`)

## 1. When to use / not use
- Use: "run a DoE", "design of experiments", "D-optimal design", "which set of
  experiments should I run next".
- Do NOT use for a single-point prediction (`ask_model_builder`) or the iterative
  Expected-Improvement search (`run_active_learning_loop`).

## 2. Tools and workflow
1. `get_doe_suggested_ranges` -- returns min/max of `growthtime`, `filamentpower`,
   `flux_ratio` from `data/train_compiled.csv`. **Always call first.**
2. Show these ranges and ask the user to confirm or override each parameter's
   `(min, max)`. Do not call `run_doe_experiment` until they answer. Any bound
   they leave unaddressed falls back to the default.
3. `run_doe_experiment(growthtime_min/max, filamentpower_min/max, flux_ratio_min/max, n_runs=12)`.
   `None` bounds fall back to defaults. `n_runs` is raised to 12 if lower
   (n_terms + 2, see section 4).
4. Report **in full**: `design_table` as a markdown table; `quadratic_fit`
   coefficients with std_error as a markdown table, plus `residual_std_dev`,
   `three_sigma`, `r_squared`, `degrees_of_freedom`; and the `optimum` inputs with
   predicted `Score_Holistic`, `EDS_ratio`, `RHEED_Quality_Film`.

Validation (`agent/validation.py`): `n_runs` and `resolution` must be positive
integers, `resolution >= 2`, and each range must have min < max. Errors are raised
as `ValueError` -- relay the message, do not retry with guessed values.

## 3. Fixed inputs and constants
Only 3 factors vary. Everything else the models need is held fixed at its
historical **median** from `data/train_compiled.csv`:
`Substrate_quality` and the 6 raw substrate numbers (`subcfwhm`, `sublfwhm`,
`subrfwhm`, `submean_width`, `subwidth_variance`, `subcurve_variance`).

| Constant | Value | Meaning |
|---|---|---|
| `FILM_WEIGHT` | 0.65 | weight on film quality |
| `EDS_WEIGHT` | 0.35 | weight on stoichiometry score |
| `EDS_TARGET_RATIO` | 3.0 | ideal Fe:Sn ratio |
| `EDS_SIGMA` | 0.5 | width of stoichiometry score |
| `CANDIDATE_RESOLUTION` | 15 | grid levels per factor (15^3 = 3375 candidates) |
| `DEFAULT_N_RUNS` | 12 | default design size |
| `N_RANDOM_STARTS` | 10 | random restarts of the exchange search |
| `N_EXCHANGE_CANDIDATES` | 200 | replacement candidates probed per exchange step |
| seed | 42 | rng seed (design is reproducible) |

## 4. Exact method (pseudo code)

```
INPUT  ranges[p] = (lo_p, hi_p) for p in {growthtime, filamentpower, flux_ratio}
       n_runs, resolution = 15, seed = 42
FIXED  sq   = median(Substrate_quality)
       sraw = [median(c) for c in 6 raw substrate columns]

# --- coding -------------------------------------------------------------
code(x_p)   = 2 * (x_p - lo_p) / (hi_p - lo_p) - 1          # real -> [-1, 1]
decode(z_p) = (z_p + 1) / 2 * (hi_p - lo_p) + lo_p          # [-1, 1] -> real

# --- model matrix row (10 terms) ----------------------------------------
f(z) = [1, z1, z2, z3, z1^2, z2^2, z3^2, z1*z2, z1*z3, z2*z3]
n_terms = 10

# --- 1. candidate pool --------------------------------------------------
C = cartesian_product( linspace(lo_p, hi_p, resolution) for each p )   # resolution^3 rows
Z = code(C)
F_all = [f(z) for z in Z]
n_runs = max(n_runs, n_terms + 2)                  # floor of 12
if n_runs > |C|: error

# --- 2. D-optimal selection: coordinate exchange ------------------------
det_of(idx) = det( F_all[idx]^T . F_all[idx] + 1e-10 * I )
best_det = -inf
repeat N_RANDOM_STARTS (=10) times:
    idx = random n_runs distinct rows of C                  # no replacement
    repeat until no swap improves:
        for i in 0 .. n_runs-1:
            cur = det_of(idx)
            probe = random min(200, |C|) rows of C
            best_j = none ; best_det_j = cur
            for j in probe, j not in idx:
                trial = idx with idx[i] replaced by j
                if det_of(trial) > best_det_j:  best_j = j ; best_det_j = det_of(trial)
            if best_j: idx[i] = best_j ; improved = true
    if det_of(idx) > best_det:  best_det = det_of(idx) ; best_idx = idx
design  = C[best_idx]         # real units
design_z = code(design)

# --- 3. evaluate every design point with the ground-truth models --------
eds  = RFR_EDS(design, sq)                      # 4 inputs, mean prediction
film = MLP_film(design, sq, sraw)               # 10 inputs, mean prediction
stoich = 100 * exp( -(eds - 3.0)^2 / (2 * 0.5^2) )
y = 0.65 * film + 0.35 * stoich                 # Score_Holistic, one per run

# --- 4. quadratic response surface (least squares) ----------------------
F      = [f(z) for z in design_z]               # (n_runs x 10)
beta   = lstsq(F, y)
resid  = y - F . beta
dof    = max(n_runs - 10, 1)
sigma2 = sum(resid^2) / dof ;  sigma = sqrt(sigma2) ;  three_sigma = 3 * sigma
R2     = 1 - sum(resid^2) / sum((y - mean(y))^2)
cov    = sigma2 * inv(F^T F)      # pinv if singular
se     = sqrt(diag(cov))          # std error of each coefficient

# --- 5. optimum of the fitted surface -----------------------------------
best = -inf
repeat 40 times:
    z0 ~ Uniform(-1, 1)^3
    z* = L-BFGS-B( minimize -(f(z) . beta), start z0, bounds [-1, 1]^3 )
    if -value > best:  best = -value ; best_z = z*
optimum = decode(best_z)
re-evaluate optimum with the two ground-truth models (EDS, film)

OUTPUT design_table, coefficients (+std_error), residual_std_dev, three_sigma,
       R2, dof, optimum (real units) with predicted Score_Holistic / EDS / film
```

## 5. Quadratic response function

With coded factors `x1 = growthtime`, `x2 = filamentpower`, `x3 = flux_ratio`
(each in [-1, 1]):

```
Score_Holistic(x) = b0
                  + b1*x1   + b2*x2   + b3*x3
                  + b11*x1^2 + b22*x2^2 + b33*x3^2
                  + b12*x1*x2 + b13*x1*x3 + b23*x2*x3
```

Matrix form: `y = F . beta`, with `beta = (F'F)^-1 F'y` (least squares) and
`Cov(beta) = sigma^2 (F'F)^-1`. D-optimality maximizes `det(F'F)`, which minimizes
the volume of the confidence region of `beta`.

Coefficient order in the output: `intercept, growthtime, filamentpower, flux_ratio,
growthtime^2, filamentpower^2, flux_ratio^2, growthtime*filamentpower,
growthtime*flux_ratio, filamentpower*flux_ratio`. Coefficients are in **coded**
units, so their sizes are comparable across factors; decode only the optimum.

Response being modeled: `Score_Holistic = 0.65 * RHEED_Quality_Film +
0.35 * 100 * exp(-(EDS_ratio - 3)^2 / (2 * 0.5^2))` -- identical to `al_loop.py`.

## 6. Outputs
- `ranges_used`, `fixed_Substrate_quality`, `fixed_substrate_raw`, `n_runs`
- `design_table`: run, the 3 factors, fixed Substrate_quality, predicted
  EDS_ratio, predicted RHEED_Quality_Film, Score_Holistic
- `quadratic_fit`: coefficients (term, coefficient, std_error), residual_std_dev,
  three_sigma, r_squared, degrees_of_freedom
- `optimum`: the 3 factors, predicted Score_Holistic / EDS_ratio / film quality

## 7. Be honest about what this is
- The "responses" are **model predictions**, not measurements. The quadratic is a
  smooth fit to the models, so R^2 says how quadratic the models are, not how
  well the real growth follows it. Run the design on the real MBE to get real
  data, then refit.
- With 12 runs and 10 terms, only **2 degrees of freedom** remain. Standard
  errors are large (often comparable to the coefficients), so R^2 is inflated and
  most terms are not significant. Say this. More `n_runs` helps.
- Film-quality error is typically ~15 points and the best films tend to be
  under-predicted; the EDS RFR is little better than the training mean on some
  growths. The optimum is a suggestion to test, not a guarantee.
- The optimum can sit at a range boundary; the surface is not trusted beyond the
  ranges, which come from only the historical growths.
- `three_sigma` is 3 x the residual std dev of the fit, not a prediction error
  bar for new growths.

## 8. Running it directly
Use an environment whose scikit-learn matches the saved models (1.7.2, see
`requirements.txt`); a mismatched version can fail to unpickle them.
```
cd agent
python -c "import doe, json; print(json.dumps(doe.run_doe(), indent=1))"
python -c "import doe; print(doe.run_doe({'growthtime': (300, 600)}, n_runs=20))"
```
Tests: `pytest tests/test_doe.py`.
