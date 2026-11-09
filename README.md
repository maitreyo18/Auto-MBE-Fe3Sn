# Auto-MBE

An LLM agent for an MBE (Molecular Beam Epitaxy) growth lab. It reads RHEED
images, predicts film quality/stoichiometry from growth parameters, looks up
and analyzes past experiments, runs an active-learning loop to suggest the
next experiment to run, and runs a D-optimal DoE to jointly optimize film
quality and stoichiometry over a parameter range.

## Installation

```bash
conda create -n Synth_KG python=3.10
conda activate Synth_KG
pip install -r requirements.txt
```

## How to run

```bash
cd agent
python agent.py
```

This opens an interactive REPL:

```
MBE agent ready. Type a request ('clear' to reset history, 'exit' to quit).
> what's the substrate quality for ../RHEED_images/film/10.3.png?
```

Type `clear` to reset conversation history, `exit`/`quit` to leave.

**Image paths:** you can also run `python agent/agent.py` from anywhere (no
`cd` needed) -- any relative image path you give the agent (e.g. for
`rheed_quality`) is resolved against `agent.py`'s own directory, not the
shell's current directory, so it works the same either way. Absolute paths
work as-is.

## Adding your Google API key

The agent uses Gemini (`gemma-4-31b-it`) via `langchain_google_genai`, which
needs a Google API key in the environment:

```bash
export GOOGLE_API_KEY="your-key-here"
```

Get a key at [aistudio.google.com](https://aistudio.google.com/apikey). Without
it, `agent.py` raises `RuntimeError: Set GOOGLE_API_KEY before running the agent.`

## Agent structure & context

`agent.py::build_agent()` wires an LLM (`ChatGoogleGenerativeAI`) and a
Python function list (`TOOLS`) into a LangGraph state graph via
`langgraph.prebuilt.create_react_agent` -- LangChain's `@tool` decorator
(in `tools.py`) turns each plain function into a schema the LLM can call,
and LangGraph loops "call LLM -> run any requested tool(s) -> feed results
back to the LLM" until it answers with no more tool calls, which is what
lets it chain tools (e.g. `rheed_quality` -> `ask_model_builder`) in one turn.

**Conversation memory** (`agent/memory.py`): each agent has its own, and the LLM
re-reads it every turn. One conversation is a user message plus the reply. When 15 have
built up, the oldest 10 are folded into a rolling summary and the latest 5 stay word for
word. The summary has fixed sections (goals and preferences, decisions, results with
exact numbers, models and files, open questions) and each one builds on the last. Exact
model ids are re-attached if the summary drops them. If the summarizing LLM call fails,
nothing is removed and it retries after the next reply; at 30 conversations a plain text
digest takes over. Tool results stay with their turn, so follow-up questions can reuse
them: they go back as text inside the earlier reply, capped at 6000 characters per tool
for the latest 5 conversations and 1500 for older ones, and the summary captures what
matters from them after that. `clear` empties both agents' memory, and nothing persists
across restarts.

## Tools

The agent is a single [LangGraph ReAct agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/)
(`create_react_agent`) that can call these tools, chaining multiple calls
together in one turn when needed:

| Tool | What it does |
|---|---|
| `rheed_quality` | Extracts `Substrate_quality` (0-100) and raw diffraction features from a RHEED image. |
| `analyze_previous_experiments` | Reads all 21 real growths with pandas, then sorts, filters, finds the runs closest to given conditions, compares runs and summarizes (`agent/experiments.py`). Flags imputed EDS values. |
| `run_active_learning_loop` | Simulates active learning against the ground-truth models, starting from 10 experiments. |
| `get_doe_suggested_ranges` | Returns min/max of growthtime, filamentpower, flux_ratio from past data, for DoE range confirmation. |
| `run_doe_experiment` | Runs a D-optimal DoE over growthtime/filamentpower/flux_ratio, fits a quadratic response surface, returns the optimum. |
| `ask_model_builder` | Delegates to the model-builder agent (below): all predictions of film quality / EDS ratio, plus training, comparing and using RFR / GPR / KRR / MLP models on the fly, or searching for better ones independently. |

## Film / substrate quality calculator

`Substrate_quality` is computed by `RHEED_images/feature_extractor.py` from
a single RHEED image:

1. Detects the central diffraction streak and fits Lorentzian peaks to get
   FWHM values (`subcfwhm` center, `sublfwhm` left, `subrfwhm` right).
2. Measures streak `submean_width` and `subcurve_variance` (how much the
   intensity profile deviates from a flat top).
3. Min-max scales these 5 features against the existing compiled dataset,
   then combines them into a weighted penalty:

```
penalty = 0.40 * mean(sublfwhm, subrfwhm)
        + 0.20 * submean_width
        + 0.20 * subcfwhm
        + 0.20 * subcurve_variance

Substrate_quality = (1 - penalty) * 100
```

Lower FWHM / narrower, flatter streaks -> lower penalty -> higher quality.

## The two lab models

Predictions, the AL loop and the DoE all use these two models, defined in
`models/lab_models.py` and trained by `models/train.py`:

| Target | Model | Inputs | Checkpoint |
|---|---|---|---|
| `RHEED_Quality_Film` | MLP (one layer of 3 nodes, `alpha=1`, batch size 4) | growth time, filament power, flux ratio, `Substrate_quality`, and the 6 raw substrate numbers from `rheed_quality` | `models/mlp_RHEED_Quality_Film.joblib` |
| `EDS_ratio` | Random Forest (100 trees, `max_depth=3`) | growth time, filament power, flux ratio, `Substrate_quality` | `models/rf_EDS_ratio.joblib` |

They are trained on the 17 initial growths (`data/train_compiled.csv`) and tested on
the 4 later ones, runs 32, 33, 35 and 36 (`data/test_compiled.csv`, from the authors'
processed CSV):

| Target | Test RMSE | Predicting the training mean | 5-fold CV on the 17 (mean baseline) |
|---|---|---|---|
| `RHEED_Quality_Film` | 10.06 | 16.57 | 15.66 (17.51) |
| `EDS_ratio` | 0.35 | 0.43 | 0.77 (0.99) |

Treat these as rough: there are only 4 test rows, the models were chosen using these
same rows, and EDS for runs 32 and 35 is imputed in the source data. On the 2 runs with
measured EDS, the RFR (RMSE 0.22) does no better than the training mean. The reported
uncertainty (spread across CV folds for the MLP, across trees for the RFR) is far
smaller than the real error.

## Active-learning (AL) algorithm

`agent/al_loop.py` simulates an Expected Improvement (EI) loop over `filamentpower`
and `flux_ratio`. Growth time and the substrate numbers are held at the median of the
observed top-quartile rows. The two saved models are the **ground truth** and supply
the result of every suggested experiment.

1. **Start** from 10 randomly chosen rows of the 17 initial experiments, with their
   measured results.
2. **Fit LOOCV models:** for each left-out row, fit the same two model definitions (MLP
   and RFR) on the rest. Each pair gives one estimate of
   ```
   Score_Holistic = 0.65 * RHEED_Quality_Film + 0.35 * stoichiometry_score(EDS_ratio)
   ```
   where `stoichiometry_score` is a Gaussian centered on 3.0 (`sigma = 0.5`), scaled to 0-100.
3. **Pick the next point:** on a 60x60 grid, take the mean and std of the score across
   the LOOCV pairs and maximize
   ```
   z  = (mu - best_so_far - xi) / std        # xi = 0.01
   EI = (mu - best_so_far - xi) * Phi(z) + std * phi(z)
   ```
4. **Query the ground truth** for that point, append the result and repeat.

Each result reports LOOCV RMSE per target and how well the AL models track the ground
truth over all 21 growths. Each iteration saves its LOOCV models and an EI plot under
`agent/al_runs/iteration_N/` (git-ignored). Because outcomes come from models, this tests
the method and does not confirm real growth conditions.

## Model-builder agent (RFR / GPR / KRR / MLP)

A second LangGraph ReAct agent, `agent/model_agent.py`, with its own tools and a system
prompt built from [`agent/skills.md`](agent/skills.md). The main agent reaches it through
`ask_model_builder`, so every request and reply goes through the LLM. It also runs alone:
`python agent/model_agent.py`, or `python agent/model_agent.py --task "..."` for one task.

- **Predict** with the two saved lab models, `lab_RHEED_Quality_Film` and `lab_EDS_ratio`.
  Film quality needs all 7 substrate numbers from `rheed_quality`.
- **Train and save several models** (`train_models`): one per target and model type on a
  shared train/test split, saved to `model_builder_results/models/<model_id>/` (git-ignored).
  Targets and training variables are never guessed: it asks if you don't give them.
- **Compare** (`compare_models`): train vs. test RMSE, ranked by test RMSE per target.
- **Reuse** saved models: `predict_with_model`, `optimize_with_model`, `get_model_info`,
  `list_models`.
- **Save results when asked** (`save_results`): "save the results" writes everything from
  the session to `model_builder_results/<name>/`: `results.json`, `comparison.csv` and
  `summary.md`. Numbers come from tool outputs, and nothing is overwritten. Without
  that request, only trained models are saved, because they must exist to be reused.
- **Search independently** (`run_autonomous_search`, `agent/auto_search.py`): given a target
  and a time budget, it tries hyperparameters for all four model types across feature
  sets, including engineered logs, squares and products. It confirms the best on new
  splits, stress-tests the winner, saves it, and returns a report.
  It never replaces the two lab models or changes the data. It needs no LLM:
  ```bash
  python agent/auto_search.py --target RHEED_Quality_Film --minutes 5
  ```
  Add `--save-results` to also write its report to `model_builder_results/`.
  With an LLM, ask for example "find a better film-quality model in 10 minutes". It
  can't read the internet. The winner is chosen on the same rows it is scored on, so
  treat its gains as optimistic until tested on new growths.

## D-optimal DoE algorithm

`agent/doe.py` runs an autonomous D-optimal Design of Experiments over
`growthtime`, `filamentpower`, `flux_ratio` (holding `Substrate_quality` and the 6 raw
substrate numbers fixed at their historical medians), jointly optimizing film quality and
stoichiometry:

1. Build a grid candidate pool over the given (or default, CSV-derived)
   ranges, coded to [-1, 1].
2. Select `n_runs` points via coordinate-exchange D-optimal search,
   maximizing `det(F'F)` of the quadratic model matrix
   `[1, x1, x2, x3, x1^2, x2^2, x3^2, x1*x2, x1*x3, x2*x3]`.
3. Evaluate each selected point with the ground-truth `EDS_ratio` RFR and
   `RHEED_Quality_Film` MLP (mean predictions only) and combine into
   the same `Score_Holistic` used by the AL loop (65% film quality + 35%
   stoichiometry closeness).
4. Fit a quadratic response surface to `Score_Holistic` via least squares;
   report each term's coefficient and standard error, the residual standard
   deviation, its 3-sigma bound, and R-squared.
5. Maximize the fitted surface (multi-start L-BFGS-B) within the given
   ranges to report the optimum `(growthtime, filamentpower, flux_ratio)`.

Defaults to 12 runs (quadratic terms + 2, the minimum for a stable fit).

## Example prompts

**`rheed_quality`**
> What's the substrate quality of RHEED_images/film/10.3.png?

**Prediction (via the model-builder agent)**
> Predict film quality and EDS ratio for growthtime=450s, filamentpower=3.5W, flux_ratio=0.6, using the substrate image RHEED_images/substrate/train/10.3.png.

**`analyze_previous_experiments`**
> What were our 5 best previous runs by film quality?
> How does flux ratio correlate with EDS ratio across past experiments?
> Which past growths are closest to filament power 3.5 W and flux ratio 0.6?
> Which runs came closest to the ideal EDS ratio of 3?
> Compare runs 14 and 24.

**`run_active_learning_loop`**
> Simulate 10 active-learning iterations to optimize filament power and flux ratio.

**Model building**
> Train RFR, GPR, KRR and MLP models to predict RHEED_Quality_Film from growthtime, filamentpower and flux_ratio, and compare their train and test RMSE.
> Using the GPR you just trained, which filament power and flux ratio maximize film quality?
> Find a better film-quality model by trying features and hyperparameters for 10 minutes.

**`run_doe_experiment`**
> Run a D-optimal DoE using the default ranges.
> Run a DoE with growth time 300-600 and 20 runs.

## Credits

Data obtained from [github.com/raghu0415/Fe3SnMLfolder](https://github.com/raghu0415/Fe3SnMLfolder).
Image processing, analysis, and the active-learning algorithm are used as
described in [arxiv.org/abs/2608.17742](https://arxiv.org/abs/2608.17742),
with necessary and adequate changes made for the agent implementation here.
