# Auto-MBE (Fe3Sn thin-films)

An LLM agent for an MBE (Molecular Beam Epitaxy) growth lab. It reads RHEED
images, predicts film (/substrate) quality/stoichiometry from growth parameters, looks up
and analyzes past experiments, and runs an active-learning loop to suggest
the next best process parameters to run. The film quality and stoichiometry is predicted
using a trained Random Forest model checkpoint from previous experimental data.

## Installation

```bash
conda create -n Auto-MBE python=3.10
conda activate Auto-MBE
git clone https://github.com/maitreyo18/Auto-MBE-Fe3Sn.git
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
lets it chain tools (e.g. `rheed_quality` -> `predict_quality`) in one turn.

Conversation memory is not a separate library -- `agent.py::run()` just
appends every `(role, message)` pair (including tool results folded back in
by the graph) to a plain Python list (`history`) and passes the whole list
back into `agent.invoke()` on each call, so the LLM re-reads the full
transcript every turn. `clear` empties that list; there's no persistence
across process restarts.

## Tools

The agent is a single [LangGraph ReAct agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/)
(`create_react_agent`) that can call these tools, chaining multiple calls
together in one turn when needed:

| Tool | What it does |
|---|---|
| `rheed_quality` | Extracts `Substrate_quality` (0-100) and raw diffraction features from a RHEED image. |
| `predict_quality` | Predicts `RHEED_Quality_Film` and `EDS_ratio` (Fe:Sn stoichiometry) from growth parameters, via pre-trained random forests. |
| `analyze_previous_experiments` | Looks up and summarizes/ranks past runs from `data/train_compiled.csv`. |
| `run_active_learning_loop` | Suggests the next set of process parameters to try, via Expected Improvement. |

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

`RHEED_Quality_Film` and `EDS_ratio` are **not** formula-based -- they come
from pre-trained random forest regressors (`models/rf_*.joblib`) trained on
`growthtime, filamentpower, flux_ratio, Substrate_quality`, called via
`models/predict.py::predict_with_uncertainty`.

## Active-learning (AL) algorithm

`agent/al_loop.py` runs a multi-dimensional Expected Improvement (EI) loop
over `filamentpower` and `flux_ratio` (holding `growthtime` and
`Substrate_quality` fixed at the top-quartile median):

1. Build a single composite target per row:
   ```
   Score_Holistic = 0.65 * RHEED_Quality_Film
                   + 0.35 * stoichiometry_score(EDS_ratio)
   ```
   where `stoichiometry_score` is a Gaussian centered at the ideal Fe:Sn
   ratio of 3.0 (`sigma = 0.5`), scaled to 0-100.
2. Fit a fresh Random Forest (`n_estimators=100, max_depth=3`) on
   `Score_Holistic`.
3. Sweep a 60x60 grid of (`filamentpower`, `flux_ratio`); for each grid
   point compute the forest's per-tree mean/std and score it with EI:
   ```
   z  = (mu - best_so_far - xi) / std        # xi = 0.01
   EI = (mu - best_so_far - xi) * Phi(z) + std * phi(z)
   ```
4. Pick the grid point with max EI as the next suggested experiment,
   predict its `EDS_ratio` / `RHEED_Quality_Film` with separate forests,
   append it to the dataset, and repeat.

Each iteration saves an EI contour plot and the fold models used under
`agent/al_runs/iteration_N/`. No pre-trained checkpoint is used -- it fits
fresh on whatever `initial_data` you give it.

NOTE: The only limitation of the current AL algorithm is that the RF models are trained 
iteratively on the synthetic data. The AL loop agent can be easily modified to give access 
to on-demand experimental data in an actual lab setting.

## Example prompts

**`rheed_quality`**
> What's the substrate quality of RHEED_images/film/10.3.png?

**`predict_quality`**
> Predict film quality and EDS ratio for growthtime=450s, filamentpower=3.5W, flux_ratio=0.6, substrate_quality=70%.

**`analyze_previous_experiments`**
> What were our 5 best previous runs by film quality?
> How does flux ratio correlate with EDS ratio across past experiments?

**`run_active_learning_loop`**
> Given these 6 past runs [...], suggest the next 3 experiments to try.
> Optimize filament power and flux ratio for the best film quality over 10 iterations.

## Credits

Data obtained from [github.com/raghu0415/Fe3SnMLfolder](https://github.com/raghu0415/Fe3SnMLfolder).
Image processing, analysis, and the active-learning algorithm are used as
described in [arxiv.org/abs/2608.17742](https://arxiv.org/abs/2608.17742),
with necessary and adequate changes made for the agent implementation here.
