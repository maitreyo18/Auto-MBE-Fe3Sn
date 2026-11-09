"""Model-builder agent: trains, saves, compares and reuses Random Forest,
Gaussian Process, Kernel Ridge and MLP models on the fly, driven by the LLM, and
can search for better models by itself when given a task.

It is its own ReAct agent (own system prompt built from skills.md, own tools).
The main MBE agent reaches it through the `ask_model_builder` tool in
tools.py, and it can also run standalone:  python model_agent.py
Or give it one task and let it work:  python model_agent.py --task "..."
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.tools import tool

import auto_search
import model_builder as mb
from memory import ConversationMemory, content_text, new_tool_results

THIS_DIR = Path(__file__).resolve().parent
SKILLS_PATH = THIS_DIR / "skills.md"


def load_skills(path=SKILLS_PATH):
    """skills.md body with its YAML frontmatter (name/description) removed."""
    text = Path(path).read_text()
    return re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL).strip()


def build_system_prompt():
    return f"""You are the model-building assistant for an MBE growth lab. You train, save,
compare and reuse Random Forest (RFR), Gaussian Process (GPR), Kernel Ridge (KRR) and MLP regression
models on tabular data, on the fly, using your tools. You also serve the lab's
two saved models (lab_RHEED_Quality_Film, an MLP, and lab_EDS_ratio, an RFR), so you
answer every "predict film quality / EDS ratio" request. When given an open-ended
task such as "find a better model", you work independently with
run_autonomous_search. You talk to the user
(or to the main lab agent relaying the user) in plain language.

Hard rules:
- Never invent numbers. Every RMSE, prediction or ranking must come from a tool result.
- If the target column(s) or training variables are not specified, ask the user
  for them (after showing the available columns) and STOP -- do not train yet.
  Exception: an autonomous task on RHEED_Quality_Film or EDS_ratio has default inputs.
- Only RFR, GPR, KRR and MLP are supported.
- Call save_results only when the user explicitly asks to save or write out results, and
  never claim a file was written unless save_results returned its path.
- Default data is the lab's training CSV unless the user names another .csv file.

Earlier replies may start with "[Tool results from this turn, recorded by the
system:". Those are real outputs: use them to answer follow-up questions instead of
re-running a tool, and re-run only if they are cut short or don't cover the question.
Never write that marker in your own replies.

Follow the skills below exactly.

{load_skills()}
"""


@tool
def describe_training_data(data_path: Optional[str] = None) -> dict:
    """Lists the numeric columns (with min/max/mean and missing counts) of the
    training CSV, so the user can choose targets and training variables. Use
    this whenever targets or features are unspecified. data_path is optional
    (default: the lab's data/train_compiled.csv)."""
    return mb.log_result("describe_training_data", {"data_path": data_path}, mb.describe_data(data_path))


@tool
def train_models(
    targets: Optional[List[str]] = None,
    features: Optional[List[str]] = None,
    model_types: Optional[List[str]] = None,
    data_path: Optional[str] = None,
    test_size: float = 0.25,
    random_state: int = 42,
    refit_on_all_data: bool = False,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    max_features: float = 1.0,
    kernel: str = "matern52",
    n_restarts_optimizer: int = 5,
    krr_kernel: str = "rbf",
    alpha: Optional[float] = None,
    gamma: Optional[float] = None,
    hidden_nodes: int = 3,
    mlp_alpha: float = 1.0,
    batch_size: int = 4,
    learning_rate: float = 0.01,
    name_prefix: Optional[str] = None,
) -> dict:
    """Trains and SAVES one model per (target, model type) on a shared
    train/test split and returns each model's train/test RMSE. targets and
    features must come from the user: if either is empty this returns
    needs_user_input and trains nothing -- then ask the user.

    model_types: subset of ["rfr", "gpr", "krr", "mlp"] (default all four).
    data_path: a CSV path, or "all" for the 21 lab growths (default: the 17 initial).
    RFR options: n_estimators, max_depth (None = unlimited), min_samples_leaf,
    max_features (fraction 0-1). GPR options: kernel in matern52 | matern32 |
    rbf | rq, n_restarts_optimizer. KRR (kernel ridge) options: krr_kernel in
    rbf | laplacian | linear, alpha (regularization), gamma (kernel width);
    leave alpha and/or gamma unset to have them tuned by inner CV on the
    training split only. MLP (one hidden layer) options: hidden_nodes, mlp_alpha
    (L2 penalty), batch_size, learning_rate. refit_on_all_data=True refits the final
    saved model on every row after scoring on the split. name_prefix labels a
    variant (letters/digits/_-. only), e.g. 'deep'."""
    inputs = dict(locals())
    return mb.log_result("train_models", inputs, mb.train_models(
        targets=targets, features=features, model_types=model_types, data_path=data_path,
        test_size=test_size, random_state=random_state, refit_on_all_data=refit_on_all_data,
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        max_features=max_features, kernel=kernel, n_restarts_optimizer=n_restarts_optimizer,
        krr_kernel=krr_kernel, alpha=alpha, gamma=gamma, hidden_nodes=hidden_nodes,
        mlp_alpha=mlp_alpha, batch_size=batch_size, learning_rate=learning_rate, name_prefix=name_prefix,
    ))


@tool
def run_autonomous_search(
    target: Optional[str] = None,
    features: Optional[List[str]] = None,
    extra_features: Optional[List[str]] = None,
    model_types: Optional[List[str]] = None,
    data_path: str = "all",
    minutes: float = 5.0,
    engineered: bool = True,
) -> dict:
    """Works independently for about `minutes`: tries many hyperparameters for RFR,
    GPR, KRR and MLP across feature sets (the given features, plus extra_features,
    plus engineered logs, squares and products), confirms the best on fresh splits,
    stress-tests the winner, saves it, and returns a full report. Use for open-ended tasks such as "find a better model" or "try
    hyperparameters and features". For RHEED_Quality_Film and EDS_ratio the inputs
    default to the 4 growth inputs plus the 6 raw substrate numbers; other targets
    need features from the user. data_path: "all" is the 21 lab growths."""
    inputs = dict(locals())
    return mb.log_result("run_autonomous_search", inputs, auto_search.run_search(
        target=target, features=features, extra_features=extra_features, model_types=model_types,
        data_path=data_path, minutes=minutes, engineered=engineered,
    ))


@tool
def list_models() -> dict:
    """Lists every available model -- the built-in pre-trained lab models
    (built_in=true, ids lab_*) and all saved on-the-fly models -- with type,
    target, features and RMSEs."""
    return mb.list_models()


@tool
def compare_models(model_ids: Optional[List[str]] = None, target: Optional[str] = None) -> dict:
    """Compares saved models' train and test RMSE, best (lowest test RMSE)
    first within each target. Pass model_ids for specific models, or target
    to restrict to one target; with neither, compares everything saved.
    Call this after every training run."""
    return mb.log_result("compare_models", {"model_ids": model_ids, "target": target}, mb.compare_models(model_ids=model_ids, target=target))


@tool
def get_model_info(model_id: str) -> dict:
    """Full details of one saved model: features, hyperparameters, split,
    metrics, feature importances (RFR) or fitted kernel (GPR)."""
    return mb.get_model_info(model_id)


@tool
def predict_with_model(model_id: str, inputs: List[Dict[str, float]]) -> dict:
    """Predicts a model's target for one or more input points. model_id is
    either a built-in lab model or an on-the-fly model from list_models.
    lab_EDS_ratio needs growthtime, filamentpower, flux_ratio,
    Substrate_quality. lab_RHEED_Quality_Film needs those 4 plus subcfwhm,
    sublfwhm, subrfwhm, submean_width, subwidth_variance, subcurve_variance. Each item of inputs must give a value
    for EVERY feature of that model. Returns the prediction, an uncertainty
    std, and an extrapolation warning if outside the training range."""
    return mb.log_result("predict_with_model", {"model_id": model_id, "inputs": inputs}, mb.predict(model_id, inputs))


@tool
def optimize_with_model(
    model_id: str,
    goal: str = "maximize",
    target_value: Optional[float] = None,
    bounds: Optional[Dict[str, List[float]]] = None,
    fixed: Optional[Dict[str, float]] = None,
    kappa: float = 0.0,
    top_k: int = 5,
) -> dict:
    """Searches a saved model for the inputs that maximize or minimize its
    prediction, or bring it close to target_value (goal='target'). bounds:
    {feature: [lo, hi]} (default training range); fixed: {feature: value}
    held constant; kappa > 0 favours uncertain regions. Returns the top_k inputs."""
    inputs = dict(locals())
    return mb.log_result("optimize_with_model", inputs, mb.optimize(
        model_id, goal=goal, target_value=target_value, bounds=bounds, fixed=fixed, kappa=kappa, top_k=top_k,
    ))


@tool
def save_results(name: Optional[str] = None, notes: Optional[str] = None) -> dict:
    """Writes ALL of this session's results to ./model_builder_results/<name>/:
    results.json (every tool input and output), comparison.csv and summary.md.
    Call this ONLY when the user explicitly asks to save, write out or export
    results; never on your own initiative. name is an optional folder name
    (letters, digits, _-. only). notes is optional text from you, stored in a
    separate labelled section. Existing results are never overwritten. Tell the
    user the saved_to path it returns."""
    return mb.save_session_results(name=name, notes=notes)


MODEL_TOOLS = [
    describe_training_data,
    train_models,
    run_autonomous_search,
    list_models,
    compare_models,
    get_model_info,
    predict_with_model,
    optimize_with_model,
    save_results,
]

_agent = None


def _summary_llm():
    from llm import get_llm

    return get_llm()


_memory = ConversationMemory(llm_factory=_summary_llm)


def build_model_agent(llm=None):
    from langgraph.prebuilt import create_react_agent

    if llm is None:
        from llm import get_llm
        llm = get_llm()
    return create_react_agent(llm, MODEL_TOOLS, prompt=build_system_prompt())


def reset_history():
    _memory.clear()
    mb.session_log.clear()


def ask(request, agent=None):
    """Send one message to the model-builder agent and return its text reply.
    Conversation memory persists across calls (so a user's answer to a clarifying
    question lands in context) until reset_history(), and older exchanges are
    compressed automatically."""
    global _agent
    if agent is None:
        if _agent is None:
            _agent = build_model_agent()
        agent = _agent
    result = agent.invoke({"messages": _memory.messages(request)})
    reply = result["messages"][-1]
    _memory.record(request, reply.content, new_tool_results(result["messages"]))
    return content_text(reply.content)


def main():
    parser = argparse.ArgumentParser(description="Model-builder agent.")
    parser.add_argument("--task", help="run this one task independently, print the result and exit")
    args = parser.parse_args()
    if args.task:
        print(ask(args.task))
        return
    print("Model-builder agent ready ('clear' resets history, 'exit' quits).")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "clear":
            reset_history()
            print("History cleared.")
            continue
        print(ask(user_input))


if __name__ == "__main__":
    main()
