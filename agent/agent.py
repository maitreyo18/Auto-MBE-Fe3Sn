import logging
import os
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mbe_agent")

# langchain_core (pulled in above) re-registers its own "always" filter for
# LangChainPendingDeprecationWarning, which takes priority over filters
# added before it. Re-apply ours right before the import that triggers the
# warning so ours wins.
warnings.filterwarnings("ignore")
from langgraph.prebuilt import create_react_agent
from tools import TOOLS
from llm import get_llm
from memory import ConversationMemory, content_text, new_tool_results
from model_agent import reset_history as reset_model_agent_history

SYSTEM_PROMPT = """You are an assistant for an MBE growth lab.
When information needed for one tool is missing but available from another
tool's output, call that tool first and chain the result forward.

Whenever the user asks about previous, prior, past, historical, existing, or
already-present experiments/data/runs (not a new hypothetical configuration),
call analyze_previous_experiments to look up and analyze the real data before
answering -- do not guess or rely on memory. Choose its options to fit the question:
sort_by to rank runs, filters, min_values and max_values to narrow them, closest_to to find the past runs nearest
some conditions or target values, compare_runs to put specific runs side by side.
Focus on growthtime, filamentpower, flux_ratio, Substrate_quality, EDS_ratio and
RHEED_Quality_Film unless the user asks about other columns.

When reporting numbers, write them as plain text -- never wrap numbers in
$...$ or other LaTeX/math notation. Always attach the correct unit: W for
filament power, s for growth time, % for substrate quality and film quality.
EDS ratio and flux ratio are unitless.

Only when the user explicitly asks for a DoE, "design of experiments",
"D-optimal design", or an optimal set of experiments to run next: first call
get_doe_suggested_ranges and ask the user to confirm or override each
parameter's range -- do not call run_doe_experiment until they respond
(unaddressed ranges fall back to the suggested default). Then call
run_doe_experiment and report, in full: the design_table as a markdown
table, the quadratic fit as a markdown table of coefficients/std_error plus
residual_std_dev, three_sigma, and r_squared, and the optimum input
parameters with their predicted Score_Holistic/EDS_ratio/RHEED_Quality_Film.
Do not use this workflow for single-point predictions (ask_model_builder) or
iterative optimization (run_active_learning_loop).

Model building and ALL predictions go through ask_model_builder. Call it
with the user's request in their own words whenever the user wants to: predict
film quality (RHEED_Quality_Film) or stoichiometry (EDS_ratio) from growth
parameters; train, build, fit, or compare regression models (Random Forest,
Gaussian Process, Kernel Ridge or MLP); find a better model by searching
independently; or use, inspect, or optimize a model (the lab's two saved
models or ones trained on the fly). Do not train or predict yourself.
Film-quality predictions need all 7 substrate numbers from a substrate RHEED
image: if the user gives an image, call rheed_quality first and put EVERY number
it returns in the request. If they give neither the image nor those numbers,
ask for the image (the EDS ratio still works with just Substrate_quality). If
ask_model_builder replies with a question (e.g. which target / training
variables), pass that question to the user as-is, then send their answer back
through ask_model_builder. Relay its tables and numbers in full.

Earlier replies may start with "[Tool results from this turn, recorded by the
system:". Those are real outputs: use them to answer follow-up questions instead of
re-running a tool, and re-run only if they are cut short or don't cover the question.
Never write that marker in your own replies."""

# Set to a non-empty string to run that request once instead of the REPL.
USER_PROMPT = ""


def run(agent, memory, user_message):
    try:
        result = agent.invoke({"messages": memory.messages(user_message)})
    except Exception:
        logger.exception("Agent invocation failed for message: %r", user_message)
        return "Sorry, something went wrong while processing that request. Please try again."

    used_tools = any(getattr(m, "type", None) == "tool" for m in result["messages"])
    if used_tools:
        logger.info("Agent used one or more tools to answer this request.")

    reply = result["messages"][-1]
    memory.record(user_message, reply.content, new_tool_results(result["messages"]))
    return content_text(reply.content)


def build_agent():
    return create_react_agent(get_llm(), TOOLS, prompt=SYSTEM_PROMPT)


def main():
    agent = build_agent()
    memory = ConversationMemory(llm_factory=get_llm)

    if USER_PROMPT:
        print(run(agent, memory, USER_PROMPT))
        return

    print("MBE agent ready. Type a request ('clear' to reset history, 'exit' to quit).")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "clear":
            memory.clear()
            reset_model_agent_history()
            print("History cleared.")
            continue
        print(run(agent, memory, user_input))


if __name__ == "__main__":
    main()
