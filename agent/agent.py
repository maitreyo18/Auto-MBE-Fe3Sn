import logging
import os
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mbe_agent")

from langchain_google_genai import ChatGoogleGenerativeAI

# langchain_core (pulled in above) re-registers its own "always" filter for
# LangChainPendingDeprecationWarning, which takes priority over filters
# added before it. Re-apply ours right before the import that triggers the
# warning so ours wins.
warnings.filterwarnings("ignore")
from langgraph.prebuilt import create_react_agent
from tools import TOOLS

SYSTEM_PROMPT = """You are an assistant for an MBE growth lab.
When information needed for one tool is missing but available from another
tool's output, call that tool first and chain the result forward.

Whenever the user asks about previous, prior, past, historical, existing, or
already-present experiments/data/runs (not a new hypothetical configuration),
call analyze_previous_experiments to look up and analyze the real data in
data/train_compiled.csv before answering -- do not guess or rely on memory.
Restrict comparisons and analysis to these six parameters unless the user
explicitly asks about a different column: growthtime, filamentpower,
flux_ratio, Substrate_quality, EDS_ratio, RHEED_Quality_Film.

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
Do not use this workflow for single-point predictions (predict_quality) or
iterative optimization (run_active_learning_loop)."""

# Set to a non-empty string to run that request once instead of the REPL.
USER_PROMPT = ""


def extract_text(content):
    """content is a plain string, or (for models with native reasoning, e.g.
    Gemma) a list of blocks like {'type': 'thinking', ...} / {'type': 'text',
    ...}. Keep only the final answer text."""
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "") for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def run(agent, history, user_message):
    history.append(("user", user_message))
    try:
        result = agent.invoke({"messages": history})
    except Exception:
        history.pop()
        logger.exception("Agent invocation failed for message: %r", user_message)
        return "Sorry, something went wrong while processing that request. Please try again."

    used_tools = any(getattr(m, "type", None) == "tool" for m in result["messages"])
    if used_tools:
        logger.info("Agent used one or more tools to answer this request.")

    reply = result["messages"][-1]
    history.append((reply.type, reply.content))
    return extract_text(reply.content)


def build_agent():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY before running the agent.")

    llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", google_api_key=api_key)
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def main():
    agent = build_agent()
    history = []

    if USER_PROMPT:
        print(run(agent, history, USER_PROMPT))
        return

    print("MBE agent ready. Type a request ('clear' to reset history, 'exit' to quit).")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "clear":
            history.clear()
            print("History cleared.")
            continue
        print(run(agent, history, user_input))


if __name__ == "__main__":
    main()
