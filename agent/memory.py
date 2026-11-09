"""Conversation memory with rolling compression.

One "conversation" is a user message plus the agent's reply. After MAX_TURNS of
them, the oldest are folded into a rolling summary and the latest KEEP_RECENT stay
word for word. The summary is structured so numbers, identifiers and decisions
survive, and each new summary builds on the previous one.

Tool results are kept with their turn so follow-up questions can reuse them. They
go back as text inside the earlier reply, because chat APIs reject replayed tool
messages unless they pair up exactly with their tool calls. Sizes are capped: the
latest turns keep up to MAX_TOOL_CHARS per tool, older ones a shorter excerpt.

Compression never loses context silently: exact model ids are re-attached if the
summary drops them, and if the LLM call fails the turns are kept and retried after
the next reply. If it keeps failing, a plain text digest stops unbounded growth.
"""
import logging
import re
from typing import NamedTuple

logger = logging.getLogger("mbe_agent.memory")

MAX_TURNS = 15
KEEP_RECENT = 5

MAX_CHARS_PER_MESSAGE = 8000
MAX_TOOL_CHARS = 6000
OLDER_TOOL_CHARS = 1500
MAX_TOOLS_PER_TURN = 8
TOOL_HEADER = "[Tool results from this turn, recorded by the system:"
DIGEST_CHARS = 300

SUMMARY_PROMPT = """You maintain the memory of an MBE growth lab assistant. Merge the
existing summary and the new conversation into one updated summary. Later statements
override earlier ones; drop anything superseded.

Use exactly these sections, and write "None" when one is empty:
USER GOALS AND PREFERENCES: what the user wants and how they want it done.
DECISIONS: what was decided, and instructions that still apply.
RESULTS: findings with exact numbers, units, run numbers and conditions, and which model or test produced each. Include tool results a follow-up question could need.
MODELS AND FILES: exact model ids, file paths and settings, copied character for character.
OPEN QUESTIONS: anything still unanswered or waiting on the user.

Copy numbers exactly and never round or invent them. Keep caveats attached to the
result they qualify. Leave out greetings and repetition. Write plain text only.

EXISTING SUMMARY:
{summary}

NEW CONVERSATION:
{conversation}

UPDATED SUMMARY:"""

# Ids of saved models end in a timestamp, e.g. auto_mlp_EDS_ratio_20261105-214105.
IDENTIFIER_RE = re.compile(r"\b(?:lab_[A-Za-z_]+|[A-Za-z][\w.-]*_\d{8}-\d{6}(?:-\d+)?)\b")


def content_text(content):
    """Message content is a string, or (for models with native reasoning, e.g.
    Gemma) a list of blocks. Keeps only the answer text."""
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    )


def _clip(text, limit=MAX_CHARS_PER_MESSAGE):
    """Keeps both ends of very long text, where requests and conclusions sit."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - limit} characters omitted ...]\n{text[-half:]}"


def new_tool_results(messages):
    """(tool name, output text) pairs from the latest request's messages, i.e. those
    after the last human message."""
    results = []
    for message in reversed(messages):
        kind = getattr(message, "type", None)
        if kind == "human":
            break
        if kind == "tool":
            results.append((getattr(message, "name", None) or "tool", content_text(message.content)))
    return results[::-1]


class Turn(NamedTuple):
    user: str
    assistant: str
    tools: tuple = ()


def _format_tools(tools, limit):
    return "\n".join(f"{name}: {_clip(text, limit)}" for name, text in tools)


class ConversationMemory:
    def __init__(self, llm_factory=None, max_turns=MAX_TURNS, keep_recent=KEEP_RECENT):
        """llm_factory returns the summarizing LLM, and is only called when compressing."""
        if not 0 < keep_recent < max_turns:
            raise ValueError("need 0 < keep_recent < max_turns")
        self.llm_factory = llm_factory
        self.max_turns = max_turns
        self.keep_recent = keep_recent
        self.hard_limit = 2 * max_turns
        self.summary = ""
        self.turns = []

    def messages(self, user_message):
        """Everything to send for this request. The summary goes in as an exchange,
        not a system message, because some chat APIs reject a second system message."""
        out = []
        if self.summary:
            out.append(("user", f"Summary of our earlier conversation, for reference:\n{self.summary}"))
            out.append(("ai", "Understood. I will use this as context."))
        first_recent = len(self.turns) - self.keep_recent
        for i, turn in enumerate(self.turns):
            out.append(("user", turn.user))
            assistant = turn.assistant
            if turn.tools:
                limit = MAX_TOOL_CHARS if i >= first_recent else OLDER_TOOL_CHARS
                assistant = f"{TOOL_HEADER}\n{_format_tools(turn.tools, limit)}\n]\n\n{assistant}"
            out.append(("ai", assistant))
        out.append(("user", user_message))
        return out

    def record(self, user_message, reply_content, tool_results=None):
        """Stores a finished exchange and compresses if it is due."""
        reply = content_text(reply_content)
        # A tool output repeated verbatim in the reply adds nothing, e.g. a relayed sub-agent answer.
        tools = tuple((name, _clip(text, MAX_TOOL_CHARS)) for name, text in (tool_results or []) if text.strip() != reply.strip())
        if len(tools) > MAX_TOOLS_PER_TURN:
            logger.info("Keeping the latest %d of %d tool results.", MAX_TOOLS_PER_TURN, len(tools))
            tools = tools[-MAX_TOOLS_PER_TURN:]
        self.turns.append(Turn(user_message, reply, tools))
        if len(self.turns) >= self.max_turns:
            self.compress()

    def clear(self):
        self.summary = ""
        self.turns = []

    def compress(self):
        """Returns True if older turns were folded into the summary."""
        old, recent = self.turns[: -self.keep_recent], self.turns[-self.keep_recent :]
        try:
            summary = self._summarize(old)
        except Exception:
            logger.warning("Context compression failed; keeping all %d turns.", len(self.turns), exc_info=True)
            if len(self.turns) < self.hard_limit:
                return False
            summary = self._digest(old)
        self.summary = self._keep_identifiers(summary, old)
        self.turns = recent
        logger.info("Compressed %d conversations into the summary; kept the latest %d.", len(old), len(recent))
        return True

    def _summarize(self, turns):
        if self.llm_factory is None:
            raise RuntimeError("no LLM available for compression")
        conversation = "\n\n".join(
            f"USER: {_clip(t.user)}\n"
            + (f"TOOL RESULTS:\n{_format_tools(t.tools, MAX_TOOL_CHARS)}\n" if t.tools else "")
            + f"ASSISTANT: {_clip(t.assistant)}"
            for t in turns
        )
        prompt = SUMMARY_PROMPT.format(summary=self.summary or "None", conversation=conversation)
        summary = content_text(self.llm_factory().invoke(prompt).content).strip()
        if not summary:
            raise RuntimeError("the LLM returned an empty summary")
        return summary

    def _digest(self, turns):
        """No-LLM fallback: user requests with the start of each reply."""
        lines = [f"USER: {t.user[:DIGEST_CHARS]}\nASSISTANT: {t.assistant[:DIGEST_CHARS]}" for t in turns]
        previous = f"{self.summary}\n\n" if self.summary else ""
        return previous + "Unsummarized earlier conversation (replies shortened):\n" + "\n".join(lines)

    def _keep_identifiers(self, summary, turns):
        """Model ids are needed verbatim later, so re-attach any the summary lost."""
        source = self.summary + " " + " ".join(f"{t.user} {t.assistant} {_format_tools(t.tools, MAX_TOOL_CHARS)}" for t in turns)
        lost = sorted(set(IDENTIFIER_RE.findall(source)) - set(IDENTIFIER_RE.findall(summary)))
        if lost:
            summary += "\nMODEL IDS ALSO MENTIONED: " + ", ".join(lost)
        return summary
