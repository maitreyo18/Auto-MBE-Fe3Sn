import types

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent

import agent as main_agent
import memory as mem


class FakeLLM:
    def __init__(self, text="RESULTS: summary text"):
        self.text, self.prompts = text, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return types.SimpleNamespace(content=self.text)


def filled(llm, n):
    m = mem.ConversationMemory(llm_factory=lambda: llm)
    for i in range(n):
        m.record(f"question {i}", f"answer {i}")
    return m


def test_defaults_are_fifteen_and_five():
    assert (mem.MAX_TURNS, mem.KEEP_RECENT) == (15, 5)


def test_no_compression_before_fifteen():
    llm = FakeLLM()
    m = filled(llm, 14)
    assert llm.prompts == [] and len(m.turns) == 14 and m.summary == ""


def test_fifteenth_conversation_compresses_oldest_ten_keeps_last_five():
    llm = FakeLLM()
    m = filled(llm, 15)
    assert len(llm.prompts) == 1 and "question 0" in llm.prompts[0] and "question 9" in llm.prompts[0]
    assert "question 10" not in llm.prompts[0]
    assert [t.user for t in m.turns] == [f"question {i}" for i in range(10, 15)]
    assert m.summary == "RESULTS: summary text"


def test_messages_order_and_alternation():
    m = filled(FakeLLM(), 15)
    msgs = m.messages("new")
    roles = [r for r, _ in msgs]
    assert roles == ["user", "ai"] * 6 + ["user"]
    assert "summary text" in msgs[0][1] and msgs[-1] == ("user", "new")
    assert msgs[2] == ("user", "question 10")


def test_summary_rolls_forward_into_next_compression():
    llm = FakeLLM()
    m = filled(llm, 25)
    assert len(llm.prompts) == 2 and "RESULTS: summary text" in llm.prompts[1]
    assert "question 10" in llm.prompts[1] and "question 19" in llm.prompts[1]
    assert len(m.turns) == 5


def test_failure_keeps_everything_then_retries():
    class Flaky(FakeLLM):
        def invoke(self, prompt):
            if not self.prompts:
                self.prompts.append(prompt)
                raise RuntimeError("API down")
            return super().invoke(prompt)

    llm = Flaky()
    m = filled(llm, 15)
    assert len(m.turns) == 15 and m.summary == ""
    m.record("question 15", "answer 15")
    assert len(m.turns) == 5 and m.summary
    assert "question 11" == m.turns[0].user


def test_empty_summary_counts_as_failure():
    m = filled(FakeLLM(text="   "), 15)
    assert len(m.turns) == 15 and m.summary == ""


def test_persistent_failure_uses_digest_at_hard_limit():
    m = mem.ConversationMemory(llm_factory=None)
    for i in range(30):
        m.record(f"question {i}", "x" * 1000)
    assert len(m.turns) == 5
    assert "question 0" in m.summary and "x" * 301 not in m.summary


def test_model_ids_survive_a_summary_that_drops_them():
    m = mem.ConversationMemory(llm_factory=lambda: FakeLLM("RESULTS: fine"), max_turns=3, keep_recent=1)
    m.record("use it", "saved auto_mlp_EDS_ratio_20261105-214105 and lab_EDS_ratio")
    m.record("b", "c")
    m.record("d", "e")
    assert "auto_mlp_EDS_ratio_20261105-214105" in m.summary and "lab_EDS_ratio" in m.summary


def test_ids_already_in_summary_are_not_repeated():
    m = mem.ConversationMemory(llm_factory=lambda: FakeLLM("MODELS: lab_EDS_ratio"), max_turns=2, keep_recent=1)
    m.record("a", "lab_EDS_ratio")
    m.record("b", "c")
    assert "MODEL IDS ALSO MENTIONED" not in m.summary


def test_long_messages_are_clipped_for_the_summarizer_only():
    llm = FakeLLM()
    m = mem.ConversationMemory(llm_factory=lambda: llm, max_turns=3, keep_recent=2)
    m.record("start" + "y" * 20000 + "end", "ok")
    assert len(m.turns[0].user) > 20000
    m.record("b", "c")
    m.record("d", "e")
    assert "start" in llm.prompts[0] and "end" in llm.prompts[0] and "omitted" in llm.prompts[0]
    assert len(llm.prompts[0]) < 20000


def test_thinking_blocks_are_not_stored():
    m = mem.ConversationMemory()
    m.record("q", [{"type": "thinking", "text": "hidden"}, {"type": "text", "text": "answer"}])
    assert m.turns == [mem.Turn("q", "answer", ())]


def test_clear_and_invalid_settings():
    m = filled(FakeLLM(), 15)
    m.clear()
    assert m.summary == "" and m.turns == []
    with pytest.raises(ValueError):
        mem.ConversationMemory(max_turns=5, keep_recent=5)


def test_failed_agent_call_is_not_recorded():
    class Broken:
        def invoke(self, state):
            raise RuntimeError("boom")

    m = mem.ConversationMemory()
    assert "Sorry" in main_agent.run(Broken(), m, "hello")
    assert m.turns == []


class ToolReadyFake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_compressed_memory_works_through_a_real_langgraph_agent():
    seen = []

    class Recording(ToolReadyFake):
        def _generate(self, messages, *args, **kwargs):
            seen.append(list(messages))
            return super()._generate(messages, *args, **kwargs)

    graph = create_react_agent(Recording(messages=iter([AIMessage(content=f"reply {i}") for i in range(20)])), [], prompt="sys")
    m = mem.ConversationMemory(llm_factory=lambda: FakeLLM("RESULTS: kept fact"))
    for i in range(16):
        assert main_agent.run(graph, m, f"q{i}") == f"reply {i}"
    last = seen[-1]
    text = " ".join(str(x.content) for x in last)
    assert "kept fact" in text and "q15" in text and "q10" in text and "q9" not in text
    assert [x.type for x in last][:1] == ["system"] and sum(x.type == "system" for x in last) == 1


# ----------------------------------------------------------- tool results ----

def _msg(kind, content="", name=None):
    return types.SimpleNamespace(type=kind, content=content, name=name)


def test_new_tool_results_takes_only_the_latest_request():
    messages = [_msg("human", "old"), _msg("tool", "old result", "a"), _msg("ai", "x"),
                _msg("human", "new"), _msg("ai", ""), _msg("tool", "r1", "rheed_quality"),
                _msg("tool", "r2", "other"), _msg("ai", "done")]
    assert mem.new_tool_results(messages) == [("rheed_quality", "r1"), ("other", "r2")]
    assert mem.new_tool_results([_msg("human", "q"), _msg("ai", "a")]) == []


def test_tool_results_return_in_the_earlier_reply_not_as_tool_messages():
    m = mem.ConversationMemory()
    m.record("score this image", "Quality is 74.5.", [("rheed_quality", '{"subcfwhm": 45.34}')])
    msgs = m.messages("what was subcfwhm?")
    assert [r for r, _ in msgs] == ["user", "ai", "user"]
    assert "subcfwhm" in msgs[1][1] and mem.TOOL_HEADER in msgs[1][1] and msgs[1][1].endswith("Quality is 74.5.")


def test_reply_returned_to_user_is_stored_without_the_tool_text():
    m = mem.ConversationMemory()
    m.record("q", "answer", [("t", "data")])
    assert m.turns[0].assistant == "answer" and m.turns[0].tools == (("t", "data"),)


def test_tool_output_identical_to_reply_is_not_duplicated():
    m = mem.ConversationMemory()
    m.record("q", "same text", [("ask_model_builder", "same text"), ("t", "other")])
    assert m.turns[0].tools == (("t", "other"),)


def test_size_caps_recent_and_older_turns():
    m = mem.ConversationMemory(llm_factory=lambda: FakeLLM())
    m.record("first", "a", [("big", "z" * 50000)])
    assert len(m.turns[0].tools[0][1]) < mem.MAX_TOOL_CHARS + 100
    for i in range(5):
        m.record(f"q{i}", "a")
    old = m.messages("next")[1][1]
    assert mem.OLDER_TOOL_CHARS <= old.count("z") <= mem.OLDER_TOOL_CHARS and "omitted" in old
    m2 = mem.ConversationMemory()
    m2.record("first", "a", [("big", "z" * 50000)])
    assert m2.messages("next")[1][1].count("z") == mem.MAX_TOOL_CHARS


def test_too_many_tools_keeps_latest():
    m = mem.ConversationMemory()
    m.record("q", "a", [(f"t{i}", "x") for i in range(12)])
    assert len(m.turns[0].tools) == mem.MAX_TOOLS_PER_TURN and m.turns[0].tools[-1][0] == "t11"


def test_summarizer_sees_tool_results_and_their_model_ids_survive():
    llm = FakeLLM("RESULTS: fine")
    m = mem.ConversationMemory(llm_factory=lambda: llm, max_turns=3, keep_recent=1)
    m.record("train", "done", [("train_models", '{"model_id": "gpr_y_20261105-101010", "test_rmse": 1.23}')])
    m.record("b", "c")
    m.record("d", "e")
    assert "TOOL RESULTS" in llm.prompts[0] and "1.23" in llm.prompts[0]
    assert "gpr_y_20261105-101010" in m.summary


def test_prompts_explain_recorded_tool_results():
    import model_agent
    assert "recorded by the" in main_agent.SYSTEM_PROMPT and "recorded by the" in model_agent.build_system_prompt()


def test_follow_up_sees_real_tool_output_through_a_real_langgraph_agent():
    from langchain_core.tools import tool

    @tool
    def lookup(x: str) -> dict:
        """Returns a number."""
        return {"subcfwhm": 45.34}

    seen = []

    class Recording(ToolReadyFake):
        def _generate(self, messages, *args, **kwargs):
            seen.append(list(messages))
            return super()._generate(messages, *args, **kwargs)

    script = iter([
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"x": "a"}, "id": "c1", "type": "tool_call"}]),
        AIMessage(content="Quality is fine."),
        AIMessage(content="It was 45.34."),
    ])
    graph = create_react_agent(Recording(messages=script), [lookup], prompt="sys")
    m = mem.ConversationMemory()
    assert main_agent.run(graph, m, "check the image") == "Quality is fine."
    assert m.turns[0].tools[0][0] == "lookup" and "45.34" in m.turns[0].tools[0][1]
    assert main_agent.run(graph, m, "what was subcfwhm?") == "It was 45.34."
    final = seen[-1]
    assert "45.34" in " ".join(str(x.content) for x in final)
    assert not any(x.type == "tool" for x in final)
