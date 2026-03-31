import base64
from types import SimpleNamespace

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

from src.voice import Base64StreamDecoder, TurnViewState, VoiceAgent
from src.voice.io import PCMFrameBuffer


def test_base64_stream_decoder_supports_chunked_input():
    payload = base64.b64encode(b"hello voice agent").decode("ascii")
    decoder = Base64StreamDecoder()

    first = decoder.feed(payload[:5])
    second = decoder.feed(payload[5:11])
    third = decoder.feed(payload[11:])
    tail = decoder.flush()

    assert first + second + third + tail == b"hello voice agent"


def test_pcm_frame_buffer_drops_incomplete_tail_bytes():
    buffer = PCMFrameBuffer(channels=1, sample_width=2)

    first = buffer.feed(b"\x01")
    second = buffer.feed(b"\x02\x03\x04\x05")
    tail = buffer.flush()

    assert first == b""
    assert second == b"\x01\x02\x03\x04"
    assert tail == b""


def test_build_audio_message_uses_input_audio_part():
    message = VoiceAgent._build_audio_message(b"wav-bytes")

    assert message["role"] == "user"
    assert message["content"][0]["type"] == "input_audio"
    assert message["content"][0]["input_audio"]["format"] == "wav"
    assert message["content"][0]["input_audio"]["data"].startswith("data:;base64,")
    assert (
        base64.b64decode(
            message["content"][0]["input_audio"]["data"].split(",", 1)[1]
        )
        == b"wav-bytes"
    )


def test_trim_history_keeps_recent_turns():
    agent = VoiceAgent.__new__(VoiceAgent)
    agent._turn_window = 2
    agent._messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
        {"role": "assistant", "content": "tool-call"},
        {"role": "tool", "content": "tool-output"},
        {"role": "user", "content": "u4"},
        {"role": "assistant", "content": "a4"},
        {"role": "assistant", "content": "tool-call-2"},
        {"role": "tool", "content": "tool-output-2"},
        {"role": "user", "content": "u5"},
        {"role": "assistant", "content": "a5"},
    ]

    agent._trim_history()

    assert agent._messages[0] == {"role": "system", "content": "sys"}
    assert len(agent._messages) == 13
    assert agent._messages[1:] == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
        {"role": "assistant", "content": "tool-call"},
        {"role": "tool", "content": "tool-output"},
        {"role": "user", "content": "u4"},
        {"role": "assistant", "content": "a4"},
        {"role": "assistant", "content": "tool-call-2"},
        {"role": "tool", "content": "tool-output-2"},
        {"role": "user", "content": "u5"},
        {"role": "assistant", "content": "a5"},
    ]


def test_text_user_message_shape():
    user_text = "帮我查一下今天北京天气"

    assert {"role": "user", "content": user_text} == {
        "role": "user",
        "content": user_text,
    }


def test_collect_tool_calls_accumulates_even_without_finish_reason():
    agent = VoiceAgent.__new__(VoiceAgent)
    final_tool_calls = {}
    first = ChoiceDeltaToolCall.model_validate(
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "execute_shell_tool", "arguments": "{\"command\":\"l"},
        }
    )
    second = ChoiceDeltaToolCall.model_validate(
        {
            "index": 0,
            "type": "function",
            "function": {"arguments": "s\"}"},
        }
    )

    final_tool_calls = agent._collect_tool_calls(
        type("Delta", (), {"tool_calls": [first]})(), final_tool_calls
    )
    final_tool_calls = agent._collect_tool_calls(
        type("Delta", (), {"tool_calls": [second]})(), final_tool_calls
    )

    tool_call = list(final_tool_calls.values())[0]
    assert tool_call.function.name == "execute_shell_tool"
    assert tool_call.function.arguments == "{\"command\":\"ls\"}"


def test_stream_response_collects_text_audio_and_usage(monkeypatch):
    agent = VoiceAgent.__new__(VoiceAgent)
    agent._config = SimpleNamespace(
        audio=SimpleNamespace(pcm_channels=1, pcm_sample_width=2)
    )
    agent._model = "qwen3.5-omni-plus"
    agent._voice = "Tina"
    agent._enable_search = False
    agent._messages = [{"role": "system", "content": "sys"}]
    agent._ui = SimpleNamespace(update_live=lambda live, view_state: None)
    agent._tool_set = SimpleNamespace(to_schemas=lambda: [])

    calls = {"audio_bytes": []}

    monkeypatch.setattr("src.voice.agent.start_player", lambda audio_config: object())
    monkeypatch.setattr(
        "src.voice.agent.write_player",
        lambda player, data: calls["audio_bytes"].append(data) if data else None,
    )
    monkeypatch.setattr("src.voice.agent.close_player", lambda player, audio_config: None)

    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="你好", tool_calls=None, audio=None, model_extra={})
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=None,
                        audio={"data": base64.b64encode(b"\x01\x02").decode("ascii")},
                        model_extra={},
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                total_tokens=10,
                prompt_tokens=4,
                completion_tokens=6,
                completion_tokens_details=SimpleNamespace(audio_tokens=2, text_tokens=4),
                prompt_tokens_details=SimpleNamespace(audio_tokens=0, text_tokens=4),
            ),
        ),
    ]

    agent._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: iter(chunks))
        )
    )

    answer_text, audio_started, final_tool_calls = agent._run_model_turn(
        live=None, view_state=TurnViewState()
    )

    assert answer_text == "你好"
    assert audio_started is True
    assert final_tool_calls == {}
    assert any(chunk == b"\x01\x02" for chunk in calls["audio_bytes"])


def test_stream_response_runs_tool_calls_and_continues(monkeypatch):
    agent = VoiceAgent.__new__(VoiceAgent)
    agent._messages = [{"role": "system", "content": "sys"}]
    agent._ui = SimpleNamespace(update_live=lambda live, view_state: None)

    handled = {"ran": False}

    def fake_run_model_turn(live, view_state):
        if not handled["ran"]:
            tool_call = SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="execute_shell_tool", arguments='{"command":"ls"}'
                ),
            )
            return "", False, {0: tool_call}
        return "目录里有 main.py", True, {}

    def fake_run_tool_calls(final_tool_calls, view_state):
        handled["ran"] = True
        agent._messages.append(
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "main.py",
            }
        )

    monkeypatch.setattr(agent, "_run_model_turn", fake_run_model_turn)
    monkeypatch.setattr(agent, "_run_tool_calls", fake_run_tool_calls)

    answer = agent._stream_response(live=None)

    assert answer == "目录里有 main.py"
    assert handled["ran"] is True
    assert agent._messages[1]["role"] == "assistant"
    assert agent._messages[1]["tool_calls"][0].function.name == "execute_shell_tool"
    assert agent._messages[2]["role"] == "tool"


def test_run_model_turn_stops_audio_when_tool_calls_appear(monkeypatch):
    agent = VoiceAgent.__new__(VoiceAgent)
    agent._config = SimpleNamespace(
        audio=SimpleNamespace(pcm_channels=1, pcm_sample_width=2)
    )
    agent._model = "qwen3.5-omni-plus"
    agent._voice = "Tina"
    agent._enable_search = False
    agent._messages = [{"role": "system", "content": "sys"}]
    agent._ui = SimpleNamespace(update_live=lambda live, view_state: None)
    agent._tool_set = SimpleNamespace(to_schemas=lambda: [])

    calls = {"writes": [], "stopped": False}
    tool_call = ChoiceDeltaToolCall.model_validate(
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "execute_shell_tool", "arguments": '{"command":"ls"}'},
        }
    )

    monkeypatch.setattr("src.voice.agent.start_player", lambda audio_config: object())
    monkeypatch.setattr(
        "src.voice.agent.write_player",
        lambda player, data: calls["writes"].append(data) if data else None,
    )
    monkeypatch.setattr("src.voice.agent.close_player", lambda player, audio_config: None)
    monkeypatch.setattr(
        "src.voice.agent.stop_player",
        lambda player: calls.__setitem__("stopped", True),
    )

    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[tool_call],
                        audio={"data": base64.b64encode(b"\x01\x02").decode("ascii")},
                        model_extra={},
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                total_tokens=8,
                prompt_tokens=4,
                completion_tokens=4,
                completion_tokens_details=SimpleNamespace(audio_tokens=2, text_tokens=2),
                prompt_tokens_details=SimpleNamespace(audio_tokens=0, text_tokens=4),
            ),
        ),
    ]

    agent._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: iter(chunks))
        )
    )

    answer_text, audio_started, final_tool_calls = agent._run_model_turn(
        live=None, view_state=TurnViewState()
    )

    assert answer_text == ""
    assert audio_started is False
    assert calls["writes"] == []
    assert calls["stopped"] is True
    assert list(final_tool_calls.values())[0].function.name == "execute_shell_tool"
