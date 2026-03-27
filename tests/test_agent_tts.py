from pathlib import Path

from agent import MiniAgent
from tools.tool_base import ToolReturnValue


def test_speak_answer_generates_audio_and_plays(monkeypatch, tmp_path):
    agent = MiniAgent.__new__(MiniAgent)
    calls = {}

    monkeypatch.setattr(
        MiniAgent,
        "_build_tts_output_path",
        staticmethod(lambda: tmp_path / "reply.mp3"),
    )

    class FakeEdgeTtsTool:
        def __call__(self, params):
            calls["text"] = params.text
            calls["output_path"] = params.output_path
            Path(params.output_path).write_bytes(b"fake-audio")
            return ToolReturnValue(output="ok", is_error=False)

    def fake_run(cmd, check, capture_output, text):
        calls["mpv_cmd"] = cmd

    monkeypatch.setattr("agent.EdgeTtsTool", FakeEdgeTtsTool)
    monkeypatch.setattr("agent.subprocess.run", fake_run)

    agent._speak_answer("你好，开始朗读")

    assert calls["text"] == "你好，开始朗读"
    assert calls["output_path"] == str(tmp_path / "reply.mp3")
    assert calls["mpv_cmd"] == ["mpv", "--really-quiet", str(tmp_path / "reply.mp3")]


def test_speak_answer_cleans_markdown_and_emoji(monkeypatch, tmp_path):
    agent = MiniAgent.__new__(MiniAgent)
    calls = {}

    monkeypatch.setattr(
        MiniAgent,
        "_build_tts_output_path",
        staticmethod(lambda: tmp_path / "reply.mp3"),
    )

    class FakeEdgeTtsTool:
        def __call__(self, params):
            calls["text"] = params.text
            return ToolReturnValue(output="ok", is_error=False)

    def fake_run(cmd, check, capture_output, text):
        calls["mpv_cmd"] = cmd

    monkeypatch.setattr("agent.EdgeTtsTool", FakeEdgeTtsTool)
    monkeypatch.setattr("agent.subprocess.run", fake_run)

    agent._speak_answer("# 标题\n- **第一条** 😀\n- 第二条 [链接](https://example.com)\n")

    assert calls["text"] == "标题\n第一条\n第二条 链接"


def test_speak_answer_skips_empty_text(monkeypatch, tmp_path):
    agent = MiniAgent.__new__(MiniAgent)
    invoked = {"tts": False}

    monkeypatch.setattr(
        MiniAgent,
        "_build_tts_output_path",
        staticmethod(lambda: tmp_path / "reply.mp3"),
    )

    class FakeEdgeTtsTool:
        def __call__(self, params):
            invoked["tts"] = True
            return ToolReturnValue(output="ok", is_error=False)

    monkeypatch.setattr("agent.EdgeTtsTool", FakeEdgeTtsTool)

    agent._speak_answer("   ")

    assert invoked["tts"] is False


def test_clean_tts_text_drops_formatting_only_content():
    cleaned = MiniAgent._clean_tts_text("## Hello\n- `code` item 🚀\n> quoted")

    assert cleaned == "Hello\ncode item\nquoted"
