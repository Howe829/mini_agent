from agent import MiniAgent


def test_speak_answer_generates_audio_and_plays(monkeypatch):
    agent = MiniAgent.__new__(MiniAgent)
    calls = {}

    async def fake_stream(self, answer):
        calls["text"] = answer

    monkeypatch.setattr(MiniAgent, "_stream_tts_to_mpv", fake_stream)

    agent._speak_answer("你好，开始朗读")

    assert calls["text"] == "你好，开始朗读"


def test_speak_answer_cleans_markdown_and_emoji(monkeypatch):
    agent = MiniAgent.__new__(MiniAgent)
    calls = {}

    async def fake_stream(self, answer):
        calls["text"] = answer

    monkeypatch.setattr(MiniAgent, "_stream_tts_to_mpv", fake_stream)

    agent._speak_answer(
        "# 标题\n- **第一条** 😀\n- 第二条 [链接](https://example.com)\n"
    )

    assert calls["text"] == "标题\n第一条\n第二条 链接"


def test_speak_answer_skips_empty_text(monkeypatch):
    agent = MiniAgent.__new__(MiniAgent)
    invoked = {"stream": False}

    async def fake_stream(self, answer):
        invoked["stream"] = True

    monkeypatch.setattr(MiniAgent, "_stream_tts_to_mpv", fake_stream)

    agent._speak_answer("   ")

    assert invoked["stream"] is False


def test_stream_tts_to_mpv_uses_pipe_and_cache(monkeypatch):
    agent = MiniAgent.__new__(MiniAgent)
    calls = {"audio": []}

    class FakePipe:
        def write(self, data):
            calls["audio"].append(data)

        def flush(self):
            calls["flushed"] = True

        def close(self):
            calls["closed"] = True

    class FakeReadPipe:
        def read(self):
            return b""

    class FakeProcess:
        def __init__(self):
            self.stdin = FakePipe()
            self.stdout = FakeReadPipe()
            self.stderr = FakeReadPipe()

        def wait(self):
            return 0

        def kill(self):
            calls["killed"] = True

    async def fake_iter(self, answer):
        calls["stream_text"] = answer
        yield b"chunk-1"
        yield b"chunk-2"

    def fake_popen(cmd, stdin, stdout, stderr):
        calls["mpv_cmd"] = cmd
        return FakeProcess()

    monkeypatch.setattr(MiniAgent, "_iter_tts_audio_chunks", fake_iter)
    monkeypatch.setattr("agent.subprocess.Popen", fake_popen)

    import asyncio

    asyncio.run(agent._stream_tts_to_mpv("你好，流式播放"))

    assert calls["stream_text"] == "你好，流式播放"
    assert calls["audio"] == [b"chunk-1", b"chunk-2"]
    assert calls["mpv_cmd"] == [
        "mpv",
        "--really-quiet",
        "--cache=yes",
        "--cache-secs=20",
        "--demuxer-max-bytes=8MiB",
        "-",
    ]
    assert calls["closed"] is True


def test_clean_tts_text_drops_formatting_only_content():
    cleaned = MiniAgent._clean_tts_text("## Hello\n- `code` item 🚀\n> quoted")

    assert cleaned == "Hello\ncode item\nquoted"
