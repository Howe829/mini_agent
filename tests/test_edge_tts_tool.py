import json
from pathlib import Path

from tools.edge_tts import EdgeTtsTool


def test_edge_tts_tool_writes_audio_file(tmp_path, monkeypatch):
    saved = {}

    class FakeCommunicate:
        def __init__(self, text, voice, rate, volume, pitch):
            saved["args"] = {
                "text": text,
                "voice": voice,
                "rate": rate,
                "volume": volume,
                "pitch": pitch,
            }

        async def save(self, output_path):
            Path(output_path).write_bytes(b"fake-mp3")

    monkeypatch.setattr(EdgeTtsTool, "_build_communicate", lambda params: FakeCommunicate(
        text=params.text,
        voice=params.voice,
        rate=params.rate,
        volume=params.volume,
        pitch=params.pitch,
    ))

    tool = EdgeTtsTool()
    output_path = tmp_path / "hello.mp3"
    func_args = json.dumps(
        {
            "text": "你好，豪哥",
            "voice": "zh-CN-XiaoxiaoNeural",
            "output_path": str(output_path),
            "rate": "+15%",
            "volume": "+5%",
            "pitch": "+10Hz",
        },
        ensure_ascii=False,
    )
    result = tool.call(func_args)

    assert result.is_error is False
    assert output_path.exists()
    assert output_path.read_bytes() == b"fake-mp3"
    assert saved["args"]["text"] == "你好，豪哥"
    assert saved["args"]["rate"] == "+15%"


def test_edge_tts_tool_rejects_empty_text():
    tool = EdgeTtsTool()

    result = tool.call('{"text":"   "}')

    assert result.is_error is True
    assert "不能为空" in result.output
