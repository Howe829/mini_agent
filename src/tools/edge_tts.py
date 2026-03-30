import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Type, override

from pydantic import BaseModel, Field

from src.tools.tool_base import ToolBase, ToolReturnValue

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_OUTPUT_DIR = Path("exports")


class EdgeTtsToolParams(BaseModel):
    text: str = Field(description="要合成为语音的文本内容")
    voice: str = Field(
        default=DEFAULT_VOICE, description="edge-tts 音色，例如 zh-CN-XiaoxiaoNeural"
    )
    rate: str = Field(default="+20%", description="语速，例如 +10% 或 -20%")
    volume: str = Field(default="+0%", description="音量，例如 +20% 或 -30%")
    pitch: str = Field(default="+0Hz", description="音高，例如 +10Hz 或 -10Hz")
    output_path: Optional[str] = Field(
        default=None, description="输出音频文件路径，默认写入 exports/ 目录"
    )


class EdgeTtsTool(ToolBase):
    name: str = "edge_tts_tool"
    description: str = "使用 edge-tts 将文本合成为语音文件"
    alias: Optional[str] = "文字转语音"
    params_class: Type[BaseModel] = EdgeTtsToolParams

    @override
    def __call__(self, params: EdgeTtsToolParams) -> ToolReturnValue:
        if not params.text.strip():
            return ToolReturnValue(output="文本内容不能为空", is_error=True)

        output_path = self._resolve_output_path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            asyncio.run(self._save_audio(params, output_path))
        except Exception as exc:
            return ToolReturnValue(output=f"edge-tts 合成失败: {exc}", is_error=True)

        return ToolReturnValue(
            output=(
                f"已生成语音文件: {output_path}\n"
                f"voice={params.voice}, rate={params.rate}, volume={params.volume}, pitch={params.pitch}"
            ),
            is_error=False,
        )

    @staticmethod
    async def _save_audio(params: EdgeTtsToolParams, output_path: Path) -> None:
        communicate = EdgeTtsTool._build_communicate(params)
        await communicate.save(str(output_path))

    @staticmethod
    def _build_communicate(params: EdgeTtsToolParams):
        from edge_tts import Communicate

        return Communicate(
            text=params.text,
            voice=params.voice,
            rate=params.rate,
            volume=params.volume,
            pitch=params.pitch,
        )

    @staticmethod
    def _resolve_output_path(output_path: str | None) -> Path:
        if output_path:
            path = Path(output_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = DEFAULT_OUTPUT_DIR / f"edge_tts_{timestamp}.mp3"

        if path.suffix:
            return path
        return path.with_suffix(".mp3")
