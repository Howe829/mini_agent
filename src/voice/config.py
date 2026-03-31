import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VoiceAudioConfig:
    input_sample_rate: int = 16000
    pcm_sample_rate: int = 24000
    pcm_channels: int = 1
    pcm_sample_width: int = 2
    tail_padding_ms: int = 180
    playback_speed: float = 1.08

    @classmethod
    def from_env(cls) -> "VoiceAudioConfig":
        return cls(
            input_sample_rate=int(os.getenv("VOICE_AGENT_INPUT_SAMPLE_RATE", "16000")),
            pcm_sample_rate=int(os.getenv("VOICE_AGENT_PCM_SAMPLE_RATE", "24000")),
            pcm_channels=int(os.getenv("VOICE_AGENT_PCM_CHANNELS", "1")),
            pcm_sample_width=int(os.getenv("VOICE_AGENT_PCM_SAMPLE_WIDTH", "2")),
            tail_padding_ms=int(os.getenv("VOICE_AGENT_TAIL_PADDING_MS", "180")),
            playback_speed=float(os.getenv("VOICE_AGENT_PLAYBACK_SPEED", "1.08")),
        )


@dataclass(frozen=True)
class VoiceAgentConfig:
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.5-omni-plus"
    voice: str = "Tina"
    enable_search: bool = False
    max_history_turns: int = 3
    audio: VoiceAudioConfig = field(default_factory=VoiceAudioConfig)

    @classmethod
    def from_env(cls) -> "VoiceAgentConfig":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")

        return cls(
            api_key=api_key,
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            model=os.getenv("QWEN_OMNI_MODEL", "qwen3.5-omni-plus"),
            voice=os.getenv("QWEN_OMNI_VOICE", "Tina"),
            enable_search=os.getenv("QWEN_ENABLE_SEARCH", "0") == "1",
            max_history_turns=int(os.getenv("VOICE_AGENT_MAX_HISTORY_TURNS", "3")),
            audio=VoiceAudioConfig.from_env(),
        )
