import base64
import binascii
import signal
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.voice.config import VoiceAudioConfig


class Base64StreamDecoder:
    def __init__(self):
        self._buffer = ""

    def feed(self, chunk: str | None) -> bytes:
        if not chunk:
            return b""

        self._buffer += "".join(chunk.split())
        complete_len = (len(self._buffer) // 4) * 4
        if complete_len == 0:
            return b""

        encoded = self._buffer[:complete_len]
        self._buffer = self._buffer[complete_len:]
        return self._decode(encoded)

    def flush(self) -> bytes:
        if not self._buffer:
            return b""

        if len(self._buffer) < 4:
            self._buffer = ""
            return b""

        padded = self._buffer + "=" * ((4 - len(self._buffer) % 4) % 4)
        self._buffer = ""
        return self._decode(padded)

    @staticmethod
    def _decode(encoded: str) -> bytes:
        try:
            return base64.b64decode(encoded, validate=False)
        except (binascii.Error, ValueError):
            return b""


class PCMFrameBuffer:
    def __init__(self, channels: int, sample_width: int):
        self._frame_size = channels * sample_width
        self._buffer = b""

    def feed(self, chunk: bytes) -> bytes:
        if not chunk:
            return b""

        self._buffer += chunk
        complete_len = (len(self._buffer) // self._frame_size) * self._frame_size
        if complete_len == 0:
            return b""

        output = self._buffer[:complete_len]
        self._buffer = self._buffer[complete_len:]
        return output

    def flush(self) -> bytes:
        output = self.feed(b"")
        self._buffer = b""
        return output


def record_audio(stop_callback, audio_config: VoiceAudioConfig) -> bytes:
    temp_path = Path(tempfile.gettempdir()) / "mini-agent-voice-input.wav"
    temp_path.unlink(missing_ok=True)

    process = _start_recording_process(temp_path, audio_config)
    try:
        stop_callback()
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    if process.returncode not in (0, 255):
        raise RuntimeError(
            stderr.strip() or f"recording backend exited with {process.returncode}"
        )
    if not temp_path.exists():
        raise RuntimeError("recorded audio file was not created")

    try:
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)


def build_audio_message(audio_bytes: bytes) -> dict:
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    return {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": f"data:;base64,{encoded_audio}",
                    "format": "wav",
                },
            }
        ],
    }


def start_player(audio_config: VoiceAudioConfig):
    return subprocess.Popen(
        _build_mpv_raw_audio_args(audio_config),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def write_player(player: subprocess.Popen | None, audio_bytes: bytes):
    if not audio_bytes or player is None or player.stdin is None:
        return

    try:
        player.stdin.write(audio_bytes)
        player.stdin.flush()
    except BrokenPipeError:
        return


def close_player(player: subprocess.Popen | None, audio_config: VoiceAudioConfig):
    if player is None:
        return

    if player.stdin is not None:
        try:
            silence_bytes = b"\x00" * (
                audio_config.pcm_sample_rate
                * audio_config.pcm_channels
                * audio_config.pcm_sample_width
                * audio_config.tail_padding_ms
                // 1000
            )
            if silence_bytes:
                player.stdin.write(silence_bytes)
                player.stdin.flush()
            player.stdin.close()
        except BrokenPipeError:
            pass

    player.wait()


def stop_player(player: subprocess.Popen | None):
    if player is None:
        return

    if player.stdin is not None:
        try:
            player.stdin.close()
        except BrokenPipeError:
            pass

    try:
        player.wait(timeout=1)
    except subprocess.TimeoutExpired:
        player.kill()


def _start_recording_process(
    output_path: Path, audio_config: VoiceAudioConfig
) -> subprocess.Popen:
    parec_path = shutil.which("parec")
    if parec_path:
        return subprocess.Popen(
            [
                parec_path,
                "--record",
                "--device=@DEFAULT_SOURCE@",
                "--channels=1",
                f"--rate={audio_config.input_sample_rate}",
                "--format=s16le",
                "--file-format=wav",
                str(output_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return subprocess.Popen(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "pulse",
                "-i",
                "default",
                "-ac",
                "1",
                "-ar",
                str(audio_config.input_sample_rate),
                "-y",
                str(output_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    raise RuntimeError("no supported recording backend found; expected parec or ffmpeg")


def _build_mpv_raw_audio_args(audio_config: VoiceAudioConfig) -> list[str]:
    return [
        "mpv",
        "--no-terminal",
        "--msg-level=all=no",
        "--force-window=no",
        "--keep-open=no",
        "--cache=yes",
        f"--speed={audio_config.playback_speed}",
        "--demuxer=rawaudio",
        "--demuxer-rawaudio-format=s16le",
        f"--demuxer-rawaudio-rate={audio_config.pcm_sample_rate}",
        f"--demuxer-rawaudio-channels={audio_config.pcm_channels}",
        "fd://0",
    ]
