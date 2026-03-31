import json
import subprocess
import time

from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from prompt_toolkit import PromptSession
from rich.live import Live

from src.prompts import build_voice_system_prompt
from src.tools.shell import ExecuteShellTool
from src.tools.tool_base import ToolSet
from src.voice.config import VoiceAgentConfig
from src.voice.io import (
    Base64StreamDecoder,
    PCMFrameBuffer,
    build_audio_message,
    close_player,
    record_audio,
    start_player,
    stop_player,
    write_player,
)
from src.voice.ui import TurnViewState, VoiceUI, tail_lines


class VoiceAgent:
    def __init__(self):
        self._config = VoiceAgentConfig.from_env()
        self._ui = VoiceUI(model=self._config.model, voice=self._config.voice)
        self._client = OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.base_url,
        )
        self._model = self._ui.model
        self._voice = self._ui.voice
        self._turn_window = self._config.max_history_turns
        self._enable_search = self._config.enable_search
        self._tool_set = ToolSet(tools=[ExecuteShellTool()])
        self._messages: list[dict] = [
            {"role": "system", "content": self._load_prompt()},
        ]
        self._session: PromptSession | None = None

    def run(self):
        self._ui.print_welcome()
        self._session = self._ui.build_session()

        while True:
            try:
                prompt_text = self._session.prompt("voice> ")
                normalized = prompt_text.strip()

                if normalized.lower() == "exit":
                    self._ui.console.print("Bye")
                    return

                if not self._append_user_input(normalized):
                    continue

                with Live(
                    self._ui.build_response_view(TurnViewState()),
                    console=self._ui.console,
                    refresh_per_second=10,
                    vertical_overflow="crop",
                ) as live:
                    answer_text = self._stream_response(live)

                self._messages.append({"role": "assistant", "content": answer_text})
                self._trim_history()
                self._ui.console.print()
            except EOFError:
                self._ui.console.print("Bye", new_line_start=True)
                return
            except KeyboardInterrupt:
                self._ui.console.print("Bye", new_line_start=True)
                return
            except Exception as exc:
                self._rollback_last_user_message()
                self._ui.print_error(f"voice request failed: {exc}")

    @staticmethod
    def _load_prompt() -> str:
        return build_voice_system_prompt()

    def _append_user_input(self, normalized_input: str) -> bool:
        if normalized_input:
            self._messages.append({"role": "user", "content": normalized_input})
            return True

        try:
            audio_bytes = record_audio(self._wait_for_record_stop, self._config.audio)
        except Exception as exc:
            self._ui.print_warning(f"recording failed: {exc}")
            return False

        if not audio_bytes:
            self._ui.print_warning("no audio captured")
            return False

        self._messages.append(build_audio_message(audio_bytes))
        return True

    def _rollback_last_user_message(self):
        if len(self._messages) > 1 and self._messages[-1]["role"] == "user":
            self._messages.pop()

    def _wait_for_record_stop(self):
        stop_prompt = "recording... press Enter to stop "
        if self._session is not None:
            self._session.prompt(stop_prompt)
        else:
            input(stop_prompt)

    @staticmethod
    def _build_audio_message(audio_bytes: bytes) -> dict:
        return build_audio_message(audio_bytes)

    def _stream_response(self, live: Live | None = None) -> str:
        view_state = TurnViewState(status_text="Thinking...")
        self._ui.update_live(live, view_state)

        while True:
            answer_text, audio_started, final_tool_calls = self._run_model_turn(
                live, view_state
            )

            if final_tool_calls:
                self._append_tool_call_messages(answer_text, final_tool_calls)
                self._run_tool_calls(final_tool_calls, view_state)
                view_state.status_text = "Running tool calls..."
                self._ui.update_live(live, view_state)
                continue

            if not audio_started:
                view_state.tool_lines.append(
                    "final response returned no playable audio chunks"
                )
            view_state.status_text = "Completed"
            self._ui.update_live(live, view_state)
            return answer_text

    def _run_model_turn(
        self, live: Live | None, view_state: TurnViewState
    ) -> tuple[str, bool, dict]:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            tools=self._tool_set.to_schemas(),
            extra_body={"enable_search": self._enable_search},
            modalities=["text", "audio"],
            audio={"voice": self._voice, "format": "wav"},
            stream=True,
            stream_options={"include_usage": True},
        )

        answer_text = ""
        audio_started = False
        final_tool_calls: dict = {}
        player = start_player(self._config.audio)
        decoder = Base64StreamDecoder()
        pcm_buffer = PCMFrameBuffer(
            channels=self._config.audio.pcm_channels,
            sample_width=self._config.audio.pcm_sample_width,
        )
        suppress_audio = False

        try:
            for chunk in completion:
                if not chunk.choices:
                    if chunk.usage:
                        final_audio = pcm_buffer.feed(decoder.flush())
                        if final_audio and not suppress_audio:
                            audio_started = True
                        if not suppress_audio:
                            write_player(player, final_audio)
                        view_state.usage_line = self._format_usage(chunk.usage)
                        self._ui.update_live(live, view_state)
                    continue

                delta = chunk.choices[0].delta
                if delta.tool_calls is not None and not suppress_audio:
                    suppress_audio = True
                    stop_player(player)
                    player = None
                answer_text = self._consume_delta(
                    delta=delta,
                    answer_text=answer_text,
                    final_tool_calls=final_tool_calls,
                    player=player,
                    decoder=decoder,
                    pcm_buffer=pcm_buffer,
                    view_state=view_state,
                    live=live,
                    suppress_audio=suppress_audio,
                )

                if self._extract_audio_data(delta) and not suppress_audio:
                    audio_started = True
        finally:
            final_audio = pcm_buffer.feed(decoder.flush())
            if final_audio and not suppress_audio:
                audio_started = True
                write_player(player, final_audio)
            pcm_tail = pcm_buffer.flush()
            if pcm_tail and not suppress_audio:
                audio_started = True
                write_player(player, pcm_tail)
            if suppress_audio:
                stop_player(player)
            else:
                close_player(player, self._config.audio)

        return answer_text, audio_started, final_tool_calls

    def _consume_delta(
        self,
        delta,
        answer_text: str,
        final_tool_calls: dict,
        player: subprocess.Popen | None,
        decoder: Base64StreamDecoder,
        pcm_buffer: PCMFrameBuffer,
        view_state: TurnViewState,
        live: Live | None,
        suppress_audio: bool,
    ) -> str:
        if delta.tool_calls is not None:
            view_state.status_text = "Collecting tool calls..."
            self._collect_tool_calls(delta, final_tool_calls)
            self._ui.update_live(live, view_state)

        content = getattr(delta, "content", None)
        if content:
            view_state.status_text = "Answering..."
            answer_text += content
            view_state.answer_text = answer_text
            self._ui.update_live(live, view_state)

        audio_data = self._extract_audio_data(delta)
        if audio_data and not suppress_audio:
            view_state.status_text = "Playing audio..."
            write_player(player, pcm_buffer.feed(decoder.feed(audio_data)))
            self._ui.update_live(live, view_state)

        return answer_text

    def _append_tool_call_messages(self, answer_text: str, final_tool_calls: dict):
        tool_calls = list(final_tool_calls.values())
        self._messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "tool_calls": tool_calls,
            }
        )

    def _run_tool_calls(self, final_tool_calls: dict, view_state: TurnViewState):
        tool_calls = list(final_tool_calls.values())
        view_state.tool_lines.append(f"model requested {len(tool_calls)} tool call(s)")

        for tool_call in tool_calls:
            if tool_call.function is None:
                continue

            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments or "{}"
            start = time.perf_counter()
            result = self._tool_set.call_tool(tool_name, tool_args)
            elapsed = time.perf_counter() - start

            summary = f"{tool_name} finished in {elapsed:.2f}s"
            if result.is_error:
                summary += f" | error: {tail_lines(result.output, 3)}"
            view_state.tool_lines.append(summary)

            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.output,
                }
            )

    @staticmethod
    def _extract_audio_data(delta) -> str | None:
        audio = getattr(delta, "audio", None)
        if isinstance(audio, dict):
            data = audio.get("data")
            if isinstance(data, str):
                return data

        extra_audio = getattr(delta, "model_extra", {}).get("audio")
        if isinstance(extra_audio, dict):
            data = extra_audio.get("data")
            if isinstance(data, str):
                return data

        return None

    def _trim_history(self):
        max_messages = 1 + self._turn_window * 6
        if len(self._messages) > max_messages:
            self._messages = [self._messages[0], *self._messages[-(max_messages - 1) :]]

    @staticmethod
    def _is_json_structured(value: str):
        try:
            result = json.loads(value)
            return isinstance(result, (dict, list))
        except (ValueError, TypeError):
            return False

    def _collect_tool_calls(self, delta: ChoiceDelta, final_tool_calls: dict):
        for tool_call in delta.tool_calls:
            if tool_call.function is None:
                continue

            index = tool_call.index
            if index not in final_tool_calls:
                final_tool_calls[index] = tool_call
                continue
            if tool_call.id:
                final_tool_calls[index].id = tool_call.id

            if final_tool_calls[index].function is None:
                final_tool_calls[index].function = tool_call.function
                continue

            if tool_call.function.name:
                final_tool_calls[index].function.name = tool_call.function.name

            arguments = final_tool_calls[index].function.arguments or ""
            incoming_arguments = tool_call.function.arguments or ""

            if self._is_json_structured(arguments):
                continue

            final_tool_calls[index].function.arguments = arguments + incoming_arguments

        return final_tool_calls

    @staticmethod
    def _format_usage(usage) -> str:
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        completion_details = getattr(usage, "completion_tokens_details", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)

        completion_audio_tokens = (
            getattr(completion_details, "audio_tokens", 0) if completion_details else 0
        ) or 0
        completion_text_tokens = (
            getattr(completion_details, "text_tokens", 0) if completion_details else 0
        ) or 0
        prompt_audio_tokens = (
            getattr(prompt_details, "audio_tokens", 0) if prompt_details else 0
        ) or 0
        prompt_text_tokens = (
            getattr(prompt_details, "text_tokens", 0) if prompt_details else 0
        ) or 0

        return (
            "[usage] "
            f"total={total_tokens} | "
            f"prompt={prompt_tokens} (text={prompt_text_tokens}, audio={prompt_audio_tokens}) | "
            f"completion={completion_tokens} (text={completion_text_tokens}, audio={completion_audio_tokens})"
        )
