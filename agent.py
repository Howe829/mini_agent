import asyncio
import time
import json
import re
import subprocess
import unicodedata
from typing import cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console, Group
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from config import load_config, MiniAgentConfig
from tools.edge_tts import EdgeTtsTool, EdgeTtsToolParams
from tools.tool_base import ToolSet
from llm.openai import OpenAiLike
from rich.live import Live
from rich.style import Style
from rich.panel import Panel
from rich.rule import Rule
from rich.columns import Columns
from rich.text import Text

console = Console()
MPV_TTS_ARGS = [
    "mpv",
    "--really-quiet",
    "--cache=yes",
    "--cache-secs=20",
    "--demuxer-max-bytes=8MiB",
    "-",
]


class MiniAgent:
    def __init__(self, tool_set: ToolSet | None = None):
        config: MiniAgentConfig = load_config()
        provider = config.get_provider(config.current.provider)
        if provider is None:
            raise Exception("Config error, provider config cannot be none")
        openai_client = OpenAI(
            base_url=provider.options.base_url, api_key=provider.options.api_key
        )
        self._model = config.current.model
        self._prompt = self._load_prompt()
        self._messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._prompt}
        ]
        self.tool_set = tool_set if tool_set is not None else ToolSet()
        self._client = OpenAiLike(openai_client, tools=self.tool_set.to_schemas())

    @staticmethod
    def _load_prompt() -> str:
        with open("system.md") as f:
            return f.read()

    def run(self):
        console.print("Welcome to Mini Agent!")
        console.print("--------------------------------------------")
        history = InMemoryHistory()
        session = PromptSession(history=history)

        while True:
            try:
                user_input = session.prompt("➜ ")
                if user_input.lower() == "exit":
                    console.print("Bye")
                    break

                self._messages.append({"role": "user", "content": user_input})
                while self._call_llm_stream():
                    pass
            except EOFError:
                console.print("Bye", new_line_start=True)
                break
            except KeyboardInterrupt:
                console.print("Bye", new_line_start=True)
                break
            # except Exception as e:
            #     console.print(f"Error Occured: {e}", new_line_start=True)
            #     break

    def _is_json_structured(self, _str):
        try:
            result = json.loads(_str)
            return isinstance(result, (dict, list))
        except (ValueError, TypeError):
            return False

    def _call_llm_stream(self):
        live = Live(Markdown(""), console=console, refresh_per_second=10)
        live.start()
        status = console.status(f"Thinking...", spinner="line")
        status.start()
        generator = self._client.stream_chat(
            messages=self._messages, model=self._model
        )
        current_thought = ""
        current_answer = ""
        tool_call_infos = []
        usage_info = None
        for event in generator:
            delta = event.choices[0].delta
            finish_reason = event.choices[0].finish_reason
            if finish_reason is not None:
                status.stop()
                usage = event.usage
                if usage is not None:
                    usage_info = Columns(
                        [
                            Text.from_markup(
                                f"[dim]Model:[/] [cyan]{event.model}[/]"
                            ),
                            Text.from_markup(
                                f"[dim]Tokens:[/] [yellow]{usage.total_tokens}[/]"
                            ),
                            Text.from_markup(
                                f"[dim]Time:[/] [green]{time.time()-event.created:.2f}s[/]"
                            ),
                        ],
                        equal=False,
                        expand=False,
                    )
                    render_group = self._get_render_group(
                        current_thought, current_answer, tool_call_infos, usage_info
                    )
                    live.update(render_group)
                if finish_reason != "tool_calls":
                    message = cast(
                        ChatCompletionMessageParam,
                        {"role": "assistant", "content": current_answer},
                    )
                    self._messages.append(message)
                    live.stop()
                    self._speak_answer(current_answer)
                    return False

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                current_thought += reasoning
                render_group = self._get_render_group(
                    current_thought, current_answer, tool_call_infos, usage_info
                )
                live.update(render_group)

            elif delta.tool_calls is not None:
                message = {
                    "role": "assistant",
                    "content": current_thought,
                    "tool_calls": delta.tool_calls,
                }
                self._messages.append(cast(ChatCompletionMessageParam, message))
                final_tool_calls = {}
                for tool_call_ in delta.tool_calls:
                    index = tool_call_.index
                    if index not in final_tool_calls:
                        final_tool_calls[index] = tool_call_
                    arguments = final_tool_calls[index].function.arguments

                    if (
                        self._is_json_structured(arguments)
                        or tool_call_.function is None
                    ):
                        continue

                    arguments += tool_call_.function.arguments

                for tool_call in final_tool_calls.values():
                    with console.status(
                        f"Using {tool_call.function.name}", spinner="dots3"
                    ):
                        start = time.perf_counter()
                        result = self.tool_set.call_tool(
                            tool_call.function.name, tool_call.function.arguments
                        )
                    tool_call_infos.append(
                        Columns(
                            [
                                Text.from_markup(
                                    f"[dim]Used: {tool_call.function.name}"
                                ),
                                Text.from_markup(
                                    f"[dim]Time: {time.perf_counter()-start:.2f}s"
                                ),
                            ],
                            equal=False,
                            expand=False,
                        )
                    )
                    if result.is_error:
                        console.print(f"Tool Call Failed: {result.output}")
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result.output,
                        }
                    )
            else:
                current_answer += delta.content or ""
                render_group = self._get_render_group(
                    current_thought, current_answer, tool_call_infos, usage_info
                )
                live.update(render_group, refresh=True)
        live.stop()
        return True

    def _speak_answer(self, answer: str):
        cleaned_answer = self._clean_tts_text(answer)
        if not cleaned_answer:
            return

        try:
            asyncio.run(self._stream_tts_to_mpv(cleaned_answer))
        except FileNotFoundError:
            console.print("[yellow]mpv not found, skipped audio playback.[/]")
        except subprocess.CalledProcessError as exc:
            error_output = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            console.print(f"[yellow]mpv playback failed:[/] {error_output}")
        except Exception as exc:
            console.print(f"[yellow]TTS failed:[/] {exc}")

    async def _stream_tts_to_mpv(self, answer: str):
        process = subprocess.Popen(
            MPV_TTS_ARGS,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if process.stdin is None:
            process.kill()
            raise RuntimeError("mpv stdin is not available")

        try:
            async for chunk in self._iter_tts_audio_chunks(answer):
                process.stdin.write(chunk) # type: ignore
                process.stdin.flush()
        finally:
            process.stdin.close()

        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read().decode("utf-8", errors="ignore") # type: ignore
            stdout = process.stdout.read().decode("utf-8", errors="ignore") # type: ignore
            raise subprocess.CalledProcessError(
                return_code, MPV_TTS_ARGS, output=stdout, stderr=stderr
            )

    async def _iter_tts_audio_chunks(self, answer: str):
        communicate = EdgeTtsTool._build_communicate(
            EdgeTtsToolParams(text=answer, output_path=None)
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk.get("data")

    @staticmethod
    def _clean_tts_text(text: str) -> str:
        cleaned = text
        cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.replace("**", "").replace("__", "")
        cleaned = cleaned.replace("*", "").replace("_", "")
        cleaned = cleaned.replace("~~", "")
        cleaned = re.sub(r"^\s*>\s?", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.replace("|", " ")
        cleaned = "".join(
            char
            for char in cleaned
            if unicodedata.category(char) not in {"So", "Sk", "Cs"}
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\s+\n", "\n", cleaned)
        return cleaned.strip()

    def _get_render_group(
        self, current_thought, current_answer, tool_call_infos, usage_info
    ):
        return Group(
            Panel(
                current_thought,
                title="[italic]思考内容[/italic]",
                style="dim",
                border_style="grey23",
            ),
            *tool_call_infos,
            Markdown(current_answer),
            usage_info or "",
        )
