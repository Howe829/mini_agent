import re
import asyncio
import subprocess
import unicodedata
from datetime import datetime
from pathlib import Path
from rich.live import Live
from rich.columns import Columns
from rich.text import Text
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from src.types.state import AgentState, ToolCallInfo, UsageInfo
from src.tools.edge_tts import EdgeTtsTool
from src.tools.edge_tts import EdgeTtsToolParams
from src.mini_agent import MiniAgent
from src.tools.shell import ExecuteShellTool
from src.tools.tool_base import ToolSet

MPV_TTS_ARGS = [
    "mpv",
    "--really-quiet",
    "--cache=yes",
    "--cache-secs=20",
    "--demuxer-max-bytes=8MiB",
    "-",
]

DEFAULT_COLLAPSED_LINES = 18
DEFAULT_THOUGHT_LINES = 6
TOOL_ERROR_PREVIEW_LINES = 3


class CLIAgent:
    def __init__(self):
        self._console = Console()
        self._status = self._console.status(f"Thinking...", spinner="line")
        tool_set = ToolSet(tools=[ExecuteShellTool(), EdgeTtsTool()])
        self._mini_agent = MiniAgent(tool_set=tool_set)

    async def run(self):
        self._print_welcome()
        history = InMemoryHistory()
        session = PromptSession(history=history)

        while True:
            try:
                user_input = await session.prompt_async("➜ ")
                if user_input.lower() == "exit":
                    self._console.print("Bye")
                    break

                user_message = {"role": "user", "content": user_input}
                self._mini_agent.add_message(user_message)
                with Live(
                    Markdown(""),
                    console=self._console,
                    refresh_per_second=10,
                    vertical_overflow="crop",
                ) as live:
                    while True:
                        agent_state = AgentState()
                        self._status.start()
                        try:
                            async for stream_state in self._mini_agent.run_stream():
                                agent_state = stream_state
                                self._update_live(agent_state, live)
                        finally:
                            self._status.stop()

                        if agent_state.is_finish is True:
                            await self._speak_answer(agent_state.current_answer)
                            break

            except EOFError:
                self._console.print("Bye", new_line_start=True)
                self._status.stop()
                break
            except KeyboardInterrupt:
                self._console.print("Bye", new_line_start=True)
                self._status.stop()
                break
            except Exception as e:
                self._console.print(f"Error Occured: {e}", new_line_start=True)
                self._status.stop()
                break

    def _print_welcome(self):
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        current_dir = Path.cwd()
        welcome_text = Text()
        welcome_text.append("Mini Agent\n", style="bold cyan")
        welcome_text.append("Async CLI session is ready.\n\n", style="dim")
        welcome_text.append("Model: ", style="bold")
        welcome_text.append(f"{self._mini_agent.model}\n", style="cyan")
        welcome_text.append("Directory: ", style="bold")
        welcome_text.append(f"{current_dir}\n", style="green")
        welcome_text.append("Datetime: ", style="bold")
        welcome_text.append(f"{now}\n", style="yellow")
        welcome_text.append("Exit: ", style="bold")
        welcome_text.append("type `exit` or press Ctrl+D", style="magenta")
        self._console.print(
            Panel(
                welcome_text,
                title="Welcome",
                border_style="blue",
                expand=False,
                padding=(1, 2),
            )
        )

    def _get_usage_info_columns(self, usage_info: UsageInfo) -> list[Columns]:
        usage_info_columns = Columns(
            [
                Text.from_markup(f"[dim]Model:[/] [cyan]{usage_info.model}[/]"),
                Text.from_markup(
                    f"[dim]Tokens:[/] [yellow]{usage_info.total_tokens}[/]"
                ),
                Text.from_markup(f"[dim]Time:[/] [green]{usage_info.elapsed:.2f}s[/]"),
            ],
            equal=False,
            expand=False,
        )
        return usage_info_columns

    @staticmethod
    def _tail_lines(text: str, max_lines: int) -> str:
        if not text.strip():
            return ""

        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text.strip()

        tail = "\n".join(lines[-max_lines:])
        hidden_count = len(lines) - max_lines
        return f"[{hidden_count} lines hidden]\n{tail}".strip()

    def _get_tool_call_info_columns(
        self, tool_call_infos: list[ToolCallInfo]
    ) -> list[Columns]:
        tool_call_info_columns = []
        for tool_call_info in tool_call_infos:
            columns = [
                Text.from_markup(f"[dim]Used: {tool_call_info.function_name}"),
                Text.from_markup(f"[dim]Time: {tool_call_info.elapsed:.2f}s"),
            ]
            if tool_call_info.is_error:
                error_preview = self._tail_lines(
                    tool_call_info.error_message or "",
                    TOOL_ERROR_PREVIEW_LINES,
                )
                columns.append(
                    Text.from_markup(f"[dim]ToolCallError: {error_preview}"),
                )
            tool_call_info_columns.append(
                Columns(
                    columns,
                    equal=False,
                    expand=False,
                )
            )
        return tool_call_info_columns

    def _update_live(self, agent_state: AgentState, live: Live):
        tool_call_info_columns = self._get_tool_call_info_columns(
            agent_state.tool_call_infos
        )
        usage_info_columns = (
            self._get_usage_info_columns(agent_state.usage_info)
            if agent_state.usage_info is not None
            else ""
        )
        thought_preview = self._tail_lines(
            agent_state.current_thought,
            DEFAULT_THOUGHT_LINES,
        )
        answer_preview = self._tail_lines(
            agent_state.current_answer,
            DEFAULT_COLLAPSED_LINES,
        )
        group = Group(
            Markdown(
                thought_preview,
                style="dim",
            ),
            *tool_call_info_columns,
            Markdown(answer_preview),
            usage_info_columns,
        )
        live.update(group)

    async def _speak_answer(self, answer: str):
        cleaned_answer = self._clean_tts_text(answer)
        if not cleaned_answer:
            return

        try:
            await self._stream_tts_to_mpv(cleaned_answer)
        except FileNotFoundError:
            self._console.print("[yellow]mpv not found, skipped audio playback.[/]")
        except subprocess.CalledProcessError as exc:
            error_output = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            self._console.print(f"[yellow]mpv playback failed:[/] {error_output}")
        except Exception as exc:
            self._console.print(f"[yellow]TTS failed:[/] {exc}")

    async def _stream_tts_to_mpv(self, answer: str):
        process = await asyncio.create_subprocess_exec(
            *MPV_TTS_ARGS,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        if process.stdin is None:
            process.kill()
            await process.wait()
            raise RuntimeError("mpv stdin is not available")

        try:
            async for chunk in self._iter_tts_audio_chunks(answer):
                if chunk is None:
                    continue
                process.stdin.write(chunk)
                await process.stdin.drain()
        finally:
            process.stdin.close()
            await process.stdin.wait_closed()

        return_code = await process.wait()
        if return_code != 0:
            stderr = await process.stderr.read() if process.stderr is not None else b""
            stdout = await process.stdout.read() if process.stdout is not None else b""
            raise subprocess.CalledProcessError(
                return_code,
                MPV_TTS_ARGS,
                output=stdout.decode("utf-8", errors="ignore"),
                stderr=stderr.decode("utf-8", errors="ignore"),
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
