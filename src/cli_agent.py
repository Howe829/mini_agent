
import re
import asyncio
import subprocess
import unicodedata
from rich.live import Live
from rich.columns import Columns
from rich.text import Text
from rich.console import Console, Group
from rich.markdown import Markdown
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

class CLIAgent:
    
    def __init__(self):
        self._console = Console()
        self._status = self._console.status(f"Thinking...", spinner="line")
        tool_set = ToolSet(tools=[ExecuteShellTool(), EdgeTtsTool()])
        self._mini_agent = MiniAgent(tool_set=tool_set)
    

    def run(self):
        self._console.print("Welcome to Mini Agent!")
        self._console.print("--------------------------------------------")
        history = InMemoryHistory()
        session = PromptSession(history=history)

        while True:
            try:
                user_input = session.prompt("➜ ")
                if user_input.lower() == "exit":
                    self._console.print("Bye")
                    break

                # Keep a static transcript line for each turn.
                self._console.print(Text("You: ", style="cyan") + Text(user_input))
                user_message = {"role": "user", "content": user_input}
                self._mini_agent.add_message(user_message)
                live = Live(Markdown(""), console=self._console, refresh_per_second=10, transient=True)
                live.start()
                try:
                    while True:
                        self._status.start()
                        try:
                            agent_state = self._mini_agent.run()
                        finally:
                            self._status.stop()

                        self._update_live(agent_state, live)
                        if agent_state.is_finish is True:
                            self._console.print(
                                Text("Assistant: ", style="green")
                                + Text(agent_state.current_answer)
                            )
                            self._speak_answer(agent_state.current_answer)
                            break
                finally:
                    live.stop()
                
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
    
    def _get_usage_info_columns(self, usage_info: UsageInfo) -> list[Columns]:
        usage_info_columns = Columns(
            [
                Text.from_markup(f"[dim]Model:[/] [cyan]{usage_info.model}[/]"),
                Text.from_markup(
                    f"[dim]Tokens:[/] [yellow]{usage_info.total_tokens}[/]"
                ),
                Text.from_markup(
                    f"[dim]Time:[/] [green]{usage_info.elapsed:.2f}s[/]"
                ),
            ],
            equal=False,
            expand=False,
        )
        return usage_info_columns
    
    def _get_tool_call_info_columns(self, tool_call_infos: list[ToolCallInfo]) -> list[Columns]:
        tool_call_info_columns = []
        for tool_call_info in tool_call_infos:
            columns = [
                        Text.from_markup(
                            f"[dim]Used: {tool_call_info.function_name}"
                        ),
                        Text.from_markup(
                            f"[dim]Time: {tool_call_info.elapsed:.2f}s"
                        ),
            ]
            if tool_call_info.is_error:
                columns.append(
                    Text.from_markup(
                        f"[dim]ToolCallError: {tool_call_info.error_message[:50]}"
                    ),
                    
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
        tool_call_info_columns = self._get_tool_call_info_columns(agent_state.tool_call_infos)
        usage_info_columns = self._get_usage_info_columns(agent_state.usage_info) if agent_state.usage_info is not None else ""
        group = Group(
            Markdown(
                agent_state.current_thought,
                style="dim",
            ),
            *tool_call_info_columns,
            Markdown(agent_state.current_answer),
            usage_info_columns,
        )
        live.update(group)

    def _speak_answer(self, answer: str):
        cleaned_answer = self._clean_tts_text(answer)
        if not cleaned_answer:
            return

        try:
            asyncio.run(self._stream_tts_to_mpv(cleaned_answer))
        except FileNotFoundError:
            self._console.print("[yellow]mpv not found, skipped audio playback.[/]")
        except subprocess.CalledProcessError as exc:
            error_output = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            self._console.print(f"[yellow]mpv playback failed:[/] {error_output}")
        except Exception as exc:
            self._console.print(f"[yellow]TTS failed:[/] {exc}")

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
                process.stdin.write(chunk)  # type: ignore
                process.stdin.flush()
        finally:
            process.stdin.close()

        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read().decode("utf-8", errors="ignore")  # type: ignore
            stdout = process.stdout.read().decode("utf-8", errors="ignore")  # type: ignore
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
