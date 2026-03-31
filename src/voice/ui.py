from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

VOICE_HISTORY_FILE_PATH = Path.home() / ".mini_agent_voice_history"
DEFAULT_COLLAPSED_LINES = 18


@dataclass
class TurnViewState:
    status_text: str = "Waiting..."
    answer_text: str = ""
    usage_line: str = ""
    tool_lines: list[str] = field(default_factory=list)


class VoiceUI:
    def __init__(self, model: str, voice: str):
        self.console = Console()
        self.model = model
        self.voice = voice

    def build_session(self) -> PromptSession:
        VOICE_HISTORY_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        return PromptSession(history=FileHistory(str(VOICE_HISTORY_FILE_PATH)))

    def print_welcome(self):
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        current_dir = Path.cwd()
        welcome_text = Text()
        welcome_text.append("Voice Agent\n", style="bold cyan")
        welcome_text.append("Qwen3.5-Omni multimodal session is ready.\n\n", style="dim")
        welcome_text.append("Model: ", style="bold")
        welcome_text.append(f"{self.model}\n", style="cyan")
        welcome_text.append("Voice: ", style="bold")
        welcome_text.append(f"{self.voice}\n", style="magenta")
        welcome_text.append("Directory: ", style="bold")
        welcome_text.append(f"{current_dir}\n", style="green")
        welcome_text.append("Datetime: ", style="bold")
        welcome_text.append(f"{now}\n", style="yellow")
        welcome_text.append("Input: ", style="bold")
        welcome_text.append("press Enter to record, or type text directly\n", style="white")
        welcome_text.append("Exit: ", style="bold")
        welcome_text.append("type `exit` or press Ctrl+D", style="magenta")
        self.console.print(
            Panel(
                welcome_text,
                title="Welcome",
                border_style="blue",
                expand=False,
                padding=(1, 2),
            )
        )

    def build_response_view(self, view_state: TurnViewState):
        status_columns = Columns(
            [
                Text.from_markup(f"[dim]Status:[/] [cyan]{view_state.status_text}[/]"),
                Text.from_markup(f"[dim]Model:[/] [green]{self.model}[/]"),
                Text.from_markup(f"[dim]Voice:[/] [magenta]{self.voice}[/]"),
            ],
            equal=False,
            expand=False,
        )
        tool_renderables = [
            Text.from_markup(f"[dim]Tool:[/] {line}")
            for line in view_state.tool_lines[-4:]
        ]
        answer_preview = tail_lines(view_state.answer_text, DEFAULT_COLLAPSED_LINES)
        answer_renderable = Markdown(answer_preview) if answer_preview else Text("...")
        usage_renderable = (
            Text.from_markup(f"[dim]{view_state.usage_line}[/]")
            if view_state.usage_line
            else Text("")
        )

        return Panel(
            Group(status_columns, *tool_renderables, answer_renderable, usage_renderable),
            title="Voice Session",
            border_style="cyan",
        )

    def update_live(self, live: Live | None, view_state: TurnViewState):
        if live is None:
            return
        live.update(self.build_response_view(view_state))

    def print_error(self, message: str):
        self.console.print(f"[red]{message}[/]")

    def print_warning(self, message: str):
        self.console.print(f"[yellow]{message}[/]")


def tail_lines(text: str, max_lines: int) -> str:
    if not text.strip():
        return ""

    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text.strip()

    hidden_count = len(lines) - max_lines
    tail = "\n".join(lines[-max_lines:])
    return f"[{hidden_count} lines hidden]\n{tail}".strip()
