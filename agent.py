from config import load_config, MiniAgentConfig
from openai import OpenAI
from rich.console import Console, Group
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from tools.tool_base import ToolSet
from llm.openai import OpenAiLike
from rich.live import Live
from rich.style import Style
from rich.panel import Panel


console = Console()


class MiniAgent:
    def __init__(self, tool_set: ToolSet = None):
        config: MiniAgentConfig = load_config()
        provider = config.get_provider(config.current.provider)
        openai_client = OpenAI(
            base_url=provider.options.base_url, api_key=provider.options.api_key
        )
        self._model = config.current.model
        self._prompt = self._load_prompt()
        self._messages = [{"role": "system", "content": self._prompt}]
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
            except Exception as e:
                console.print(f"Error Occured: {e}", new_line_start=True)
                break

    def _call_llm_sync(self):
        with console.status(f"Thinking...", spinner="line"):
            message = self._client.chat(
                model=self._model,
                messages=self._messages,
            )
        self._messages.append(message)
        if message.tool_calls:
            self.tool_set.call(message.tool_calls.func)
            return True
        else:
            content = Markdown(message.content)
            console.print("● ", end="", new_line_start=True)
            console.print(content)
            return False

    def _call_llm_stream(self):
        with Live(Markdown(""), console=console, refresh_per_second=10) as live:
            status = console.status(f"Thinking...", spinner="line")
            status.start()
            generator = self._client.stream_chat(
                messages=self._messages, model=self._model
            )
            current_thought = ""
            current_answer = ""
            for event in generator:
                delta = event.choices[0].delta
                if hasattr(delta, "reasoning_content"):
                    current_thought += delta.reasoning_content
                    render_group = Group(
                        Panel(
                            current_thought,
                            title="[italic]思考内容[/italic]",
                            style="grey50",
                            border_style="grey23",
                        ),
                        Markdown(current_answer),
                    )
                    live.update(render_group)
                elif delta.tool_calls is not None:
                    status.stop()
                    message = {
                        "role": "assistant",
                        "content": current_thought,
                        "tool_calls": delta.tool_calls,
                    }
                    self._messages.append(message)
                    for tool in delta.tool_calls:
                        result = self.tool_set.call_tool(
                            tool.function.name, tool.function.arguments
                        )
                        console.print(
                            f"Used Tool: {tool.function.name}({tool.function.arguments})"
                        )
                        if result.is_error:
                            console.print(f"Tool Call Failed: {result.output}")
                        self._messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool.id,
                                "content": result.output,
                            }
                        )
                else:
                    current_answer += delta.content
                    render_group = Group(
                        # 思考内容已经完成，保持浅色显示
                        Panel(
                            current_thought,
                            title="[italic]思考内容[/italic]",
                            style="dim",
                            border_style="grey23",
                        ),
                        Markdown(current_answer),
                    )
                    live.update(render_group)
        if current_answer.strip():
            status.stop()
            message = {"role": "assistant", "content": current_answer}
            self._messages.append(message)
            return False

        return True
