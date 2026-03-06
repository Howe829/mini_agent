from config import load_config, MiniAgentConfig
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from tools.tool_base import ToolSet


console = Console()



class MiniAgent:
    def __init__(self, tool_set: ToolSet = None):
        config: MiniAgentConfig = load_config()
        provider = config.get_provider(config.current.provider)
        self._client = OpenAI(
            base_url=provider.options.base_url, api_key=provider.options.api_key
        )
        self._model = config.current.model
        self._prompt = self._load_prompt()
        self._messages = [{"role": "system", "content": self._prompt}]
        self.tool_set = tool_set if tool_set is not None else ToolSet()


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
                while True:
                    with console.status(f"Thinking...", spinner="line"):
                        response = self._client.chat.completions.create(
                            model=self._model,
                            messages=self._messages,
                            tools=self._tool_schemas,
                        )
                    message = response.choices[0].message
                    self._messages.append(message)
                    if message.tool_calls:
                        self.tool_set.call(message.tool_calls.func)
                    else:
                        content = Markdown(message.content)
                        console.print("● ", end="", new_line_start=True)
                        console.print(content)
                        break
            except EOFError:
                console.print("Bye", new_line_start=True)
                break
            except KeyboardInterrupt:
                console.print("Bye", new_line_start=True)
                break
            except Exception as e:
                console.print(f"Error Occured: {e}", new_line_start=True)
                break
