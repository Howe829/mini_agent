from pydantic import BaseModel, Field
from typing import Type, override
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from src.tools.tool_base import ToolBase, ToolReturnValue


class AskUserToolParams(BaseModel):
    question: str = Field(description="向用户发起询问时的问题")
    options: list[str] = Field(description="提供给用户选择的选项")


class AskUserTool(ToolBase):
    name: str = "ask_user_tool"
    description: str = """
    询问用户选择工具，当你需要向用户发起询问选择时使用的工具，除了问题外你还需要给用户提供两个或以上的选项
    注意不要包含“其它”这个选项，因为系统会默认包含
    """
    params_class: Type[BaseModel] = AskUserToolParams

    @override
    def __call__(self, params: AskUserToolParams) -> ToolReturnValue:
        completer = WordCompleter(params.options, ignore_case=True)
        result = prompt(
            f"{params.question} （按tab查看选项或者输入你的回答）\n",
            completer=completer,
            complete_while_typing=True,
            default="",
        )

        return ToolReturnValue(
            output=f"对于问题{params.question}，用户的答案是： {result}",
            is_error=False,
        )
