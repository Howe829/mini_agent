import subprocess
from pydantic import BaseModel, Field
from typing import Type, override

from tools.tool_base import ToolBase, ToolReturnValue

SHELL_EXECUTE_TIMEOUT = 30


class ExecuteShellToolParams(BaseModel):
    command: str = Field(description="用以执行的shell命令")


class ExecuteShellTool(ToolBase):
    name: str = "execute_shell_tool"
    description: str = "用以执行shell终端命令的工具"
    params_class: Type[BaseModel] = ExecuteShellToolParams

    @override
    def __call__(self, params: ExecuteShellToolParams) -> ToolReturnValue:
        try:
            result = subprocess.run(
                params.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=SHELL_EXECUTE_TIMEOUT,
            )
            if result.stderr:
                return ToolReturnValue(output=result.stderr, is_error=True)
            return ToolReturnValue(output=result.stdout, is_error=False)
        except subprocess.TimeoutExpired:
            return ToolReturnValue(
                output=f"执行命令：{params.command} 超时({SHELL_EXECUTE_TIMEOUT}s)",
                is_error=True,
            )
