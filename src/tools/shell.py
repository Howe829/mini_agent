import asyncio
import subprocess
from pydantic import BaseModel, Field
from typing import Type, override, Optional

from src.tools.tool_base import ToolBase, ToolReturnValue

SHELL_EXECUTE_TIMEOUT = 30


class ExecuteShellToolParams(BaseModel):
    command: str = Field(description="用以执行的shell命令")


class ExecuteShellTool(ToolBase):
    name: str = "execute_shell_tool"
    description: str = "用以执行shell终端命令的工具"
    alias: Optional[str] = "执行shell命令"
    params_class: Type[BaseModel] = ExecuteShellToolParams

    @override
    async def __call__(self, params: ExecuteShellToolParams) -> ToolReturnValue:
        try:
            process = await asyncio.create_subprocess_shell(
                params.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=SHELL_EXECUTE_TIMEOUT
            )
            std_content = stdout.decode() + stderr.decode()
            if process.returncode != 0:
                return ToolReturnValue(output=std_content, is_error=True)
            return ToolReturnValue(output=std_content, is_error=False)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ToolReturnValue(
                output=f"执行命令：{params.command} 超时({SHELL_EXECUTE_TIMEOUT}s)",
                is_error=True,
            )
