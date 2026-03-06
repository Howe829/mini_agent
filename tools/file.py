import pathlib
import os
from pydantic import BaseModel, Field
from typing import Type, override

from tools.tool_base import ToolBase, ToolReturnValue


class ReadFileToolParams(BaseModel):
    path: str = Field(description="用以读取的文件路径")


class WriteFileToolParams(BaseModel):
    path: str = Field(description="用以写入的文件路径")
    content: str = Field(description="用以写入文件的文本内容")


class ReadFileTool(ToolBase):
    name: str = "read_file_tool"
    description: str = "读取某个文本文件内容"
    params_class: Type[BaseModel] = ReadFileToolParams

    @override
    def __call__(self, params: ReadFileToolParams) -> ToolReturnValue:
        path = pathlib.Path(params.path)
        if not path.exists():
            return ToolReturnValue(output="File not exists", is_error=True)
        with path.open(mode="r", encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()
            numbered_lines = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines)]
            return ToolReturnValue(output="\n".join(numbered_lines), is_error=False)


class WriteFileTool(ToolBase):
    name: str = "write_file_tool"
    description: str = "将文本内容写入某个文件"
    params_class: Type[BaseModel] = WriteFileToolParams

    @override
    def __call__(self, params: WriteFileToolParams) -> ToolReturnValue:
        path = pathlib.Path(params.path)
        os.makedirs(path.absolute().parent, exist_ok=True)
        path.write_text(params.content, encoding="utf-8")
        return ToolReturnValue(
            output=f"Successfully wrote {len(params.content)} chars to {params.path}",
            is_error=False,
        )
