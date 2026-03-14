from pydantic import BaseModel, Field, field_validator
from typing import Type
from abc import ABC, abstractmethod

_DEFAULT_TOOL_OUTPUT_MAX_LEN = 5000


class ToolReturnValue(BaseModel):
    output: str = Field(description="工具输出", max_length=_DEFAULT_TOOL_OUTPUT_MAX_LEN)
    is_error: bool = Field(description="工具执行是否出错")

    @field_validator("output", mode="before")
    @classmethod
    def truncate_output(cls, v: str) -> str:
        if isinstance(v, str) and len(v) > _DEFAULT_TOOL_OUTPUT_MAX_LEN:
            return (
                v[:_DEFAULT_TOOL_OUTPUT_MAX_LEN] + "...（工具输出内容过长，已被省略）"
            )
        return v

    def __str__(self):
        if self.is_error:
            return f"[Tool Error] {self.output}"
        return self.output


class ToolBase(ABC):
    name: str
    description: str
    params_class: Type[BaseModel]

    def to_schema(self):
        parameters = self.params_class.model_json_schema()

        if "properties" in parameters:
            for prop in parameters["properties"].values():
                prop.pop("title", None)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def call(self, func_args: str) -> ToolReturnValue:
        parameters = self.params_class.model_validate_json(func_args)
        return self.__call__(parameters)

    @abstractmethod
    def __call__(self, *args, **kwds) -> ToolReturnValue:
        raise NotImplementedError


class ToolSet:
    _tools: dict[str, ToolBase]

    def __init__(self, tools: list[ToolBase] = None):
        if tools is None:
            return
        
        for tool in tools:
            self.add_tool(tool)

    def add_tool(self, tool: ToolBase):
        if tool.name in self._tools.keys():
            return
        self._tools[tool.name] = tool

    def call_tool(self, tool_name: str, func_args: str) -> ToolReturnValue:
        if tool_name not in self._tools.keys():
            return ToolReturnValue(
                output=f"Tool ({tool_name}) not exists.", is_error=True
            )
        return self._tools[tool_name].call(func_args)
    
    def to_schemas(self):
        return [tool.to_schema() for tool in self._tools.values()]
