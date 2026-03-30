from pydantic import BaseModel, Field


class UsageInfo(BaseModel):
    model: str
    total_tokens: int
    elapsed: float


class ToolCallInfo(BaseModel):
    function_name: str
    elapsed: float
    is_error: bool = Field(default=False)
    error_message: str | None = Field(None)


class AgentState(BaseModel):
    current_thought: str = Field(default="")
    current_answer: str = Field(default="")
    tool_call_infos: list[ToolCallInfo] = Field(default_factory=list)
    usage_info: UsageInfo | None = Field(default=None)
    is_finish: bool = Field(default=False)
