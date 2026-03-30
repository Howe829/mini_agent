import time
import json
from typing import cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from config import load_config, MiniAgentConfig
from src.tools.tool_base import ToolSet
from src.llm.openai import OpenAiLike
from src.types.state import AgentState, ToolCallInfo, UsageInfo



class MiniAgent:
    def __init__(self, tool_set: ToolSet | None = None):
        config: MiniAgentConfig = load_config()
        provider = config.get_provider(config.current.provider)
        if provider is None:
            raise Exception("Config error, provider config cannot be none")
        openai_client = OpenAI(
            base_url=provider.options.base_url, api_key=provider.options.api_key
        )
        self._model = config.current.model
        self._prompt = self._load_prompt()
        self._messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._prompt}
        ]
        self.tool_set = tool_set if tool_set is not None else ToolSet()
        self._client = OpenAiLike(openai_client, tools=self.tool_set.to_schemas())

    @staticmethod
    def _load_prompt() -> str:
        with open("system.md") as f:
            return f.read()

    def _is_json_structured(self, _str: str):
        try:
            result = json.loads(_str)
            return isinstance(result, (dict, list))
        except (ValueError, TypeError):
            return False
    
    def add_message(self, message: dict):
        message = cast(
            ChatCompletionMessageParam,
            message
        )
        self._messages.append(message)
        
    def _collect_usage_info(self, event):
        usage = event.usage
        
        if usage is None:
            return None
        
        usage_info = UsageInfo(
            model=event.model,
            total_tokens=usage.total_tokens,
            elapsed=time.time() - event.created
        )
        return usage_info
    
    def _collect_tool_calls(self, delta: ChoiceDelta):
        final_tool_calls = {}
        for tool_call_ in delta.tool_calls:
            
            if tool_call_.function is None:
                continue
            
            index = tool_call_.index
            if index not in final_tool_calls:
                final_tool_calls[index] = tool_call_
            arguments = final_tool_calls[index].function.arguments

            if self._is_json_structured(arguments):
                continue

            final_tool_calls[index].function.arguments += tool_call_.function.arguments
        return final_tool_calls
    
    def _handle_tool_calls(self, delta: ChoiceDelta) -> list:
        final_tool_calls = self._collect_tool_calls(delta)
        tool_call_infos = []
        for tool_call in final_tool_calls.values():
            start = time.perf_counter()
            result = self.tool_set.call_tool(
                tool_call.function.name, tool_call.function.arguments
            )
            tool_call_infos.append(
                ToolCallInfo(
                    function_name=tool_call.function.name,
                    elapsed=time.perf_counter() - start,
                    is_error=result.is_error,
                    error_message=result.output if result.is_error else ""
                )
            )
            self.add_message(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.output,
                }
            )
        return tool_call_infos
    
        
        
    def run(self) -> AgentState:
        state = AgentState()

        generator = self._client.stream_chat(messages=self._messages, model=self._model)

        for event in generator:
            delta = event.choices[0].delta
            finish_reason = event.choices[0].finish_reason
            if finish_reason is not None:
                state.usage_info = self._collect_usage_info(event)
                    
                if finish_reason != "tool_calls":
                    self.add_message({"role": "assistant", "content": state.current_answer})
                    state.is_finish = True
                    return state

            reasoning = getattr(delta, "reasoning_content", None)

            if reasoning is not None:
                state.current_thought += reasoning

            elif delta.tool_calls is not None:
                _message = {
                    "role": "assistant",
                    "content": state.current_thought,
                    "tool_calls": delta.tool_calls,
                }
                self.add_message(_message)
                state.tool_call_infos = self._handle_tool_calls(delta)
            else:
                state.current_answer += delta.content or ""
                
        return state
