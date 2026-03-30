import time
import json
from typing import cast
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from config import load_config, MiniAgentConfig
from src.tools.tool_base import ToolSet
from src.llm.openai import AsyncOpenAiLike
from src.types.state import AgentState, ToolCallInfo, UsageInfo


class MiniAgent:
    def __init__(self, tool_set: ToolSet | None = None):
        config: MiniAgentConfig = load_config()
        provider = config.get_provider(config.current.provider)
        if provider is None:
            raise Exception("Config error, provider config cannot be none")
        openai_client = AsyncOpenAI(
            base_url=provider.options.base_url, api_key=provider.options.api_key
        )
        self.model = config.current.model
        self._prompt = self._load_prompt()
        self._messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._prompt}
        ]
        self.tool_set = tool_set if tool_set is not None else ToolSet()
        self._client = AsyncOpenAiLike(openai_client, tools=self.tool_set.to_schemas())

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
        message = cast(ChatCompletionMessageParam, message)
        self._messages.append(message)

    def _collect_usage_info(self, event):
        usage = event.usage

        if usage is None:
            return None

        usage_info = UsageInfo(
            model=event.model,
            total_tokens=usage.total_tokens,
            elapsed=time.time() - event.created,
        )
        return usage_info

    def _collect_tool_calls(self, delta: ChoiceDelta, final_tool_calls: dict):
        for tool_call_ in delta.tool_calls:
            if tool_call_.function is None:
                continue

            index = tool_call_.index
            if index not in final_tool_calls:
                final_tool_calls[index] = tool_call_
            elif tool_call_.id:
                final_tool_calls[index].id = tool_call_.id

            if tool_call_.function is None:
                continue

            if final_tool_calls[index].function is None:
                final_tool_calls[index].function = tool_call_.function
                continue

            if tool_call_.function.name:
                final_tool_calls[index].function.name = tool_call_.function.name

            arguments = final_tool_calls[index].function.arguments or ""
            incoming_arguments = tool_call_.function.arguments or ""

            if self._is_json_structured(arguments):
                continue

            final_tool_calls[index].function.arguments = arguments + incoming_arguments
        return final_tool_calls

    async def _handle_tool_calls(self, final_tool_calls: list) -> list:
        tool_call_infos = []
        for tool_call in final_tool_calls:
            if tool_call.function is None:
                continue
            start = time.perf_counter()
            result = await self.tool_set.call_tool_async(
                tool_call.function.name, tool_call.function.arguments
            )
            tool_call_infos.append(
                ToolCallInfo(
                    function_name=tool_call.function.name,
                    elapsed=time.perf_counter() - start,
                    is_error=result.is_error,
                    error_message=result.output if result.is_error else "",
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

    async def run_stream(self):
        state = AgentState()
        final_tool_calls: dict = {}

        generator = await self._client.stream_chat(
            messages=self._messages, model=self.model
        )

        async for event in generator:
            delta = event.choices[0].delta
            finish_reason = event.choices[0].finish_reason

            reasoning = getattr(delta, "reasoning_content", None)
            has_update = False

            if reasoning is not None:
                state.current_thought += reasoning
                has_update = True

            if delta.content is not None:
                state.current_answer += delta.content or ""
                has_update = True

            if delta.tool_calls is not None:
                final_tool_calls = self._collect_tool_calls(delta, final_tool_calls)
                has_update = True

            usage_info = self._collect_usage_info(event)
            if usage_info is not None:
                state.usage_info = usage_info
                has_update = True

            if has_update:
                yield state.model_copy(deep=True)

            if finish_reason is not None:
                if finish_reason == "tool_calls":
                    tool_calls = list(final_tool_calls.values())
                    _message = {
                        "role": "assistant",
                        "content": state.current_thought,
                        "tool_calls": tool_calls,
                    }
                    self.add_message(_message)
                    state.tool_call_infos = await self._handle_tool_calls(tool_calls)
                    yield state.model_copy(deep=True)
                    return

                self.add_message({"role": "assistant", "content": state.current_answer})
                state.is_finish = True
                yield state.model_copy(deep=True)
                return

        yield state.model_copy(deep=True)

    async def run(self) -> AgentState:
        final_state = AgentState()
        async for state in self.run_stream():
            final_state = state
        return final_state
