import time
import pytest
from typing import cast

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionMessageParam

from src.llm.openai import OpenAiLike, AsyncOpenAiLike
from src.tools.weather_tool import WeatherTool
from src.tools.ask_user_tool import AskUserTool
from src.tools.shell import ExecuteShellTool
from config import load_config


@pytest.fixture
def model_config():
    _config = load_config()
    return _config


@pytest.fixture
def current_model(model_config):
    return model_config.current.model


@pytest.fixture
def openai_client(model_config):
    provider = model_config.get_provider(model_config.current.provider)
    return OpenAI(base_url=provider.options.base_url, api_key=provider.options.api_key)


@pytest.fixture
def async_openai_client(model_config):
    provider = model_config.get_provider(model_config.current.provider)
    return AsyncOpenAI(
        base_url=provider.options.base_url, api_key=provider.options.api_key
    )


def test_openai_like_chat(openai_client, current_model):
    client = OpenAiLike(client=openai_client, tools=[])
    client.chat([{"role": "user", "content": "Hi"}], model=current_model)


def test_openai_like_weather_tool(openai_client, current_model):
    weather_tool = WeatherTool()
    tools = [weather_tool.to_schema()]
    client = OpenAiLike(client=openai_client, tools=tools)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "上海天气怎么样"}
    ]
    message = client.chat(messages=messages, model=current_model)

    assert message.tool_calls is not None
    tool_call = cast(ChatCompletionMessageToolCall, message.tool_calls.pop())
    func_name = tool_call.function.name
    assert func_name == weather_tool.name
    func_args = tool_call.function.arguments
    result = weather_tool.call(func_args)
    assert result.is_error is False


@pytest.mark.skip
def test_openai_like_ask_user_tool(openai_client, current_model):
    ask_user_tool = AskUserTool()
    tools = [ask_user_tool.to_schema()]
    client = OpenAiLike(client=openai_client, tools=tools)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "请问我一个问题"}
    ]
    message = client.chat(messages=messages, model=current_model)

    assert message.tool_calls is not None
    tool_call = cast(ChatCompletionMessageToolCall, message.tool_calls.pop())
    func_name = tool_call.function.name
    assert func_name == ask_user_tool.name
    func_args = tool_call.function.arguments
    result = ask_user_tool.call(func_args)
    assert result.is_error is False


def test_openai_like_execute_shell_tool(openai_client, current_model):
    execute_shell_tool = ExecuteShellTool()
    tools = [execute_shell_tool.to_schema()]
    client = OpenAiLike(client=openai_client, tools=tools)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "请执行一下`ls -la`命令"}
    ]
    message = client.chat(messages=messages, model=current_model)

    assert message.tool_calls is not None
    tool_call = cast(ChatCompletionMessageToolCall, message.tool_calls.pop())
    func_name = tool_call.function.name
    assert func_name == execute_shell_tool.name
    func_args = tool_call.function.arguments
    result = execute_shell_tool.call(func_args)
    assert result.is_error is False


def test_openai_like_stream_chat(openai_client, current_model):
    weather_tool = WeatherTool()
    tools = [weather_tool.to_schema()]
    client = OpenAiLike(client=openai_client, tools=tools)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "上海天气怎么样"}
    ]
    generator = client.stream_chat(messages=messages, model=current_model)
    start = time.perf_counter()
    for _ in generator:
        pass
        # print(event)
    print(f"Sync Elapsed: {time.perf_counter() - start:.2f}s")


@pytest.mark.asyncio
async def test_openai_like_stream_chat_async(async_openai_client, current_model):
    weather_tool = WeatherTool()
    tools = [weather_tool.to_schema()]
    client = AsyncOpenAiLike(client=async_openai_client, tools=tools)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "上海天气怎么样"}
    ]
    generator = await client.stream_chat(messages=messages, model=current_model)

    start = time.perf_counter()
    async for _ in generator:
        pass
        # print(event)
    print(f"ASync Elapsed: {time.perf_counter() - start:.2f}s")
