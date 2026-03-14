import pytest
from openai import OpenAI

from llm.openai import OpenAiLike
from tools.weather_tool import WeatherTool
from tools.ask_user_tool import AskUserTool
from tools.shell import ExecuteShellTool
from config import load_config


@pytest.fixture
def config():
    config = load_config()
    return config


@pytest.fixture
def current_model(config):
    return config.current.model


@pytest.fixture
def openai_client(config):
    provider = config.get_provider(config.current.provider)
    return OpenAI(base_url=provider.options.base_url, api_key=provider.options.api_key)


def test_openai_like_chat(openai_client, current_model):
    client = OpenAiLike(client=openai_client, tools=[])
    client.chat([{"role": "user", "content": "Hi"}], model=current_model)


def test_openai_like_weather_tool(openai_client, current_model):
    weather_tool = WeatherTool()
    tools = [weather_tool.to_schema()]
    client = OpenAiLike(client=openai_client, tools=tools)
    messages = [{"role": "user", "content": "上海天气怎么样"}]
    message = client.chat(messages=messages, model=current_model)

    assert message.tool_calls is not None
    tool_call = message.tool_calls.pop()
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
    messages = [{"role": "user", "content": "请问我一个问题"}]
    message = client.chat(messages=messages, model=current_model)

    assert message.tool_calls is not None
    tool_call = message.tool_calls.pop()
    func_name = tool_call.function.name
    assert func_name == ask_user_tool.name
    func_args = tool_call.function.arguments
    result = ask_user_tool.call(func_args)
    assert result.is_error is False


def test_openai_like_execute_shell_tool(openai_client, current_model):
    execute_shell_tool = ExecuteShellTool()
    tools = [execute_shell_tool.to_schema()]
    client = OpenAiLike(client=openai_client, tools=tools)
    messages = [{"role": "user", "content": "请执行一下`ls -la`命令"}]
    message = client.chat(messages=messages, model=current_model)

    assert message.tool_calls is not None
    tool_call = message.tool_calls.pop()
    func_name = tool_call.function.name
    assert func_name == execute_shell_tool.name
    func_args = tool_call.function.arguments
    result = execute_shell_tool.call(func_args)
    assert result.is_error is False
