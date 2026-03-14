import pytest
from tools.ask_user_tool import AskUserTool


@pytest.mark.skip
def test_ask_user_tool():
    ask_user_tool = AskUserTool()
    func_args = '{"question": "你今天开心吗？", "options": ["开心", " 不开心"]}'
    result = ask_user_tool.call(func_args=func_args)
    assert result.is_error is False
