from agent import MiniAgent
from tools.shell import ExecuteShellTool
from tools.tool_base import ToolSet


def run_mini_agent():
    tool_set = ToolSet(tools=[ExecuteShellTool()])
    mini_agent = MiniAgent(tool_set=tool_set)
    mini_agent.run()


if __name__ == "__main__":
    run_mini_agent()
