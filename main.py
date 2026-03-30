import asyncio

from src.cli_agent import CLIAgent



if __name__ == "__main__":
    agent = CLIAgent()
    asyncio.run(agent.run())
