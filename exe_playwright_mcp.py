import os, asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, create_mcp_server_session, mcp_server_tools
from dotenv import load_dotenv


load_dotenv()

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4.1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-04-01-preview",
    model="gpt-4.1",
)

async def main() -> None:
    server_params = StdioServerParams(
        command="npx",
        args=[
            "@playwright/mcp@latest",
            "--headless",
        ],
        read_timeout_seconds=60,
    )

    async with create_mcp_server_session(server_params) as session:
        await session.initialize()
        tools = await mcp_server_tools(server_params=server_params, session=session)
        print(f"Tools: {[tool.name for tool in tools]}")

        agent = AssistantAgent(
            name="Assistant",
            model_client=model_client,
            tools=tools,  # type: ignore
        )

        termination = TextMentionTermination("TERMINATE")
        team = RoundRobinGroupChat([agent], termination_condition=termination)
        await Console(
            team.run_stream(
                task="「https://weather.yahoo.co.jp/weather/jp」を開いて、明日の東京の天気を調べて報告する"
            )
        )


# must use asyncio.run to run the main function
# because main() is an async function
if __name__ == "__main__":
    asyncio.run(main())
