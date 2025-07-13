import asyncio
from dotenv import load_dotenv
load_dotenv()
from browser_use import Agent
from browser_use.llm import ChatAzureOpenAI

async def main():
    agent = Agent(
        task="明日の天気を調べて、それを報告する",
        llm=ChatAzureOpenAI(model="gpt-4.1", temperature=1.0),
    )
    result = await agent.run()
    await agent.close()
    print(result)

asyncio.run(main())