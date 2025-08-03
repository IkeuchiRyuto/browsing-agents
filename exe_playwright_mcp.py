import os, asyncio, time

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
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
    start_time = time.time()
    
    server_params = StdioServerParams(
        command="npx",
        args=[
            "@playwright/mcp@latest",
            "--browser",
            "chromium"
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
            description="あなたはMCPを使用して、ウェブサイトの情報を取得したり操作を行うエージェントです。",
            tools=tools,
        )

        # シングルエージェントでタスクを実行
        user_message = TextMessage(
            content="""
            以下のタスクを正確に実行してください:
            1. Qiitaにアクセス
            2. メールアドレス=<your email>、パスワード=<your password>でログイン
            3. 画面右上の「投稿する」をクリックして、「記事を新規作成」をクリック
            4. タイトルに「あああああ」を入力
            5. 本文に「いいいいい」を入力
            6. 「下書き保存」をクリック
            7. 保存が完了したら、トップ画面に戻って、「トレンド」タブをクリック
            8. トレンドのトップの記事を開いて、いいねを押してください。
            9. 最後にユーザーアイコンをクリックして、ログアウトしてください。
            """,
            source="user"
        )

        termination = TextMentionTermination("TERMINATE")
        team = RoundRobinGroupChat([agent], termination_condition=termination)
        await Console(
            team.run_stream(
                task=user_message.content,
            )
        )
        
        # 処理時間を計算
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 最終的なトークン使用量を取得
        final_usage = model_client.total_usage()
        
        print(f"\n=== 実行ログ ===")
        print(f"処理時間: {elapsed_time:.2f}秒")
        print(f"合計消費トークン数: {final_usage.prompt_tokens + final_usage.completion_tokens}")
        print(f"  - プロンプトトークン: {final_usage.prompt_tokens}")
        print(f"  - 補完トークン: {final_usage.completion_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
