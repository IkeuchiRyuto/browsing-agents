import asyncio
import time
from dotenv import load_dotenv
load_dotenv()
from browser_use import Agent
from browser_use.llm import ChatAzureOpenAI

async def execute_browser_task(task_description, model="gpt-4.1", temperature=0.1):
    """Execute a browser task using browser-use library."""
    # 処理時間とトークン数の計測開始
    start_time = time.time()
 
    print("=== Browser-Use タスク実行開始 ===")
    print(f"タスク: {task_description}")
    print("=" * 60)
    
    # タスクの定義
    task_input = task_description

    try:
        # エージェントの作成と実行
        agent = Agent(
            task=task_input,
            llm=ChatAzureOpenAI(model=model, temperature=temperature),
        )
        
        print("エージェントがタスクの実行を開始します...")
        result = await agent.run()
        await agent.close()
        
        print("\n" + "=" * 60)
        print("=== タスク実行結果 ===")
        print(result.final_result())
        print("=" * 60)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        result = None
    
    # 処理時間とトークン数の計測終了
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("=== 実行ログ ===")
    print(f"処理時間: {elapsed_time:.2f}秒")
    
    if result and hasattr(result, 'usage'):
        usage = result.usage
        if hasattr(usage, 'total_tokens'):
            print(f"実際の消費トークン数: {usage.total_tokens}")
        if hasattr(usage, 'total_prompt_tokens'):
            print(f"  - プロンプトトークン: {usage.total_prompt_tokens}")
        if hasattr(usage, 'total_completion_tokens'):
            print(f"  - 補完トークン: {usage.total_completion_tokens}")
        if hasattr(usage, 'total_cost'):
            print(f"  - 総コスト: ${usage.total_cost}")
        if hasattr(usage, 'entry_count'):
            print(f"  - API呼び出し回数: {usage.entry_count}")
    
    print("=" * 60)
    
    return {
        "result": result.final_result() if result else None,
        "execution_time": elapsed_time,
        "usage": result.usage if result and hasattr(result, 'usage') else None
    }

async def main():
    """メイン関数 - デモ用のタスクを実行"""
    # 天気情報取得タスクの例
    weather_task = """
    明日の東京都新宿区の天気と最高気温、最低気温を調べて報告する
    """

    # デフォルトは天気情報取得タスクを実行
    result = await execute_browser_task(weather_task)
    return result

if __name__ == "__main__":
    asyncio.run(main())
