"""第1回: LLMへの単発質問と、履歴を保持する対話。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # /exit で終了、/reset で履歴を消去

通常はツールも会話履歴も持たない「素の」呼び出し。
--chat の履歴は実行中のメモリだけに保持し、ファイルには保存しない。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(response: types.GenerateContentResponse) -> None:
    """API が報告した各応答のトークン数を表示する。未取得は0と区別する。"""
    usage = response.usage_metadata
    input_count = usage.prompt_token_count if usage else None
    output_count = usage.candidates_token_count if usage else None
    print(f"[tokens] input={input_count if input_count is not None else 'N/A'} "
          f"output={output_count if output_count is not None else 'N/A'}")


def ask(question: str) -> str:
    with genai.Client(api_key=get_api_key()) as client:
        response = client.models.generate_content(model=MODEL, contents=question)
        print_usage(response)
        return response.text or "（テキストの回答が返りませんでした）"


def chat(initial_question: str | None = None) -> None:
    """同じSDKチャットを再利用して、過去の質問と回答を次の質問に渡す。"""
    with genai.Client(api_key=get_api_key()) as client:
        session = client.chats.create(model=MODEL)
        print("対話モード：/exit で終了、/reset で履歴を消去。履歴は終了時に消えます。")
        pending = initial_question
        while True:
            try:
                question = pending if pending is not None else input("あなた> ")
                pending = None
                question = question.strip()
                if not question:
                    continue
                if question == "/exit":
                    break
                if question == "/reset":
                    session = client.chats.create(model=MODEL)
                    print("会話履歴を消去しました。")
                    continue
                response = session.send_message(question)
                print(f"AI> {response.text or '（テキストの回答が返りませんでした）'}")
                print_usage(response)
            except (EOFError, KeyboardInterrupt):
                print("\n対話を終了します。")
                break
            except errors.APIError as e:
                # 応答の詳細にはリクエスト情報が含まれうるため表示しない。
                print(api_error_message(e.code))
                print("自動再試行はしません。質問を再入力するか /exit で終了してください。")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保持して対話する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            chat(args.question)
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
