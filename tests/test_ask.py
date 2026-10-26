from copy import deepcopy

import pytest
from google.genai import Client, errors, types
from google.genai.models import Models

from app import ask


def response(text="テスト用の回答", usage=True):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            finish_reason="STOP",
        )],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=5,
        ) if usage else None,
    )


@pytest.fixture
def requests(monkeypatch):
    monkeypatch.setenv("GEMINIAPIKEY", "test-placeholder-not-a-real-key")
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: Client(
        **kwargs, http_options=types.HttpOptions(
            client_args={"trust_env": False},
            async_client_args={"trust_env": False},
        ),
    ))
    calls = []

    def generate(self, **kwargs):
        calls.append(deepcopy(kwargs))
        return response()

    monkeypatch.setattr(Models, "generate_content", generate)
    return calls


def inputs(monkeypatch, values):
    values = iter(values)

    def read(_):
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", read)


def test_second_request_contains_question_and_model_reply(monkeypatch, requests):
    inputs(monkeypatch, ["私の名前はこうきです", "私の名前は？", "/exit"])
    ask.chat()
    contents = requests[1]["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert [c.parts[0].text for c in contents] == [
        "私の名前はこうきです", "テスト用の回答", "私の名前は？",
    ]
    assert requests[1]["model"] == ask.MODEL


def test_reset_drops_history_and_blank_input_makes_no_request(monkeypatch, requests):
    inputs(monkeypatch, ["最初の質問", "  ", "/reset", "次の質問", "/exit"])
    ask.chat()
    assert len(requests) == 2
    assert len(requests[1]["contents"]) == 1
    assert requests[1]["contents"][0].parts[0].text == "次の質問"


@pytest.mark.parametrize("end", [EOFError(), KeyboardInterrupt()])
def test_end_input_exits_cleanly(monkeypatch, requests, end):
    inputs(monkeypatch, [end])
    ask.chat()
    assert not requests


def test_api_error_preserves_successful_history(monkeypatch, requests, capsys):
    def generate(self, **kwargs):
        requests.append(deepcopy(kwargs))
        if len(requests) == 2:
            raise errors.ClientError(429, {"error": {"message": "private-detail"}})
        return response()

    monkeypatch.setattr(Models, "generate_content", generate)
    inputs(monkeypatch, ["成功する質問", "失敗する質問", "再入力", "/exit"])
    ask.chat()
    assert [c.parts[0].text for c in requests[2]["contents"]] == [
        "成功する質問", "テスト用の回答", "再入力",
    ]
    output = capsys.readouterr().out
    assert "レート制限" in output
    assert "private-detail" not in output


def test_one_shot_and_usage(requests, capsys):
    assert ask.ask("LoRAとは？") == "テスト用の回答"
    assert requests[0]["contents"] == "LoRAとは？"
    assert "input=12 output=5" in capsys.readouterr().out


def test_missing_usage_is_not_reported_as_zero(capsys):
    ask.print_usage(response(usage=False))
    assert "input=N/A output=N/A" in capsys.readouterr().out


def test_cli_chat_initial_question(monkeypatch, requests):
    monkeypatch.setattr("sys.argv", ["app.ask", "--chat", "最初の質問"])
    inputs(monkeypatch, ["続き", "/exit"])
    ask.main()
    assert len(requests[1]["contents"]) == 3


def test_cli_default_question(monkeypatch, requests):
    monkeypatch.setattr("sys.argv", ["app.ask"])
    ask.main()
    assert requests[0]["contents"] == ask.DEFAULT_QUESTION


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINIAPIKEY", raising=False)
    with pytest.raises(SystemExit, match="GEMINIAPIKEY"):
        ask.chat()
