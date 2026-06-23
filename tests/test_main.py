from __future__ import annotations

import argparse
import builtins
from pathlib import Path

from app import main
from app.config import AppConfig
from app.qa import AnswerResult


def _base_config() -> AppConfig:
    return AppConfig(
        ollama_base_url="http://localhost:11434",
        chat_model="qwen2.5:7b",
        embedding_model="nomic-embed-text",
        index_dir_name=".storage",
        chunk_size=1200,
        chunk_overlap=150,
        top_k=8,
    )


def test_main_index_command_prints_index_path(monkeypatch, capsys, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    config = _base_config()

    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        main,
        "parse_args",
        lambda: argparse.Namespace(repo_path=str(repo_path), command="index", rebuild=False, question=None),
    )
    monkeypatch.setattr(main.AppConfig, "from_env", lambda: config)
    monkeypatch.setattr(main, "build_or_load_index", lambda repo_path, config, rebuild: "fake-index")

    main.main()
    output = capsys.readouterr().out

    assert "Index ready at:" in output
    assert str(config.resolve_index_dir(repo_path.resolve())) in output


def test_main_ask_question_invokes_single_question_handler(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    config = _base_config()
    seen: dict[str, object] = {}

    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        main,
        "parse_args",
        lambda: argparse.Namespace(
            repo_path=str(repo_path),
            command="ask",
            rebuild=False,
            question="Which file contains argparse and the main function?",
        ),
    )
    monkeypatch.setattr(main.AppConfig, "from_env", lambda: config)
    monkeypatch.setattr(main, "build_or_load_index", lambda repo_path, config, rebuild: "fake-index")

    def fake_handle_question(index, question: str, config: AppConfig, repo_path: Path) -> None:
        seen["index"] = index
        seen["question"] = question
        seen["config"] = config
        seen["repo_path"] = repo_path

    monkeypatch.setattr(main, "_handle_question", fake_handle_question)
    monkeypatch.setattr(main, "_interactive_loop", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("interactive loop should not run")))

    main.main()

    assert seen["index"] == "fake-index"
    assert seen["question"] == "Which file contains argparse and the main function?"
    assert seen["config"] == config
    assert seen["repo_path"] == repo_path.resolve()


def test_interactive_loop_stops_on_exit(monkeypatch, capsys, tmp_path: Path) -> None:
    config = _base_config()
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    prompts = iter(["exit"])

    monkeypatch.setattr(builtins, "input", lambda _: next(prompts))
    monkeypatch.setattr(main, "answer_question", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("answer_question should not run")))

    main._interactive_loop(index="fake-index", config=config, repo_path=repo_path)
    output = capsys.readouterr().out

    assert "Interactive mode started. Type 'exit' to quit." in output


def test_handle_question_prints_answer_and_sources(monkeypatch, capsys, tmp_path: Path) -> None:
    config = _base_config()
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    monkeypatch.setattr(
        main,
        "answer_question",
        lambda **kwargs: AnswerResult(
            answer="Answer: app/main.py\nWhy:\n- app/main.py contains argparse.\n\nSources:\n- app/main.py",
            sources=["app/main.py"],
            search_question="Which file contains argparse and the main function?",
            evidence=[],
            call_chain_summary="",
            confidence_label="High confidence",
            confidence_score=90,
            risk_note="Focused evidence.",
        ),
    )

    main._handle_question(index="fake-index", question="Which file contains argparse and the main function?", config=config, repo_path=repo_path)
    output = capsys.readouterr().out

    assert "Answer:" in output
    assert "Sources:" in output
    assert "- app/main.py" in output
