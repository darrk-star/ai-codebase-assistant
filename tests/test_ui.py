from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import streamlit as st

from app.config import AppConfig
from app import ui
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


def test_submit_suggested_question_without_index_adds_guard_message(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo_key = str(repo_path.resolve())
    workspace = {
        "repo_path": repo_key,
        "index": None,
        "messages": [],
    }

    st.session_state.clear()
    st.session_state["workspaces"] = {repo_key: workspace}
    st.session_state["workspace_order"] = [repo_key]
    st.session_state["active_repo_key"] = ""
    st.session_state["pending_question"] = ""
    st.session_state["pending_repo_key"] = repo_key
    st.session_state["question_in_flight"] = True
    st.session_state["active_question_run_id"] = 1

    rerun_called = {"value": False}

    def fake_rerun() -> None:
        rerun_called["value"] = True

    monkeypatch.setattr(ui.st, "rerun", fake_rerun)

    ui._submit_suggested_question(
        question="Which file contains argparse and the main function?",
        workspace=workspace,
        repo_path=repo_path,
        config=_base_config(),
    )

    assert rerun_called["value"] is True
    assert len(workspace["messages"]) == 1
    assert workspace["messages"][0]["content"].startswith("Index is not ready yet.")
    assert st.session_state["active_repo_key"] == repo_key


def test_submit_suggested_question_stores_call_chain_summary(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo_key = str(repo_path.resolve())
    workspace = {
        "repo_path": repo_key,
        "index": "fake-index",
        "messages": [{"role": "user", "content": "Which file calls compute_digest and where is it defined?"}],
    }

    st.session_state.clear()
    st.session_state["workspaces"] = {repo_key: workspace}
    st.session_state["workspace_order"] = [repo_key]
    st.session_state["active_repo_key"] = ""
    st.session_state["pending_question"] = ""
    st.session_state["pending_repo_key"] = repo_key
    st.session_state["question_in_flight"] = True
    st.session_state["active_question_run_id"] = 1

    class FakeSpinner:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(ui.st, "spinner", lambda _: FakeSpinner())
    monkeypatch.setattr(ui.st, "rerun", lambda: None)
    monkeypatch.setattr(
        ui,
        "answer_question",
        lambda **kwargs: AnswerResult(
            answer="Answer: app/service.py calls compute_digest.\nWhy:\n- app/service.py calls it.\n\nSources:\n- app/service.py",
            sources=["app/service.py"],
            search_question="Which file calls compute_digest and where is it defined?",
            evidence=[],
            call_chain_summary="- `app/service.py` -> `compute_digest()` -> `app/helpers.py`",
            confidence_label="High confidence",
            confidence_score=88,
            risk_note="Focused evidence.",
        ),
    )

    ui._submit_suggested_question(
        question="Which file calls compute_digest and where is it defined?",
        workspace=workspace,
        repo_path=repo_path,
        config=_base_config(),
        run_id=1,
    )

    assert len(workspace["messages"]) == 2
    assert workspace["messages"][1]["call_chain_summary"].startswith("- `app/service.py` -> `compute_digest()` -> `app/helpers.py`")


def test_submit_suggested_question_accepts_result_without_call_chain_summary(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo_key = str(repo_path.resolve())
    workspace = {
        "repo_path": repo_key,
        "index": "fake-index",
        "messages": [{"role": "user", "content": "Which file contains argparse and the main function?"}],
    }

    st.session_state.clear()
    st.session_state["workspaces"] = {repo_key: workspace}
    st.session_state["workspace_order"] = [repo_key]
    st.session_state["active_repo_key"] = ""
    st.session_state["pending_question"] = ""
    st.session_state["pending_repo_key"] = repo_key
    st.session_state["question_in_flight"] = True
    st.session_state["active_question_run_id"] = 1

    class FakeSpinner:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(ui.st, "spinner", lambda _: FakeSpinner())
    monkeypatch.setattr(ui.st, "rerun", lambda: None)
    monkeypatch.setattr(
        ui,
        "answer_question",
        lambda **kwargs: SimpleNamespace(
            answer="Answer: app/main.py\nWhy:\n- app/main.py contains argparse.\n\nSources:\n- app/main.py",
            sources=["app/main.py"],
            search_question="Which file contains argparse and the main function?",
            evidence=[],
            confidence_label="High confidence",
            confidence_score=90,
            risk_note="Focused evidence.",
        ),
    )

    ui._submit_suggested_question(
        question="Which file contains argparse and the main function?",
        workspace=workspace,
        repo_path=repo_path,
        config=_base_config(),
        run_id=1,
    )

    assert len(workspace["messages"]) == 2
    assert workspace["messages"][1]["call_chain_summary"] == ""


def test_render_message_content_preserves_why_block_for_assistant(monkeypatch) -> None:
    captured: dict[str, str] = {}

    monkeypatch.setattr(ui.st, "markdown", lambda value: captured.setdefault("markdown", value))
    monkeypatch.setattr(ui.st, "write", lambda value: captured.setdefault("write", value))

    ui._render_message_content(
        "Answer: Based on the retrieved implementation, the main design risks are hard-coded failure classification rules.\nWhy:\n- Evidence line.\n\nSources:\n- app/ci_failure_analysis.py",
        "assistant",
    )

    assert "write" not in captured
    assert "\n\nWhy:\n\n- Evidence line." in captured["markdown"]
    assert "Sources:" not in captured["markdown"]


def test_render_message_content_keeps_user_messages_unchanged(monkeypatch) -> None:
    captured: dict[str, str] = {}

    monkeypatch.setattr(ui.st, "markdown", lambda value: captured.setdefault("markdown", value))
    monkeypatch.setattr(ui.st, "write", lambda value: captured.setdefault("write", value))

    ui._render_message_content("What design risks do you see in this project?", "user")

    assert captured["write"] == "What design risks do you see in this project?"
    assert "markdown" not in captured


def test_suggested_questions_cover_multiple_question_types() -> None:
    prompts = ui.DEFAULT_SUGGESTED_QUESTIONS

    assert "How is the weekly digest built?" in prompts
    assert "Where is the Ollama base URL configured?" in prompts
    assert "What calls summarize_workflow_runs across files?" in prompts
    assert "What design risks do you see in this project?" in prompts


def test_suggested_questions_for_repo_uses_detected_repo_signals(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "main.py").write_text(
        "import argparse\n"
        "from app.github_client import GitHubClient\n"
        "from app.metrics import summarize_workflow_runs\n"
        "from app.report import write_weekly_digest_report, write_markdown_report\n"
        "from app.charts import write_failure_trend_chart\n"
        "def main():\n"
        "    summarize_workflow_runs([])\n"
        "    write_weekly_digest_report('outputs/weekly_digest.md', 'repo', 7, {})\n",
        encoding="utf-8",
    )
    (app_dir / "config.py").write_text("OLLAMA_BASE_URL = 'http://localhost:11434'\n", encoding="utf-8")
    (app_dir / "metrics.py").write_text(
        "def summarize_workflow_runs(records):\n"
        "    category_counts = {}\n"
        "    workflow_failures = {}\n"
        "    return category_counts, workflow_failures\n",
        encoding="utf-8",
    )
    (app_dir / "github_client.py").write_text("def fetch_workflow_runs():\n    return []\n", encoding="utf-8")
    (app_dir / "report.py").write_text(
        "def write_weekly_digest_report(*args):\n    return 'outputs/weekly_digest.md'\n"
        "def write_markdown_report(*args):\n    return 'outputs/summary.md'\n",
        encoding="utf-8",
    )
    (app_dir / "charts.py").write_text("def write_failure_trend_chart(*args):\n    return True\n", encoding="utf-8")
    (app_dir / "ci_failure_analysis.py").write_text("patterns = [('unknown_failure', ['error'])]\n", encoding="utf-8")

    ui._suggested_questions_for_repo.cache_clear()
    prompts = ui._suggested_questions_for_repo(tmp_path)

    assert prompts[0] == "How is the weekly digest built?"
    assert prompts[1] == "Which file fetches GitHub workflow runs and where are they summarized?"
    assert prompts[2] == "What calls summarize_workflow_runs across files?"
    assert "Which file contains argparse and the main function?" in prompts
    assert "Where is the Ollama base URL configured?" in prompts
    assert "How is the summary report built?" in prompts
    assert "Where are CI charts generated?" in prompts
    assert "What design risks do you see in this project?" in prompts


def test_suggested_questions_for_repo_falls_back_to_defaults(tmp_path: Path) -> None:
    ui._suggested_questions_for_repo.cache_clear()
    prompts = ui._suggested_questions_for_repo(tmp_path)

    assert prompts == ui.DEFAULT_SUGGESTED_QUESTIONS


def test_queue_question_stores_pending_prompt_and_user_message(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo_key = str(repo_path.resolve())
    rerun_called = {"value": False}

    st.session_state.clear()
    st.session_state["workspaces"] = {repo_key: {"repo_path": repo_key, "index": "fake-index", "messages": []}}
    st.session_state["workspace_order"] = [repo_key]
    st.session_state["active_repo_key"] = repo_key
    st.session_state["pending_question"] = ""
    st.session_state["pending_repo_key"] = ""
    st.session_state["question_in_flight"] = False
    st.session_state["active_question_run_id"] = 0

    def fake_rerun() -> None:
        rerun_called["value"] = True

    monkeypatch.setattr(ui.st, "rerun", fake_rerun)

    ui._queue_question("  Which file contains argparse?  ", repo_path)

    assert rerun_called["value"] is True
    assert st.session_state["pending_question"] == "Which file contains argparse?"
    assert st.session_state["pending_repo_key"] == repo_key
    assert st.session_state["question_in_flight"] is True
    assert st.session_state["active_question_run_id"] == 1
    assert st.session_state["workspaces"][repo_key]["messages"][0]["content"] == "Which file contains argparse?"


def test_process_pending_question_consumes_matching_prompt(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo_key = str(repo_path.resolve())
    workspace = {
        "repo_path": repo_key,
        "index": "fake-index",
        "messages": [{"role": "user", "content": "Where is the Ollama base URL configured?"}],
    }
    captured: dict[str, object] = {}

    st.session_state.clear()
    st.session_state["workspaces"] = {repo_key: workspace}
    st.session_state["workspace_order"] = [repo_key]
    st.session_state["active_repo_key"] = repo_key
    st.session_state["pending_question"] = "Where is the Ollama base URL configured?"
    st.session_state["pending_repo_key"] = repo_key
    st.session_state["question_in_flight"] = True
    st.session_state["active_question_run_id"] = 7

    monkeypatch.setattr(
        ui,
        "_submit_suggested_question",
        lambda question, workspace, repo_path, config, run_id=None, thinking_placeholder=None: captured.update(
            {
                "question": question,
                "workspace": workspace,
                "repo_path": repo_path,
                "config": config,
                "run_id": run_id,
                "thinking_placeholder": thinking_placeholder,
            }
        ),
    )

    ui._process_pending_question(workspace, repo_path, _base_config())

    assert captured["question"] == "Where is the Ollama base URL configured?"
    assert captured["workspace"] is workspace
    assert captured["repo_path"] == repo_path
    assert captured["run_id"] == 7
    assert captured["thinking_placeholder"] is None
    assert st.session_state["pending_question"] == ""
    assert st.session_state["pending_repo_key"] == ""
    assert st.session_state["question_in_flight"] is False


def test_process_pending_question_ignores_other_workspace(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    other_repo = tmp_path / "other"
    other_repo.mkdir()
    repo_key = str(repo_path.resolve())
    workspace = {
        "repo_path": repo_key,
        "index": "fake-index",
        "messages": [],
    }

    st.session_state.clear()
    st.session_state["pending_question"] = "How is the weekly digest built?"
    st.session_state["pending_repo_key"] = str(other_repo.resolve())
    st.session_state["question_in_flight"] = True
    st.session_state["active_question_run_id"] = 3

    ui._process_pending_question(workspace, repo_path, _base_config())

    assert st.session_state["pending_question"] == "How is the weekly digest built?"
    assert st.session_state["pending_repo_key"] == str(other_repo.resolve())
    assert st.session_state["question_in_flight"] is True


def test_cancel_pending_question_clears_matching_repo(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    rerun_called = {"value": False}

    st.session_state.clear()
    st.session_state["pending_question"] = "How is the weekly digest built?"
    st.session_state["pending_repo_key"] = str(repo_path.resolve())
    st.session_state["question_in_flight"] = True
    st.session_state["active_question_run_id"] = 5

    def fake_rerun() -> None:
        rerun_called["value"] = True

    monkeypatch.setattr(ui.st, "rerun", fake_rerun)

    ui._cancel_pending_question(repo_path)

    assert rerun_called["value"] is True
    assert st.session_state["pending_question"] == ""
    assert st.session_state["pending_repo_key"] == ""
    assert st.session_state["question_in_flight"] is False
    assert st.session_state["active_question_run_id"] == 6


def test_question_busy_for_repo_checks_matching_running_state(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    other_repo = tmp_path / "other"
    other_repo.mkdir()

    st.session_state.clear()
    st.session_state["pending_repo_key"] = str(repo_path.resolve())
    st.session_state["question_in_flight"] = False

    assert ui._question_busy_for_repo(repo_path) is False
    assert ui._question_busy_for_repo(other_repo) is False

    st.session_state["pending_repo_key"] = str(repo_path.resolve())
    st.session_state["question_in_flight"] = True

    assert ui._question_busy_for_repo(repo_path) is True
    assert ui._question_busy_for_repo(other_repo) is False


def test_submit_suggested_question_drops_late_answer_after_cancel(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo_key = str(repo_path.resolve())
    workspace = {
        "repo_path": repo_key,
        "index": "fake-index",
        "messages": [{"role": "user", "content": "Where is the Ollama base URL configured?"}],
    }

    st.session_state.clear()
    st.session_state["workspaces"] = {repo_key: workspace}
    st.session_state["workspace_order"] = [repo_key]
    st.session_state["active_repo_key"] = repo_key
    st.session_state["pending_question"] = "Where is the Ollama base URL configured?"
    st.session_state["pending_repo_key"] = repo_key
    st.session_state["question_in_flight"] = True
    st.session_state["active_question_run_id"] = 11

    class FakeSpinner:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(ui.st, "spinner", lambda _: FakeSpinner())
    monkeypatch.setattr(ui.st, "rerun", lambda: None)

    def fake_answer_question(**kwargs):
        st.session_state["pending_question"] = ""
        st.session_state["pending_repo_key"] = ""
        st.session_state["question_in_flight"] = False
        st.session_state["active_question_run_id"] = 12
        return SimpleNamespace(
            answer="Late answer that should be ignored",
            sources=["app/config.py"],
            search_question="Where is the Ollama base URL configured?",
            evidence=[],
            confidence_label="High confidence",
            confidence_score=90,
            risk_note="Focused evidence.",
        )

    monkeypatch.setattr(ui, "answer_question", fake_answer_question)

    ui._submit_suggested_question(
        question="Where is the Ollama base URL configured?",
        workspace=workspace,
        repo_path=repo_path,
        config=_base_config(),
        run_id=11,
    )

    assert len(workspace["messages"]) == 1
