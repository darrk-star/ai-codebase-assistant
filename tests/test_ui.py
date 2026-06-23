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
        "messages": [],
    }

    st.session_state.clear()
    st.session_state["workspaces"] = {repo_key: workspace}
    st.session_state["workspace_order"] = [repo_key]
    st.session_state["active_repo_key"] = ""

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
        "messages": [],
    }

    st.session_state.clear()
    st.session_state["workspaces"] = {repo_key: workspace}
    st.session_state["workspace_order"] = [repo_key]
    st.session_state["active_repo_key"] = ""

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
    )

    assert len(workspace["messages"]) == 2
    assert workspace["messages"][1]["call_chain_summary"] == ""


def test_suggested_questions_cover_multiple_question_types() -> None:
    prompts = ui.SUGGESTED_QUESTIONS

    assert "How is the weekly digest built?" in prompts
    assert "Where is the Ollama base URL configured?" in prompts
    assert "What calls compute_digest across files?" in prompts
    assert "What design risks do you see in this project?" in prompts
