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
        "source_url": "",
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
        "source_url": "",
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
        "source_url": "",
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


def test_build_search_explainer_describes_rewritten_query() -> None:
    lines = ui._build_search_explainer("build_weekly_ci_digest writer output path")

    assert lines[0].startswith("Started from your question")
    assert "concrete code evidence" in lines[1]
    assert lines[2] == "build_weekly_ci_digest writer output path"


def test_evidence_bucket_label_prefers_direct_implementation() -> None:
    label = ui._evidence_bucket_label(
        "app/metrics.py defines `build_weekly_ci_digest()` and contains a call to `write_summary()`.",
        "app/metrics.py",
    )

    assert label == "Direct implementation evidence"


def test_evidence_bucket_label_detects_supporting_scripts() -> None:
    label = ui._evidence_bucket_label(
        "Sibling script supplement selected from the same runtime folder.",
        "skills/brainstorming/scripts/start-server.sh",
    )

    assert label == "Supporting sibling script"


def test_build_source_descriptors_adds_kind_labels() -> None:
    descriptors = ui._build_source_descriptors(
        [
            "app/metrics.py",
            "skills/brainstorming/scripts/start-server.sh",
            "README.md",
        ]
    )

    assert descriptors[0]["label"] == "Python source"
    assert descriptors[1]["label"] == "Shell script"
    assert descriptors[2]["label"] == "Documentation"


def test_build_source_descriptors_deduplicates_sources() -> None:
    descriptors = ui._build_source_descriptors(
        [
            "app/metrics.py",
            "app/metrics.py",
            "app/ui.py",
        ]
    )

    assert [item["path"] for item in descriptors] == ["app/metrics.py", "app/ui.py"]


def test_suggested_questions_cover_multiple_question_types() -> None:
    prompts = ui.DEFAULT_SUGGESTED_QUESTIONS

    assert "How is the weekly digest built?" in prompts
    assert "Which configuration values are defined in this project?" in prompts
    assert "What calls summarize_workflow_runs across files?" in prompts
    assert "What design risks do you see in this project?" in prompts


def test_parse_github_repo_url_extracts_owner_and_repo() -> None:
    parsed = ui._parse_github_repo_url("https://github.com/openai/codex.git")

    assert parsed is not None
    assert parsed["kind"] == "github_url"
    assert parsed["owner"] == "openai"
    assert parsed["repo"] == "codex"


def test_resolve_repo_input_maps_github_url_to_clone_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ui, "CLONED_REPOS_ROOT", tmp_path / ".repos")

    repo_path, repo_spec = ui._resolve_repo_input("https://github.com/openai/codex")

    assert repo_spec["kind"] == "github_url"
    assert repo_path == (tmp_path / ".repos" / "openai" / "codex").resolve()


def test_validate_repo_input_accepts_github_url_before_clone(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ui, "CLONED_REPOS_ROOT", tmp_path / ".repos")
    repo_path, repo_spec = ui._resolve_repo_input("https://github.com/openai/codex")

    valid, message = ui._validate_repo_input("https://github.com/openai/codex", repo_path, repo_spec)

    assert valid is True
    assert message == ""


def test_apply_pending_repo_input_value_updates_widget_state() -> None:
    st.session_state.clear()
    st.session_state["pending_repo_input_value"] = "C:/repos/demo"
    st.session_state["repo_input_value"] = "old"

    ui._apply_pending_repo_input_value()

    assert st.session_state["repo_input_value"] == "C:/repos/demo"
    assert st.session_state["pending_repo_input_value"] == ""


def test_apply_pending_workspace_selector_updates_widget_state() -> None:
    st.session_state.clear()
    st.session_state["pending_workspace_selector"] = "C:/repos/demo"
    st.session_state["workspace_selector"] = "old"

    ui._apply_pending_workspace_selector()

    assert st.session_state["workspace_selector"] == "C:/repos/demo"
    assert st.session_state["pending_workspace_selector"] == ""


def test_save_workspace_queues_workspace_selector(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    st.session_state.clear()
    st.session_state["workspaces"] = {}
    st.session_state["workspace_order"] = []
    st.session_state["active_repo_key"] = ""
    st.session_state["pending_workspace_selector"] = ""

    ui._save_workspace(repo_path=repo_path, index="fake-index")

    repo_key = str(repo_path.resolve())
    assert st.session_state["active_repo_key"] == repo_key
    assert st.session_state["pending_workspace_selector"] == repo_key
    assert repo_key in st.session_state["workspace_order"]


def test_save_workspace_persists_github_source_url(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    st.session_state.clear()
    st.session_state["workspaces"] = {}
    st.session_state["workspace_order"] = []
    st.session_state["active_repo_key"] = ""
    st.session_state["pending_workspace_selector"] = ""

    ui._save_workspace(
        repo_path=repo_path,
        index="fake-index",
        source_url="https://github.com/openai/codex",
    )

    repo_key = str(repo_path.resolve())
    assert st.session_state["workspaces"][repo_key]["source_url"] == "https://github.com/openai/codex"


def test_get_workspace_does_not_auto_save_unsaved_repo(tmp_path: Path) -> None:
    repo_path = tmp_path / "unsaved-repo"
    repo_path.mkdir()
    repo_key = str(repo_path.resolve())

    st.session_state.clear()
    st.session_state["workspaces"] = {}
    st.session_state["workspace_order"] = []
    st.session_state["active_repo_key"] = ""

    workspace = ui._get_workspace(repo_path)

    assert workspace["repo_path"] == repo_key
    assert workspace["index"] is None
    assert workspace["messages"] == []
    assert repo_key not in st.session_state["workspaces"]
    assert repo_key not in st.session_state["workspace_order"]


def test_workspace_source_url_returns_github_input_for_github_repo() -> None:
    source_url = ui._workspace_source_url(
        {
            "kind": "github_url",
            "input": "https://github.com/openai/codex",
            "owner": "openai",
            "repo": "codex",
        }
    )

    assert source_url == "https://github.com/openai/codex"


def test_workspace_source_url_ignores_local_path_repo() -> None:
    source_url = ui._workspace_source_url({"kind": "local_path", "input": "C:/repos/codex"})

    assert source_url == ""


def test_infer_github_source_url_from_cloned_repo_path(monkeypatch, tmp_path: Path) -> None:
    cloned_root = tmp_path / ".repos"
    repo_path = cloned_root / "obra" / "superpowers"
    repo_path.mkdir(parents=True)
    monkeypatch.setattr(ui, "CLONED_REPOS_ROOT", cloned_root)

    source_url = ui._infer_github_source_url(repo_path)

    assert source_url == "https://github.com/obra/superpowers"


def test_infer_github_source_url_returns_empty_for_non_clone_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ui, "CLONED_REPOS_ROOT", tmp_path / ".repos")
    repo_path = tmp_path / "local-repo"
    repo_path.mkdir()

    source_url = ui._infer_github_source_url(repo_path)

    assert source_url == ""


def test_display_source_url_prefers_stored_value(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ui, "CLONED_REPOS_ROOT", tmp_path / ".repos")
    repo_path = tmp_path / ".repos" / "obra" / "superpowers"
    repo_path.mkdir(parents=True)

    source_url = ui._display_source_url(repo_path, {"source_url": "https://github.com/custom/url"})

    assert source_url == "https://github.com/custom/url"


def test_display_source_url_falls_back_to_inferred_clone_url(monkeypatch, tmp_path: Path) -> None:
    cloned_root = tmp_path / ".repos"
    repo_path = cloned_root / "obra" / "superpowers"
    repo_path.mkdir(parents=True)
    monkeypatch.setattr(ui, "CLONED_REPOS_ROOT", cloned_root)

    source_url = ui._display_source_url(repo_path, {"source_url": ""})

    assert source_url == "https://github.com/obra/superpowers"


def test_prepare_repo_for_indexing_clones_github_repo(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ui, "CLONED_REPOS_ROOT", tmp_path / ".repos")
    repo_path, repo_spec = ui._resolve_repo_input("https://github.com/openai/codex")
    captured: dict[str, object] = {}

    def fake_run_git_command(command: list[str], repo_input: str) -> None:
        captured["command"] = command
        captured["repo_input"] = repo_input
        repo_path.mkdir(parents=True, exist_ok=True)
        (repo_path / ".git").mkdir()

    monkeypatch.setattr(ui, "_run_git_command", fake_run_git_command)

    resolved, force_rebuild = ui._prepare_repo_for_indexing("https://github.com/openai/codex", repo_path, repo_spec)

    assert resolved == repo_path
    assert force_rebuild is True
    assert captured["command"] == ["git", "clone", "https://github.com/openai/codex.git", str(repo_path)]


def test_prepare_repo_for_indexing_marks_existing_github_repo_for_rebuild_after_pull(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ui, "CLONED_REPOS_ROOT", tmp_path / ".repos")
    repo_path, repo_spec = ui._resolve_repo_input("https://github.com/openai/codex")
    repo_path.mkdir(parents=True, exist_ok=True)
    (repo_path / ".git").mkdir()
    captured: dict[str, object] = {}

    def fake_run_git_command(command: list[str], repo_input: str) -> None:
        captured["command"] = command
        captured["repo_input"] = repo_input

    monkeypatch.setattr(ui, "_run_git_command", fake_run_git_command)

    resolved, force_rebuild = ui._prepare_repo_for_indexing("https://github.com/openai/codex", repo_path, repo_spec)

    assert resolved == repo_path
    assert force_rebuild is True
    assert captured["command"] == ["git", "-C", str(repo_path), "pull", "--ff-only"]


def test_humanize_git_error_explains_timeout() -> None:
    message = ui._humanize_git_error(
        "fatal: unable to access 'https://github.com/openai/codex.git/': Failed to connect to github.com port 443 after 21090 ms: Timed out",
        "https://github.com/openai/codex",
    )

    assert "GitHub network request timed out" in message
    assert "clone the repo locally first" in message
    assert "Original git detail:" in message


def test_humanize_git_error_explains_missing_repo() -> None:
    message = ui._humanize_git_error(
        "remote: Repository not found.",
        "https://github.com/openai/missing-repo",
    )

    assert "could not find that repository" in message
    assert "Double-check the URL" in message


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

    ui._build_repo_profile.cache_clear()
    ui._recommended_question_entries.cache_clear()
    ui._suggested_questions_for_repo.cache_clear()
    prompts = ui._suggested_questions_for_repo(tmp_path)

    assert prompts[0] == "How is the weekly digest built?"
    assert prompts[1] == "Which file fetches GitHub workflow runs and where are they summarized?"
    assert prompts[2] == "What calls summarize_workflow_runs across files?"
    assert "Which file contains argparse and the main function?" in prompts
    assert "Where is `OLLAMA_BASE_URL` configured?" in prompts
    assert "How is the summary report built?" in prompts
    assert "Where are CI charts generated?" in prompts
    assert "What design risks do you see in this project?" in prompts


def test_build_repo_profile_extracts_core_repo_signals(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "main.py").write_text(
        "import argparse\n"
        "from app.metrics import summarize_workflow_runs\n"
        "def main():\n"
        "    summarize_workflow_runs([])\n",
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

    ui._build_repo_profile.cache_clear()
    profile = ui._build_repo_profile(tmp_path)

    assert profile["has_entrypoint"] is True
    assert "app/main.py" in profile["entrypoint_reason"]
    assert profile["config_target"] == "OLLAMA_BASE_URL"
    assert "app/config.py" in profile["config_reason"]
    assert profile["relationship_target"] == "summarize_workflow_runs"
    assert "app/main.py" in profile["relationship_reason"]
    assert profile["has_workflow_fetch_and_summary"] is True
    assert profile["has_weekly_digest"] is True
    assert profile["has_summary_report"] is True
    assert profile["has_ci_charts"] is True
    assert profile["has_design_risk_signals"] is True


def test_suggested_questions_for_repo_falls_back_to_defaults(tmp_path: Path) -> None:
    ui._build_repo_profile.cache_clear()
    ui._recommended_question_entries.cache_clear()
    ui._suggested_questions_for_repo.cache_clear()
    prompts = ui._suggested_questions_for_repo(tmp_path)

    assert prompts == ui.DEFAULT_SUGGESTED_QUESTIONS


def test_recommended_question_entries_include_implementation_reasons(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "main.py").write_text(
        "import argparse\n"
        "from app.metrics import summarize_workflow_runs\n"
        "def parse_args():\n"
        "    return argparse.ArgumentParser()\n"
        "def main():\n"
        "    summarize_workflow_runs([])\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (app_dir / "config.py").write_text("OLLAMA_BASE_URL = 'http://localhost:11434'\n", encoding="utf-8")
    (app_dir / "metrics.py").write_text(
        "def summarize_workflow_runs(records):\n"
        "    workflow_failures = {}\n"
        "    return workflow_failures\n",
        encoding="utf-8",
    )
    (app_dir / "github_client.py").write_text("def fetch_workflow_runs():\n    return []\n", encoding="utf-8")
    (app_dir / "report.py").write_text(
        "def write_weekly_digest_report(*args):\n    return 'outputs/weekly_digest.md'\n",
        encoding="utf-8",
    )

    ui._build_repo_profile.cache_clear()
    ui._recommended_question_entries.cache_clear()
    entries = ui._recommended_question_entries(tmp_path)

    entry_map = {entry["prompt"]: entry["reason"] for entry in entries}
    assert "app/report.py" in entry_map["How is the weekly digest built?"]
    assert "`summarize_workflow_runs()` is defined in `app/metrics.py` and called from `app/main.py`." == entry_map[
        "What calls summarize_workflow_runs across files?"
    ]
    assert "app/config.py" in entry_map["Where is `OLLAMA_BASE_URL` configured?"]


def test_recommended_questions_skip_irrelevant_ollama_fallback(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "main.py").write_text(
        "import argparse\n"
        "from app.metrics import summarize_workflow_runs\n"
        "def main():\n"
        "    summarize_workflow_runs([])\n",
        encoding="utf-8",
    )
    (app_dir / "config.py").write_text(
        "GITHUB_API_BASE = 'https://api.github.com'\n"
        "GITHUB_TOKEN = None\n",
        encoding="utf-8",
    )
    (app_dir / "metrics.py").write_text(
        "def summarize_workflow_runs(records):\n"
        "    category_counts = {}\n"
        "    return category_counts\n",
        encoding="utf-8",
    )
    (app_dir / "github_client.py").write_text("def fetch_workflow_runs():\n    return []\n", encoding="utf-8")
    (app_dir / "report.py").write_text(
        "def write_weekly_digest_report(*args):\n    return 'outputs/weekly_digest.md'\n"
        "def write_markdown_report(*args):\n    return 'outputs/summary.md'\n",
        encoding="utf-8",
    )
    (app_dir / "charts.py").write_text("def write_failure_trend_chart(*args):\n    return True\n", encoding="utf-8")

    ui._build_repo_profile.cache_clear()
    ui._recommended_question_entries.cache_clear()
    ui._suggested_questions_for_repo.cache_clear()
    prompts = ui._suggested_questions_for_repo(tmp_path)

    assert "Where is `OLLAMA_BASE_URL` configured?" not in prompts
    assert "Where is `GITHUB_API_BASE` configured?" in prompts
    assert prompts.index("Where is `GITHUB_API_BASE` configured?") < 5


def test_build_repo_profile_detects_env_backed_config_targets(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "config.py").write_text(
        "import os\n"
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class AppConfig:\n"
        "    github_token: str | None = None\n"
        "    github_api_base: str = 'https://api.github.com'\n"
        "    @classmethod\n"
        "    def from_env(cls):\n"
        "        return cls(\n"
        "            github_token=os.getenv('GITHUB_TOKEN') or None,\n"
        "            github_api_base=os.getenv('GITHUB_API_BASE', 'https://api.github.com'),\n"
        "        )\n",
        encoding="utf-8",
    )

    ui._build_repo_profile.cache_clear()
    profile = ui._build_repo_profile(tmp_path)

    assert profile["config_target"] == "GITHUB_API_BASE"
    assert "env config" in profile["config_reason"]


def test_queue_question_stores_pending_prompt_and_user_message(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo_key = str(repo_path.resolve())
    rerun_called = {"value": False}

    st.session_state.clear()
    st.session_state["workspaces"] = {repo_key: {"repo_path": repo_key, "index": "fake-index", "messages": [], "source_url": ""}}
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


def test_queue_question_keeps_other_repo_in_flight_state_isolated(monkeypatch, tmp_path: Path) -> None:
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    repo_a_key = str(repo_a.resolve())
    repo_b_key = str(repo_b.resolve())
    rerun_calls = {"count": 0}

    st.session_state.clear()
    st.session_state["workspaces"] = {
        repo_a_key: {"repo_path": repo_a_key, "index": "index-a", "messages": [], "source_url": ""},
        repo_b_key: {"repo_path": repo_b_key, "index": "index-b", "messages": [], "source_url": ""},
    }
    st.session_state["workspace_order"] = [repo_a_key, repo_b_key]
    st.session_state["active_repo_key"] = repo_a_key
    st.session_state["pending_question"] = ""
    st.session_state["pending_repo_key"] = ""
    st.session_state["question_in_flight"] = False
    st.session_state["active_question_run_id"] = 0

    monkeypatch.setattr(ui.st, "rerun", lambda: rerun_calls.__setitem__("count", rerun_calls["count"] + 1))

    ui._queue_question("Question for repo A", repo_a)
    ui._queue_question("Question for repo B", repo_b)

    assert rerun_calls["count"] == 2
    assert ui._question_busy_for_repo(repo_a) is True
    assert ui._question_busy_for_repo(repo_b) is True
    assert st.session_state["workspaces"][repo_a_key]["messages"][-1]["content"] == "Question for repo A"
    assert st.session_state["workspaces"][repo_b_key]["messages"][-1]["content"] == "Question for repo B"


def test_process_pending_question_consumes_matching_prompt(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo_key = str(repo_path.resolve())
    workspace = {
        "repo_path": repo_key,
        "index": "fake-index",
        "messages": [{"role": "user", "content": "Where is the Ollama base URL configured?"}],
        "source_url": "",
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
        "source_url": "",
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


def test_cancel_pending_question_only_clears_matching_repo(monkeypatch, tmp_path: Path) -> None:
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    rerun_calls = {"count": 0}

    st.session_state.clear()
    st.session_state["workspaces"] = {}
    st.session_state["workspace_order"] = []
    st.session_state["active_repo_key"] = str(repo_a.resolve())
    st.session_state["pending_question"] = ""
    st.session_state["pending_repo_key"] = ""
    st.session_state["question_in_flight"] = False
    st.session_state["active_question_run_id"] = 0

    monkeypatch.setattr(ui.st, "rerun", lambda: rerun_calls.__setitem__("count", rerun_calls["count"] + 1))

    ui._queue_question("Question for repo A", repo_a)
    ui._queue_question("Question for repo B", repo_b)
    ui._cancel_pending_question(repo_b)

    assert rerun_calls["count"] == 3
    assert ui._question_busy_for_repo(repo_a) is True
    assert ui._question_busy_for_repo(repo_b) is False


def test_question_busy_for_repo_checks_matching_running_state(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    other_repo = tmp_path / "other"
    other_repo.mkdir()

    st.session_state.clear()
    st.session_state["question_runs"] = {
        str(repo_path.resolve()): {
            "pending_question": "",
            "question_in_flight": False,
            "active_question_run_id": 0,
        }
    }

    assert ui._question_busy_for_repo(repo_path) is False
    assert ui._question_busy_for_repo(other_repo) is False

    st.session_state["question_runs"][str(repo_path.resolve())] = {
        "pending_question": "How is the weekly digest built?",
        "question_in_flight": True,
        "active_question_run_id": 1,
    }

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
        "source_url": "",
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
