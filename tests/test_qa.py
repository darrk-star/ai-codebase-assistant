from __future__ import annotations

from pathlib import Path

import pytest

from app.config import AppConfig
from app import qa


class FakeNode:
    def __init__(self, file_path: str, text: str, extension: str = ".py") -> None:
        self.metadata = {
            "file_path": file_path,
            "extension": extension,
        }
        self.text = text


def test_answer_question_requires_index(tmp_path: Path) -> None:
    config = AppConfig(
        ollama_base_url="http://localhost:11434",
        chat_model="qwen2.5:7b",
        embedding_model="nomic-embed-text",
        index_dir_name=".storage",
        chunk_size=1200,
        chunk_overlap=150,
        top_k=8,
    )

    with pytest.raises(ValueError, match="Index is not ready"):
        qa.answer_question(
            index=None,
            question="Which file contains argparse and the main function?",
            config=config,
            repo_path=tmp_path,
        )


def test_rerank_nodes_prioritizes_entrypoint_file() -> None:
    nodes = [
        FakeNode(
            "app/metrics.py",
            "def build_weekly_ci_digest(records):\n    return []",
        ),
        FakeNode(
            "app/main.py",
            "import argparse\n\ndef main() -> None:\n    parser = argparse.ArgumentParser()\n"
            "if __name__ == '__main__':\n    main()\n",
        ),
        FakeNode(
            "app/config.py",
            "def from_env() -> AppConfig:\n    return AppConfig()",
        ),
    ]

    ranked = qa._rerank_nodes(
        question="Which file contains argparse and the main function?",
        search_question="Which file contains argparse and the main function?",
        nodes=nodes,
    )

    assert ranked[0].metadata["file_path"] == "app/main.py"


def test_collect_keyword_contexts_finds_config_file(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "config.py").write_text(
        "import os\n\n"
        "def from_env():\n"
        "    return os.getenv('OLLAMA_BASE_URL')\n",
        encoding="utf-8",
    )
    (app_dir / "main.py").write_text(
        "def main():\n    print('hello')\n",
        encoding="utf-8",
    )

    matches = qa._collect_keyword_contexts(
        repo_path=tmp_path,
        search_text="Where is the Ollama base URL configured in config?",
    )

    assert matches
    assert matches[0][0] == "app/config.py"
    assert "OLLAMA_BASE_URL" in matches[0][2]


def test_rerank_nodes_prioritizes_indexing_file_for_index_question() -> None:
    nodes = [
        FakeNode(
            "app/metrics.py",
            "def build_weekly_ci_digest(records):\n    return []",
        ),
        FakeNode(
            "app/indexing.py",
            "from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage\n"
            "def build_or_load_index():\n"
            "    index = VectorStoreIndex.from_documents([])\n"
            "    index.storage_context.persist(persist_dir='.storage')\n"
            "    return load_index_from_storage(StorageContext.from_defaults(persist_dir='.storage'))\n",
        ),
        FakeNode(
            "app/ui.py",
            "index = build_or_load_index(repo_path=repo_path, config=config, rebuild=rebuild)\n",
        ),
    ]

    ranked = qa._rerank_nodes(
        question="How is the index built and persisted?",
        search_question="How is the index built and persisted?",
        nodes=nodes,
    )

    assert ranked[0].metadata["file_path"] == "app/indexing.py"


def test_finalize_answer_adds_why_and_sources() -> None:
    result = qa._finalize_answer(
        answer_text="app/main.py contains the main function.",
        source_paths=["app/main.py", "app/config.py"],
        evidence_blocks=[
            {"file_path": "app/main.py", "reason": "Keyword-based code match", "snippet": "def main() -> None"},
        ],
        question="Which file contains argparse and the main function?",
        call_chain_summary="",
    )

    assert result.startswith("Answer:")
    assert "\nWhy:\n" in result
    assert "Sources:" in result
    assert "- app/main.py" in result


def test_finalize_answer_prefers_call_chain_for_flow_questions() -> None:
    result = qa._finalize_answer(
        answer_text="Answer: A generic summary that should be overridden.",
        source_paths=["app/main.py", "app/report.py", "outputs/weekly_digest.md"],
        evidence_blocks=[],
        question="How is weekly_digest.md built?",
        call_chain_summary="- `app/main.py` -> `write_weekly_digest_report()` -> `app/report.py`\n- `app/report.py` -> writes `outputs/weekly_digest.md`",
    )

    assert "multi-step repository flow" in result
    assert "`app/main.py` -> `write_weekly_digest_report()` -> `app/report.py`" in result
    assert "outputs/weekly_digest.md" in result
    assert "`app/report.py` -> writes `outputs/weekly_digest.md`" in result
    assert "`app/main.py` -> `write_weekly_digest_report()` -> `app/report.py` -> writes" not in result


def test_finalize_answer_formats_entity_location_question() -> None:
    result = qa._finalize_answer(
        answer_text="app/config.py contains the setting.",
        source_paths=["app/config.py", "app/main.py"],
        evidence_blocks=[
            {"file_path": "app/config.py", "reason": "Keyword-based code match", "snippet": "OLLAMA_BASE_URL"},
        ],
        question="Where is the Ollama base URL configured?",
        call_chain_summary="",
    )

    assert "Answer: The most relevant location for this question is `app/config.py`." in result
    assert "- `app/config.py`: Keyword-based code match" in result


def test_finalize_answer_formats_relationship_trace_question() -> None:
    result = qa._finalize_answer(
        answer_text="Answer: generic",
        source_paths=["app/service.py", "app/helpers.py"],
        evidence_blocks=[],
        question="What calls compute_digest across files?",
        call_chain_summary="- `app/service.py` -> `compute_digest()` -> `app/helpers.py`",
    )

    assert "cross-file relationship" in result
    assert "`app/service.py` -> `compute_digest()` -> `app/helpers.py`" in result


def test_finalize_answer_formats_open_analysis_question() -> None:
    result = qa._finalize_answer(
        answer_text="The design is simple but tightly coupled around the CLI and retrieval layers.",
        source_paths=["app/main.py", "app/qa.py"],
        evidence_blocks=[
            {"file_path": "app/main.py", "reason": "Vector retrieval result", "snippet": "def main() -> None"},
            {"file_path": "app/qa.py", "reason": "Vector retrieval result", "snippet": "def answer_question("},
        ],
        question="What design risks do you see in this project?",
        call_chain_summary="",
    )

    assert result.startswith("Answer:")
    assert "\nWhy:\n" in result
    assert "runtime proof" not in result
    assert "Based on the retrieved implementation" in result
    assert "defines `main()`" in result or "implementation detail" in result


def test_finalize_answer_rebuilds_open_analysis_why_even_if_model_supplies_one() -> None:
    result = qa._finalize_answer(
        answer_text=(
            "Answer: Based on the retrieved implementation, the project faces several design risks.\n"
            "Why:\n"
            "- Generic model summary.\n"
            "- Another generic summary."
        ),
        source_paths=["app/metrics.py", "app/ci_failure_analysis.py"],
        evidence_blocks=[
            {
                "file_path": "app/metrics.py",
                "reason": "Vector retrieval result",
                "snippet": "category_counts: dict[str, int] = {}\nworkflow_failures: dict[str, int] = {}",
            },
            {
                "file_path": "app/ci_failure_analysis.py",
                "reason": "Vector retrieval result",
                "snippet": "patterns: list[tuple[str, list[str]]] = [(\"permission_failure\", [\"permission denied\"])]",
            },
        ],
        question="What design risks do you see in this project?",
        call_chain_summary="",
    )

    assert "Generic model summary" not in result
    assert "aggregates workflow signals through in-memory counters" in result
    assert "hard-codes failure classification rules" in result


def test_compress_chain_lines_rewrites_writer_flow() -> None:
    lines = qa._compress_chain_lines(
        [
            "- `app/main.py` -> `build_weekly_ci_digest()` -> `app/metrics.py`",
            "- `app/main.py` -> `write_weekly_digest_report()` -> `app/report.py`",
            "- `app/main.py` -> `write_weekly_digest_report()` -> `app/report.py` -> writes `weekly_digest.md`",
        ]
    )

    assert lines == [
        "- `app/main.py` -> `build_weekly_ci_digest()` -> `app/metrics.py`",
        "- `app/main.py` -> `write_weekly_digest_report()` -> `app/report.py`",
        "- `app/report.py` -> writes `outputs/weekly_digest.md`",
    ]


def test_build_search_text_expands_semantic_aliases() -> None:
    search_text = qa._build_search_text(
        question="How is the weekly digest built?",
        search_question="How is the weekly digest built?",
    )

    assert "build_weekly_ci_digest" in search_text
    assert "write_weekly_digest_report" in search_text
    assert "weekly_digest.md" in search_text


def test_keyword_patterns_for_open_analysis_include_high_signal_terms() -> None:
    patterns = qa._keyword_patterns_for_query("What design risks do you see in this project?")

    assert "what" not in patterns
    assert "category_counts" in patterns
    assert "workflow_failures" in patterns
    assert "unknown_failure" in patterns


def test_collect_keyword_contexts_finds_call_chain_files(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "service.py").write_text(
        "from app.helpers import compute_digest\n\n"
        "def build_report() -> str:\n"
        "    return compute_digest()\n",
        encoding="utf-8",
    )
    (app_dir / "helpers.py").write_text(
        "def compute_digest() -> str:\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    matches = qa._collect_keyword_contexts(
        repo_path=tmp_path,
        search_text="Which file calls compute_digest and where is compute_digest defined?",
    )

    by_file = {file_path: reason for file_path, reason, _ in matches}
    matched_files = list(by_file)

    assert "app/service.py" in matched_files
    assert "app/helpers.py" in matched_files
    assert by_file["app/service.py"] == "Identifier call-site match"
    assert by_file["app/helpers.py"] == "Identifier definition match"


def test_build_call_chain_summary_describes_cross_file_relationships(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "main.py").write_text(
        "from app.service import build_report\n\n"
        "def main() -> None:\n"
        "    build_report()\n",
        encoding="utf-8",
    )
    (app_dir / "service.py").write_text(
        "from app.helpers import compute_digest\n\n"
        "def build_report() -> str:\n"
        "    return compute_digest()\n",
        encoding="utf-8",
    )
    (app_dir / "helpers.py").write_text(
        "def compute_digest() -> str:\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    summary = qa._build_call_chain_summary(
        repo_path=tmp_path,
        search_text="How does main call build_report across files?",
    )

    assert "`app/main.py` -> `build_report()` -> `app/service.py`" in summary


def test_build_call_chain_summary_includes_output_writer_path(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "main.py").write_text(
        "from app.metrics import build_weekly_ci_digest\n"
        "from app.report import write_weekly_digest_report\n\n"
        "def run() -> None:\n"
        "    digest = build_weekly_ci_digest([])\n"
        "    write_weekly_digest_report('outputs/weekly_digest.md', 'repo', 30, digest)\n",
        encoding="utf-8",
    )
    (app_dir / "metrics.py").write_text(
        "def build_weekly_ci_digest(records):\n"
        "    return {}\n",
        encoding="utf-8",
    )
    (app_dir / "report.py").write_text(
        "def write_weekly_digest_report(output_path, repo, days, digest):\n"
        "    output_path = 'outputs/weekly_digest.md'\n"
        "    return output_path\n",
        encoding="utf-8",
    )

    summary = qa._build_call_chain_summary(
        repo_path=tmp_path,
        search_text="How is weekly_digest.md built?",
    )

    assert "`app/main.py` -> `write_weekly_digest_report()` -> `app/report.py`" in summary
    assert "writes `outputs/weekly_digest.md`" in summary


def test_build_call_chain_summary_filters_tests_and_low_signal_nodes(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    tests_dir = tmp_path / "tests"
    app_dir.mkdir()
    tests_dir.mkdir()
    (app_dir / "main.py").write_text(
        "from app.config import from_env\n"
        "from app.report import write_weekly_digest_report\n\n"
        "def parse_args():\n"
        "    return {}\n\n"
        "def run() -> None:\n"
        "    from_env()\n"
        "    write_weekly_digest_report('outputs/weekly_digest.md', 'repo', 30, {})\n",
        encoding="utf-8",
    )
    (app_dir / "config.py").write_text(
        "def from_env():\n"
        "    return {}\n",
        encoding="utf-8",
    )
    (app_dir / "report.py").write_text(
        "def write_weekly_digest_report(output_path, repo, days, digest):\n"
        "    output_path = 'outputs/weekly_digest.md'\n"
        "    return output_path\n",
        encoding="utf-8",
    )
    (tests_dir / "test_metrics.py").write_text(
        "def _dt():\n"
        "    return '2026-01-01'\n",
        encoding="utf-8",
    )

    summary = qa._build_call_chain_summary(
        repo_path=tmp_path,
        search_text=qa._build_search_text(
            question="How is weekly_digest.md built?",
            search_question="How is weekly_digest.md built?",
        ),
    )

    assert "tests/test_metrics.py" not in summary
    assert "parse_args()" not in summary
    assert "from_env()" not in summary


def test_build_call_chain_summary_filters_fetch_functions(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "main.py").write_text(
        "from app.github_client import fetch_pull_requests\n"
        "from app.report import write_weekly_digest_report\n\n"
        "def run() -> None:\n"
        "    fetch_pull_requests('repo')\n"
        "    write_weekly_digest_report('outputs/weekly_digest.md', 'repo', 30, {})\n",
        encoding="utf-8",
    )
    (app_dir / "github_client.py").write_text(
        "def fetch_pull_requests(repo):\n"
        "    return []\n",
        encoding="utf-8",
    )
    (app_dir / "report.py").write_text(
        "def write_weekly_digest_report(output_path, repo, days, digest):\n"
        "    output_path = 'outputs/weekly_digest.md'\n"
        "    return output_path\n",
        encoding="utf-8",
    )

    summary = qa._build_call_chain_summary(
        repo_path=tmp_path,
        search_text=qa._build_search_text(
            question="How is the weekly digest built?",
            search_question="How is the weekly digest built?",
        ),
    )

    assert "fetch_pull_requests()" not in summary
    assert "github_client.py" not in summary


def test_build_call_chain_summary_filters_row_builder_functions(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "main.py").write_text(
        "from app.metrics import build_pr_rows, build_weekly_ci_digest\n"
        "from app.report import write_weekly_digest_report\n\n"
        "def run() -> None:\n"
        "    build_pr_rows([])\n"
        "    digest = build_weekly_ci_digest([])\n"
        "    write_weekly_digest_report('outputs/weekly_digest.md', 'repo', 30, digest)\n",
        encoding="utf-8",
    )
    (app_dir / "metrics.py").write_text(
        "def build_pr_rows(records):\n"
        "    return []\n\n"
        "def build_weekly_ci_digest(records):\n"
        "    return {}\n",
        encoding="utf-8",
    )
    (app_dir / "report.py").write_text(
        "def write_weekly_digest_report(output_path, repo, days, digest):\n"
        "    output_path = 'outputs/weekly_digest.md'\n"
        "    return output_path\n",
        encoding="utf-8",
    )

    summary = qa._build_call_chain_summary(
        repo_path=tmp_path,
        search_text=qa._build_search_text(
            question="How is the weekly digest built?",
            search_question="How is the weekly digest built?",
        ),
    )

    assert "build_pr_rows()" not in summary


def test_filter_sources_for_flow_question_removes_readme_noise() -> None:
    filtered = qa._filter_sources_for_question(
        source_paths=[
            "app/main.py",
            "app/report.py",
            "app/metrics.py",
            "README.md",
            "outputs/weekly_digest.md",
        ],
        question="How is the weekly digest built?",
        call_chain_summary="- `app/main.py` -> `build_weekly_ci_digest()` -> `app/metrics.py`\n- `app/main.py` -> `write_weekly_digest_report()` -> `app/report.py`\n- `app/report.py` -> writes `outputs/weekly_digest.md`",
    )

    assert "README.md" not in filtered
    assert "app/main.py" in filtered
    assert "outputs/weekly_digest.md" in filtered


def test_filter_evidence_for_flow_question_removes_summarize_noise() -> None:
    filtered = qa._filter_evidence_for_question(
        evidence_blocks=[
            {"file_path": "app/main.py", "reason": "Vector retrieval result", "snippet": "summarize_pull_requests(pr_records)"},
            {"file_path": "app/metrics.py", "reason": "Vector retrieval result", "snippet": "build_weekly_ci_digest(workflow_records)"},
        ],
        question="How is the weekly digest built?",
    )

    assert len(filtered) == 1
    assert filtered[0]["file_path"] == "app/metrics.py"


def test_filter_sources_for_open_analysis_removes_readme_tests_and_outputs() -> None:
    filtered = qa._filter_sources_for_question(
        source_paths=[
            "README.md",
            "app/metrics.py",
            "tests/test_metrics.py",
            "app/ci_failure_analysis.py",
            "outputs/weekly_digest.md",
        ],
        question="What design risks do you see in this project?",
        call_chain_summary="",
    )

    assert filtered == ["app/ci_failure_analysis.py", "app/metrics.py"]


def test_filter_sources_for_open_analysis_prioritizes_analysis_files_over_client_and_report() -> None:
    filtered = qa._filter_sources_for_question(
        source_paths=[
            "app/ci_failure_analysis.py",
            "app/metrics.py",
            "app/github_client.py",
            "app/report.py",
        ],
        question="What design risks do you see in this project?",
        call_chain_summary="",
    )

    assert filtered == ["app/ci_failure_analysis.py", "app/metrics.py"]


def test_filter_evidence_for_open_analysis_removes_readme_tests_and_outputs() -> None:
    filtered = qa._filter_evidence_for_question(
        evidence_blocks=[
            {"file_path": "README.md", "reason": "Vector retrieval result", "snippet": "overview"},
            {"file_path": "app/metrics.py", "reason": "Vector retrieval result", "snippet": "def build_weekly_ci_digest("},
            {"file_path": "tests/test_metrics.py", "reason": "Vector retrieval result", "snippet": "assert digest"},
            {"file_path": "outputs/weekly_digest.md", "reason": "Vector retrieval result", "snippet": "# Weekly CI Digest"},
        ],
        question="What design risks do you see in this project?",
    )

    assert len(filtered) == 1
    assert filtered[0]["file_path"] == "app/metrics.py"


def test_filter_evidence_for_open_analysis_prioritizes_analysis_files_over_client_and_report() -> None:
    filtered = qa._filter_evidence_for_question(
        evidence_blocks=[
            {"file_path": "app/ci_failure_analysis.py", "reason": "Vector retrieval result", "snippet": "patterns: list[tuple[str, list[str]]] = []"},
            {"file_path": "app/metrics.py", "reason": "Vector retrieval result", "snippet": "category_counts: dict[str, int] = {}"},
            {"file_path": "app/github_client.py", "reason": "Vector retrieval result", "snippet": "def fetch_workflow_runs("},
            {"file_path": "app/report.py", "reason": "Vector retrieval result", "snippet": "def write_weekly_digest_report("},
        ],
        question="What design risks do you see in this project?",
    )

    assert [item["file_path"] for item in filtered] == [
        "app/ci_failure_analysis.py",
        "app/metrics.py",
    ]


def test_build_open_analysis_why_lines_prefers_implementation_signals() -> None:
    why_lines = qa._build_open_analysis_why_lines(
        [
            {
                "file_path": "app/ci_failure_analysis.py",
                "reason": "Vector retrieval result",
                "snippet": "patterns: list[tuple[str, list[str]]] = [(\"permission_failure\", [\"permission denied\"])]",
            },
            {
                "file_path": "app/metrics.py",
                "reason": "Vector retrieval result",
                "snippet": "category_counts: dict[str, int] = {}\nworkflow_failures: dict[str, int] = {}",
            },
        ]
    )

    assert "hard-codes failure classification rules" in why_lines[0]
    assert "aggregates workflow signals through in-memory counters" in why_lines[1]


def test_extract_best_snippet_prefers_open_analysis_aggregation_patterns() -> None:
    text = (
        "def _average_or_none(values):\n"
        "    return None\n\n"
        "def summarize_workflow_runs(records):\n"
        "    category_counts: dict[str, int] = {}\n"
        "    workflow_failures: dict[str, int] = {}\n"
        "    return category_counts, workflow_failures\n"
    )

    snippet = qa._extract_best_snippet(
        text=text,
        patterns=("risk", "failure"),
        preferred_patterns=("category_counts", "workflow_failures"),
    )

    assert "category_counts" in snippet
    assert "workflow_failures" in snippet


def test_confidence_score_boosts_focused_flow_answers() -> None:
    evidence_blocks = [
        {
            "file_path": "call-chain-summary",
            "reason": "Cross-file relationship summary",
            "snippet": "- `app/main.py` -> `build_weekly_ci_digest()` -> `app/metrics.py`",
        },
        {
            "file_path": "app/main.py",
            "reason": "Identifier call-site match",
            "snippet": "weekly_digest = build_weekly_ci_digest(workflow_records)",
        },
        {
            "file_path": "app/report.py",
            "reason": "Identifier definition match",
            "snippet": "def write_weekly_digest_report(output_path, repo, days, digest):",
        },
    ]
    source_paths = [
        "app/main.py",
        "app/report.py",
        "app/metrics.py",
        "outputs/weekly_digest.md",
    ]

    score = qa._confidence_score(
        source_paths=source_paths,
        evidence_blocks=evidence_blocks,
        question="How is the weekly digest built?",
        search_question="How is the weekly digest built?",
    )

    assert score >= 80
    assert qa._confidence_label(source_paths, evidence_blocks, "How is the weekly digest built?", "How is the weekly digest built?") == "High confidence"


def test_confidence_score_boosts_focused_entity_location_answers() -> None:
    score = qa._confidence_score(
        source_paths=["app/config.py", "app/main.py"],
        evidence_blocks=[
            {"file_path": "app/config.py", "reason": "Keyword-based code match", "snippet": "OLLAMA_BASE_URL"},
        ],
        question="Where is the Ollama base URL configured?",
        search_question="Where is the Ollama base URL configured?",
    )

    assert score >= 70


def test_confidence_score_keeps_open_analysis_more_conservative() -> None:
    score = qa._confidence_score(
        source_paths=["app/main.py", "app/qa.py", "README.md"],
        evidence_blocks=[
            {"file_path": "app/main.py", "reason": "Vector retrieval result", "snippet": "def main() -> None"},
            {"file_path": "app/qa.py", "reason": "Vector retrieval result", "snippet": "def answer_question("},
        ],
        question="What design risks do you see in this project?",
        search_question="What design risks do you see in this project?",
    )

    assert score < 80
    assert qa._confidence_label(
        ["app/main.py", "app/qa.py", "README.md"],
        [
            {"file_path": "app/main.py", "reason": "Vector retrieval result", "snippet": "def main() -> None"},
            {"file_path": "app/qa.py", "reason": "Vector retrieval result", "snippet": "def answer_question("},
        ],
        "What design risks do you see in this project?",
        "What design risks do you see in this project?",
    ) != "High confidence"


def test_risk_note_reflects_question_type() -> None:
    risk = qa._risk_note(
        source_paths=["app/main.py", "app/qa.py"],
        evidence_blocks=[
            {"file_path": "app/main.py", "reason": "Vector retrieval result", "snippet": "def main() -> None"},
        ],
        question="What design risks do you see in this project?",
        search_question="What design risks do you see in this project?",
    )

    assert "interpretation" in risk
