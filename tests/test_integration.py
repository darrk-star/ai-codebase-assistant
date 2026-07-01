from __future__ import annotations

from pathlib import Path

from app import main
from app import qa
from app.config import AppConfig
from app.loaders import load_codebase_documents


class FakeNode:
    def __init__(self, file_path: str, text: str, extension: str = ".py") -> None:
        self.metadata = {
            "file_path": file_path,
            "extension": extension,
        }
        self.text = text


class FakeRetriever:
    def __init__(self, nodes: list[FakeNode]) -> None:
        self._nodes = nodes

    def retrieve(self, search_question: str) -> list[FakeNode]:
        return self._nodes


class FakeIndex:
    def __init__(self, nodes: list[FakeNode]) -> None:
        self._nodes = nodes

    def as_retriever(self, similarity_top_k: int):
        return FakeRetriever(self._nodes)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakePipeline:
    def invoke(self, payload: dict[str, str]) -> FakeResponse:
        return FakeResponse(
            "Answer: build_report in app/service.py calls compute_digest.\n"
            "Why:\n"
            "- app/service.py contains compute_digest().\n"
            "- app/helpers.py defines compute_digest.\n"
        )


class FakePrompt:
    def __or__(self, other) -> FakePipeline:
        return FakePipeline()


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


def _make_sample_repo(tmp_path: Path) -> Path:
    repo_path = tmp_path / "sample_repo"
    app_dir = repo_path / "app"
    app_dir.mkdir(parents=True)
    (app_dir / "main.py").write_text(
        "import argparse\n"
        "from app.service import build_report\n\n"
        "def main() -> None:\n"
        "    parser = argparse.ArgumentParser()\n"
        "    build_report()\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
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
        "    return 'digest'\n",
        encoding="utf-8",
    )
    (app_dir / "config.py").write_text(
        "import os\n\n"
        "def from_env():\n"
        "    return os.getenv('OLLAMA_BASE_URL')\n",
        encoding="utf-8",
    )
    (app_dir / "report.py").write_text(
        "def write_weekly_digest_report(output_path, repo, days, digest):\n"
        "    output_path = 'outputs/weekly_digest.md'\n"
        "    return output_path\n",
        encoding="utf-8",
    )
    (app_dir / "metrics.py").write_text(
        "def build_weekly_ci_digest(records):\n"
        "    return {}\n",
        encoding="utf-8",
    )
    return repo_path


def test_load_codebase_documents_reads_sample_repo(tmp_path: Path) -> None:
    repo_path = _make_sample_repo(tmp_path)

    documents = load_codebase_documents(repo_path)
    file_paths = sorted(doc.metadata["file_path"] for doc in documents)

    assert "app/main.py" in file_paths
    assert "app/service.py" in file_paths
    assert "app/helpers.py" in file_paths
    assert "app/config.py" in file_paths
    assert "app/report.py" in file_paths
    assert "app/metrics.py" in file_paths


def test_load_codebase_documents_includes_typescript_files(tmp_path: Path) -> None:
    repo_path = tmp_path / "ts_repo"
    src_dir = repo_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "agent.ts").write_text("export function runAgent() { return true }\n", encoding="utf-8")
    (src_dir / "types.ts").write_text("export interface AgentContext { workspace: string }\n", encoding="utf-8")

    documents = load_codebase_documents(repo_path)
    file_paths = sorted(doc.metadata["file_path"] for doc in documents)

    assert "src/agent.ts" in file_paths
    assert "src/types.ts" in file_paths


def test_answer_question_uses_sample_repo_context(monkeypatch, tmp_path: Path) -> None:
    repo_path = _make_sample_repo(tmp_path)
    config = _base_config()
    fake_index = FakeIndex(
        [
            FakeNode("app/service.py", "from app.helpers import compute_digest\n\ndef build_report() -> str:\n    return compute_digest()\n"),
            FakeNode("app/helpers.py", "def compute_digest() -> str:\n    return 'digest'\n"),
        ]
    )

    monkeypatch.setattr(qa, "_rewrite_question", lambda question, history, llm: question)
    monkeypatch.setattr(qa, "ANSWER_PROMPT", FakePrompt())
    monkeypatch.setattr(qa, "ChatOllama", lambda *args, **kwargs: object())

    result = qa.answer_question(
        index=fake_index,
        question="Which file calls compute_digest and where is it defined?",
        config=config,
        repo_path=repo_path,
    )

    assert "Answer:" in result.answer
    assert "Sources:" in result.answer
    assert "app/service.py" in result.sources
    assert "app/helpers.py" in result.sources
    assert "README.md" not in result.sources
    assert result.call_chain_summary
    assert "`app/service.py` -> `compute_digest()` -> `app/helpers.py`" in result.call_chain_summary
    assert any(item["reason"] == "Cross-file relationship summary" for item in result.evidence)
    assert any("`app/service.py` -> `compute_digest()` -> `app/helpers.py`" in item["snippet"] for item in result.evidence if item["reason"] == "Cross-file relationship summary")


def test_build_call_chain_summary_prefers_writer_flow_for_output_question(tmp_path: Path) -> None:
    repo_path = _make_sample_repo(tmp_path)

    summary = qa._build_call_chain_summary(
        repo_path=repo_path,
        search_text="How is weekly_digest.md built?",
    )

    assert "`app/report.py` -> writes `outputs/weekly_digest.md`" in summary or "`app/main.py` -> `write_weekly_digest_report()` -> `app/report.py`" in summary


def test_build_call_chain_summary_uses_semantic_alias_for_weekly_digest(tmp_path: Path) -> None:
    repo_path = _make_sample_repo(tmp_path)

    summary = qa._build_call_chain_summary(
        repo_path=repo_path,
        search_text=qa._build_search_text(
            question="How is the weekly digest built?",
            search_question="How is the weekly digest built?",
        ),
    )

    assert "write_weekly_digest_report" in summary or "weekly_digest.md" in summary


def test_build_call_chain_summary_triggers_for_reporting_flow_question(tmp_path: Path) -> None:
    repo_path = _make_sample_repo(tmp_path)

    summary = qa._build_call_chain_summary(
        repo_path=repo_path,
        search_text=qa._build_search_text(
            question="How is the weekly digest built?",
            search_question="How is the weekly digest built?",
        ),
    )

    assert summary
    assert "weekly_digest.md" in summary or "write_weekly_digest_report" in summary
    assert "parse_args()" not in summary
    assert "tests/" not in summary
    assert "from_env()" not in summary
    assert "fetch_" not in summary
    assert "summarize_" not in summary


def test_handle_question_prints_realistic_result(monkeypatch, capsys, tmp_path: Path) -> None:
    repo_path = _make_sample_repo(tmp_path)
    config = _base_config()

    monkeypatch.setattr(
        main,
        "answer_question",
        lambda **kwargs: qa.AnswerResult(
            answer="Answer: build_report is in app/service.py.\nWhy:\n- app/service.py calls compute_digest.\n\nSources:\n- app/service.py",
            sources=["app/service.py"],
            search_question="Which file calls compute_digest and where is it defined?",
            evidence=[],
            call_chain_summary="- `app/service.py` -> `compute_digest()` -> `app/helpers.py`",
            confidence_label="High confidence",
            confidence_score=88,
            risk_note="Focused evidence.",
        ),
    )

    main._handle_question(index="fake-index", question="Which file calls compute_digest and where is it defined?", config=config, repo_path=repo_path)
    output = capsys.readouterr().out

    assert "Answer:" in output
    assert "- app/service.py" in output
