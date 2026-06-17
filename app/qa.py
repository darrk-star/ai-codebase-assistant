from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from llama_index.core import VectorStoreIndex

from app.config import AppConfig
from app.loaders import IGNORED_DIR_NAMES


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a codebase assistant. Answer only from the provided repository context. "
            "If the context is insufficient, say so clearly. Cite relevant file paths in your answer.",
        ),
        (
            "human",
            "Question:\n{question}\n\nRepository context:\n{context}",
        ),
    ]
)


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[str]


def answer_question(
    index: VectorStoreIndex,
    question: str,
    config: AppConfig,
    repo_path: Path,
) -> AnswerResult:
    retriever = index.as_retriever(similarity_top_k=config.top_k)
    nodes = retriever.retrieve(question)
    nodes = _rerank_nodes(question=question, nodes=nodes)

    source_paths: list[str] = []
    context_blocks: list[str] = []

    keyword_context_blocks = _collect_keyword_contexts(repo_path=repo_path, question=question)
    for file_path, snippet in keyword_context_blocks:
        if file_path not in source_paths:
            source_paths.append(file_path)
        context_blocks.append(f"File: {file_path}\n{snippet}")

    for node in nodes:
        metadata = node.metadata or {}
        file_path = metadata.get("file_path", "unknown")
        if file_path not in source_paths:
            source_paths.append(file_path)
        context_blocks.append(
            f"File: {file_path}\n{node.text.strip()}"
        )

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No context found."

    llm = ChatOllama(
        model=config.chat_model,
        base_url=config.ollama_base_url,
        temperature=0,
    )

    response = (ANSWER_PROMPT | llm).invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return AnswerResult(answer=response.content, sources=source_paths)


def _rerank_nodes(question: str, nodes: list[Any]) -> list[Any]:
    question_lower = question.lower()
    scored_nodes: list[tuple[float, Any]] = []

    for node in nodes:
        metadata = node.metadata or {}
        file_path = str(metadata.get("file_path", "unknown")).lower()
        extension = str(metadata.get("extension", "")).lower()
        text = node.text.lower()
        score = 0.0

        if extension == ".py":
            score += 2.0
        elif extension == ".md":
            score += 1.0
        elif extension == ".txt":
            score += 0.2
        else:
            score -= 0.5

        if "requirements.txt" in file_path:
            score -= 2.0
        if "__init__.py" in file_path:
            score -= 0.5

        if any(keyword in question_lower for keyword in ("entrypoint", "main", "cli", "command")):
            if "main.py" in file_path:
                score += 5.0
            if "argparse" in text:
                score += 4.0
            if "__name__" in text and "__main__" in text:
                score += 4.0
            if "def main" in text:
                score += 3.0

        if any(keyword in question_lower for keyword in ("chart", "plot", "graph")):
            if "chart" in file_path or "charts.py" in file_path:
                score += 5.0
            if "matplotlib" in text:
                score += 3.0

        if any(keyword in question_lower for keyword in ("workflow", "ci", "github actions")):
            if "github_client.py" in file_path:
                score += 5.0
            if "actions/runs" in text or "workflow" in text:
                score += 3.0

        if any(keyword in question_lower for keyword in ("digest", "summary", "report")):
            if "report.py" in file_path or "metrics.py" in file_path:
                score += 4.0

        scored_nodes.append((score, node))

    scored_nodes.sort(key=lambda item: item[0], reverse=True)
    return [node for _, node in scored_nodes]


def _collect_keyword_contexts(repo_path: Path, question: str) -> list[tuple[str, str]]:
    question_lower = question.lower()
    if not any(keyword in question_lower for keyword in ("entrypoint", "main", "cli", "command")):
        return []

    patterns = ("argparse", "argumentparser", "parse_args", "def main", "__main__")
    matches: list[tuple[int, str, str]] = []

    for path in repo_path.rglob("*.py"):
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        text_lower = text.lower()
        score = sum(1 for pattern in patterns if pattern in text_lower)
        if score == 0:
            continue

        snippet = _extract_best_snippet(text=text, patterns=patterns)
        relative_path = path.relative_to(repo_path).as_posix()
        matches.append((score, relative_path, snippet))

    matches.sort(key=lambda item: (-item[0], item[1]))
    return [(file_path, snippet) for _, file_path, snippet in matches[:3]]


def _extract_best_snippet(text: str, patterns: tuple[str, ...]) -> str:
    text_lower = text.lower()
    for pattern in patterns:
        index = text_lower.find(pattern)
        if index != -1:
            start = max(0, index - 300)
            end = min(len(text), index + 900)
            return text[start:end].strip()
    return text[:1200].strip()
