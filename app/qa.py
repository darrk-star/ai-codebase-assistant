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
            "If the context is insufficient, say so clearly. Use the recent conversation only to resolve "
            "references such as 'it', 'that file', or follow-up questions. Cite relevant file paths in your answer.",
        ),
        (
            "human",
            "Recent conversation:\n{history}\n\nQuestion:\n{question}\n\nRepository context:\n{context}",
        ),
    ]
)

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the latest user question into a standalone repository-search question. "
            "Use the recent conversation only to resolve missing references. "
            "Return only the rewritten question.",
        ),
        (
            "human",
            "Recent conversation:\n{history}\n\nLatest question:\n{question}",
        ),
    ]
)


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[str]
    search_question: str
    evidence: list[dict[str, str]]


def answer_question(
    index: VectorStoreIndex,
    question: str,
    config: AppConfig,
    repo_path: Path,
    history: list[dict[str, str]] | None = None,
) -> AnswerResult:
    history = history or []
    llm = ChatOllama(
        model=config.chat_model,
        base_url=config.ollama_base_url,
        temperature=0,
    )

    search_question = _rewrite_question(
        question=question,
        history=history,
        llm=llm,
    )

    retriever = index.as_retriever(similarity_top_k=config.top_k)
    nodes = retriever.retrieve(search_question)
    nodes = _rerank_nodes(question=search_question, nodes=nodes)

    source_paths: list[str] = []
    context_blocks: list[str] = []
    evidence_blocks: list[dict[str, str]] = []
    history_text = _format_history(history)

    keyword_context_blocks = _collect_keyword_contexts(repo_path=repo_path, question=search_question)
    for file_path, snippet in keyword_context_blocks:
        if file_path not in source_paths:
            source_paths.append(file_path)
        evidence_blocks.append(
            {
                "file_path": file_path,
                "reason": "Keyword-based code match",
                "snippet": snippet,
            }
        )
        context_blocks.append(f"File: {file_path}\n{snippet}")

    for node in nodes:
        metadata = node.metadata or {}
        file_path = metadata.get("file_path", "unknown")
        if file_path not in source_paths:
            source_paths.append(file_path)
        if not any(item["file_path"] == file_path for item in evidence_blocks):
            evidence_blocks.append(
                {
                    "file_path": str(file_path),
                    "reason": "Vector retrieval result",
                    "snippet": _trim_snippet(node.text.strip()),
                }
            )
        context_blocks.append(
            f"File: {file_path}\n{node.text.strip()}"
        )

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No context found."

    response = (ANSWER_PROMPT | llm).invoke(
        {
            "history": history_text,
            "question": question,
            "context": context,
        }
    )

    return AnswerResult(
        answer=response.content,
        sources=source_paths,
        search_question=search_question,
        evidence=evidence_blocks[:5],
    )


def _rewrite_question(question: str, history: list[dict[str, str]], llm: ChatOllama) -> str:
    history_text = _format_history(history)
    if history_text == "No prior conversation.":
        return question

    response = (QUERY_REWRITE_PROMPT | llm).invoke(
        {
            "history": history_text,
            "question": question,
        }
    )
    rewritten = str(response.content).strip()
    return rewritten or question


def _format_history(history: list[dict[str, str]], max_turns: int = 6) -> str:
    if not history:
        return "No prior conversation."

    recent_messages = history[-max_turns:]
    lines: list[str] = []
    for message in recent_messages:
        role = message.get("role", "user").capitalize()
        content = message.get("content", "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "No prior conversation."


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


def _trim_snippet(text: str, limit: int = 500) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."
