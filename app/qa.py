from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from llama_index.core import VectorStoreIndex

from app.config import AppConfig
from app.config import DEFAULT_EXTENSIONS
from app.loaders import IGNORED_DIR_NAMES


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a codebase assistant. Answer only from the provided repository context. "
            "If the context is insufficient, say so clearly. Use the recent conversation only to resolve "
            "references such as 'it', 'that file', or follow-up questions. Do not mention files that are not "
            "present in the repository context. Keep the answer concise and use this exact structure:\n"
            "Answer: <short answer>\n"
            "Why:\n"
            "- <fact grounded in the retrieved context>\n"
            "- <fact grounded in the retrieved context>\n"
            "If the context is insufficient, say so in the Answer line and keep the Why section brief.",
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
    call_chain_summary: str
    confidence_label: str
    confidence_score: int
    risk_note: str


INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "entrypoint": ("entrypoint", "main", "cli", "command", "startup", "run"),
    "workflow": ("workflow", "github actions", "ci", "pipeline", "job", "run"),
    "reporting": ("digest", "summary", "report", "chart", "plot", "graph", "metric"),
    "configuration": ("config", "setting", "env", "environment", "variable"),
    "indexing": ("index", "indexed", "indexing", "persist", "persisted", "storage", "retriever", "retrieve"),
    "callchain": ("call", "called", "uses", "used", "flow", "interaction", "chain", "invokes"),
    "tests": ("test", "pytest", "coverage", "fixture"),
    "structure": ("how", "where", "which", "call", "flow", "built", "generated", "implemented"),
}


SEMANTIC_ALIASES: dict[str, tuple[str, ...]] = {
    "weekly digest": ("build_weekly_ci_digest", "write_weekly_digest_report", "weekly_digest.md"),
    "weekly_digest": ("build_weekly_ci_digest", "write_weekly_digest_report", "weekly_digest.md"),
    "summary report": ("write_markdown_report", "summary.md"),
    "markdown report": ("write_markdown_report", "summary.md"),
    "ci charts": ("write_failure_trend_chart", "write_failed_workflow_chart", "ci_failure_trend.png", "unstable_workflows.png"),
}

LOW_SIGNAL_FUNCTIONS = {
    "parse_args",
    "_dt",
    "_fmt",
    "_format_pair",
    "__init__",
    "main",
    "from_env",
    "load_dotenv",
}

LOW_SIGNAL_PREFIXES = (
    "fetch_",
    "summarize_",
)

LOW_SIGNAL_PATTERNS = (
    "build_pr_rows",
    "build_workflow_rows",
)


def answer_question(
    index: VectorStoreIndex | None,
    question: str,
    config: AppConfig,
    repo_path: Path,
    history: list[dict[str, str]] | None = None,
) -> AnswerResult:
    if index is None:
        raise ValueError("Index is not ready. Build or load the repository index first.")

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
    search_text = _build_search_text(question=question, search_question=search_question)

    retriever = index.as_retriever(similarity_top_k=config.top_k)
    nodes = retriever.retrieve(search_question)
    nodes = _rerank_nodes(
        question=question,
        search_question=search_question,
        nodes=nodes,
    )
    nodes = _select_best_nodes(nodes)

    source_paths: list[str] = []
    context_blocks: list[str] = []
    evidence_blocks: list[dict[str, str]] = []
    history_text = _format_history(history)
    call_chain_summary = _build_call_chain_summary(repo_path=repo_path, search_text=search_text)

    keyword_context_blocks = _collect_keyword_contexts(
        repo_path=repo_path,
        search_text=search_text,
    )
    for file_path, reason, snippet in keyword_context_blocks:
        if file_path not in source_paths:
            source_paths.append(file_path)
        evidence_blocks.append(
            {
                "file_path": file_path,
                "reason": reason,
                "snippet": snippet,
            }
        )
        context_blocks.append(_format_context_block(file_path=file_path, snippet=snippet))

    if call_chain_summary:
        evidence_blocks.insert(
            0,
            {
                "file_path": "call-chain-summary",
                "reason": "Cross-file relationship summary",
                "snippet": call_chain_summary,
            },
        )
        context_blocks.insert(0, f"Cross-file relationship summary:\n{call_chain_summary}")

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
        context_blocks.append(_format_context_block(file_path=str(file_path), snippet=node.text.strip()))

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No context found."

    response = (ANSWER_PROMPT | llm).invoke(
        {
            "history": history_text,
            "question": question,
            "context": context,
        }
    )

    source_paths = _filter_sources_for_question(
        source_paths=source_paths,
        question=question,
        call_chain_summary=call_chain_summary,
    )
    evidence_blocks = _filter_evidence_for_question(
        evidence_blocks=evidence_blocks,
        question=question,
    )

    return AnswerResult(
        answer=_finalize_answer(
            answer_text=str(response.content),
            source_paths=source_paths,
            evidence_blocks=evidence_blocks,
            question=question,
            call_chain_summary=call_chain_summary,
        ),
        sources=source_paths,
        search_question=search_question,
        evidence=evidence_blocks[:5],
        call_chain_summary=call_chain_summary,
        confidence_label=_confidence_label(source_paths, evidence_blocks, question, search_question),
        confidence_score=_confidence_score(source_paths, evidence_blocks, question, search_question),
        risk_note=_risk_note(source_paths, evidence_blocks, question, search_question),
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


def _build_search_text(question: str, search_question: str) -> str:
    if search_question.strip().lower() == question.strip().lower():
        return _expand_semantic_aliases(question)
    return _expand_semantic_aliases(f"{question}\n{search_question}")


def _expand_semantic_aliases(text: str) -> str:
    expanded_parts = [text]
    text_lower = text.lower()
    for phrase, expansions in SEMANTIC_ALIASES.items():
        if phrase in text_lower:
            expanded_parts.extend(expansions)
    return "\n".join(dict.fromkeys(expanded_parts))


def _finalize_answer(
    answer_text: str,
    source_paths: list[str],
    evidence_blocks: list[dict[str, str]],
    question: str,
    call_chain_summary: str,
) -> str:
    chain_first_answer = _build_chain_first_answer(
        question=question,
        call_chain_summary=call_chain_summary,
        source_paths=source_paths,
    )
    if chain_first_answer:
        return chain_first_answer

    typed_answer = _build_typed_answer(
        answer_text=answer_text,
        source_paths=source_paths,
        evidence_blocks=evidence_blocks,
        question=question,
        call_chain_summary=call_chain_summary,
    )
    if typed_answer:
        return typed_answer

    cleaned = answer_text.strip()
    if not cleaned:
        cleaned = "Answer: I do not have enough repository context to answer reliably.\nWhy:\n- No answer text was generated."
    if not cleaned.lower().startswith("answer:"):
        cleaned = f"Answer: {cleaned}"
    if "\nWhy:" not in cleaned:
        why_lines = []
        for item in evidence_blocks[:2]:
            if item["file_path"] == "call-chain-summary":
                why_lines.append(f"- {item['snippet']}")
                continue
            why_lines.append(f"- {item['file_path']}: {item['reason']}")
        if not why_lines:
            why_lines.append("- The retrieved context was too weak to support a stronger answer.")
        cleaned = f"{cleaned}\nWhy:\n" + "\n".join(why_lines)

    if source_paths:
        source_section = "Sources:\n" + "\n".join(f"- {path}" for path in source_paths[:5])
        if "Sources:" not in cleaned:
            cleaned = f"{cleaned}\n\n{source_section}"

    return cleaned


def _build_typed_answer(
    answer_text: str,
    source_paths: list[str],
    evidence_blocks: list[dict[str, str]],
    question: str,
    call_chain_summary: str,
) -> str:
    question_type = _classify_question_type(question, call_chain_summary)
    if question_type == "artifact_flow":
        if call_chain_summary:
            return ""
        answer_line = "Answer: The retrieved implementation points to a repository flow for this artifact, but the full cross-file chain was not recovered exactly."
        why_lines = _build_flow_evidence_why_lines(evidence_blocks)
        if not why_lines:
            why_lines = [
                "- The retrieved files are implementation files, but they do not expose a complete writer chain for this artifact.",
            ]
        return _format_typed_answer(
            answer_line=answer_line,
            why_lines=why_lines,
            source_paths=source_paths,
        )

    if question_type == "relationship_trace":
        if call_chain_summary:
            chain_lines = [line.strip() for line in call_chain_summary.splitlines() if line.strip()]
            if not chain_lines:
                return ""
            why_lines = _compress_chain_lines(chain_lines[:3])
            relationship_sources = source_paths
        else:
            relationship_evidence = _prioritize_relationship_evidence(evidence_blocks, question)
            why_lines = _build_relationship_evidence_why_lines(relationship_evidence)
            relationship_sources = [
                str(item.get("file_path", ""))
                for item in relationship_evidence
                if str(item.get("file_path", "")).strip() and str(item.get("file_path", "")) != "call-chain-summary"
            ]
            if not why_lines:
                identifiers = ", ".join(f"`{term}`" for term in _extract_identifier_terms(question))
                target = identifiers or "the requested identifier"
                why_lines = [
                    f"- I could not find precise implementation evidence for {target} in the retrieved repository context.",
                ]
        return _format_typed_answer(
            answer_line="Answer: This behavior is implemented through a cross-file relationship rather than a single isolated file.",
            why_lines=why_lines,
            source_paths=list(dict.fromkeys(relationship_sources)),
        )

    if question_type == "entity_location":
        composite_answer = _build_composite_entity_location_answer(source_paths, evidence_blocks, question)
        if composite_answer:
            return composite_answer
        location = source_paths[0] if source_paths else "the retrieved repository context"
        why_lines = _build_evidence_why_lines(evidence_blocks)
        if not why_lines:
            why_lines = [f"- The most relevant match points to `{location}`."]
        return _format_typed_answer(
            answer_line=f"Answer: The most relevant location for this question is `{location}`.",
            why_lines=why_lines,
            source_paths=source_paths,
        )

    if question_type == "open_analysis":
        cleaned = answer_text.strip()
        if not cleaned:
            return ""
        why_lines = _build_open_analysis_why_lines(evidence_blocks)
        answer_line = _build_open_analysis_answer_line(why_lines)
        if not why_lines:
            why_lines = [
                "- This is a repository-grounded summary based on the most relevant retrieved files, not a runtime proof."
            ]
        return _format_typed_answer(
            answer_line=answer_line,
            why_lines=why_lines,
            source_paths=source_paths,
        )

    return ""


def _classify_question_type(question: str, call_chain_summary: str) -> str:
    question_lower = question.lower()
    if _is_flow_question(question_lower) and call_chain_summary:
        return "artifact_flow"
    if _is_flow_question(question_lower):
        return "artifact_flow"
    if any(
        token in question_lower
        for token in ("what calls", "called by", "across files", "interact", "interaction", "relationship")
    ):
        return "relationship_trace"
    if any(
        token in question_lower
        for token in ("which file", "where is", "where are", "contains", "defined", "definition")
    ):
        return "entity_location"
    if any(
        token in question_lower
        for token in ("why", "should", "risk", "optimize", "improve", "design", "architecture", "good", "bad")
    ):
        return "open_analysis"
    if call_chain_summary:
        return "relationship_trace"
    return "open_analysis"


def _build_evidence_why_lines(evidence_blocks: list[dict[str, str]], limit: int = 2) -> list[str]:
    why_lines: list[str] = []
    for item in evidence_blocks[:limit]:
        if item["file_path"] == "call-chain-summary":
            why_lines.append(f"- {item['snippet']}")
            continue
        why_lines.append(f"- `{item['file_path']}`: {item['reason']}")
    return why_lines


def _build_flow_evidence_why_lines(
    evidence_blocks: list[dict[str, str]],
    limit: int = 3,
) -> list[str]:
    why_lines: list[str] = []
    for item in evidence_blocks[:limit]:
        file_path = str(item.get("file_path", "")).strip()
        if not file_path or file_path == "call-chain-summary":
            continue
        snippet = str(item.get("snippet", "")).strip()
        why_lines.append(_describe_flow_evidence(file_path, snippet))
    return why_lines


def _build_relationship_evidence_why_lines(
    evidence_blocks: list[dict[str, str]],
    limit: int = 3,
) -> list[str]:
    why_lines: list[str] = []
    for item in evidence_blocks[:limit]:
        file_path = str(item.get("file_path", "")).strip()
        if not file_path or file_path == "call-chain-summary":
            continue
        snippet = str(item.get("snippet", "")).strip()
        why_lines.append(_describe_relationship_evidence(file_path, snippet))
    return why_lines


def _build_open_analysis_why_lines(
    evidence_blocks: list[dict[str, str]],
    limit: int = 3,
) -> list[str]:
    why_lines: list[str] = []
    for item in evidence_blocks[:limit]:
        file_path = str(item.get("file_path", "")).strip()
        if not file_path or file_path == "call-chain-summary":
            continue
        snippet = str(item.get("snippet", "")).strip()
        why_lines.append(_describe_open_analysis_evidence(file_path, snippet))
    return why_lines


def _describe_relationship_evidence(file_path: str, snippet: str) -> str:
    condensed = " ".join(snippet.split())

    function_match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", snippet)
    call_targets = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\(", snippet)
    defined_name = function_match.group(1) if function_match else ""
    meaningful_calls = [name for name in call_targets if name not in LOW_SIGNAL_FUNCTIONS and name != defined_name]

    if function_match and meaningful_calls:
        function_name = function_match.group(1)
        target_name = meaningful_calls[0]
        return f"- `{file_path}` defines `{function_name}()` and contains a call to `{target_name}()`, which is direct implementation evidence for the cross-file relationship."
    if function_match:
        function_name = function_match.group(1)
        return f"- `{file_path}` defines `{function_name}()`, which is one of the implementation points retrieved for this cross-file relationship."
    if meaningful_calls:
        target_name = meaningful_calls[0]
        return f"- `{file_path}` contains a call to `{target_name}()`, which suggests this file participates in the cross-file behavior."
    if condensed:
        preview = condensed[:120].rstrip()
        if len(condensed) > 120:
            preview += "..."
        return f"- `{file_path}` includes implementation detail such as `{preview}`, which is more concrete than a generic summary for this relationship."
    return f"- `{file_path}` is one of the implementation files retrieved for this cross-file relationship."


def _describe_flow_evidence(file_path: str, snippet: str) -> str:
    condensed = " ".join(snippet.split())

    writer_match = re.search(r"writes?\s+`?([A-Za-z0-9_./-]+)`?", snippet)
    output_path_match = re.search(r"output_path\s*=\s*['\"]([^'\"]+)['\"]", snippet)
    function_match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", snippet)
    call_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\(", snippet)

    if writer_match:
        output_path = writer_match.group(1)
        normalized_output = output_path if "/" in output_path else f"outputs/{output_path}"
        return f"- `{file_path}` contains the write step for `{normalized_output}`, so it is part of the artifact generation path."
    if output_path_match:
        output_path = output_path_match.group(1)
        normalized_output = output_path if "/" in output_path else f"outputs/{output_path}"
        return f"- `{file_path}` contains the write step for `{normalized_output}`, so it is part of the artifact generation path."
    if function_match:
        function_name = function_match.group(1)
        return f"- `{file_path}` defines `{function_name}()`, which is one of the implementation entry points retrieved for this artifact flow."
    if call_match and any(token in snippet for token in ("build_", "write_", "compute_", "generate_")):
        function_name = call_match.group(1)
        return f"- `{file_path}` contains implementation calls such as `{function_name}()`, which indicates this file participates in the build path."
    if condensed:
        preview = condensed[:120].rstrip()
        if len(condensed) > 120:
            preview += "..."
        return f"- `{file_path}` includes implementation detail such as `{preview}`, which is more concrete than a README-level summary for this flow."
    return f"- `{file_path}` is one of the implementation files retrieved for this artifact flow."


def _describe_open_analysis_evidence(file_path: str, snippet: str) -> str:
    condensed = " ".join(snippet.split())

    function_match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", snippet)
    if "patterns:" in snippet or "keywords in" in snippet or "permission_failure" in snippet:
        return f"- `{file_path}` hard-codes failure classification rules in one place, so new failure modes require code changes instead of configuration."
    if "unknown_failure" in snippet or "fallback_detail" in snippet:
        return f"- `{file_path}` falls back to a generic `unknown_failure` path when no rule matches, which suggests edge cases can collapse into a coarse bucket."
    if any(token in snippet for token in ("category_counts", "workflow_failures", "grouped:", "top_failed_workflows")):
        return f"- `{file_path}` aggregates workflow signals through in-memory counters and grouped summaries, concentrating the reporting logic in a small set of data structures."
    if function_match:
        function_name = function_match.group(1)
        return f"- `{file_path}` defines `{function_name}()`, which is one of the implementation points this answer is based on."
    if condensed:
        preview = condensed[:120].rstrip()
        if len(condensed) > 120:
            preview += "..."
        return f"- `{file_path}` includes implementation detail such as `{preview}`, which informs this interpretation."
    return f"- `{file_path}` is one of the implementation files used to support this interpretation."


def _soften_open_analysis_answer(answer_text: str) -> str:
    cleaned = answer_text.strip()
    if not cleaned:
        return cleaned

    if cleaned.lower().startswith("answer:"):
        prefix, _, remainder = cleaned.partition(":")
        softened_remainder = remainder.strip()
        if softened_remainder and not softened_remainder.lower().startswith(
            ("based on the retrieved implementation", "the current implementation suggests")
        ):
            softened_remainder = f"Based on the retrieved implementation, {softened_remainder[:1].lower()}{softened_remainder[1:]}"
        return f"{prefix}: {softened_remainder}".strip()

    if cleaned.lower().startswith(("based on the retrieved implementation", "the current implementation suggests")):
        return cleaned
    return f"Based on the retrieved implementation, {cleaned[:1].lower()}{cleaned[1:]}"


def _build_open_analysis_answer_line(why_lines: list[str]) -> str:
    if not why_lines:
        return "Answer: Based on the retrieved implementation, the current code suggests a few design risks worth reviewing."

    risk_phrases: list[str] = []
    for line in why_lines:
        text = line.lstrip("- ").strip().rstrip(".")
        if "hard-codes failure classification rules" in text:
            risk_phrases.append("hard-coded failure classification rules")
        elif "falls back to a generic `unknown_failure` path" in text:
            risk_phrases.append("coarse unknown-failure fallback behavior")
        elif "aggregates workflow signals through in-memory counters" in text:
            risk_phrases.append("concentrated in-memory aggregation logic")
        elif "defines `" in text and "implementation points" in text:
            risk_phrases.append("single-function coupling around core analysis paths")

    unique_phrases = list(dict.fromkeys(risk_phrases))
    if unique_phrases:
        top = ", ".join(unique_phrases[:2])
        return f"Answer: Based on the retrieved implementation, the main design risks are {top}."

    return "Answer: Based on the retrieved implementation, the current code suggests a few design risks worth reviewing."


def _extract_answer_line(answer_text: str) -> str:
    cleaned = answer_text.strip()
    if not cleaned:
        return "Answer: Based on the retrieved implementation, the current code suggests a few design risks worth reviewing."
    first_line = cleaned.splitlines()[0].strip()
    first_line = _strip_trailing_why_marker(first_line)
    if first_line.lower().startswith("answer:"):
        return first_line
    return f"Answer: {first_line}"


def _format_typed_answer(answer_line: str, why_lines: list[str], source_paths: list[str]) -> str:
    answer_line = _strip_trailing_why_marker(answer_line)
    source_section = ""
    if source_paths:
        source_section = "\n\nSources:\n" + "\n".join(f"- {path}" for path in source_paths[:5])
    return f"{answer_line}\nWhy:\n" + "\n".join(why_lines) + source_section


def _strip_trailing_why_marker(line: str) -> str:
    cleaned = line.strip()
    if not cleaned:
        return cleaned

    marker_pattern = r"(?:\s*[\*\_`#>\-]*)\bwhy\b(?:\s*[:：\.])\s*$"
    had_marker = bool(re.search(marker_pattern, cleaned, flags=re.IGNORECASE))
    if not had_marker:
        return cleaned

    cleaned = re.sub(
        marker_pattern,
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).rstrip()
    cleaned = cleaned.rstrip(":：.").rstrip()
    if cleaned and cleaned[-1].isalnum():
        cleaned = f"{cleaned}."
    return cleaned


def _build_composite_entity_location_answer(
    source_paths: list[str],
    evidence_blocks: list[dict[str, str]],
    question: str,
) -> str:
    if not _is_fetch_summary_location_query(question):
        return ""

    evidence_by_path = {
        str(item.get("file_path", "")).replace("\\", "/"): item
        for item in evidence_blocks
        if str(item.get("file_path", "")).strip()
    }
    normalized_sources = [path.replace("\\", "/") for path in source_paths]

    fetch_path = next((path for path in normalized_sources if path.endswith("github_client.py")), "")
    summary_metric_path = next((path for path in normalized_sources if path.endswith("metrics.py")), "")
    summary_report_path = next((path for path in normalized_sources if path.endswith("report.py")), "")
    main_path = next((path for path in normalized_sources if path.endswith("main.py")), "")

    if not fetch_path:
        fetch_path = next(
            (
                path
                for path, item in evidence_by_path.items()
                if "fetch_workflow_runs" in str(item.get("snippet", "")).lower()
                or path.endswith("github_client.py")
            ),
            "",
        )
    if not summary_metric_path:
        summary_metric_path = next(
            (
                path
                for path, item in evidence_by_path.items()
                if "summarize_workflow_runs" in str(item.get("snippet", "")).lower()
            ),
            "",
        )
    if not summary_report_path:
        summary_report_path = next(
            (
                path
                for path, item in evidence_by_path.items()
                if path.endswith("report.py")
                and "write_markdown_report" in str(item.get("snippet", "")).lower()
            ),
            "",
        )
    if not summary_report_path:
        main_snippet = str(evidence_by_path.get(main_path, {}).get("snippet", "")).lower() if main_path else ""
        if "write_markdown_report" in main_snippet and "workflow_summary" in main_snippet:
            summary_report_path = "app/report.py"

    if not fetch_path or not (summary_metric_path or summary_report_path or main_path):
        return ""

    if summary_metric_path and summary_report_path:
        answer_line = (
            f"Answer: GitHub workflow runs are fetched in `{fetch_path}`, then summarized in "
            f"`{summary_metric_path}` and written into the report flow in `{summary_report_path}`."
        )
    elif summary_metric_path:
        answer_line = (
            f"Answer: GitHub workflow runs are fetched in `{fetch_path}` and summarized in `{summary_metric_path}`."
        )
    elif summary_report_path:
        answer_line = (
            f"Answer: GitHub workflow runs are fetched in `{fetch_path}` and then written into the summary report flow in `{summary_report_path}`."
        )
    else:
        answer_line = (
            f"Answer: GitHub workflow runs are fetched in `{fetch_path}` and passed through the summary flow rooted in `{main_path}`."
        )

    why_lines: list[str] = [f"- `{fetch_path}` contains the workflow-run fetching implementation."]
    if main_path:
        why_lines.append(f"- `{main_path}` calls the workflow-fetching logic and wires the results into downstream summary generation.")
    if summary_metric_path:
        why_lines.append(f"- `{summary_metric_path}` contains `summarize_workflow_runs()`, which aggregates the fetched workflow-run data.")
    if summary_report_path:
        why_lines.append(f"- `{summary_report_path}` contains the report-writing step that consumes the summarized workflow data.")

    composite_sources = [path for path in [fetch_path, main_path, summary_metric_path, summary_report_path] if path]
    return _format_typed_answer(
        answer_line=answer_line,
        why_lines=why_lines[:4],
        source_paths=list(dict.fromkeys(composite_sources)),
    )


def _build_chain_first_answer(
    question: str,
    call_chain_summary: str,
    source_paths: list[str],
) -> str:
    question_lower = question.lower()
    if not call_chain_summary:
        return ""
    if not any(keyword in question_lower for keyword in ("how", "built", "generated", "written", "produced")):
        return ""

    chain_lines = [line.strip() for line in call_chain_summary.splitlines() if line.strip()]
    if not chain_lines:
        return ""

    answer_line = "Answer: This artifact is built through the following repository flow."
    if any(token in question_lower for token in ("weekly digest", "weekly_digest.md", "summary.md")):
        answer_line = "Answer: This artifact is built through a multi-step repository flow that starts in the CLI entrypoint, then calls builder functions, and finally writes the output file."

    why_lines = _describe_chain_lines(_compress_chain_lines(chain_lines[:3]))
    source_section = ""
    if source_paths:
        source_section = "\n\nSources:\n" + "\n".join(f"- {path}" for path in source_paths[:5])

    return f"{answer_line}\nWhy:\n" + "\n".join(why_lines) + source_section


def _compress_chain_lines(chain_lines: list[str]) -> list[str]:
    cleaned_lines = [line.lstrip("- ").strip() for line in chain_lines if line.strip()]
    compressed: list[str] = []
    writer_targets: set[str] = set()

    for line in cleaned_lines:
        writer_match = re.match(
            r"`([^`]+)` -> `([^`]+)\(\)` -> `([^`]+)` -> writes `([^`]+)`",
            line,
        )
        if writer_match:
            _, _, writer_path, output_path = writer_match.groups()
            normalized_output = output_path if "/" in output_path else f"outputs/{output_path}"
            writer_targets.add(writer_path)
            compressed.append(f"- `{writer_path}` -> writes `{normalized_output}`")
            continue

        simple_writer_match = re.match(r"`([^`]+)` -> writes `([^`]+)`", line)
        if simple_writer_match:
            writer_path, output_path = simple_writer_match.groups()
            normalized_output = output_path if "/" in output_path else f"outputs/{output_path}"
            writer_targets.add(writer_path)
            compressed.append(f"- `{writer_path}` -> writes `{normalized_output}`")
            continue

        compressed.append(f"- {line}")

    deduped: list[str] = []
    for line in compressed:
        if "-> `write_" in line:
            target_match = re.search(r"-> `([^`]+\.py)`$", line)
            if target_match and target_match.group(1) in writer_targets:
                if line not in deduped:
                    deduped.append(line)
                continue
        if line not in deduped:
            deduped.append(line)

    return deduped


def _describe_chain_lines(chain_lines: list[str]) -> list[str]:
    described: list[str] = []
    for line in chain_lines:
        clean = line.lstrip("- ").strip()
        call_match = re.match(r"`([^`]+)` -> `([^`]+)\(\)` -> `([^`]+)`", clean)
        if call_match:
            caller_path, function_name, callee_path = call_match.groups()
            described.append(
                f"- `{caller_path}` calls `{function_name}()`, whose implementation is in `{callee_path}`."
            )
            continue

        writer_match = re.match(r"`([^`]+)` -> writes `([^`]+)`", clean)
        if writer_match:
            writer_path, output_path = writer_match.groups()
            described.append(f"- `{writer_path}` writes the generated artifact to `{output_path}`.")
            continue

        described.append(f"- {clean}")
    return described


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


def _filter_sources_for_question(
    source_paths: list[str],
    question: str,
    call_chain_summary: str,
) -> list[str]:
    deduped = list(dict.fromkeys(source_paths))
    question_lower = question.lower()
    question_type = _classify_question_type(question, call_chain_summary)
    if question_type == "relationship_trace" and not call_chain_summary:
        identifiers = _extract_identifier_terms(question)
        if identifiers:
            exact_paths = [
                path
                for path in deduped
                if not path.replace("\\", "/").lower().endswith("readme.md")
                and any(identifier in path.replace("\\", "/").lower() for identifier in identifiers)
            ]
            if exact_paths:
                return exact_paths[:5]
    if question_type == "entity_location":
        return _prioritize_entity_location_paths(deduped, question)

    if not _is_flow_question(question) and not call_chain_summary:
        if question_type != "open_analysis":
            return deduped

    filtered: list[str] = []
    chain_lower = call_chain_summary.lower()
    for path in deduped:
        normalized = path.replace("\\", "/").lower()
        if normalized.endswith("readme.md"):
            continue
        if normalized.startswith("tests/") or "/tests/" in normalized:
            continue
        if question_type == "open_analysis" and not _is_flow_question(question_lower):
            if normalized.endswith("readme.md"):
                continue
            if normalized.startswith("outputs/") or "/outputs/" in normalized:
                continue
            if not normalized.endswith(".py"):
                continue
            filtered.append(path)
            continue
        if chain_lower and normalized not in chain_lower:
            continue
        filtered.append(path)

    if question_type == "open_analysis" and not _is_flow_question(question_lower):
        ranked = _prioritize_open_analysis_paths(filtered)
        return ranked or deduped[:5]

    return filtered or deduped[:5]


def _filter_evidence_for_question(
    evidence_blocks: list[dict[str, str]],
    question: str,
) -> list[dict[str, str]]:
    question_lower = question.lower()
    question_type = _classify_question_type(question, "")
    is_flow = _is_flow_question(question_lower)
    if question_type == "relationship_trace":
        return _prioritize_relationship_evidence(evidence_blocks, question)
    if question_type == "entity_location":
        return _prioritize_entity_location_evidence(evidence_blocks, question)

    if not is_flow and question_type != "open_analysis":
        return evidence_blocks

    filtered: list[dict[str, str]] = []
    for item in evidence_blocks:
        file_path = str(item.get("file_path", "")).replace("\\", "/").lower()
        snippet = str(item.get("snippet", "")).lower()
        if question_type == "open_analysis" and not is_flow:
            if file_path.endswith("readme.md"):
                continue
            if file_path.startswith("tests/") or "/tests/" in file_path:
                continue
            if file_path.startswith("outputs/") or "/outputs/" in file_path:
                continue
            if file_path == "call-chain-summary":
                continue
            filtered.append(item)
            continue
        if is_flow:
            if file_path.endswith("readme.md"):
                continue
            if file_path.startswith("tests/") or "/tests/" in file_path:
                continue
            if any(token in snippet for token in ("summarize_pull_requests", "summarize_workflow_runs")):
                continue
            filtered.append(item)

    if question_type == "open_analysis" and not is_flow:
        ranked = _prioritize_open_analysis_evidence(filtered)
        return ranked or evidence_blocks

    return filtered or evidence_blocks


def _prioritize_entity_location_paths(paths: list[str], question: str, limit: int = 2) -> list[str]:
    if _is_fetch_summary_location_query(question):
        return _prioritize_fetch_summary_paths(paths) or list(dict.fromkeys(paths))[:4]

    unique_paths = list(dict.fromkeys(paths))
    ranked = sorted(
        unique_paths,
        key=lambda path: (_entity_location_path_score(path, question), path.replace("\\", "/").lower()),
        reverse=True,
    )
    filtered = [
        path
        for path in ranked
        if _entity_location_path_score(path, question) > 0
        and not path.replace("\\", "/").lower().endswith("readme.md")
        and not path.replace("\\", "/").lower().startswith(("tests/", "outputs/"))
    ]
    question_lower = question.lower()
    entrypoint_query = any(token in question_lower for token in ("argparse", "main function", "entrypoint", "cli", "command"))
    if entrypoint_query:
        strong_main = [path for path in filtered if path.replace("\\", "/").lower().endswith("main.py")]
        if strong_main:
            return strong_main[:1]
    return filtered[:limit] or list(dict.fromkeys(paths))[:limit]


def _prioritize_entity_location_evidence(
    evidence_blocks: list[dict[str, str]],
    question: str,
    limit: int = 2,
) -> list[dict[str, str]]:
    if _is_fetch_summary_location_query(question):
        return _prioritize_fetch_summary_evidence(evidence_blocks) or evidence_blocks[:4]

    ranked = sorted(
        enumerate(evidence_blocks),
        key=lambda entry: (
            _entity_location_evidence_score(entry[1], question),
            -entry[0],
        ),
        reverse=True,
    )
    filtered = [
        item
        for _, item in ranked
        if _entity_location_evidence_score(item, question) > 0
        and not str(item.get("file_path", "")).replace("\\", "/").lower().endswith("readme.md")
        and not str(item.get("file_path", "")).replace("\\", "/").lower().startswith(("tests/", "outputs/"))
    ]
    question_lower = question.lower()
    entrypoint_query = any(token in question_lower for token in ("argparse", "main function", "entrypoint", "cli", "command"))
    if entrypoint_query:
        strong_main = [
            item
            for item in filtered
            if str(item.get("file_path", "")).replace("\\", "/").lower().endswith("main.py")
        ]
        if strong_main:
            return strong_main[:1]
    return filtered[:limit] or evidence_blocks[:limit]


def _is_fetch_summary_location_query(question: str) -> bool:
    question_lower = question.lower()
    return "fetch" in question_lower and "summarized" in question_lower


def _prioritize_fetch_summary_paths(paths: list[str]) -> list[str]:
    normalized_paths = list(dict.fromkeys(path.replace("\\", "/") for path in paths))
    preferred_suffixes = (
        "main.py",
        "report.py",
        "metrics.py",
        "github_client.py",
    )
    prioritized: list[str] = []
    for suffix in preferred_suffixes:
        for path in normalized_paths:
            if path.endswith(suffix) and path not in prioritized:
                prioritized.append(path)
    return prioritized


def _prioritize_fetch_summary_evidence(
    evidence_blocks: list[dict[str, str]],
) -> list[dict[str, str]]:
    preferred_suffixes = (
        "main.py",
        "report.py",
        "metrics.py",
        "github_client.py",
    )
    prioritized: list[dict[str, str]] = []
    for suffix in preferred_suffixes:
        for item in evidence_blocks:
            file_path = str(item.get("file_path", "")).replace("\\", "/")
            if file_path.endswith(suffix) and item not in prioritized:
                prioritized.append(item)
                break
    return prioritized


def _entity_location_path_score(path: str, question: str) -> int:
    normalized = path.replace("\\", "/").lower()
    question_lower = question.lower()
    config_query = any(token in question_lower for token in ("setting", "configured", "config", "environment", "base url"))
    entrypoint_query = any(token in question_lower for token in ("argparse", "main function", "entrypoint", "cli", "command"))
    score = 0
    if normalized.endswith(".py"):
        score += 1
    if config_query and "config" in normalized:
        score += 8
    if entrypoint_query and "main.py" in normalized:
        score += 10
    elif "main.py" in normalized:
        score += 4
    if config_query:
        if any(token in normalized for token in ("config", "settings", "env", "main.py")):
            score += 4
    for term in _extract_identifier_terms(question):
        if term in normalized:
            score += 4
    return score


def _entity_location_evidence_score(item: dict[str, str], question: str) -> int:
    file_path = str(item.get("file_path", "")).replace("\\", "/").lower()
    snippet = str(item.get("snippet", "")).lower()
    question_lower = question.lower()
    score = _entity_location_path_score(file_path, question)
    entrypoint_query = any(token in question_lower for token in ("argparse", "main function", "entrypoint", "cli", "command"))
    if "ollama" in question_lower:
        if "ollama" in snippet:
            score += 8
        if "base_url" in snippet or "base url" in snippet:
            score += 6
    if entrypoint_query:
        if "argparse" in snippet or "argumentparser" in snippet:
            score += 10
        if "def main" in snippet or "__main__" in snippet:
            score += 8
    if any(token in snippet for token in ("os.getenv", "from_env", "load_dotenv", "env")):
        score += 4
    for term in _extract_identifier_terms(question):
        if term in snippet:
            score += 5
    return score


def _prioritize_relationship_evidence(
    evidence_blocks: list[dict[str, str]],
    question: str,
) -> list[dict[str, str]]:
    identifiers = _extract_identifier_terms(question)
    if not identifiers:
        return evidence_blocks

    ranked: list[tuple[int, int, dict[str, str]]] = []
    for index, item in enumerate(evidence_blocks):
        file_path = str(item.get("file_path", "")).replace("\\", "/").lower()
        snippet = str(item.get("snippet", "")).lower()
        if file_path.endswith("readme.md"):
            continue
        if file_path.startswith("tests/") or "/tests/" in file_path:
            continue
        if file_path.startswith("outputs/") or "/outputs/" in file_path:
            continue

        score = 0
        for identifier in identifiers:
            if f"def {identifier}" in snippet or f"class {identifier}" in snippet:
                score += 8
            if f"{identifier}(" in snippet:
                score += 6
            if identifier in snippet:
                score += 3
            if identifier in file_path:
                score += 2
        if score > 0:
            ranked.append((-score, index, item))

    if not ranked:
        return []

    ranked.sort(key=lambda entry: (entry[0], entry[1]))
    return [item for _, _, item in ranked[:4]]


def _prioritize_open_analysis_paths(paths: list[str]) -> list[str]:
    ranked = sorted(
        dict.fromkeys(paths),
        key=lambda path: (_open_analysis_path_rank(path), path.replace("\\", "/").lower()),
    )
    high_priority = [path for path in ranked if _open_analysis_path_rank(path) == 0]
    if high_priority:
        return high_priority[:4]
    return ranked[:4]


def _prioritize_open_analysis_evidence(
    evidence_blocks: list[dict[str, str]],
) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for item in evidence_blocks:
        normalized = str(item.get("file_path", "")).replace("\\", "/").lower()
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        deduped.append(item)

    ranked = sorted(
        deduped,
        key=lambda item: (
            _open_analysis_path_rank(str(item.get("file_path", ""))),
            str(item.get("file_path", "")).replace("\\", "/").lower(),
        ),
    )
    high_priority = [item for item in ranked if _open_analysis_path_rank(str(item.get("file_path", ""))) == 0]
    if high_priority:
        return high_priority[:4]
    return ranked[:4]


def _open_analysis_path_rank(path: str) -> int:
    normalized = path.replace("\\", "/").lower()
    high_signal_tokens = ("metrics", "analysis", "rules", "scoring", "evaluate", "failure")
    low_signal_tokens = ("github_client", "report", "ui", "main")
    if any(token in normalized for token in high_signal_tokens):
        return 0
    if any(token in normalized for token in low_signal_tokens):
        return 2
    return 1


def _rerank_nodes(question: str, search_question: str, nodes: list[Any]) -> list[Any]:
    search_text = _build_search_text(question=question, search_question=search_question)
    question_lower = search_text.lower()
    question_terms = _extract_search_terms(question_lower)
    identifier_terms = _extract_identifier_terms(search_text)
    path_terms = _extract_path_terms(search_text)
    intents = _detect_intents(question_lower)
    scored_nodes: list[tuple[float, Any]] = []

    for node in nodes:
        metadata = node.metadata or {}
        file_path = str(metadata.get("file_path", "unknown")).lower()
        file_name = Path(file_path).name.lower()
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

        file_term_hits = sum(1 for term in question_terms if term in file_path)
        text_term_hits = sum(1 for term in question_terms if term in text)
        identifier_file_hits = sum(1 for term in identifier_terms if term in file_path or term in file_name)
        identifier_text_hits = sum(1 for term in identifier_terms if term in text)
        path_hits = sum(1 for term in path_terms if term in file_path or term in file_name)
        definition_hits = sum(1 for term in identifier_terms if f"def {term}" in text or f"class {term}" in text)
        invocation_hits = sum(1 for term in identifier_terms if f"{term}(" in text)
        import_hits = sum(1 for term in identifier_terms if f"import {term}" in text or f" import {term}" in text)

        score += min(file_term_hits, 4) * 1.5
        score += min(text_term_hits, 6) * 0.8
        score += min(identifier_file_hits, 3) * 4.0
        score += min(identifier_text_hits, 4) * 2.0
        score += min(path_hits, 3) * 5.0
        score += min(definition_hits, 3) * 4.0
        score += min(invocation_hits, 4) * 2.5
        score += min(import_hits, 3) * 1.5

        if any(keyword in question_lower for keyword in ("which file", "what file", "where is")):
            score += min(file_term_hits, 3) * 1.5
            if identifier_file_hits:
                score += 3.0

        if any(keyword in question_lower for keyword in ("defined", "definition", "implemented", "implementation")):
            if "def " in text or "class " in text:
                score += 1.5

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
            if "main.py" in file_path:
                score += 2.5
            if any(token in text for token in ("write_weekly_digest_report", "write_markdown_report", "weekly_digest.md", "summary.md")):
                score += 4.0

        if any(keyword in question_lower for keyword in ("how", "built", "generated", "produce")):
            if any(token in text for token in ("return", "build_", "generate_", "create_", "class ", "def ")):
                score += 1.5

        if "configuration" in intents:
            if any(token in file_name for token in ("config", ".env", "settings")):
                score += 4.0
            if any(token in text for token in ("os.getenv", "load_dotenv", "from_env", "environment")):
                score += 3.0

        if "indexing" in intents:
            if any(token in file_name for token in ("index", "indexing", "storage")):
                score += 5.0
            if any(token in text for token in ("vectorstoreindex", "storagecontext", "persist", "load_index_from_storage", "as_retriever")):
                score += 4.0

        if "callchain" in intents:
            if definition_hits:
                score += 2.0
            if invocation_hits:
                score += 4.0
            if import_hits:
                score += 2.0

        if "tests" in intents:
            if "test" in file_name or "/tests/" in file_path.replace("\\", "/"):
                score += 4.0
            if any(token in text for token in ("pytest", "fixture", "assert ")):
                score += 2.5

        if "structure" in intents:
            if any(token in text for token in ("def ", "class ", "return ", "import ")):
                score += 1.5

        scored_nodes.append((score, node))

    scored_nodes.sort(key=lambda item: item[0], reverse=True)
    return [node for _, node in scored_nodes]


def _select_best_nodes(nodes: list[Any], max_nodes: int = 6, max_per_file: int = 2) -> list[Any]:
    selected: list[Any] = []
    file_counts: dict[str, int] = {}

    for node in nodes:
        metadata = node.metadata or {}
        file_path = str(metadata.get("file_path", "unknown"))
        if file_counts.get(file_path, 0) >= max_per_file:
            continue
        selected.append(node)
        file_counts[file_path] = file_counts.get(file_path, 0) + 1
        if len(selected) >= max_nodes:
            break

    return selected


def _build_call_chain_summary(repo_path: Path, search_text: str) -> str:
    detected_intents = _detect_intents(search_text.lower())
    mentioned_outputs = _extract_output_targets(search_text)
    flow_question = _is_flow_question(search_text)
    reporting_flow = flow_question and "reporting" in detected_intents
    if "callchain" not in detected_intents and not mentioned_outputs and not reporting_flow:
        return ""

    identifier_terms = _extract_identifier_terms(search_text)
    if not identifier_terms and not mentioned_outputs:
        return ""

    graph = _build_static_relationship_graph(repo_path)

    multi_hop = _build_multi_hop_relationships(
        identifier_terms=identifier_terms,
        function_definitions=graph["function_definitions"],
        function_callers=graph["function_callers"],
        function_calls_by_file=graph["function_calls_by_file"],
        output_writers=graph["output_writers"],
        writer_functions=graph["writer_functions"],
        mentioned_outputs=mentioned_outputs,
    )
    if multi_hop:
        return "\n".join(f"- {item}" for item in multi_hop[:6])

    relationships = _build_single_hop_relationships(
        identifier_terms=identifier_terms,
        function_definitions=graph["function_definitions"],
        function_callers=graph["function_callers"],
        importer_map=graph["importer_map"],
    )
    return "\n".join(f"- {item}" for item in relationships[:6])


def _is_flow_question(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ("how", "built", "generated", "written", "produced"))


def _build_static_relationship_graph(repo_path: Path) -> dict[str, dict[str, list[str]]]:
    function_definitions: dict[str, list[str]] = {}
    function_callers: dict[str, list[str]] = {}
    importer_map: dict[str, list[str]] = {}
    function_calls_by_file: dict[str, list[str]] = {}
    output_writers: dict[str, list[str]] = {}
    writer_functions: dict[str, list[str]] = {}

    for path in repo_path.rglob("*.py"):
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        relative_path = path.relative_to(repo_path).as_posix()
        if _should_exclude_path_from_chain(relative_path):
            continue
        text_lower = text.lower()

        definitions = re.findall(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", text, flags=re.MULTILINE)
        imports = re.findall(r"^\s*from\s+[^\n]+\s+import\s+([a-zA-Z_][a-zA-Z0-9_]*)", text, flags=re.MULTILINE)
        calls = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\(", text)
        output_mentions = re.findall(r"['\"]([^'\"]+\.(?:md|csv|png|json|txt))['\"]", text, flags=re.IGNORECASE)

        for name in definitions:
            function_definitions.setdefault(name.lower(), []).append(relative_path)
            if name.lower().startswith("write_"):
                writer_functions.setdefault(relative_path, []).append(name.lower())

        local_calls = [
            call.lower()
            for call in calls
            if call not in {"def", "if", "for", "while", "return", "print"}
            and call.lower() not in LOW_SIGNAL_FUNCTIONS
            and not any(call.lower().startswith(prefix) for prefix in LOW_SIGNAL_PREFIXES)
            and call.lower() not in LOW_SIGNAL_PATTERNS
        ]
        function_calls_by_file[relative_path] = list(dict.fromkeys(local_calls))
        for call in local_calls:
            if f"def {call}" not in text_lower:
                function_callers.setdefault(call, []).append(relative_path)

        for imported in imports:
            importer_map.setdefault(imported.lower(), []).append(relative_path)

        if "write_" in text_lower or "output_path" in text_lower or "output_dir" in text_lower:
            normalized_outputs = [output.replace("\\", "/") for output in output_mentions]
            if normalized_outputs:
                output_writers[relative_path] = list(dict.fromkeys(normalized_outputs))

    return {
        "function_definitions": function_definitions,
        "function_callers": function_callers,
        "importer_map": importer_map,
        "function_calls_by_file": function_calls_by_file,
        "output_writers": output_writers,
        "writer_functions": writer_functions,
    }


def _build_multi_hop_relationships(
    identifier_terms: list[str],
    function_definitions: dict[str, list[str]],
    function_callers: dict[str, list[str]],
    function_calls_by_file: dict[str, list[str]],
    output_writers: dict[str, list[str]],
    writer_functions: dict[str, list[str]],
    mentioned_outputs: list[str],
) -> list[str]:
    relationships: list[str] = []

    for term in identifier_terms:
        normalized_term = term.lower()
        if normalized_term in LOW_SIGNAL_FUNCTIONS:
            continue
        if any(normalized_term.startswith(prefix) for prefix in LOW_SIGNAL_PREFIXES):
            continue
        if normalized_term in LOW_SIGNAL_PATTERNS:
            continue
        term_definitions = list(dict.fromkeys(function_definitions.get(normalized_term, [])))
        term_callers = list(dict.fromkeys(function_callers.get(normalized_term, [])))

        for caller_path in term_callers[:2]:
            for definition_path in term_definitions[:2]:
                relationships.append(
                    f"`{caller_path}` -> `{normalized_term}()` -> `{definition_path}`"
                )

                for writer_path, outputs in output_writers.items():
                    if writer_path != definition_path and writer_path != caller_path:
                        continue
                    writer_names = writer_functions.get(writer_path, [])
                    writer_name = next(
                        (
                            name for name in writer_names
                            if any(_writer_name_matches_target(name, output, identifier_terms) for output in outputs)
                        ),
                        writer_names[0] if writer_names else "",
                    )
                    if writer_path == caller_path and not writer_name:
                        continue
                    writer_callers = function_callers.get(writer_name, []) if writer_name else []
                    chain_prefix = (
                        f"`{writer_callers[0]}` -> `{writer_name}()` -> `{writer_path}`"
                        if writer_callers
                        else f"`{writer_path}`"
                    )
                    for output in outputs[:2]:
                        if mentioned_outputs and output not in mentioned_outputs:
                            continue
                        relationships.append(f"{chain_prefix} -> writes `{output}`")

        for definition_path in term_definitions[:2]:
            for writer_path, outputs in output_writers.items():
                if writer_path == definition_path:
                    for output in outputs[:2]:
                        if mentioned_outputs and output not in mentioned_outputs:
                            continue
                        relationships.append(
                            f"`{definition_path}` -> writes `{output}`"
                        )

    if mentioned_outputs:
        caller_output_mentions: dict[str, set[str]] = {}
        for writer_path, outputs in output_writers.items():
            caller_output_mentions[writer_path] = set(outputs)
            writer_names = writer_functions.get(writer_path, [])
            for output in outputs:
                writer_name = next(
                    (
                        name for name in writer_names
                        if _writer_name_matches_target(name, output, identifier_terms)
                    ),
                    "",
                )
                writer_callers = function_callers.get(writer_name, []) if writer_name else []
                output_name = Path(output).name.lower()
                if output not in mentioned_outputs and output_name not in mentioned_outputs:
                    continue
                if writer_callers and writer_name:
                    relationships.append(
                        f"`{writer_callers[0]}` -> `{writer_name}()` -> `{writer_path}` -> writes `{output}`"
                    )
                elif not _has_matching_writer_call(
                    caller_path=writer_path,
                    mentioned_output=output,
                    writer_functions=writer_functions,
                    function_callers=function_callers,
                    identifier_terms=identifier_terms,
                ):
                    relationships.append(f"`{writer_path}` -> writes `{output}`")

        for writer_path, writer_names in writer_functions.items():
            if not writer_names:
                continue
            for writer_name in writer_names:
                writer_callers = function_callers.get(writer_name, [])
                if not writer_callers:
                    continue
                caller_path = writer_callers[0]
                caller_outputs = caller_output_mentions.get(caller_path, set())
                for mentioned_output in mentioned_outputs:
                    if not _writer_name_matches_target(writer_name, mentioned_output, identifier_terms):
                        continue
                    matched_output = next(
                        (
                            output
                            for output in caller_outputs
                            if output == mentioned_output or Path(output).name.lower() == mentioned_output
                        ),
                        None,
                    )
                    if matched_output:
                        relationships.append(
                            f"`{caller_path}` -> `{writer_name}()` -> `{writer_path}` -> writes `{matched_output}`"
                        )

    return list(dict.fromkeys(relationships))


def _writer_name_matches_target(
    writer_name: str,
    mentioned_output: str,
    identifier_terms: list[str],
) -> bool:
    writer_tokens = set(re.findall(r"[a-z0-9]+", writer_name.lower()))
    output_tokens = set(re.findall(r"[a-z0-9]+", Path(mentioned_output).stem.lower()))
    identifier_tokens = {
        token
        for term in identifier_terms
        for token in re.findall(r"[a-z0-9]+", term.lower())
    }
    target_tokens = {token for token in output_tokens | identifier_tokens if len(token) >= 4}
    matched_tokens = writer_tokens & target_tokens
    specific_tokens = {
        token for token in output_tokens if len(token) >= 4 and token not in {"report", "write"}
    }
    if specific_tokens:
        return len(writer_tokens & specific_tokens) >= 2
    return len(matched_tokens) >= 2


def _has_matching_writer_call(
    caller_path: str,
    mentioned_output: str,
    writer_functions: dict[str, list[str]],
    function_callers: dict[str, list[str]],
    identifier_terms: list[str],
) -> bool:
    for writer_names in writer_functions.values():
        for writer_name in writer_names:
            if not _writer_name_matches_target(writer_name, mentioned_output, identifier_terms):
                continue
            if caller_path in function_callers.get(writer_name, []):
                return True
    return False


def _build_single_hop_relationships(
    identifier_terms: list[str],
    function_definitions: dict[str, list[str]],
    function_callers: dict[str, list[str]],
    importer_map: dict[str, list[str]],
) -> list[str]:
    relationships: list[str] = []

    for term in identifier_terms:
        normalized_term = term.lower()
        if normalized_term in LOW_SIGNAL_FUNCTIONS:
            continue
        if any(normalized_term.startswith(prefix) for prefix in LOW_SIGNAL_PREFIXES):
            continue
        if normalized_term in LOW_SIGNAL_PATTERNS:
            continue
        term_callers = list(dict.fromkeys(function_callers.get(normalized_term, [])))
        term_definitions = list(dict.fromkeys(function_definitions.get(normalized_term, [])))
        term_importers = list(dict.fromkeys(importer_map.get(normalized_term, [])))

        if term_callers and term_definitions:
            for caller_path in term_callers[:2]:
                for definition_path in term_definitions[:2]:
                    relationships.append(
                        f"`{caller_path}` -> `{normalized_term}()` -> `{definition_path}`"
                    )

        for caller_path in term_callers[:2]:
            if not term_definitions:
                relationships.append(f"`{caller_path}` calls `{normalized_term}()`.")

        for definition_path in term_definitions[:2]:
            if not term_callers:
                relationships.append(f"`{definition_path}` defines `{normalized_term}`.")

        for importer_path in term_importers[:2]:
            relationships.append(f"`{importer_path}` imports `{normalized_term}`.")

    return list(dict.fromkeys(relationships))


def _extract_output_targets(search_text: str) -> list[str]:
    outputs = re.findall(r"[A-Za-z0-9_.-]+\.(?:md|csv|png|json|txt)", search_text, flags=re.IGNORECASE)
    normalized: list[str] = []
    for item in outputs:
        clean = item.replace("\\", "/")
        normalized.append(clean)
        normalized.append(Path(clean).name.lower())
    return list(dict.fromkeys(normalized))


def _should_exclude_path_from_chain(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/").lower()
    return normalized.startswith("tests/") or "/tests/" in normalized


def _collect_keyword_contexts(repo_path: Path, search_text: str) -> list[tuple[str, str, str]]:
    patterns = _keyword_patterns_for_query(search_text)
    if not patterns:
        return []

    extensions = set(DEFAULT_EXTENSIONS)
    matches: list[tuple[int, str, str, str]] = []
    identifier_terms = _extract_identifier_terms(search_text)
    preferred_snippet_patterns = _preferred_snippet_patterns_for_query(search_text)

    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() not in extensions:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        text_lower = text.lower()
        score = sum(1 for pattern in patterns if pattern in text_lower)
        relative_path = path.relative_to(repo_path).as_posix()
        file_path_lower = relative_path.lower()
        score += sum(2 for pattern in patterns if pattern in file_path_lower)
        reason = "Keyword-based code match"

        definition_hits = [term for term in identifier_terms if f"def {term}" in text_lower or f"class {term}" in text_lower]
        invocation_hits = [term for term in identifier_terms if f"{term}(" in text_lower]
        import_hits = [term for term in identifier_terms if f"import {term}" in text_lower or f" import {term}" in text_lower]

        if definition_hits:
            score += len(definition_hits) * 3
            reason = "Identifier definition match"
        if invocation_hits:
            score += len(invocation_hits) * 4
            if not definition_hits:
                reason = "Identifier call-site match"
        if import_hits:
            score += len(import_hits) * 2
            if reason == "Keyword-based code match":
                reason = "Identifier import match"

        if score == 0 and not any(term in file_path_lower for term in _extract_path_terms(search_text)):
            continue

        snippet = _extract_best_snippet(
            text=text,
            patterns=patterns,
            preferred_terms=tuple(invocation_hits or definition_hits or import_hits),
            preferred_patterns=preferred_snippet_patterns,
        )
        matches.append((score, relative_path, reason, snippet))

    matches.sort(key=lambda item: (-item[0], item[1]))
    return [(file_path, reason, snippet) for _, file_path, reason, snippet in matches[:5]]


def _keyword_patterns_for_query(search_text: str) -> tuple[str, ...]:
    search_lower = search_text.lower()
    patterns: list[str] = []

    if "entrypoint" in _detect_intents(search_lower):
        patterns.extend(["argparse", "argumentparser", "parse_args", "def main", "__main__"])
    if "workflow" in _detect_intents(search_lower):
        patterns.extend(["workflow", "github actions", "actions/runs", "jobs:", "on:"])
    if "reporting" in _detect_intents(search_lower):
        patterns.extend(["chart", "plot", "graph", "digest", "summary", "report"])
    if "configuration" in _detect_intents(search_lower):
        patterns.extend(["os.getenv", "load_dotenv", "from_env", "config", ".env"])
    if "indexing" in _detect_intents(search_lower):
        patterns.extend(["vectorstoreindex", "storagecontext", "persist", "load_index_from_storage", "as_retriever"])
    if "tests" in _detect_intents(search_lower):
        patterns.extend(["pytest", "fixture", "assert ", "test_"])
    if _classify_question_type(search_text, "") == "open_analysis":
        patterns.extend(
            [
                "risk",
                "failure",
                "fallback",
                "unknown_failure",
                "category_counts",
                "workflow_failures",
                "top_failed_workflows",
                "patterns:",
                "grouped",
            ]
        )

    patterns.extend(_extract_identifier_terms(search_text))
    patterns.extend(_extract_path_terms(search_text))

    deduped_patterns = [pattern.lower() for pattern in patterns if len(pattern.strip()) >= 3]
    return tuple(dict.fromkeys(deduped_patterns))


def _preferred_snippet_patterns_for_query(search_text: str) -> tuple[str, ...]:
    if _classify_question_type(search_text, "") != "open_analysis":
        return ()
    return (
        "category_counts",
        "workflow_failures",
        "top_failed_workflows",
        "patterns:",
        "unknown_failure",
        "fallback_detail",
        "grouped:",
        "failure_categories",
    )


def _extract_best_snippet(
    text: str,
    patterns: tuple[str, ...],
    preferred_terms: tuple[str, ...] = (),
    preferred_patterns: tuple[str, ...] = (),
) -> str:
    text_lower = text.lower()
    for pattern in preferred_patterns:
        index = text_lower.find(pattern)
        if index != -1:
            start = max(0, index - 300)
            end = min(len(text), index + 900)
            return text[start:end].strip()
    for term in preferred_terms:
        for pattern in (f"def {term}", f"class {term}", f"{term}(", f"import {term}"):
            index = text_lower.find(pattern)
            if index != -1:
                start = max(0, index - 300)
                end = min(len(text), index + 900)
                return text[start:end].strip()
    for pattern in patterns:
        index = text_lower.find(pattern)
        if index != -1:
            start = max(0, index - 300)
            end = min(len(text), index + 900)
            return text[start:end].strip()
    return text[:1200].strip()


def _format_context_block(file_path: str, snippet: str, limit: int = 1800) -> str:
    trimmed = snippet.strip()
    if len(trimmed) > limit:
        trimmed = f"{trimmed[:limit].rstrip()}..."
    return f"File: {file_path}\n{trimmed}"


def _trim_snippet(text: str, limit: int = 500) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _extract_search_terms(question_lower: str) -> list[str]:
    stop_words = {
        "the", "a", "an", "is", "are", "to", "of", "in", "for", "and", "or", "how",
        "what", "which", "where", "when", "why", "does", "do", "file", "files",
        "function", "module", "code", "repository", "this", "that", "built",
    }
    terms = re.findall(r"[a-zA-Z_][a-zA-Z0-9_./-]*", question_lower)
    cleaned_terms = [term for term in terms if len(term) >= 3 and term not in stop_words]
    return list(dict.fromkeys(cleaned_terms))


def _extract_identifier_terms(question: str) -> list[str]:
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question)
    ignored_terms = {"what", "which", "where", "when", "why", "does", "should"}
    strong_terms = [
        term.lower()
        for term in identifiers
        if ("_" in term or term[:1].isupper() or len(term) >= 8) and term.lower() not in ignored_terms
    ]
    return list(dict.fromkeys(strong_terms))


def _extract_path_terms(question: str) -> list[str]:
    raw_terms = re.findall(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+|[A-Za-z0-9_.-]+\\[A-Za-z0-9_.\\-]+|[A-Za-z0-9_.-]+\.[A-Za-z0-9]+", question)
    normalized: list[str] = []
    for term in raw_terms:
        lowered = term.replace("\\", "/").lower().strip("./")
        if len(lowered) >= 3:
            normalized.append(lowered)
            if "/" in lowered:
                normalized.extend(part for part in lowered.split("/") if len(part) >= 3)
    return list(dict.fromkeys(normalized))


def _detect_intents(search_lower: str) -> set[str]:
    intents: set[str] = set()
    for intent, patterns in INTENT_PATTERNS.items():
        if any(pattern in search_lower for pattern in patterns):
            intents.add(intent)
    return intents


def _confidence_score(
    source_paths: list[str],
    evidence_blocks: list[dict[str, str]],
    question: str,
    search_question: str,
) -> int:
    score = 45
    question_lower = question.lower()
    search_lower = search_question.lower()
    flow_question = _is_flow_question(question_lower)
    question_type = _classify_question_type(question, "")
    source_lower = [path.replace("\\", "/").lower() for path in source_paths]
    has_call_chain_evidence = any(
        item.get("reason") == "Cross-file relationship summary" for item in evidence_blocks
    )
    has_output_source = any(path.startswith("outputs/") for path in source_lower)
    focused_python_sources = [path for path in source_lower if path.endswith(".py")]

    if evidence_blocks:
        score += min(len(evidence_blocks), 3) * 8
    if source_paths:
        score += 8
    if len(source_paths) == 1:
        score += 10
    if any(item.get("reason") == "Keyword-based code match" for item in evidence_blocks):
        score += 12

    if any(keyword in search_lower for keyword in ("entrypoint", "main", "cli", "command")):
        if any("main.py" in path.lower() for path in source_paths):
            score += 10

    if any(keyword in question_lower for keyword in ("which file", "where")):
        if source_paths:
            score += 5
    if flow_question and has_call_chain_evidence:
        score += 10
    if flow_question and has_output_source:
        score += 6
    if flow_question and 2 <= len(focused_python_sources) <= 3:
        score += 5
    if question_type == "entity_location" and source_paths:
        score += 8
    if question_type == "entity_location" and len(source_paths) <= 2:
        score += 4
    if question_type == "relationship_trace" and has_call_chain_evidence:
        score += 10
    if question_type == "relationship_trace" and 2 <= len(source_paths) <= 3:
        score += 4
    if question_type == "open_analysis":
        score -= 8

    if len(source_paths) >= 4:
        score -= 10
    if flow_question and len(source_paths) <= 4:
        score += 6
    if not evidence_blocks:
        score -= 15
    if question_type == "open_analysis" and len(source_paths) >= 3:
        score -= 4

    return max(0, min(100, score))


def _confidence_label(
    source_paths: list[str],
    evidence_blocks: list[dict[str, str]],
    question: str,
    search_question: str,
) -> str:
    score = _confidence_score(source_paths, evidence_blocks, question, search_question)
    if score >= 80:
        return "High confidence"
    if score >= 60:
        return "Medium confidence"
    return "Low confidence"


def _risk_note(
    source_paths: list[str],
    evidence_blocks: list[dict[str, str]],
    question: str,
    search_question: str,
) -> str:
    score = _confidence_score(source_paths, evidence_blocks, question, search_question)
    search_lower = search_question.lower()
    question_lower = question.lower()
    question_type = _classify_question_type(question, "")

    if score >= 80:
        return "The answer is backed by focused matches and is likely reliable for this repository question."
    if question_type == "open_analysis":
        return "This answer is an evidence-backed codebase interpretation, not a definitive runtime or architectural proof."
    if question_type == "entity_location":
        return "This answer points to the strongest matching file location, but you should still open the cited file to confirm surrounding context."
    if question_type == "relationship_trace":
        return "This answer traces likely cross-file relationships from retrieved code, but it may still miss indirect runtime behavior."
    if _is_flow_question(question_lower) and any(
        item.get("reason") == "Cross-file relationship summary" for item in evidence_blocks
    ):
        return "The answer is grounded in a focused cross-file path, but you should still verify the cited files if runtime behavior matters."
    if any(keyword in search_lower for keyword in ("flow", "interaction", "across files", "call chain", "called")):
        return "This answer may miss runtime behavior or cross-file interactions because the assistant relies on retrieved code snippets, not full program execution."
    if len(source_paths) >= 4:
        return "The answer pulled in several files, so treat it as a best-effort summary and verify the cited sources."
    if not evidence_blocks:
        return "The answer has weak retrieval support. Check the cited files before relying on it."
    return "The answer is plausible, but you should still verify the cited files for edge cases or missing context."
