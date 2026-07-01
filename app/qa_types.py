from __future__ import annotations

from pathlib import Path
import re
from typing import Callable


def is_flow_question(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ("how", "built", "generated", "written", "produced"))


def classify_question_type(question: str, call_chain_summary: str) -> str:
    question_lower = question.lower()
    if is_flow_question(question_lower) and call_chain_summary:
        return "artifact_flow"
    if is_flow_question(question_lower):
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


def is_fetch_summary_location_query(question: str) -> bool:
    question_lower = question.lower()
    return "fetch" in question_lower and "summarized" in question_lower


def format_typed_answer(answer_line: str, why_lines: list[str], source_paths: list[str]) -> str:
    answer_line = strip_trailing_why_marker(answer_line)
    source_section = ""
    if source_paths:
        source_section = "\n\nSources:\n" + "\n".join(f"- {path}" for path in source_paths[:5])
    return f"{answer_line}\nWhy:\n" + "\n".join(why_lines) + source_section


def strip_trailing_why_marker(line: str) -> str:
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
    return cleaned


def guard_target_identifiers(question: str, extract_identifier_terms: Callable[[str], list[str]]) -> list[str]:
    identifiers = extract_identifier_terms(question)
    filtered: list[str] = []
    for identifier in identifiers:
        if identifier.upper() == identifier:
            continue
        if "_" not in identifier and len(identifier) < 12:
            continue
        filtered.append(identifier)
    return list(dict.fromkeys(filtered))


def evidence_mentions_identifier(evidence_blocks: list[dict[str, str]], identifier: str) -> bool:
    normalized = identifier.lower()
    for item in evidence_blocks:
        snippet = str(item.get("snippet", "")).lower()
        file_path = str(item.get("file_path", "")).replace("\\", "/").lower()
        if normalized in snippet or normalized in file_path:
            return True
    return False


def source_path_mentions_identifier(source_paths: list[str], identifier: str) -> bool:
    normalized = identifier.lower()
    return any(normalized in path.replace("\\", "/").lower() for path in source_paths)


def build_workspace_mismatch_guard_answer(
    question: str,
    repo_path: Path,
    source_paths: list[str],
    evidence_blocks: list[dict[str, str]],
    call_chain_summary: str,
    repo_symbols: set[str] | frozenset[str],
    extract_identifier_terms: Callable[[str], list[str]],
) -> str:
    question_type = classify_question_type(question, call_chain_summary)
    if question_type == "relationship_trace":
        target_identifiers = guard_target_identifiers(question, extract_identifier_terms)
    elif question_type == "entity_location" and is_fetch_summary_location_query(question):
        target_identifiers = list(
            dict.fromkeys(
                guard_target_identifiers(question, extract_identifier_terms)
                + ["fetch_workflow_runs", "summarize_workflow_runs"]
            )
        )
    else:
        return ""

    if not target_identifiers:
        return ""
    if call_chain_summary.strip():
        return ""
    if any(evidence_mentions_identifier(evidence_blocks, identifier) for identifier in target_identifiers):
        return ""
    if any(source_path_mentions_identifier(source_paths, identifier) for identifier in target_identifiers):
        return ""
    if any(identifier in repo_symbols for identifier in target_identifiers):
        return ""

    joined_targets = ", ".join(f"`{identifier}`" for identifier in target_identifiers[:3])
    answer_line = (
        f"Answer: This question does not appear to match the current workspace `{repo_path.name}`. "
        f"I could not find {joined_targets} in this repository."
    )
    why_lines = [
        f"- I checked the current repository for definitions, call sites, and imports tied to {joined_targets}, but did not find them.",
        "- This usually means the wrong workspace is selected, the GitHub import resolved to a different repository, or the index is stale.",
        "- Switch to the repository that should contain those implementation targets, or rebuild the current index before asking again.",
    ]
    return format_typed_answer(answer_line=answer_line, why_lines=why_lines, source_paths=[])
