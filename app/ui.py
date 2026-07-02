from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLONED_REPOS_ROOT = PROJECT_ROOT / ".repos"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig
from app.indexing import build_or_load_index
from app.qa import answer_question


load_dotenv()


DEFAULT_SUGGESTED_QUESTIONS = (
    "How is the weekly digest built?",
    "How is the summary report built?",
    "Which configuration values are defined in this project?",
    "Which file contains argparse and the main function?",
    "What calls summarize_workflow_runs across files?",
    "Which file fetches GitHub workflow runs and where are they summarized?",
    "Where are CI charts generated?",
    "What design risks do you see in this project?",
)


class RepoProfile(dict[str, object]):
    pass


class PythonFileFacts(dict[str, object]):
    pass

APP_BUILD_TAG = "relationship-filter-v5"

STOP_BUTTON_CSS = """
<style>
.st-key-stop_question {
    display: flex;
    justify-content: flex-end;
}
.st-key-stop_question button {
    width: 3rem !important;
    height: 3rem !important;
    min-width: 3rem !important;
    max-width: 3rem !important;
    min-height: 3rem !important;
    max-height: 3rem !important;
    aspect-ratio: 1 / 1 !important;
    border-radius: 50% !important;
    padding: 0 !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    background: rgba(255, 255, 255, 0.05) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    position: relative !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.04) inset !important;
}
.st-key-stop_question button::after {
    content: "";
    width: 0.72rem;
    height: 0.72rem;
    border-radius: 0.08rem;
    background: currentColor;
    display: block;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}
.st-key-stop_question button p {
    display: none !important;
    margin: 0 !important;
    line-height: 1 !important;
}
</style>
"""


def main() -> None:
    st.set_page_config(page_title="AI Codebase Assistant", page_icon=":books:", layout="wide")
    st.markdown(STOP_BUTTON_CSS, unsafe_allow_html=True)
    st.title("AI Codebase Assistant")
    st.caption("Ask natural language questions about a local repository with LangChain, LlamaIndex, and Ollama.")
    st.caption(f"Build: {APP_BUILD_TAG}")

    config = AppConfig.from_env()
    _init_session_state()
    _apply_pending_repo_input_value()
    _apply_pending_workspace_selector()

    default_repo = str((Path(__file__).resolve().parents[2] / "github-efficiency-analyzer").resolve())
    active_repo = _get_active_repo(default_repo)
    repo_path = Path(active_repo).resolve()

    st.markdown("### Workspace")
    top_left, top_right = st.columns([2, 1], gap="large")

    with top_left:
        workspace_options = st.session_state["workspace_order"] or [active_repo]
        st.selectbox(
            "Saved workspaces",
            options=workspace_options,
            index=workspace_options.index(active_repo) if active_repo in workspace_options else 0,
            key="workspace_selector",
            on_change=_handle_workspace_change,
        )
        repo_input = st.text_input("Repository path or GitHub URL", key="repo_input_value")
        repo_path, repo_spec = _resolve_repo_input(repo_input)
        repo_ready, repo_status_message = _validate_repo_input(repo_input, repo_path, repo_spec)
        index_dir = config.resolve_index_dir(repo_path)
        persisted_index_exists = index_dir.exists()
        rebuild = st.checkbox("Rebuild index", value=False)
        current_workspace = _get_workspace(repo_path)
        workspace_source_url = _display_source_url(repo_path, current_workspace)

        if repo_ready:
            if repo_spec["kind"] == "github_url" and not repo_path.exists():
                st.caption(f"GitHub repository will be cloned to: {repo_path}")
            else:
                st.caption(f"Repository found: {repo_path}")
            if workspace_source_url:
                st.caption(f"Imported from GitHub URL: {workspace_source_url}")
        else:
            st.error(repo_status_message)
        st.caption(f"Index directory: {index_dir}")
        st.caption(f"Persisted index on disk: {'Yes' if persisted_index_exists else 'No'}")

        action_col1, action_col2, action_col3 = st.columns([2, 1, 1])
        with action_col1:
            if st.button("Build / Load Index", use_container_width=True, type="primary", disabled=not repo_ready):
                try:
                    with st.spinner("Preparing index..."):
                        repo_path, force_rebuild = _prepare_repo_for_indexing(repo_input, repo_path, repo_spec)
                        index = build_or_load_index(repo_path=repo_path, config=config, rebuild=rebuild or force_rebuild)
                    _save_workspace(repo_path=repo_path, index=index, source_url=_workspace_source_url(repo_spec))
                    _queue_repo_input_value(str(repo_path))
                    st.success(f"Index ready at: {config.resolve_index_dir(repo_path)}")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to prepare index: {exc}")
        with action_col2:
            if st.button("Clear chat", use_container_width=True, disabled=not _get_workspace(repo_path)["messages"]):
                workspace = _get_workspace(repo_path)
                workspace["messages"] = []
                st.session_state["workspaces"][_repo_key(repo_path)] = workspace
                st.session_state["active_repo_key"] = _repo_key(repo_path)
                st.rerun()
        with action_col3:
            if st.button("Save workspace", use_container_width=True, disabled=not repo_ready or not repo_path.exists()):
                workspace = _get_workspace(repo_path)
                st.session_state["workspaces"][_repo_key(repo_path)] = workspace
                if _repo_key(repo_path) not in st.session_state["workspace_order"]:
                    st.session_state["workspace_order"].append(_repo_key(repo_path))
                st.session_state["active_repo_key"] = _repo_key(repo_path)
                st.success(f"Saved workspace: {repo_path.name}")
                st.rerun()

    with top_right:
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Chat model", config.chat_model)
        metric_col2.metric("Embedding model", config.embedding_model)
        st.caption(f"Active repo: {repo_path.name}")
        active_source_url = _display_source_url(repo_path, _get_workspace(repo_path))
        if active_source_url:
            st.caption(f"GitHub source: {active_source_url}")
        st.caption(f"Saved workspaces: {len(st.session_state['workspace_order'])}")
        current_workspace = _get_workspace(repo_path)
        st.caption(f"Index ready: {'Yes' if current_workspace['index'] is not None else 'No'}")
        st.caption(f"Persisted index: {'Yes' if config.resolve_index_dir(repo_path).exists() else 'No'}")
        st.caption(f"Messages in this workspace: {len(current_workspace['messages'])}")

    workspace = _get_workspace(repo_path)
    question_busy = _question_busy_for_repo(repo_path)

    st.markdown("### Suggested questions")
    q_col1, q_col2 = st.columns(2, gap="large")
    suggested_entries = _recommended_question_entries(repo_path)
    suggested_questions = tuple(entry["prompt"] for entry in suggested_entries)
    for idx, entry in enumerate(suggested_entries):
        prompt = str(entry["prompt"])
        target_column = q_col1 if idx % 2 == 0 else q_col2
        with target_column:
            if st.button(
                prompt,
                use_container_width=True,
                key=f"suggest_q_{idx}",
                disabled=workspace["index"] is None or question_busy,
            ):
                _queue_question(prompt, repo_path)
    if suggested_entries:
        with st.expander("Why these questions were recommended"):
            reason_col1, reason_col2 = st.columns(2, gap="large")
            for idx, entry in enumerate(suggested_entries):
                prompt = str(entry["prompt"])
                reason = str(entry.get("reason", "")).strip()
                target_column = reason_col1 if idx % 2 == 0 else reason_col2
                with target_column:
                    st.markdown(f"**{prompt}**")
                    if reason:
                        st.caption(reason)

    st.markdown("### Conversation")

    if workspace["index"] is None:
        st.info("Build or load an index first. Once the index is ready, suggested questions and chat input will become available.")
        return

    status_col, stop_col = st.columns([6, 1], gap="small")
    with status_col:
        if question_busy:
            st.caption("Thinking... recommended questions are temporarily locked for this workspace.")
        else:
            st.caption("Questions run synchronously in the current page session. For config or model changes, stop and restart Streamlit.")
    with stop_col:
        pass

    _render_messages(workspace["messages"])
    question = st.chat_input(
        "Ask a question about the repository",
        disabled=question_busy,
    )

    if question_busy:
        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
        thinking_col, stop_col = st.columns([7, 1], gap="small", vertical_alignment="center")
        with thinking_col:
            thinking_placeholder = st.empty()
        with stop_col:
            if st.button(" ", key="stop_question", help="Stop the current question", use_container_width=True):
                _cancel_pending_question(repo_path)
    else:
        thinking_placeholder = None

    if question:
        _queue_question(question, repo_path)

    _process_pending_question(workspace, repo_path, config, thinking_placeholder)


def _init_session_state() -> None:
    st.session_state.setdefault("workspaces", {})
    st.session_state.setdefault("workspace_order", [])
    st.session_state.setdefault("active_repo_key", "")
    st.session_state.setdefault("question_runs", {})
    st.session_state.setdefault("pending_question", "")
    st.session_state.setdefault("pending_repo_key", "")
    st.session_state.setdefault("question_in_flight", False)
    st.session_state.setdefault("active_question_run_id", 0)
    st.session_state.setdefault("pending_repo_input_value", "")
    st.session_state.setdefault("pending_workspace_selector", "")
    if "repo_input_value" not in st.session_state:
        default_repo = str((Path(__file__).resolve().parents[2] / "github-efficiency-analyzer").resolve())
        st.session_state["repo_input_value"] = st.session_state["active_repo_key"] or default_repo


def _queue_repo_input_value(value: str) -> None:
    st.session_state["pending_repo_input_value"] = value


def _queue_workspace_selector(value: str) -> None:
    st.session_state["pending_workspace_selector"] = value


def _apply_pending_repo_input_value() -> None:
    pending_value = st.session_state.get("pending_repo_input_value", "")
    if pending_value:
        st.session_state["repo_input_value"] = pending_value
        st.session_state["pending_repo_input_value"] = ""


def _apply_pending_workspace_selector() -> None:
    pending_value = st.session_state.get("pending_workspace_selector", "")
    if pending_value:
        st.session_state["workspace_selector"] = pending_value
        st.session_state["pending_workspace_selector"] = ""


def _render_messages(messages: list[dict[str, object]]) -> None:
    if not messages:
        st.caption("No messages yet. Ask a question after the index is ready.")
        return

    for message in messages:
        with st.chat_message(message["role"]):
            _render_message_content(str(message["content"]), str(message["role"]))
            search_question = message.get("search_question", "").strip()
            evidence = message.get("evidence") or []
            call_chain_summary = message.get("call_chain_summary", "").strip()
            confidence_label = message.get("confidence_label", "").strip()
            confidence_score = message.get("confidence_score", 0)
            risk_note = message.get("risk_note", "").strip()

            if confidence_label:
                if confidence_score >= 80:
                    st.success(f"{confidence_label} ({confidence_score}/100)")
                elif confidence_score >= 60:
                    st.warning(f"{confidence_label} ({confidence_score}/100)")
                else:
                    st.error(f"{confidence_label} ({confidence_score}/100)")

            if risk_note:
                st.caption(risk_note)

            if call_chain_summary:
                with st.expander("Cross-file relationships"):
                    st.markdown(call_chain_summary)

            if search_question:
                with st.expander("How the assistant searched"):
                    search_lines = _build_search_explainer(search_question)
                    st.caption(search_lines[0])
                    st.caption(search_lines[1])
                    st.markdown("**Retrieval query**")
                    st.code(search_lines[2])

            if evidence:
                with st.expander("Why these files were selected"):
                    for item in evidence:
                        file_path = str(item.get("file_path", "")).strip()
                        reason = str(item.get("reason", "")).strip()
                        snippet = str(item.get("snippet", "")).strip()
                        st.markdown(f"**{file_path}**")
                        st.caption(_evidence_bucket_label(reason, file_path))
                        if reason:
                            st.write(reason)
                        if snippet:
                            st.code(snippet)

            sources = message.get("sources") or []
            if sources:
                with st.expander("Sources"):
                    st.caption("These are the repository paths the final answer cited directly.")
                    for descriptor in _build_source_descriptors(sources):
                        st.markdown(f"**{descriptor['path']}**")
                        st.caption(descriptor["label"])


def _render_message_content(content: str, role: str) -> None:
    if role != "assistant":
        st.write(content)
        return

    body = re.split(r"\n\s*Sources:\s*\n", content, maxsplit=1)[0].strip()
    body = body.replace("\nWhy:\n", "\n\nWhy:\n\n")
    st.markdown(body)


def _build_search_explainer(search_question: str) -> tuple[str, str, str]:
    cleaned = search_question.strip()
    return (
        "Started from your question, then rewrote it into a repository search query.",
        "The rewritten query biases retrieval toward concrete code evidence such as definitions, calls, branches, and runtime behavior.",
        cleaned,
    )


def _evidence_bucket_label(reason: str, file_path: str) -> str:
    reason_lower = reason.lower()
    normalized_path = file_path.lower()
    if any(
        token in reason_lower
        for token in ("sibling script", "supplement", "same runtime folder", "companion script")
    ):
        return "Supporting sibling script"
    if reason_lower == "cross-file relationship summary":
        return "Cross-file relationship summary"
    if any(token in reason_lower for token in ("vector retrieval", "semantic", "keyword-based code match")):
        return "Retrieval match"
    if any(
        token in reason_lower
        for token in (
            "direct implementation evidence",
            "implementation",
            "defines `",
            "contains a call",
            "exports `",
            "control-flow",
            "runtime behavior",
            "entry point",
            "entrypoint",
            "startup and shutdown",
        )
    ):
        return "Direct implementation evidence"
    if normalized_path.endswith((".sh", ".cmd", ".bat")):
        return "Runtime script evidence"
    if normalized_path.endswith((".js", ".cjs", ".mjs", ".ts", ".tsx", ".py")):
        return "Relevant code evidence"
    if normalized_path.endswith((".md", ".rst")):
        return "Documentation context"
    return "Relevant repository evidence"


def _build_source_descriptors(sources: list[str]) -> list[dict[str, str]]:
    descriptors: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_source in sources:
        source = str(raw_source).strip()
        if not source or source in seen:
            continue
        seen.add(source)
        descriptors.append({"path": source, "label": _source_kind_label(source)})
    return descriptors


def _source_kind_label(path: str) -> str:
    normalized = path.lower()
    if normalized.endswith(".py"):
        return "Python source"
    if normalized.endswith((".js", ".cjs", ".mjs", ".ts", ".tsx")):
        return "JavaScript / TypeScript source"
    if normalized.endswith((".sh", ".bash")):
        return "Shell script"
    if normalized.endswith((".cmd", ".bat", ".ps1")):
        return "Command script"
    if normalized.endswith((".md", ".rst")) or normalized.endswith(("readme", "readme.md")):
        return "Documentation"
    if normalized.endswith((".json", ".toml", ".yaml", ".yml", ".ini", ".env")):
        return "Configuration / metadata"
    return "Repository file"


@lru_cache(maxsize=16)
def _suggested_questions_for_repo(repo_path: Path) -> tuple[str, ...]:
    return tuple(entry["prompt"] for entry in _recommended_question_entries(repo_path))


@lru_cache(maxsize=16)
def _recommended_question_entries(repo_path: Path) -> tuple[dict[str, str], ...]:
    profile = _build_repo_profile(repo_path)
    if not profile:
        return tuple(
            {"prompt": prompt, "reason": "Default fallback because no repository signals were detected."}
            for prompt in DEFAULT_SUGGESTED_QUESTIONS
        )

    prompt_entries: dict[str, dict[str, str | int]] = {}

    def add_prompt(prompt: str, score: int, reason: str) -> None:
        if not prompt.strip():
            return
        existing = prompt_entries.get(prompt)
        if existing is None or score > int(existing["score"]):
            prompt_entries[prompt] = {"prompt": prompt, "score": score, "reason": reason}

    if profile.get("has_entrypoint"):
        add_prompt(
            "Which file contains argparse and the main function?",
            60,
            str(profile.get("entrypoint_reason", "")),
        )

    config_target = str(profile.get("config_target", "")).strip()
    if config_target:
        config_reason = str(profile.get("config_reason", "")).strip()
        add_prompt(_config_question_for_target(config_target), 89, config_reason)

    relationship_target = str(profile.get("relationship_target", "")).strip()
    if relationship_target:
        relationship_score = 90 if relationship_target == "summarize_workflow_runs" else 75
        add_prompt(
            f"What calls {relationship_target} across files?",
            relationship_score,
            str(profile.get("relationship_reason", "")),
        )

    if profile.get("has_workflow_fetch_and_summary"):
        add_prompt(
            "Which file fetches GitHub workflow runs and where are they summarized?",
            95,
            str(profile.get("workflow_reason", "")),
        )

    if profile.get("has_weekly_digest"):
        add_prompt("How is the weekly digest built?", 100, str(profile.get("weekly_digest_reason", "")))

    if profile.get("has_summary_report"):
        add_prompt("How is the summary report built?", 85, str(profile.get("summary_report_reason", "")))

    if profile.get("has_ci_charts"):
        add_prompt("Where are CI charts generated?", 65, str(profile.get("ci_charts_reason", "")))

    if profile.get("has_design_risk_signals"):
        add_prompt("What design risks do you see in this project?", 88, str(profile.get("design_risk_reason", "")))

    merged = [
        prompt_entries[prompt]
        for prompt, _ in sorted(
            prompt_entries.items(),
            key=lambda item: (-int(item[1]["score"]), item[0]),
        )
    ]
    if not merged:
        return tuple(
            {"prompt": prompt, "reason": "Default fallback because no repository signals were detected."}
            for prompt in DEFAULT_SUGGESTED_QUESTIONS
        )
    merged_prompts = {str(entry["prompt"]) for entry in merged}
    for fallback in _fallback_questions_for_profile(profile):
        if fallback not in merged_prompts:
            merged.append(
                {
                    "prompt": fallback,
                    "score": 0,
                    "reason": "Default fallback matched to the repository signals already detected.",
                }
            )
        if len(merged) >= 8:
            break
    return tuple({"prompt": str(entry["prompt"]), "reason": str(entry["reason"])} for entry in merged[:8])


def _fallback_questions_for_profile(profile: RepoProfile) -> tuple[str, ...]:
    fallbacks: list[str] = []
    config_target = str(profile.get("config_target", "")).strip()

    if profile.get("has_weekly_digest"):
        fallbacks.append("How is the weekly digest built?")
    if profile.get("has_workflow_fetch_and_summary"):
        fallbacks.append("Which file fetches GitHub workflow runs and where are they summarized?")
    if profile.get("relationship_target"):
        relationship_target = str(profile.get("relationship_target", "")).strip()
        if relationship_target:
            fallbacks.append(f"What calls {relationship_target} across files?")
    if profile.get("has_design_risk_signals"):
        fallbacks.append("What design risks do you see in this project?")
    if profile.get("has_summary_report"):
        fallbacks.append("How is the summary report built?")
    if profile.get("has_ci_charts"):
        fallbacks.append("Where are CI charts generated?")
    if profile.get("has_entrypoint"):
        fallbacks.append("Which file contains argparse and the main function?")
    if config_target:
        fallbacks.append(_config_question_for_target(config_target))

    for prompt in DEFAULT_SUGGESTED_QUESTIONS:
        if prompt not in fallbacks:
            fallbacks.append(prompt)
        if len(fallbacks) >= 8:
            break
    return tuple(fallbacks[:8])


def _config_question_for_target(config_target: str) -> str:
    cleaned_target = config_target.strip()
    if not cleaned_target:
        return "Which configuration values are defined in this project?"
    return f"Where is `{cleaned_target}` configured?"


@lru_cache(maxsize=16)
def _build_repo_profile(repo_path: Path) -> RepoProfile:
    repo_root = Path(repo_path).resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        return RepoProfile()

    python_facts = _collect_python_file_facts(repo_root)
    if not python_facts:
        return RepoProfile()

    main_facts = python_facts.get("app/main.py")
    config_facts = python_facts.get("app/config.py")
    metrics_facts = python_facts.get("app/metrics.py")
    github_client_facts = python_facts.get("app/github_client.py")
    report_facts = python_facts.get("app/report.py")
    charts_facts = python_facts.get("app/charts.py")
    risk_facts = python_facts.get("app/ci_failure_analysis.py")

    has_entrypoint = False
    entrypoint_reason = ""
    if main_facts:
        has_entrypoint = bool(main_facts.get("has_argparse_import")) and (
            bool(main_facts.get("has_main_function"))
            or bool(main_facts.get("has_parse_args_function"))
            or bool(main_facts.get("has_main_guard"))
        )
        if has_entrypoint:
            entrypoint_reason = "Found `argparse` and entrypoint flow in `app/main.py` (`parse_args`, `main`, `__main__`)."

    config_target = ""
    config_reason = ""
    if config_facts:
        assigned_names = tuple(config_facts.get("assigned_names", ()))
        env_var_names = tuple(config_facts.get("env_var_names", ()))
        config_target = _pick_first_name(
            env_var_names,
            ("OLLAMA_BASE_URL", "OLLAMA_CHAT_MODEL", "GITHUB_API_BASE", "GITHUB_TOKEN"),
        )
        if not config_target:
            config_target = _pick_first_name(
                assigned_names,
                ("OLLAMA_BASE_URL", "ollama_base_url", "OLLAMA_CHAT_MODEL"),
            )
        if not config_target:
            config_target = _pick_first_name(
                env_var_names,
                tuple(name for name in env_var_names if "ollama" in name.lower() or "base_url" in name.lower()),
            )
        if not config_target:
            config_target = _pick_first_name(
                assigned_names,
                tuple(name for name in assigned_names if "ollama" in name.lower() or "base_url" in name.lower()),
            )
        if not config_target:
            config_target = _pick_first_name(
                env_var_names,
                tuple(
                    name
                    for name in env_var_names
                    if name.isupper() and any(token in name for token in ("URL", "BASE", "TOKEN", "MODEL", "API", "KEY"))
                ),
            )
        if not config_target:
            config_target = _pick_first_name(
                assigned_names,
                tuple(
                    name
                    for name in assigned_names
                    if name.isupper() and any(token in name for token in ("URL", "BASE", "TOKEN", "MODEL", "API", "KEY"))
                ),
            )
        if not config_target:
            config_target = _pick_first_name(
                assigned_names,
                tuple(
                    name
                    for name in assigned_names
                    if any(token in name.lower() for token in ("base", "url", "token", "model", "api", "key"))
                ),
            )
        if config_target:
            if config_target in env_var_names:
                config_reason = f"Found env config `{config_target}` in `app/config.py`."
            else:
                config_reason = f"Found `{config_target}` in `app/config.py`."

    relationship_target = ""
    relationship_reason = ""
    metrics_defs = tuple(metrics_facts.get("function_defs", ())) if metrics_facts else ()
    main_calls = tuple(main_facts.get("calls", ())) if main_facts else ()
    for candidate in ("summarize_workflow_runs", "build_weekly_ci_digest", "summarize_pull_requests"):
        if candidate in metrics_defs and candidate in main_calls:
            relationship_target = candidate
            relationship_reason = f"`{candidate}()` is defined in `app/metrics.py` and called from `app/main.py`."
            break
    if not relationship_target:
        for candidate in ("summarize_workflow_runs", "build_weekly_ci_digest", "summarize_pull_requests"):
            if candidate in metrics_defs:
                relationship_target = candidate
                relationship_reason = f"Found `{candidate}()` in `app/metrics.py`."
                break

    has_workflow_fetch_and_summary = bool(github_client_facts and metrics_facts) and (
        "fetch_workflow_runs" in tuple(github_client_facts.get("function_defs", ()))
        and "summarize_workflow_runs" in metrics_defs
    )
    workflow_reason = ""
    if has_workflow_fetch_and_summary:
        workflow_reason = "`fetch_workflow_runs()` is in `app/github_client.py`; `summarize_workflow_runs()` is in `app/metrics.py`."

    has_weekly_digest = False
    weekly_digest_reason = ""
    has_summary_report = False
    summary_report_reason = ""
    if report_facts:
        report_defs = set(report_facts.get("function_defs", ()))
        report_strings = set(report_facts.get("string_literals", ()))
        has_weekly_digest = "write_weekly_digest_report" in report_defs or "weekly_digest.md" in report_strings
        has_summary_report = "write_markdown_report" in report_defs or "summary.md" in report_strings
        if has_weekly_digest:
            weekly_digest_reason = "Found `write_weekly_digest_report()` and `weekly_digest.md` in `app/report.py`."
        if has_summary_report:
            summary_report_reason = "Found `write_markdown_report()` and `summary.md` in `app/report.py`."

    has_ci_charts = False
    ci_charts_reason = ""
    if charts_facts:
        chart_defs = set(charts_facts.get("function_defs", ()))
        has_ci_charts = bool({"write_failure_trend_chart", "write_failed_workflow_chart"} & chart_defs)
        if has_ci_charts:
            ci_charts_reason = "Found CI chart writers in `app/charts.py`."

    has_design_risk_signals = False
    design_risk_reason = ""
    if metrics_facts:
        metrics_names = set(metrics_facts.get("assigned_names", ())) | set(metrics_facts.get("name_refs", ()))
        if {"category_counts", "workflow_failures"} & metrics_names:
            has_design_risk_signals = True
            design_risk_reason = "Found centralized failure buckets like `category_counts` or `workflow_failures` in `app/metrics.py`."
    if not has_design_risk_signals and risk_facts:
        risk_names = set(risk_facts.get("assigned_names", ())) | set(risk_facts.get("name_refs", ()))
        risk_strings = set(risk_facts.get("string_literals", ()))
        if "patterns" in risk_names or {"permission_failure", "unknown_failure"} & risk_strings:
            has_design_risk_signals = True
            design_risk_reason = "Found rule-based failure buckets in `app/ci_failure_analysis.py`."

    return RepoProfile(
        has_entrypoint=has_entrypoint,
        entrypoint_reason=entrypoint_reason,
        config_target=config_target,
        config_reason=config_reason,
        relationship_target=relationship_target,
        relationship_reason=relationship_reason,
        has_workflow_fetch_and_summary=has_workflow_fetch_and_summary,
        workflow_reason=workflow_reason,
        has_weekly_digest=has_weekly_digest,
        weekly_digest_reason=weekly_digest_reason,
        has_summary_report=has_summary_report,
        summary_report_reason=summary_report_reason,
        has_ci_charts=has_ci_charts,
        ci_charts_reason=ci_charts_reason,
        has_design_risk_signals=has_design_risk_signals,
        design_risk_reason=design_risk_reason,
    )


def _collect_python_file_facts(repo_root: Path) -> dict[str, PythonFileFacts]:
    facts_by_file: dict[str, PythonFileFacts] = {}
    for path in repo_root.rglob("*.py"):
        try:
            relative_path = path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        facts = _parse_python_file_facts(path)
        if facts:
            facts_by_file[relative_path] = facts
    return facts_by_file


def _parse_python_file_facts(path: Path) -> PythonFileFacts:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return PythonFileFacts()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return PythonFileFacts()

    function_defs: list[str] = []
    assigned_names: list[str] = []
    calls: list[str] = []
    name_refs: list[str] = []
    string_literals: list[str] = []
    imports: list[str] = []
    env_var_names: list[str] = []
    has_main_guard = False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_defs.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                assigned_names.extend(_extract_assigned_names(target))
        elif isinstance(node, ast.AnnAssign):
            assigned_names.extend(_extract_assigned_names(node.target))
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name:
                calls.append(call_name)
            env_name = _env_var_name(node)
            if env_name:
                env_var_names.append(env_name)
        elif isinstance(node, ast.Name):
            name_refs.append(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_literals.append(node.value)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.If) and _is_main_guard(node.test):
            has_main_guard = True

    unique_function_defs = tuple(dict.fromkeys(function_defs))
    unique_assigned_names = tuple(dict.fromkeys(assigned_names))
    unique_calls = tuple(dict.fromkeys(calls))
    unique_name_refs = tuple(dict.fromkeys(name_refs))
    unique_string_literals = tuple(dict.fromkeys(string_literals))
    unique_imports = tuple(dict.fromkeys(imports))
    unique_env_var_names = tuple(dict.fromkeys(env_var_names))

    return PythonFileFacts(
        function_defs=unique_function_defs,
        assigned_names=unique_assigned_names,
        calls=unique_calls,
        name_refs=unique_name_refs,
        string_literals=unique_string_literals,
        imports=unique_imports,
        env_var_names=unique_env_var_names,
        has_argparse_import=any(name == "argparse" or name.endswith(".argparse") for name in unique_imports)
        or "ArgumentParser" in unique_name_refs,
        has_main_function="main" in unique_function_defs,
        has_parse_args_function="parse_args" in unique_function_defs,
        has_main_guard=has_main_guard,
    )


def _extract_assigned_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in target.elts:
            names.extend(_extract_assigned_names(element))
        return names
    return []


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _env_var_name(node: ast.Call) -> str:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return ""
    if func.attr != "getenv":
        return ""
    if not node.args:
        return ""
    first_arg = node.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value
    return ""


def _is_main_guard(node: ast.expr) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    if not isinstance(node.ops[0], ast.Eq):
        return False
    left = node.left
    right = node.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    )


def _pick_first_name(candidates: tuple[str, ...], preferred_order: tuple[str, ...]) -> str:
    candidate_set = {candidate for candidate in candidates if candidate}
    for preferred in preferred_order:
        if preferred in candidate_set:
            return preferred
    return ""


def _get_active_repo(default_repo: str) -> str:
    active_repo_key = st.session_state.get("active_repo_key", "")
    if active_repo_key:
        return active_repo_key
    return default_repo


def _repo_key(repo_path: Path) -> str:
    return str(repo_path.resolve())


def _new_workspace(repo_path: Path) -> dict[str, object]:
    repo_key = _repo_key(repo_path)
    return {
        "repo_path": repo_key,
        "index": None,
        "messages": [],
        "source_url": "",
    }


def _get_workspace(repo_path: Path) -> dict[str, object]:
    repo_key = _repo_key(repo_path)
    workspace = st.session_state["workspaces"].get(repo_key)
    if workspace is None:
        workspace = _new_workspace(repo_path)
    workspace.setdefault("source_url", "")
    return workspace


def _save_workspace(repo_path: Path, index: object, source_url: str = "") -> None:
    workspace = _get_workspace(repo_path)
    workspace["index"] = index
    workspace["repo_path"] = _repo_key(repo_path)
    if source_url:
        workspace["source_url"] = source_url
    st.session_state["workspaces"][_repo_key(repo_path)] = workspace
    if _repo_key(repo_path) not in st.session_state["workspace_order"]:
        st.session_state["workspace_order"].append(_repo_key(repo_path))
    st.session_state["active_repo_key"] = _repo_key(repo_path)
    _queue_workspace_selector(_repo_key(repo_path))


def _workspace_source_url(repo_spec: dict[str, str]) -> str:
    if repo_spec.get("kind") == "github_url":
        return str(repo_spec.get("input", "")).strip()
    return ""


def _display_source_url(repo_path: Path, workspace: dict[str, object]) -> str:
    stored = str(workspace.get("source_url", "")).strip()
    if stored:
        return stored
    return _infer_github_source_url(repo_path)


def _infer_github_source_url(repo_path: Path) -> str:
    try:
        relative = repo_path.resolve().relative_to(CLONED_REPOS_ROOT.resolve())
    except ValueError:
        return ""

    parts = relative.parts
    if len(parts) < 2:
        return ""
    owner, repo = parts[0], parts[1]
    if not owner or not repo:
        return ""
    return f"https://github.com/{owner}/{repo}"


def _question_run_defaults() -> dict[str, object]:
    return {
        "pending_question": "",
        "question_in_flight": False,
        "active_question_run_id": 0,
    }


def _question_runs_store() -> dict[str, dict[str, object]]:
    store = st.session_state.setdefault("question_runs", {})
    legacy_repo_key = str(st.session_state.get("pending_repo_key", "")).strip()
    if legacy_repo_key and not store:
        store[legacy_repo_key] = {
            "pending_question": st.session_state.get("pending_question", ""),
            "question_in_flight": bool(st.session_state.get("question_in_flight", False)),
            "active_question_run_id": int(st.session_state.get("active_question_run_id", 0)),
        }
    return store


def _question_run_state(repo_path: Path) -> dict[str, object]:
    repo_key = _repo_key(repo_path)
    store = _question_runs_store()
    state = store.get(repo_key)
    if state is None:
        state = _question_run_defaults()
        store[repo_key] = state
    return state


def _sync_legacy_question_state(repo_path: Path) -> None:
    state = _question_run_state(repo_path)
    st.session_state["pending_question"] = str(state.get("pending_question", ""))
    st.session_state["pending_repo_key"] = _repo_key(repo_path) if state.get("question_in_flight") else ""
    st.session_state["question_in_flight"] = bool(state.get("question_in_flight", False))
    st.session_state["active_question_run_id"] = int(state.get("active_question_run_id", 0))


def _question_busy_for_repo(repo_path: Path) -> bool:
    state = _question_run_state(repo_path)
    return bool(state.get("question_in_flight", False))


def _queue_question(question: str, repo_path: Path) -> None:
    cleaned_question = question.strip()
    if not cleaned_question or _question_busy_for_repo(repo_path):
        return

    workspace = _get_workspace(repo_path)
    workspace["messages"].append(
        {
            "role": "user",
            "content": cleaned_question,
        }
    )
    if _repo_key(repo_path) not in st.session_state["workspace_order"]:
        st.session_state["workspace_order"].append(_repo_key(repo_path))
    st.session_state["workspaces"][_repo_key(repo_path)] = workspace
    state = _question_run_state(repo_path)
    state["pending_question"] = cleaned_question
    state["question_in_flight"] = True
    state["active_question_run_id"] = int(state.get("active_question_run_id", 0)) + 1
    _sync_legacy_question_state(repo_path)
    st.rerun()


def _cancel_pending_question(repo_path: Path) -> None:
    if not _question_busy_for_repo(repo_path):
        return

    state = _question_run_state(repo_path)
    state["pending_question"] = ""
    state["question_in_flight"] = False
    state["active_question_run_id"] = int(state.get("active_question_run_id", 0)) + 1
    _sync_legacy_question_state(repo_path)
    st.rerun()


def _process_pending_question(
    workspace: dict[str, object],
    repo_path: Path,
    config: AppConfig,
    thinking_placeholder: object | None = None,
) -> None:
    state = _question_run_state(repo_path)
    pending_question = str(state.get("pending_question", "")).strip()
    if not pending_question:
        return
    if not state.get("question_in_flight"):
        return

    run_id = int(state.get("active_question_run_id", 0))
    try:
        _submit_suggested_question(pending_question, workspace, repo_path, config, run_id, thinking_placeholder)
    finally:
        current_state = _question_run_state(repo_path)
        if int(current_state.get("active_question_run_id", 0)) == run_id:
            current_state["pending_question"] = ""
            current_state["question_in_flight"] = False
        _sync_legacy_question_state(repo_path)


def _submit_suggested_question(
    question: str,
    workspace: dict[str, object],
    repo_path: Path,
    config: AppConfig,
    run_id: int | None = None,
    thinking_placeholder: object | None = None,
    ) -> None:
    if workspace["index"] is None:
        workspace["messages"].append(
            {
                "role": "assistant",
                "content": "Index is not ready yet. Click `Build / Load Index` before asking a question.",
                "sources": [],
                "search_question": "",
                "evidence": [],
                "call_chain_summary": "",
                "confidence_label": "Low confidence",
                "confidence_score": 0,
                "risk_note": "No repository index is loaded in the current workspace, so the assistant cannot answer reliably yet.",
            }
        )
        st.session_state["workspaces"][_repo_key(repo_path)] = workspace
        st.session_state["active_repo_key"] = _repo_key(repo_path)
        st.rerun()
        return

    if _question_run_canceled(repo_path, run_id):
        return

    spinner_host = thinking_placeholder if thinking_placeholder is not None else st
    with spinner_host.spinner("Thinking..."):
        try:
            if _question_run_canceled(repo_path, run_id):
                return
            result = answer_question(
                index=workspace["index"],
                question=question,
                config=config,
                repo_path=repo_path,
                history=workspace["messages"],
            )
            if _question_run_canceled(repo_path, run_id):
                return
            workspace["messages"].append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.sources,
                    "search_question": result.search_question,
                    "evidence": result.evidence,
                    "call_chain_summary": getattr(result, "call_chain_summary", ""),
                    "confidence_label": result.confidence_label,
                    "confidence_score": result.confidence_score,
                    "risk_note": result.risk_note,
                }
            )
        except Exception as exc:  # noqa: BLE001
            if _question_run_canceled(repo_path, run_id):
                return
            workspace["messages"].append(
                {
                    "role": "assistant",
                    "content": f"Failed to answer question: {exc}",
                    "sources": [],
                    "search_question": "",
                    "evidence": [],
                    "call_chain_summary": "",
                    "confidence_label": "Low confidence",
                    "confidence_score": 0,
                    "risk_note": "The assistant failed to answer the question, so no reliable confidence estimate is available.",
                }
            )
    st.session_state["workspaces"][_repo_key(repo_path)] = workspace
    st.session_state["active_repo_key"] = _repo_key(repo_path)
    st.rerun()


def _question_run_canceled(repo_path: Path, run_id: int | None) -> bool:
    if run_id is None:
        return False
    state = _question_run_state(repo_path)
    if int(state.get("active_question_run_id", 0)) != run_id or not bool(state.get("question_in_flight", False)):
        return True

    legacy_run_id = int(st.session_state.get("active_question_run_id", 0))
    legacy_in_flight = bool(st.session_state.get("question_in_flight", False))
    legacy_repo_key = str(st.session_state.get("pending_repo_key", "")).strip()
    active_repo_key = str(st.session_state.get("active_repo_key", "")).strip()
    current_repo_key = _repo_key(repo_path)
    legacy_targets_current_repo = legacy_repo_key == current_repo_key or (
        not legacy_repo_key and active_repo_key == current_repo_key
    )
    if legacy_targets_current_repo and (legacy_run_id != run_id or not legacy_in_flight):
        return True

    return False


def _handle_workspace_change() -> None:
    selected_workspace = st.session_state.get("workspace_selector", "")
    if selected_workspace:
        st.session_state["active_repo_key"] = selected_workspace
        _queue_repo_input_value(selected_workspace)


def _validate_repo_path(repo_path: Path) -> tuple[bool, str]:
    if not repo_path.exists():
        return False, "Repository path does not exist yet."
    if not repo_path.is_dir():
        return False, "Repository path must point to a folder."
    return True, ""


def _validate_repo_input(repo_input: str, repo_path: Path, repo_spec: dict[str, str]) -> tuple[bool, str]:
    cleaned_input = repo_input.strip()
    if not cleaned_input:
        return False, "Enter a local repository path or a GitHub repository URL."
    if repo_spec["kind"] == "github_url":
        return True, ""
    return _validate_repo_path(repo_path)


def _resolve_repo_input(repo_input: str) -> tuple[Path, dict[str, str]]:
    cleaned_input = repo_input.strip()
    github_spec = _parse_github_repo_url(cleaned_input)
    if github_spec:
        return _clone_target_dir(github_spec["owner"], github_spec["repo"]), github_spec
    if cleaned_input:
        return Path(cleaned_input).resolve(), {"kind": "local_path", "input": cleaned_input}
    return Path(".").resolve(), {"kind": "local_path", "input": ""}


def _parse_github_repo_url(repo_input: str) -> dict[str, str] | None:
    if not repo_input:
        return None
    parsed = urlparse(repo_input)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != "github.com":
        return None
    path_parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(path_parts) < 2:
        return None
    owner = path_parts[0]
    repo = path_parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        return None
    return {
        "kind": "github_url",
        "input": repo_input,
        "owner": owner,
        "repo": repo,
        "clone_url": f"https://github.com/{owner}/{repo}.git",
    }


def _clone_target_dir(owner: str, repo: str) -> Path:
    return (CLONED_REPOS_ROOT / owner / repo).resolve()


def _prepare_repo_for_indexing(repo_input: str, repo_path: Path, repo_spec: dict[str, str]) -> tuple[Path, bool]:
    if repo_spec["kind"] != "github_url":
        return repo_path, False

    clone_url = repo_spec["clone_url"]
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    force_rebuild = False
    if (repo_path / ".git").exists():
        _run_git_command(["git", "-C", str(repo_path), "pull", "--ff-only"], repo_input)
        force_rebuild = True
    elif repo_path.exists() and any(repo_path.iterdir()):
        raise RuntimeError(f"Clone target already exists and is not an empty git repository: {repo_path}")
    else:
        _run_git_command(["git", "clone", clone_url, str(repo_path)], repo_input)
        force_rebuild = True

    valid, message = _validate_repo_path(repo_path)
    if not valid:
        raise RuntimeError(message)
    return repo_path, force_rebuild


def _run_git_command(command: list[str], repo_input: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Git is not installed or is not available on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or f"git command failed for {repo_input}"
        raise RuntimeError(_humanize_git_error(detail, repo_input)) from exc


def _humanize_git_error(detail: str, repo_input: str) -> str:
    lowered = detail.lower()
    if any(
        token in lowered
        for token in (
            "timed out",
            "failed to connect to github.com",
            "could not resolve host: github.com",
            "proxy connect aborted",
            "ssl connect error",
            "connection was reset",
            "connection reset",
        )
    ):
        return (
            "GitHub network request timed out while preparing this repository. "
            "Check whether this machine can access github.com, verify proxy or VPN settings, "
            "or clone the repo locally first and then paste the local folder path. "
            f"Original git detail: {detail}"
        )
    if "repository not found" in lowered or "not found" in lowered:
        return (
            "GitHub could not find that repository or this machine does not have permission to access it. "
            "Double-check the URL and repository visibility. "
            f"Original git detail: {detail}"
        )
    if "authentication failed" in lowered:
        return (
            "GitHub rejected the request during authentication. "
            "If the repository is private, authenticate Git on this machine first or use a local clone. "
            f"Original git detail: {detail}"
        )
    return detail


if __name__ == "__main__":
    main()
