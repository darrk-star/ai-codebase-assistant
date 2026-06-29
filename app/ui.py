from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import sys

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig
from app.indexing import build_or_load_index
from app.qa import answer_question


load_dotenv()


DEFAULT_SUGGESTED_QUESTIONS = (
    "How is the weekly digest built?",
    "How is the summary report built?",
    "Where is the Ollama base URL configured?",
    "Which file contains argparse and the main function?",
    "What calls summarize_workflow_runs across files?",
    "Which file fetches GitHub workflow runs and where are they summarized?",
    "Where are CI charts generated?",
    "What design risks do you see in this project?",
)

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
        repo_input = st.text_input("Repository path", key="repo_input_value")
        repo_path = Path(repo_input).resolve()
        repo_ready, repo_status_message = _validate_repo_path(repo_path)
        index_dir = config.resolve_index_dir(repo_path)
        persisted_index_exists = index_dir.exists()
        rebuild = st.checkbox("Rebuild index", value=False)

        if repo_ready:
            st.caption(f"Repository found: {repo_path}")
        else:
            st.error(repo_status_message)
        st.caption(f"Index directory: {index_dir}")
        st.caption(f"Persisted index on disk: {'Yes' if persisted_index_exists else 'No'}")

        action_col1, action_col2, action_col3 = st.columns([2, 1, 1])
        with action_col1:
            if st.button("Build / Load Index", use_container_width=True, type="primary", disabled=not repo_ready):
                try:
                    with st.spinner("Preparing index..."):
                        index = build_or_load_index(repo_path=repo_path, config=config, rebuild=rebuild)
                    _save_workspace(repo_path=repo_path, index=index)
                    st.success(f"Index ready at: {config.resolve_index_dir(repo_path)}")
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
            if st.button("Save workspace", use_container_width=True, disabled=not repo_ready):
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
        st.caption(f"Saved workspaces: {len(st.session_state['workspace_order'])}")
        current_workspace = _get_workspace(repo_path)
        st.caption(f"Index ready: {'Yes' if current_workspace['index'] is not None else 'No'}")
        st.caption(f"Persisted index: {'Yes' if config.resolve_index_dir(repo_path).exists() else 'No'}")
        st.caption(f"Messages in this workspace: {len(current_workspace['messages'])}")

    workspace = _get_workspace(repo_path)
    question_busy = _question_busy_for_repo(repo_path)

    st.markdown("### Suggested questions")
    q_col1, q_col2 = st.columns(2, gap="large")
    suggested_questions = _suggested_questions_for_repo(repo_path)
    for idx, prompt in enumerate(suggested_questions):
        target_column = q_col1 if idx % 2 == 0 else q_col2
        with target_column:
            if st.button(
                prompt,
                use_container_width=True,
                key=f"suggest_q_{idx}",
                disabled=workspace["index"] is None or question_busy,
            ):
                _queue_question(prompt, repo_path)

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
    st.session_state.setdefault("pending_question", "")
    st.session_state.setdefault("pending_repo_key", "")
    st.session_state.setdefault("question_in_flight", False)
    st.session_state.setdefault("active_question_run_id", 0)
    if "repo_input_value" not in st.session_state:
        default_repo = str((Path(__file__).resolve().parents[2] / "github-efficiency-analyzer").resolve())
        st.session_state["repo_input_value"] = st.session_state["active_repo_key"] or default_repo


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
                    st.caption("Rewritten retrieval question")
                    st.code(search_question)

            if evidence:
                with st.expander("Why these files were selected"):
                    for item in evidence:
                        st.markdown(f"**{item['file_path']}**")
                        st.caption(item["reason"])
                        st.code(item["snippet"])

            sources = message.get("sources") or []
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.code(source)


def _render_message_content(content: str, role: str) -> None:
    if role != "assistant":
        st.write(content)
        return

    body = re.split(r"\n\s*Sources:\s*\n", content, maxsplit=1)[0].strip()
    body = body.replace("\nWhy:\n", "\n\nWhy:\n\n")
    st.markdown(body)


@lru_cache(maxsize=16)
def _suggested_questions_for_repo(repo_path: Path) -> tuple[str, ...]:
    repo_root = Path(repo_path).resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        return DEFAULT_SUGGESTED_QUESTIONS

    app_dir = repo_root / "app"
    main_path = app_dir / "main.py"
    config_path = app_dir / "config.py"
    metrics_path = app_dir / "metrics.py"
    github_client_path = app_dir / "github_client.py"
    report_path = app_dir / "report.py"
    charts_path = app_dir / "charts.py"

    prompt_scores: dict[str, int] = {}

    def add_prompt(prompt: str, score: int) -> None:
        if not prompt.strip():
            return
        existing = prompt_scores.get(prompt)
        if existing is None or score > existing:
            prompt_scores[prompt] = score

    if _file_contains(main_path, ("argparse", "ArgumentParser", "def main(")):
        add_prompt("Which file contains argparse and the main function?", 60)

    config_target = _pick_first_identifier(
        config_path,
        ("OLLAMA_BASE_URL", "ollama_base_url", "OLLAMA_CHAT_MODEL"),
    )
    if config_target:
        if "base_url" in config_target.lower():
            add_prompt("Where is the Ollama base URL configured?", 80)
        else:
            add_prompt(f"Where is `{config_target}` configured?", 70)

    relationship_target = _pick_first_identifier(
        metrics_path,
        ("summarize_workflow_runs", "build_weekly_ci_digest", "summarize_pull_requests"),
    )
    if relationship_target:
        relationship_score = 90 if relationship_target == "summarize_workflow_runs" else 75
        add_prompt(f"What calls {relationship_target} across files?", relationship_score)

    if _file_contains(github_client_path, ("fetch_workflow_runs",)) and _file_contains(metrics_path, ("summarize_workflow_runs",)):
        add_prompt("Which file fetches GitHub workflow runs and where are they summarized?", 95)

    if _file_contains(report_path, ("write_weekly_digest_report", "weekly_digest.md")):
        add_prompt("How is the weekly digest built?", 100)

    if _file_contains(report_path, ("write_markdown_report", "summary.md")):
        add_prompt("How is the summary report built?", 85)

    if _file_contains(charts_path, ("write_failure_trend_chart", "write_failed_workflow_chart")):
        add_prompt("Where are CI charts generated?", 65)

    if _file_contains(metrics_path, ("category_counts", "workflow_failures")) or _file_contains(
        repo_root / "app" / "ci_failure_analysis.py",
        ("patterns", "permission_failure", "unknown_failure"),
    ):
        add_prompt("What design risks do you see in this project?", 88)

    merged = [
        prompt
        for prompt, _ in sorted(
            prompt_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    if not merged:
        return DEFAULT_SUGGESTED_QUESTIONS
    for fallback in DEFAULT_SUGGESTED_QUESTIONS:
        if fallback not in merged:
            merged.append(fallback)
        if len(merged) >= 8:
            break
    return tuple(merged[:8])


def _file_contains(path: Path, patterns: tuple[str, ...]) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
    return any(pattern in text for pattern in patterns)


def _pick_first_identifier(path: Path, candidates: tuple[str, ...]) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""
    for candidate in candidates:
        if candidate in text:
            return candidate
    return ""


def _get_active_repo(default_repo: str) -> str:
    active_repo_key = st.session_state.get("active_repo_key", "")
    if active_repo_key:
        return active_repo_key
    return default_repo


def _repo_key(repo_path: Path) -> str:
    return str(repo_path.resolve())


def _get_workspace(repo_path: Path) -> dict[str, object]:
    repo_key = _repo_key(repo_path)
    workspace = st.session_state["workspaces"].get(repo_key)
    if workspace is None:
        workspace = {
            "repo_path": repo_key,
            "index": None,
            "messages": [],
        }
        st.session_state["workspaces"][repo_key] = workspace
        if repo_key not in st.session_state["workspace_order"]:
            st.session_state["workspace_order"].append(repo_key)
    return workspace


def _save_workspace(repo_path: Path, index: object) -> None:
    workspace = _get_workspace(repo_path)
    workspace["index"] = index
    workspace["repo_path"] = _repo_key(repo_path)
    st.session_state["workspaces"][_repo_key(repo_path)] = workspace
    if _repo_key(repo_path) not in st.session_state["workspace_order"]:
        st.session_state["workspace_order"].append(_repo_key(repo_path))
    st.session_state["active_repo_key"] = _repo_key(repo_path)


def _question_busy_for_repo(repo_path: Path) -> bool:
    return bool(st.session_state.get("question_in_flight")) and st.session_state.get("pending_repo_key", "") == _repo_key(repo_path)


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
    st.session_state["workspaces"][_repo_key(repo_path)] = workspace
    st.session_state["pending_question"] = cleaned_question
    st.session_state["pending_repo_key"] = _repo_key(repo_path)
    st.session_state["question_in_flight"] = True
    st.session_state["active_question_run_id"] = st.session_state.get("active_question_run_id", 0) + 1
    st.rerun()


def _cancel_pending_question(repo_path: Path) -> None:
    if not _question_busy_for_repo(repo_path):
        return

    st.session_state["pending_question"] = ""
    st.session_state["pending_repo_key"] = ""
    st.session_state["question_in_flight"] = False
    st.session_state["active_question_run_id"] = st.session_state.get("active_question_run_id", 0) + 1
    st.rerun()


def _process_pending_question(
    workspace: dict[str, object],
    repo_path: Path,
    config: AppConfig,
    thinking_placeholder: object | None = None,
) -> None:
    pending_question = st.session_state.get("pending_question", "").strip()
    pending_repo_key = st.session_state.get("pending_repo_key", "")

    if not pending_question or pending_repo_key != _repo_key(repo_path):
        return
    if not st.session_state.get("question_in_flight"):
        return

    run_id = st.session_state.get("active_question_run_id", 0)
    try:
        _submit_suggested_question(pending_question, workspace, repo_path, config, run_id, thinking_placeholder)
    finally:
        if st.session_state.get("active_question_run_id", 0) == run_id:
            st.session_state["pending_question"] = ""
            st.session_state["pending_repo_key"] = ""
            st.session_state["question_in_flight"] = False


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

    spinner_host = thinking_placeholder if thinking_placeholder is not None else st
    with spinner_host.spinner("Thinking..."):
        try:
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
    return (
        st.session_state.get("active_question_run_id", 0) != run_id
        or st.session_state.get("pending_repo_key", "") != _repo_key(repo_path)
        or not st.session_state.get("question_in_flight", False)
    )


def _handle_workspace_change() -> None:
    selected_workspace = st.session_state.get("workspace_selector", "")
    if selected_workspace:
        st.session_state["active_repo_key"] = selected_workspace
        st.session_state["repo_input_value"] = selected_workspace


def _validate_repo_path(repo_path: Path) -> tuple[bool, str]:
    if not repo_path.exists():
        return False, "Repository path does not exist yet."
    if not repo_path.is_dir():
        return False, "Repository path must point to a folder."
    return True, ""


if __name__ == "__main__":
    main()
