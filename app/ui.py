from __future__ import annotations

from pathlib import Path
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


SUGGESTED_QUESTIONS = (
    "How is the weekly digest built?",
    "How is the summary report built?",
    "Where is the Ollama base URL configured?",
    "Which file contains argparse and the main function?",
    "What calls compute_digest across files?",
    "Which file fetches GitHub workflow runs and where are they summarized?",
    "Where are CI charts generated?",
    "What design risks do you see in this project?",
)


def main() -> None:
    st.set_page_config(page_title="AI Codebase Assistant", page_icon=":books:", layout="wide")
    st.title("AI Codebase Assistant")
    st.caption("Ask natural language questions about a local repository with LangChain, LlamaIndex, and Ollama.")

    config = AppConfig.from_env()
    _init_session_state()

    default_repo = str((Path(__file__).resolve().parents[2] / "github-efficiency-analyzer").resolve())
    active_repo = _get_active_repo(default_repo)
    repo_path = Path(active_repo).resolve()

    st.markdown("### Workspace")
    top_left, top_right = st.columns([2, 1], gap="large")

    with top_left:
        workspace_options = st.session_state["workspace_order"] or [active_repo]
        selected_workspace = st.selectbox(
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

    st.markdown("### Suggested questions")
    q_col1, q_col2 = st.columns(2, gap="large")
    for idx, question in enumerate(SUGGESTED_QUESTIONS):
        target_column = q_col1 if idx % 2 == 0 else q_col2
        with target_column:
            if st.button(
                question,
                use_container_width=True,
                key=f"suggest_q_{idx}",
                disabled=workspace["index"] is None,
            ):
                _submit_suggested_question(
                    question,
                    workspace,
                    repo_path,
                    config,
                )

    st.markdown("### Conversation")

    if workspace["index"] is None:
        st.info("Build or load an index first. Once the index is ready, suggested questions and chat input will become available.")
        return

    st.caption("Questions run synchronously in the current page session. For config or model changes, stop and restart Streamlit.")

    _render_messages(workspace["messages"])
    question = st.chat_input("Ask a question about the repository")

    if question:
        _submit_suggested_question(question, workspace, repo_path, config)


def _init_session_state() -> None:
    st.session_state.setdefault("workspaces", {})
    st.session_state.setdefault("workspace_order", [])
    st.session_state.setdefault("active_repo_key", "")
    if "repo_input_value" not in st.session_state:
        default_repo = str((Path(__file__).resolve().parents[2] / "github-efficiency-analyzer").resolve())
        st.session_state["repo_input_value"] = st.session_state["active_repo_key"] or default_repo


def _render_messages(messages: list[dict[str, object]]) -> None:
    if not messages:
        st.caption("No messages yet. Ask a question after the index is ready.")
        return

    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
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


def _submit_suggested_question(
    question: str,
    workspace: dict[str, object],
    repo_path: Path,
    config: AppConfig,
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

    workspace["messages"].append(
        {
            "role": "user",
            "content": question,
        }
    )
    with st.spinner("Thinking..."):
        try:
            result = answer_question(
                index=workspace["index"],
                question=question,
                config=config,
                repo_path=repo_path,
                history=workspace["messages"],
            )
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
