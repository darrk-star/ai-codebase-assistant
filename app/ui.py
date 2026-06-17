from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app.config import AppConfig
from app.indexing import build_or_load_index
from app.qa import answer_question


load_dotenv()


def main() -> None:
    st.set_page_config(page_title="AI Codebase Assistant", page_icon=":books:", layout="wide")
    st.title("AI Codebase Assistant")
    st.caption("Ask natural language questions about a local repository with LangChain, LlamaIndex, and Ollama.")

    config = AppConfig.from_env()
    _init_session_state()

    default_repo = str((Path(__file__).resolve().parents[2] / "github-efficiency-analyzer").resolve())
    left_col, right_col = st.columns([1, 2], gap="large")
    repo_path = Path(default_repo).resolve()

    with left_col:
        st.subheader("Workspace")
        repo_input = st.text_input("Repository path", value=default_repo)
        repo_path = Path(repo_input).resolve()
        rebuild = st.checkbox("Rebuild index", value=False)

        if st.button("Build / Load Index", use_container_width=True, type="primary"):
            try:
                with st.spinner("Preparing index..."):
                    index = build_or_load_index(repo_path=repo_path, config=config, rebuild=rebuild)
                st.session_state["index"] = index
                st.session_state["repo_path"] = repo_path
                st.success(f"Index ready at: {config.resolve_index_dir(repo_path)}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to prepare index: {exc}")

        st.markdown("### Models")
        st.write(f"Chat: `{config.chat_model}`")
        st.write(f"Embedding: `{config.embedding_model}`")

        st.markdown("### Suggested questions")
        st.code("Which file contains argparse and the main function?")
        st.code("Which file fetches GitHub workflow runs?")
        st.code("Where are CI charts generated?")
        st.code("How is the weekly digest built?")

        if st.button("Clear chat history", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    with right_col:
        st.subheader("Conversation")

        if "index" not in st.session_state:
            st.info("Build or load an index first.")
            return

        _render_messages()
        question = st.chat_input("Ask a question about the repository")

        if question:
            st.session_state["messages"].append(
                {
                    "role": "user",
                    "content": question,
                }
            )
            with st.spinner("Thinking..."):
                try:
                    result = answer_question(
                        index=st.session_state["index"],
                        question=question,
                        config=config,
                        repo_path=st.session_state["repo_path"],
                    )
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": result.sources,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": f"Failed to answer question: {exc}",
                            "sources": [],
                        }
                    )
            st.rerun()


def _init_session_state() -> None:
    st.session_state.setdefault("messages", [])


def _render_messages() -> None:
    if not st.session_state["messages"]:
        st.caption("No messages yet. Ask a question after the index is ready.")
        return

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            sources = message.get("sources") or []
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.code(source)


if __name__ == "__main__":
    main()
