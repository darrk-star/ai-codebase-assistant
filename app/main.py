from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from app.config import AppConfig
from app.indexing import build_or_load_index
from app.qa import answer_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a codebase index and answer repository questions with LangChain + LlamaIndex."
    )
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Path to the local repository to analyze. Default: current directory.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build or rebuild the local codebase index.")
    index_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a rebuild even if an index already exists.",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question about the indexed repository.")
    ask_parser.add_argument(
        "--question",
        help="Question to ask. If omitted, an interactive prompt starts.",
    )
    ask_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a rebuild before answering.",
    )

    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    repo_path = Path(args.repo_path).resolve()
    config = AppConfig.from_env()

    index = build_or_load_index(
        repo_path=repo_path,
        config=config,
        rebuild=getattr(args, "rebuild", False),
    )

    if args.command == "index":
        print(f"Index ready at: {config.resolve_index_dir(repo_path)}")
        return

    if args.question:
        _handle_question(index, args.question, config, repo_path)
        return

    _interactive_loop(index, config, repo_path)


def _handle_question(index, question: str, config: AppConfig, repo_path: Path) -> None:
    result = answer_question(
        index=index,
        question=question,
        config=config,
        repo_path=repo_path,
    )
    print("\nAnswer:\n")
    print(result.answer)
    print("\nSources:")
    for source in result.sources:
        print(f"- {source}")


def _interactive_loop(index, config: AppConfig, repo_path: Path) -> None:
    print("Interactive mode started. Type 'exit' to quit.")
    history: list[dict[str, str]] = []
    while True:
        question = input("\nQuestion> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = answer_question(
            index=index,
            question=question,
            config=config,
            repo_path=repo_path,
            history=history,
        )
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.answer})
        print("\nAnswer:\n")
        print(result.answer)
        print("\nSearch question:")
        print(result.search_question)
        if result.evidence:
            print("\nEvidence:")
            for item in result.evidence:
                print(f"- {item['file_path']} ({item['reason']})")
        print("\nSources:")
        for source in result.sources:
            print(f"- {source}")


if __name__ == "__main__":
    main()
