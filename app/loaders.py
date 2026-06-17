from __future__ import annotations

from pathlib import Path

from llama_index.core.schema import Document

from app.config import DEFAULT_EXTENSIONS


IGNORED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".storage",
    "node_modules",
    "dist",
    "build",
}


def load_codebase_documents(
    repo_path: Path,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> list[Document]:
    documents: list[Document] = []

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

        relative_path = path.relative_to(repo_path).as_posix()
        documents.append(
            Document(
                text=text,
                metadata={
                    "file_path": relative_path,
                    "extension": path.suffix.lower(),
                },
            )
        )

    return documents
