from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding

from app.config import AppConfig, DEFAULT_EXTENSIONS
from app.loaders import IGNORED_DIR_NAMES
from app.loaders import load_codebase_documents

INDEX_STATE_FILE = "index-state.json"


def build_or_load_index(
    repo_path: Path,
    config: AppConfig,
    rebuild: bool = False,
) -> VectorStoreIndex:
    index_dir = config.resolve_index_dir(repo_path)
    _configure_llama_index(config)
    repo_state = _repo_state(repo_path)

    if index_dir.exists() and not rebuild and _stored_repo_state(index_dir) == repo_state:
        storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
        return load_index_from_storage(storage_context)

    documents = load_codebase_documents(repo_path)
    if not documents:
        raise ValueError("No supported files were found in the target repository.")

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(index_dir))
    _write_repo_state(index_dir, repo_state)
    return index


def _configure_llama_index(config: AppConfig) -> None:
    Settings.embed_model = OllamaEmbedding(
        config.embedding_model,
        base_url=config.ollama_base_url,
    )
    Settings.text_splitter = SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )


def _repo_state(repo_path: Path) -> dict[str, object]:
    file_entries: list[dict[str, object]] = []
    for path in sorted(repo_path.rglob("*")):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() not in DEFAULT_EXTENSIONS:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        file_entries.append(
            {
                "path": path.relative_to(repo_path).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    digest = hashlib.sha256(json.dumps(file_entries, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "version": 1,
        "repo_digest": digest,
        "file_count": len(file_entries),
    }


def _state_file_path(index_dir: Path) -> Path:
    return index_dir / INDEX_STATE_FILE


def _stored_repo_state(index_dir: Path) -> dict[str, object] | None:
    state_path = _state_file_path(index_dir)
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_repo_state(index_dir: Path, repo_state: dict[str, object]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    _state_file_path(index_dir).write_text(json.dumps(repo_state, indent=2, sort_keys=True), encoding="utf-8")
