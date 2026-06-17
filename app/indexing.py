from __future__ import annotations

from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding

from app.config import AppConfig
from app.loaders import load_codebase_documents


def build_or_load_index(
    repo_path: Path,
    config: AppConfig,
    rebuild: bool = False,
) -> VectorStoreIndex:
    index_dir = config.resolve_index_dir(repo_path)
    _configure_llama_index(config)

    if index_dir.exists() and not rebuild:
        storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
        return load_index_from_storage(storage_context)

    documents = load_codebase_documents(repo_path)
    if not documents:
        raise ValueError("No supported files were found in the target repository.")

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(index_dir))
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
