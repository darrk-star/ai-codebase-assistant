from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXTENSIONS = (
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
)


@dataclass(frozen=True)
class AppConfig:
    ollama_base_url: str
    chat_model: str
    embedding_model: str
    index_dir_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b"),
            embedding_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            index_dir_name=os.getenv("INDEX_DIR_NAME", ".storage"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            top_k=int(os.getenv("TOP_K", "8")),
        )

    def resolve_index_dir(self, repo_path: Path) -> Path:
        return repo_path / self.index_dir_name
