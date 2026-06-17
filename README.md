# AI Codebase Assistant

A local codebase Q&A assistant built with LangChain, LlamaIndex, and Ollama.

This project is designed for a common engineering problem:
- a code repository is too large for new contributors to understand quickly
- source code, docs, and configuration are spread across many files
- keyword search is often not enough to answer structural questions like:
  - Where is the CLI entrypoint?
  - Which file fetches GitHub workflow runs?
  - How is the weekly digest generated?

This assistant indexes a local repository, retrieves relevant code and documents, and answers natural language questions with cited source files.

## Why this project

I wanted a project that is more practical than a generic chatbot demo. Instead of building a plain chat UI, this project focuses on repository understanding:
- codebase onboarding
- developer productivity
- engineering knowledge retrieval
- local AI application development

It also shows how to combine retrieval-augmented generation with code-aware heuristics, which is much more realistic than relying on vector similarity alone.

## What it does

- scans a local repository and loads source code and docs
- builds a local vector index with LlamaIndex
- retrieves relevant context for a natural language question
- uses LangChain to assemble prompts and generate answers
- runs fully locally with Ollama, so no external API key is required
- returns cited file paths for traceability

## Real project value

This is not only a document search demo. It is built around real engineering questions:
- locating entrypoints
- finding implementation files
- understanding how modules interact
- identifying where reports, workflows, or charts are generated

The project also includes a retrieval improvement layer:
- vector retrieval for broad semantic recall
- heuristic reranking for code-specific queries
- keyword-driven context injection for patterns such as `argparse`, `def main`, and `__main__`

That extra layer matters because pure vector retrieval often misses the right file for code structure questions.

## Architecture

- `app/loaders.py`
  - scans repository files and converts them into LlamaIndex `Document` objects
- `app/indexing.py`
  - builds or loads the persisted vector index
- `app/qa.py`
  - retrieves, reranks, injects keyword-matched code snippets, and generates answers
- `app/main.py`
  - CLI entrypoint with `index` and `ask` commands

## Why use both LangChain and LlamaIndex

The two frameworks are used for different responsibilities:

- LlamaIndex
  - file ingestion
  - document chunking
  - vector indexing
  - base retrieval

- LangChain
  - prompt template management
  - local LLM invocation through Ollama
  - final answer generation

This separation makes the project easier to reason about and extend.

## Tech stack

- Python 3.11
- LangChain
- LlamaIndex
- Ollama
- local embedding model: `nomic-embed-text`
- local chat model: `qwen2.5:7b`

## Project structure

```text
ai-codebase-assistant/
  app/
    config.py
    indexing.py
    loaders.py
    main.py
    qa.py
  .env.example
  .gitignore
  requirements.txt
  README.md
```

## Local setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Install and prepare Ollama:

```powershell
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
ollama serve
```

Default `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen2.5:7b
OLLAMA_EMBED_MODEL=nomic-embed-text
INDEX_DIR_NAME=.storage
CHUNK_SIZE=1200
CHUNK_OVERLAP=150
TOP_K=8
```

## Usage

Build an index for a repository:

```powershell
python -m app.main --repo-path C:\path\to\repo index
```

Force a rebuild:

```powershell
python -m app.main --repo-path C:\path\to\repo index --rebuild
```

Ask one question:

```powershell
python -m app.main --repo-path C:\path\to\repo ask --question "Which file contains argparse and the main function?"
```

Start interactive mode:

```powershell
python -m app.main --repo-path C:\path\to\repo ask
```

Start the Streamlit UI:

```powershell
streamlit run app/ui.py
```

## Example questions

- Which file contains argparse and the main function?
- Which file fetches GitHub workflow runs?
- Where are CI charts generated?
- How is the weekly digest built?
- How do I start this project locally?

## Example result

When asked:

```text
Which file contains argparse and the main function?
```

The assistant correctly identified:

```text
app/main.py
```

and surfaced supporting evidence such as:
- `import argparse`
- `def main() -> None`

This is important because it demonstrates that the system is not only returning approximate matches; it can be tuned toward code-structure questions with better retrieval logic.

## Engineering highlights

- local-first AI workflow with no external API key requirement
- persisted repository index for repeated querying
- source citation in answers
- retrieval tuning for code-oriented questions
- minimal web UI with Streamlit
- clear modular separation between loading, indexing, retrieval, and generation

## Good next extensions

- support GitHub Issues and PR comments as extra context
- add repository filters by directory or file type
- support multiple repositories
- add follow-up question memory beyond single-session UI state

## Resume-ready description

Built a local codebase Q&A assistant with LangChain, LlamaIndex, and Ollama that indexes source code and documentation, retrieves relevant repository context, and answers natural language questions with cited file paths. Improved retrieval quality for code-structure questions by combining vector search with heuristic reranking and keyword-based context injection.
