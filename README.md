# AI Codebase Assistant

AI Codebase Assistant is a local-first repository Q&A tool that helps users understand a codebase with source-grounded answers.

It indexes a local repository, retrieves relevant implementation files, and answers natural-language questions with explicit file citations, confidence labels, evidence panels, and cross-file relationship summaries.

The project is built with LangChain, LlamaIndex, Ollama, and Streamlit, and is designed for practical repository inspection rather than generic AI summarization.

## What this project does

This assistant helps users ask questions about a local codebase and get answers that are easier to verify in real files.

It supports tasks like:

- finding where a function, config, or entrypoint is defined
- tracing how data or logic moves across multiple files
- explaining how a report, chart, or output file is generated
- giving cautious design-risk analysis based on retrieved implementation evidence

## Main features

### 1. Source-grounded Q&A

The assistant answers repository questions with cited source files instead of only giving a free-form summary.

Example questions:

- `Which file contains argparse and the main function?`
- `Where is the Ollama base URL configured?`
- `How is the index built and persisted?`

### 2. Cross-file tracing

It can trace relationships across files when a question depends on definitions, imports, callers, or output writers.

Example questions:

- `What calls compute_digest across files?`
- `Which file fetches GitHub workflow runs and where are they summarized?`

### 3. Artifact-flow explanation

It can explain how a specific artifact is produced by following builder and writer logic.

Example questions:

- `How is the weekly digest built?`
- `Where are CI charts generated?`

### 4. Evidence-aware analysis

For broader design questions, the assistant gives a more conservative answer and tries to ground its reasoning in implementation evidence.

Example questions:

- `What design risks do you see in this project?`
- `What should be optimized next?`

### 5. Streamlit UI

The project includes a local Streamlit interface for interactive use.

The UI supports:

- repository path input
- build/load index flow
- saved workspaces
- suggested questions
- conversation history
- evidence and source panels
- confidence display
- cross-file relationship display

## Recent improvements

The latest version improves the project in several ways:

- added clearer answer formatting with `Answer`, `Why`, and `Sources`
- added confidence labels and evidence panels
- improved artifact-flow tracing for report and output questions
- improved cross-file relationship summaries
- added question-type routing for different kinds of questions
- reduced noisy sources in open-ended analysis answers
- improved VS Code startup flow with virtual environment support and Ollama checks

## Example improvement

For a question like:

```text
How is the weekly digest built?
```

the assistant can now produce a more concrete flow such as:

```text
app/main.py -> build_weekly_ci_digest() -> app/metrics.py
app/main.py -> write_weekly_digest_report() -> app/report.py
app/report.py -> writes outputs/weekly_digest.md
```

This makes it easier for a user to open the files and verify the answer manually.

## Project structure

```text
ai-codebase-assistant/
  app/
    config.py
    indexing.py
    loaders.py
    main.py
    qa.py
    ui.py
  tests/
    test_integration.py
    test_main.py
    test_qa.py
    test_ui.py
  assets/
    ui_preview.png
  .vscode/
    launch.json
    tasks.json
  .env.example
  requirements.txt
  README.md
```

## Tech stack

- Python 3.11
- LangChain
- LlamaIndex
- Ollama
- Streamlit
- pytest

Default local models:

- chat model: `qwen2.5:7b`
- embedding model: `nomic-embed-text`

## How it works

The assistant works in four main steps:

1. load files from a local repository
2. build or load a vector index
3. retrieve and rerank relevant snippets for a question
4. generate a structured answer with sources and evidence

Important modules:

- `app/loaders.py`: loads repository files into documents
- `app/indexing.py`: builds or loads the vector index
- `app/qa.py`: handles retrieval, reranking, answer formatting, and evidence logic
- `app/main.py`: CLI entrypoint
- `app/ui.py`: Streamlit UI

## Local setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create the environment file

```powershell
Copy-Item .env.example .env
```

## Ollama setup

Pull the required models:

```powershell
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Start Ollama if it is not already running:

```powershell
ollama serve
```

Example `.env` values:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen2.5:7b
OLLAMA_EMBED_MODEL=nomic-embed-text
INDEX_DIR_NAME=.storage
CHUNK_SIZE=1200
CHUNK_OVERLAP=150
TOP_K=8
```

## CLI usage

Build an index:

```powershell
python -m app.main --repo-path C:\path\to\repo index
```

Force rebuild:

```powershell
python -m app.main --repo-path C:\path\to\repo index --rebuild
```

Ask one question:

```powershell
python -m app.main --repo-path C:\path\to\repo ask --question "Which file contains argparse and the main function?"
```

Start interactive CLI mode:

```powershell
python -m app.main --repo-path C:\path\to\repo ask
```

## Streamlit usage

Start the UI:

```powershell
python -m streamlit run app\ui.py
```

Then open:

```text
http://localhost:8501
```

Typical workflow:

1. enter a repository path
2. click `Build / Load Index`
3. ask a question
4. review the answer, sources, and evidence
5. open cited files for manual verification

## VS Code usage

This project includes VS Code launch files for easier startup.

Files included:

- `.vscode/launch.json`
- `.vscode/tasks.json`

Recommended flow:

1. open the project folder in VS Code
2. select the project Python interpreter
3. open `Run and Debug`
4. choose `Run Streamlit UI`
5. press `F5`

## Example questions

- `Which file contains argparse and the main function?`
- `Where is the Ollama base URL configured?`
- `How is the index built and persisted?`
- `Which file fetches GitHub workflow runs?`
- `Where are CI charts generated?`
- `How is the weekly digest built?`
- `What calls compute_digest across files?`
- `What design risks do you see in this project?`

## Example answer format

```text
Answer: The main CLI entrypoint is in app/main.py.

Why:
- app/main.py imports argparse.
- app/main.py defines def main() and the __main__ entry flow.

Sources:
- app/main.py
```

## Testing

Run tests with:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_main.py tests\test_qa.py tests\test_ui.py -q
```

## Known limitations

- answers depend on retrieved snippets, not runtime execution
- multi-file tracing is still based on static relationships
- answer quality depends on the local Ollama model in use
- Streamlit state can be lost after reload in some cases

## Future improvements

- stronger AST-based cross-file reasoning
- better filtering by directory and file type
- more stable workspace persistence in the UI
- broader retrieval and answer-formatting tests

## GitHub description

Suggested short GitHub description:

```text
Local-first codebase analysis assistant with source-grounded Q&A, cross-file tracing, artifact-flow explanation, and evidence-backed confidence signals.
```

Suggested GitHub topics:

```text
python
streamlit
rag
ollama
langchain
llamaindex
code-search
developer-tools
repository-analysis
question-answering
```
