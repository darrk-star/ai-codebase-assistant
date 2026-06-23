# AI Codebase Assistant

A local-first codebase Q&A assistant built with LangChain, LlamaIndex, Ollama, and Streamlit.

It indexes a local repository, retrieves relevant code and docs, and answers natural-language questions with explicit file citations.

It is tuned for practical repository inspection tasks where a user needs to verify whether an answer is actually grounded in the checked-out codebase.

## Highlights

- built a local-first codebase assistant that combines LlamaIndex retrieval, LangChain prompting, Ollama models, and a Streamlit review UI
- upgraded the answer pipeline from generic semantic Q&A into a verification-oriented workflow with explicit `Answer`, `Why`, `Sources`, confidence labels, and evidence panels
- added cross-file relationship summaries so artifact questions such as `How is the weekly digest built?` resolve into concrete code paths instead of vague summaries
- introduced question-type routing for `artifact_flow`, `entity_location`, `relationship_trace`, and `open_analysis`, so different question styles now produce different answer shapes
- tightened source selection and evidence ranking so analysis questions prefer implementation files over `README`, `tests`, and generated outputs
- expanded regression coverage around retrieval, answer formatting, UI behavior, and confidence heuristics

## Example improvements

Representative improvements from the current iteration:

- `How is the weekly digest built?`
  now resolves to a compact chain like `app/main.py -> build_weekly_ci_digest() -> app/metrics.py`, `app/main.py -> write_weekly_digest_report() -> app/report.py`, and `app/report.py -> writes outputs/weekly_digest.md`
- `What design risks do you see in this project?`
  now uses implementation-grounded evidence such as aggregation logic in `app/metrics.py` and hard-coded failure classification rules in `app/ci_failure_analysis.py`
- VS Code launch flow now prefers the project virtual environment and includes an Ollama prelaunch check for a more reliable one-click startup path

## What the project does

- indexes source files and project docs from a local repository
- stores a persisted vector index under the target repository
- answers repository questions in CLI or Streamlit UI
- combines vector retrieval with code-aware reranking and keyword context injection
- shows supporting evidence, rewritten search questions, confidence labels, cited source paths, and cross-file relationship summaries

This project is aimed at practical repository understanding tasks such as:

- finding the CLI entrypoint
- locating where configuration is loaded
- tracing where indexing is built and persisted
- identifying where reports, charts, or workflows are generated

## Current behavior

The retrieval pipeline is not pure semantic search. It includes:

- question rewriting for follow-up questions
- heuristic reranking by file path, identifiers, and intent
- keyword-based context injection for code-oriented questions
- intent-specific boosts for entrypoint, configuration, indexing, workflow, reporting, and test questions
- extra scoring for identifier definitions, call sites, and import relationships in cross-file questions
- generated chain-style summaries such as `file A -> function() -> file B` for call-chain style questions
- output-aware flow summaries for artifact questions such as `How is the weekly digest built?`
- answer post-processing so the final output consistently includes `Answer`, `Why`, and `Sources`

The Streamlit UI currently includes:

- repository path input and validation
- build/load index workflow
- saved workspaces
- suggested questions
- conversation history
- a dedicated cross-file relationship panel for call-chain style questions
- evidence and source expanders
- protection against asking questions before an index is ready

## Verification-oriented workflow

This assistant is designed to help you verify codebase claims instead of only producing summaries.

For a question like `How is the weekly digest built?`, the current flow aims to:

- expand semantic aliases such as `weekly digest` into repository-specific identifiers like `build_weekly_ci_digest`, `write_weekly_digest_report`, and `weekly_digest.md`
- prioritize files that define, call, or write those identifiers
- filter out low-signal helpers such as `parse_args`, `from_env`, `fetch_*`, `summarize_*`, and test-only references
- surface a concise relationship chain and a tight `Sources` list that the user can manually open and verify

The intended verification loop is:

1. Ask a concrete code-path question.
2. Inspect the `Cross-file relationships` panel.
3. Open the cited files and confirm the function calls or output writes.
4. Treat lower-confidence answers as hints rather than final truth.

## Supported question styles

The assistant now uses a small question-type router before formatting the final answer. In practice, this means different questions are allowed to produce different answer shapes instead of forcing everything into one template.

### 1. Artifact flow questions

Examples:

- `How is the weekly digest built?`
- `How is the summary report built?`
- `Where are CI charts generated?`

Expected behavior:

- prioritizes entrypoint -> builder -> writer -> output style chains
- surfaces concise cross-file relationships
- keeps `Sources` narrow and verification-friendly
- tends to score higher confidence when the chain and output file are both explicit

### 2. Entity location questions

Examples:

- `Where is the Ollama base URL configured?`
- `Which file contains argparse and the main function?`
- `Where is compute_digest defined?`

Expected behavior:

- answers with the strongest matching file first
- uses direct file-level evidence instead of long call chains
- usually gives higher confidence when the match is concentrated in one or two files

### 3. Relationship trace questions

Examples:

- `What calls compute_digest across files?`
- `Which file fetches GitHub workflow runs and where are they summarized?`
- `How does config get loaded before indexing starts?`

Expected behavior:

- focuses on cross-file call sites, imports, and definitions
- uses relationship summaries when the repository structure is explicit enough
- stays more conservative than artifact-flow questions because static retrieval can miss indirect runtime behavior

### 4. Open analysis questions

Examples:

- `What design risks do you see in this project?`
- `What should be optimized next?`
- `Is this architecture too tightly coupled?`

Expected behavior:

- keeps the answer analytic rather than pretending there is one exact code path
- attaches `Why` lines and `Sources` so the judgment stays repository-grounded
- uses more conservative confidence and risk notes because this is interpretation, not execution proof

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
  .env.example
  requirements.txt
  README.md
```

## Architecture

- `app/loaders.py`
  - scans repository files and converts them into LlamaIndex `Document` objects
- `app/indexing.py`
  - configures embeddings and chunking, then builds or loads the persisted vector index
- `app/qa.py`
  - rewrites search questions, reranks retrieved nodes, injects keyword matches, and formats final answers
- `app/main.py`
  - CLI entrypoint for `index` and `ask`
- `app/ui.py`
  - Streamlit UI for building/loading the index and asking questions interactively

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

## Local setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Install and start Ollama:

```powershell
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
ollama serve
```

Default `.env` values:

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

Build an index for a target repository:

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

Start interactive CLI mode:

```powershell
python -m app.main --repo-path C:\path\to\repo ask
```

## Streamlit usage

Start the UI:

```powershell
python -m streamlit run app\ui.py
```

VS Code one-click run:

- open the project folder in VS Code
- select the `.venv` interpreter if prompted
- open `Run and Debug`
- choose `Run Streamlit UI`
- press `F5` or click the green run button

This project now includes:

- `.vscode/launch.json` for one-click debug launch
- `.vscode/tasks.json` for one-click task-based launch

Typical flow:

1. Enter a repository path.
2. Click `Build / Load Index`.
3. Ask a question or use one of the suggested prompts.
4. Inspect the answer, evidence, rewritten search question, and cited sources.

If the current workspace does not have an index loaded, the app now blocks the request and tells you to build/load the index first.

## Example questions

- `Which file contains argparse and the main function?`
- `Where is the Ollama base URL configured?`
- `How is the index built and persisted?`
- `Which file fetches GitHub workflow runs?`
- `Where are CI charts generated?`
- `How is the weekly digest built?`

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

Run the focused test suite with the project virtual environment:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_main.py tests\test_qa.py tests\test_ui.py -q
```

Current tests cover:

- small-repository integration paths
- CLI entry behavior
- index-required guardrails
- entrypoint/config/indexing retrieval rules
- cross-file call-site and definition matching
- reporting-flow and artifact-output chain generation
- UI protection when no index is loaded
- final answer formatting and source inclusion

## Known limitations

- answers still depend on retrieved snippets, not runtime execution
- multi-file call-chain questions still rely on static relationships rather than full AST or runtime tracing
- Streamlit requests run synchronously in the current page session
- the in-memory workspace index can be lost across reloads, even if the persisted `.storage` directory still exists
- answer quality still depends on the local Ollama model you run

## Next improvements

- add stronger AST-based reasoning for cross-file and artifact-output questions
- support filtering by directory or file type
- persist workspace metadata more explicitly across UI reloads
- add broader tests around retrieval scoring and answer formatting

