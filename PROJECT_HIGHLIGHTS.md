# AI Codebase Assistant Highlights

## Project framing

AI Codebase Assistant is a local-first repository Q&A tool for code verification, not just generic summarization. It indexes a checked-out codebase, retrieves relevant implementation files, and answers code questions with explicit evidence, source paths, confidence labels, and cross-file relationships.

## What changed in this iteration

- shifted the product from "semantic code search with a chat box" toward a verification-oriented assistant
- added artifact-flow tracing so output questions resolve to concrete entrypoint -> builder -> writer chains
- added question-type routing so flow, location, relationship, and analysis questions no longer share one brittle answer template
- tightened source and evidence filtering so analysis answers prefer implementation files over `README`, tests, or generated outputs
- refined confidence scoring so narrow, verifiable flows score higher while open-ended analysis stays more conservative
- improved VS Code startup flow to use the project virtual environment and check whether Ollama is already running

## Representative capabilities

### Artifact flow

Example question:

- `How is the weekly digest built?`

Expected answer shape:

- `app/main.py` -> `build_weekly_ci_digest()` -> `app/metrics.py`
- `app/main.py` -> `write_weekly_digest_report()` -> `app/report.py`
- `app/report.py` -> writes `outputs/weekly_digest.md`

### Entity location

Example question:

- `Where is the Ollama base URL configured?`

Expected answer shape:

- points to the strongest matching file directly
- keeps sources narrow
- avoids unnecessary call-chain formatting

### Relationship trace

Example question:

- `What calls compute_digest across files?`

Expected answer shape:

- cross-file relationship summary
- conservative confidence because static retrieval can still miss indirect runtime behavior

### Open analysis

Example question:

- `What design risks do you see in this project?`

Expected answer shape:

- repository-grounded interpretation instead of a fake precise trace
- `Why` lines tied to implementation evidence such as aggregation logic, rule tables, fallback branches, or centralized data structures
- conservative confidence and explicit risk note

## Short project-description options

### Option A

Built a local-first AI codebase assistant with LangChain, LlamaIndex, Ollama, and Streamlit that answers repository questions with explicit sources, confidence labels, and cross-file reasoning instead of generic summaries.

### Option B

Improved a repository Q&A assistant by adding verification-oriented answer formatting, artifact-flow tracing, question-type routing, and evidence filtering so code questions resolve into concrete implementation-backed explanations.

### Option C

Developed an AI assistant for local repositories that can trace output-generation flows, locate implementation points, summarize cross-file relationships, and produce cautious design analysis grounded in retrieved code evidence.
