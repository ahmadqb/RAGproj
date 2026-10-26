# Manufacturing Knowledge Assistant (Hybrid RAG + Agent)

A fully local assistant for a manufacturing floor. It answers questions from two different sources and automatically picks the right one:

- **Documentation (RAG):** maintenance manuals, safety procedures and incident reports are chunked, embedded and stored in ChromaDB. Answers are generated only from the retrieved passages, with sources shown.
- **Live machine data (tool-calling agent):** a LangGraph ReAct agent calls a REST API to get machine status, torque readings and error summaries.
- **Router:** an LLM classifier sends each question to one of the two paths.

Everything runs locally with Ollama. No API keys, no data leaves your machine.

## Architecture

```
                     +-------------------+
   user question --> |  Router (LLM)     |
                     +---------+---------+
                               |
              +----------------+----------------+
              |                                 |
         DOCS route                         SQL route*
              |                                 |
   +----------v-----------+         +-----------v-----------+
   |  RAG pipeline        |         |  Data agent (ReAct)   |
   |  ChromaDB retrieval  |         |  tools -> REST API    |
   |  + grounded LLM      |         |                       |
   +----------+-----------+         +-----------+-----------+
              |                                 |
        answer + sources                  answer from tool results
```

\*The route is called `SQL` in the code, but it queries a REST API, not a SQL database.

## Tech stack

- Python, LangChain, LangGraph
- Ollama: `qwen2.5:7b` (RAG answers and routing), `llama3.2` (data agent), `nomic-embed-text` (embeddings)
- ChromaDB (vector store)
- `RecursiveCharacterTextSplitter` (800 chars, 100 overlap), top-k similarity retrieval (k=4)
- Mock manufacturing REST API built with Python's `http.server`

## Project structure

```
RAGproj/
├── docs/                    # knowledge base (.txt files)
│   ├── maintenance_manual.txt
│   ├── safety_procedures.txt
│   └── incident_reports.txt
├── rag_demo.py              # load, chunk, embed, retrieve, answer
├── data_agent.py            # tool-calling agent for live machine data
├── hybrid_router.py         # routes each question to RAG or the agent
├── mock_api.py              # fake manufacturing API (port 8000)
├── test_rag.py              # retrieval, attribution and refusal checks
├── test_router.py           # routing accuracy checks
└── requirements.txt
```

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:

   ```bash
   ollama pull qwen2.5:7b
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

1. **Build the vector database** (first run only, or after changing files in `docs/`). Type `quit` once it's ready:

   ```bash
   python rag_demo.py
   ```

2. **Start the mock API** in a separate terminal:

   ```bash
   python mock_api.py
   ```

3. **Run the hybrid assistant:**

   ```bash
   python hybrid_router.py
   ```

By default the data agent talks to `http://localhost:8000`. Override it with the `API_URL` environment variable.

### Example questions

| Question | Route |
|---|---|
| What does error code E-451 mean? | DOCS |
| What is the lockout/tagout procedure? | DOCS |
| What happened to MCH-03 in September 2024? | DOCS |
| What is the status of MCH-01? | SQL (API) |
| Which machines have errors? | SQL (API) |
| What is the capital of France? | DOCS, refused (not in documentation) |

## Tests

```bash
python test_router.py   # needs Ollama
python test_rag.py      # needs Ollama; rebuilds chroma_db, so close the hybrid app first
```

`test_rag.py` checks that the right source documents are retrieved and that out-of-scope questions are refused. `test_router.py` checks routing on six sample questions.

## Known limitations

- **Single route per question.** A question that needs both documents and live data (for example "MCH-03 shows E-451, what does it mean and what's its current status?") is sent to only one path.
- **Mock data.** The API serves a few hard-coded readings. It has no history, no cycle counts and no downtime durations.
- **The data agent can answer questions the API has no data for.** Questions such as "which machine had the most downtime last month?" may get a confident but unsupported answer, because the small local model doesn't always follow the "only use tool results" instruction. A keyword guard or a larger model would reduce this.
- **Small knowledge base.** With only three short documents, retrieval tests prove less than they would on a large corpus.
- **Refusals rely on the prompt.** Out-of-scope questions are refused by instruction to the LLM, not by a similarity-score threshold.

## Possible improvements

- Run both paths for mixed questions and combine the answers
- Add a similarity threshold so irrelevant questions never reach the LLM
- Add a larger document set and convert the test scripts to `pytest`
- Rename the `SQL` route to `DATA`
