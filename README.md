# BJJ Agent

**An AI agent that answers Brazilian Jiu-Jitsu questions — techniques, defenses, escapes and competition stats — with grounded, verified answers.**

Ask it *"How do I escape an armlock from closed guard?"* and it reasons about the question, retrieves the relevant technique knowledge, checks its own answer against the evidence, and can point you to a video if you want to see it in action.

> **Status:** feature-complete (Nov 2026). All five phases, the CLI and the Streamlit interface are done. See the [Roadmap](#roadmap) and the [technical reports](docs/).

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Data Sources](#data-sources)
- [Disclaimer](#disclaimer)
- [Author](#author)

---

## Features

| Capability | What it does |
| --- | --- |
| **ReAct reasoning** | Decides step by step whether to answer directly or call a tool. |
| **Technique knowledge base (RAG)** | Semantic search over curated BJJ technique sheets using FAISS. |
| **Competition stats (Text-to-SQL)** | Turns natural-language questions into SQL over historical ADCC match data. |
| **IBJJF scoring** | Exact point values from a hardcoded rules table (no hallucinated numbers). |
| **Video lookup** | Finds a relevant instructional video through the YouTube Data API. |
| **Encyclopedic context** | Pulls background on techniques and athletes from the Wikipedia API. |
| **Conversational memory** | Keeps context across turns — follow-up questions just work. |
| **Answer verification** | An NLI model checks each answer against the retrieved evidence to catch hallucinations. |
| **Systematic evaluation** | Response quality and tool-call trajectories evaluated with LangSmith. |

---

## How It Works

```
                ┌─────────────────────────────┐
  User  ──────▶ │        Agent Executor       │ ◀── Conversation memory
                │   (reasoning + tool choice) │
                └──────────────┬──────────────┘
                               │
     ┌──────────────┬──────────┼──────────────┬──────────────┐
     ▼              ▼          ▼              ▼              ▼
 Technique RAG   ADCC SQL   YouTube API   Wikipedia API   IBJJF points
   (FAISS)      (SQLite)                                  (hardcoded)
                               │
                               ▼
                 ┌─────────────────────────────┐
                 │     Answer Verification     │
                 │     (NLI + logic check)     │
                 └──────────────┬──────────────┘
                                ▼
                         Final answer
                                │
                     CLI (cli.py) · Streamlit (app.py)
```

The final agent is a `create_openai_tools_agent` + `AgentExecutor` with 5 tools in 4 categories (hardcoded, semantic/RAG, external API, text-to-SQL), wrapped in `RunnableWithMessageHistory` (one history per session). Every answer goes through an NLI verifier: if it contradicts the retrieved technique sheets, it is revised once; if it is weakly supported or has no source, the student is warned.

The project evolves in five phases, from a simple structured chain to a fully integrated, evaluated agent:

1. **MVP chain** — reusable prompt templates and structured JSON output.
2. **ReAct agent** — tools, external API integration, advanced prompting and reliability components.
3. **Memory** — short- and long-term memory for multi-turn conversations.
4. **External tools & RAG** — text-to-SQL, retrieval over an existing FAISS index and query transformation.
5. **Integrated agent** — answer verification, single executor and LangSmith evaluation.

---

## Tech Stack

- **Language:** Python 3.13
- **Orchestration:** [LangChain](https://www.langchain.com/) 1.4 (`langchain-classic` for the legacy APIs required by the course: `initialize_agent`, `ConversationBufferMemory`, `MultiPromptChain`, `create_openai_tools_agent`) and LangGraph (`create_agent`)
- **LLM:** `openai/gpt-oss-120b` via [Groq](https://groq.com/) (free tier); `gpt-oss-20b` for text-to-SQL and the eval judge; `qwen/qwen3.8-27b` only for the legacy text ReAct agent
- **Embeddings & NLI:** `paraphrase-multilingual-MiniLM-L12-v2` and `mDeBERTa-v3-base-xnli-multilingual` (Hugging Face), running locally on CPU
- **Vector store:** FAISS
- **Database:** SQLite
- **Observability & evaluation:** LangSmith
- **Interfaces:** CLI and Streamlit

> The entire project runs at **zero cost** — no paid APIs required.

---

## Project Structure

```
bjj-agent/
├── notebooks/
│   ├── 00_setup.ipynb                # Connectivity tests
│   ├── 01_mvp_chain.ipynb            # Phase 1: prompts, parsers, @chain vs LCEL
│   ├── 02_react_agent.ipynb          # Phase 2: tools, function calling, initialize_agent ReAct
│   ├── 02b_chains_reliability.ipynb  # Phase 2: chains, router, PTCF, parsers, memory, FAISS, callbacks
│   ├── 03_memory.ipynb               # Phase 3: memory types, intent + entities, multi-turn test
│   ├── 04_tools_rag.ipynb            # Phase 4: tool typology, text-to-SQL, retrieval agent, multi-query
│   └── 05_agent_eval.ipynb           # Phase 5: NLI verifier, final agent, memory, LangSmith eval
├── src/
│   ├── tools.py          # All tools (+ USE_DUMMY responses)
│   ├── chains.py         # LLM factory, DummyChatModel, Mestre prompt/schema, intent chain
│   ├── text_to_sql.py    # SQL generation, validation and answer chain
│   ├── verifier.py       # NLI faithfulness + logical coherence checks
│   └── agent.py          # Final AgentExecutor, verification, session memory
├── app.py            # Streamlit interface
├── cli.py            # Terminal interface
├── data/
│   ├── raw/          # Source datasets (e.g. ADCC matches CSV)
│   ├── kb/           # Curated technique knowledge base (Markdown)
│   ├── db/           # SQLite database
│   └── index/        # FAISS vector index
├── docs/             # Architecture, orchestrator comparison, one report per phase, evidence
├── .env.example      # Required environment variables (no secrets)
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.13 (the pinned `requirements.txt` was tested on 3.13 / Windows 11)
- Free accounts on [Groq](https://console.groq.com/), [LangSmith](https://smith.langchain.com/) and [Google Cloud](https://console.cloud.google.com/) (YouTube Data API v3)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/GiovanniAsantos/bjj-agent.git
cd bjj-agent

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
source .venv/bin/activate         # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your keys:

```env
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=bjj-agent
YOUTUBE_API_KEY=
USE_DUMMY=false
```

> **No API keys?** Set `USE_DUMMY=true`. Every tool returns a fixed `(dummy)` response and `build_llm()` returns a `DummyChatModel` that simulates the tool-calling loop, so the final agent, the NLI verifier, the session memory, the CLI and the Streamlit app all run without any key. Internet is only needed once, to download the local Hugging Face models (embeddings and NLI).
>
> The notebooks of phases 1–4 call the Groq LLM directly and are meant to be read with their saved outputs; in dummy mode, use `05_agent_eval.ipynb` (last cell), `cli.py` or `app.py`.

---

## Usage

Run the notebooks in order (`notebooks/00_setup.ipynb` onward), or talk to the final agent directly from the `bjj-agent/` folder:

```bash
python cli.py               # Terminal chat ('nova' = new conversation, 'sair' = quit)
python -m streamlit run app.py   # Web interface at http://localhost:8501
```

The first start takes ~30 s (FAISS + NLI model load). Each answer shows the tools used, their categories, the verifier verdict (`ok`, `pouco_sustentada`, `contradiz_fonte`, `incoerente`, `sem_fonte`), the NLI support rate and whether it was revised. Each browser tab is a separate session with its own memory.

> **Groq free tier:** 8,000 tokens/min and 200k tokens/day for `gpt-oss-120b`. A `429` is shown as an error message without crashing the interface; wait a minute (or until the next day for the daily limit).

**Example conversation:**

```
You:   How do I escape an armlock from closed guard?
Agent: [search_technique] Step-by-step escape, citing (fonte: armlock_closed_guard.md)
       [verifier: ok]

You:   And if he switches to a triangle?
Agent: [remembers you are still inside closed guard with an arm trapped]
       [search_technique] Triangle defense from that same position...

You:   Is the armlock common at ADCC?
Agent: [query_adcc] Count of armlock wins, with the SQL used...

You:   Show me a video of that
Agent: [search_video] YouTube link for the armlock defense
```

---

## Roadmap

- [x] Environment setup and connectivity tests
- [x] **Phase 1** — MVP chain with structured output (imperative vs. declarative composition) — [report](docs/03_relatorio_fase1.md)
- [x] **Phase 2** — ReAct agent, external APIs, PTCF prompting and reliability components — [report](docs/04_relatorio_fase2.md)
- [x] **Phase 3** — Short- and long-term memory, intent recognition and entity extraction — [report](docs/05_relatorio_fase3.md)
- [x] **Phase 4** — Text-to-SQL, retrieval agent and query transformation — [report](docs/06_relatorio_fase4.md)
- [x] **Phase 5** — Answer verification, integrated executor and LangSmith evaluation — [report](docs/07_relatorio_fase5.md)
- [x] CLI and Streamlit interfaces

**Results (LangSmith, dataset `bjj-mestre-eval`, 8 questions):** tool recall 1.00 · tool precision 0.88 · answer quality 0.84 · NLI-verified 0.62. Details and analysis in the [Phase 5 report](docs/07_relatorio_fase5.md).

**Future work:** verify Wikipedia answers with NLI too, an aliases section in the technique sheets, persistent chat history (SQLite), a larger eval dataset, and migrating the final agent to `create_agent` + LangGraph checkpointer.

---

## Data Sources

- **ADCC match history (1998–2022)** — public dataset on [Kaggle](https://www.kaggle.com/datasets/bjagrelli/adcc-historical-dataset), originally scraped from [BJJ Heroes](https://www.bjjheroes.com/).
- **Technique knowledge base** — curated technique sheets, AI-assisted and reviewed by a BJJ purple belt.
- **Wikipedia** — encyclopedic content under CC BY-SA.
- **YouTube Data API v3** — video search results.

---

## Disclaimer

This project is for **educational purposes only**. It does not replace guidance from a qualified instructor. Always train under supervision and tap early.

---

## Author

**Giovanni Azevedo** — Fullstack Developer · BJJ purple belt

Developed as the course project for *AI Agents with LangChain* — MBA in Artificial Intelligence, Machine Learning & Deep Learning at **Infnet**.