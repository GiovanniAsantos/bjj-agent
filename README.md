# BJJ Agent

**An AI agent that answers Brazilian Jiu-Jitsu questions — techniques, defenses, escapes and competition stats — with grounded, verified answers.**

Ask it *"How do I escape an armlock from closed guard?"* and it reasons about the question, retrieves the relevant technique knowledge, checks its own answer against the evidence, and can point you to a video if you want to see it in action.

> **Status:** in active development (Oct–Nov 2026). See the [Roadmap](#roadmap) for progress.

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
 Technique RAG   ADCC SQL   YouTube API   Wikipedia API   (more tools)
   (FAISS)      (SQLite)
                               │
                               ▼
                 ┌─────────────────────────────┐
                 │     Answer Verification     │
                 │     (NLI + logic check)     │
                 └──────────────┬──────────────┘
                                ▼
                         Final answer
```

The project evolves in five phases, from a simple structured chain to a fully integrated, evaluated agent:

1. **MVP chain** — reusable prompt templates and structured JSON output.
2. **ReAct agent** — tools, external API integration, advanced prompting and reliability components.
3. **Memory** — short- and long-term memory for multi-turn conversations.
4. **External tools & RAG** — text-to-SQL, retrieval over an existing FAISS index and query transformation.
5. **Integrated agent** — answer verification, single executor and LangSmith evaluation.

---

## Tech Stack

- **Language:** Python 3.12
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **LLM:** `openai/gpt-oss-120b` via [Groq](https://groq.com/) (free tier)
- **Embeddings & NLI:** Hugging Face models, running locally on CPU
- **Vector store:** FAISS
- **Database:** SQLite
- **Observability & evaluation:** LangSmith
- **Interfaces:** CLI and Streamlit

> The entire project runs at **zero cost** — no paid APIs required.

---

## Project Structure

```
bjj-agent/
├── notebooks/        # One notebook per phase, from setup to the final agent
├── src/              # Shared code: configuration, tools, chains
├── data/
│   ├── raw/          # Source datasets (e.g. ADCC matches CSV)
│   ├── kb/           # Curated technique knowledge base (Markdown)
│   ├── db/           # SQLite database
│   └── index/        # FAISS vector index
├── docs/             # Technical report and evidence
├── .env.example      # Required environment variables (no secrets)
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.12
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

> **No API keys?** Set `USE_DUMMY=true` to run the full pipeline with mocked LLM and API responses.

### Running on Google Colab

The notebooks also run on Colab. Clone the repository, install the requirements and add your keys through **Colab Secrets** — the configuration detects the environment automatically.

---

## Usage

Run the notebooks in order (`notebooks/00_setup.ipynb` onward), or talk to the final agent directly:

```bash
python chat_cli.py          # Terminal chat
streamlit run app.py        # Web interface
```

**Example conversation:**

```
You:   How do I escape an armlock from closed guard?
Agent: [searches the technique knowledge base]
       Step-by-step escape, common mistakes and key details...

You:   And if he switches to a triangle?
Agent: [remembers you are still inside closed guard with an arm trapped]
       Triangle defense from that same position...

You:   Is the armlock common at ADCC?
Agent: [queries ADCC match data]
       Submission statistics by year and weight class...
```

---

## Roadmap

- [x] Environment setup and connectivity tests
- [ ] **Phase 1** — MVP chain with structured output (imperative vs. declarative composition)
- [ ] **Phase 2** — ReAct agent, external APIs, PTCF prompting and reliability components
- [ ] **Phase 3** — Short- and long-term memory, intent recognition and entity extraction
- [ ] **Phase 4** — Text-to-SQL, retrieval agent and query transformation
- [ ] **Phase 5** — Answer verification, integrated executor and LangSmith evaluation
- [ ] CLI and Streamlit interfaces

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