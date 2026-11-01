import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_core.tools import tool

USE_DUMMY = os.getenv("USE_DUMMY", "false").lower() == "true"
WIKI_HEADERS = {"User-Agent": "bjj-agent/0.1 (Infnet MBA study project)"}

# retry só p/ falhas transitórias (rate limit / servidor), com backoff 0.5s, 1s
_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
    ),
)

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index" / "techniques"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ---------- YouTube ----------


@lru_cache(maxsize=128)
def _youtube_lookup(query: str) -> str:
    # só HTTP; erro sobe como exceção (não entra no cache)
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": 1,
        "key": os.getenv("YOUTUBE_API_KEY"),
    }
    r = _session.get(
        "https://www.googleapis.com/youtube/v3/search", params=params, timeout=10
    )
    r.raise_for_status()

    items = r.json().get("items", [])
    if not items:
        return "No video found."
    item = items[0]
    title = item["snippet"]["title"]
    link = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
    return f"{title} — {link}"


@tool
def search_video(query: str) -> str:
    """Search YouTube for a BJJ instructional video.
    Use ONLY when the student wants to see the technique in a video.
    Input: short English keywords (technique + position + bjj), e.g. 'kimura closed guard bjj'.
    Returns 'title — link' of the first result."""
    if USE_DUMMY:
        return "BJJ Mount Escape (dummy) — https://www.youtube.com/watch?v=dummy"
    try:
        return _youtube_lookup(query.lower().strip())
    except Exception as e:
        return f"YouTube API error: {e}"


# ---------- Wikipedia ----------


@lru_cache(maxsize=128)
def _wiki_lookup(query: str) -> str:
    # 1) busca: termo -> chave do artigo
    r = _session.get(
        "https://en.wikipedia.org/w/rest.php/v1/search/page",
        params={"q": query, "limit": 1},
        headers=WIKI_HEADERS,
        timeout=10,
    )
    r.raise_for_status()
    pages = r.json().get("pages", [])
    if not pages:
        return "No Wikipedia article found."

    # 2) resumo do artigo
    key = quote(pages[0]["key"], safe="")
    r = _session.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{key}",
        headers=WIKI_HEADERS,
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    return f"{data['title']}: {data.get('extract', '')[:800]}"


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for background on BJJ/MMA history, athletes, academies and general concepts.
    Do NOT use it to teach the step-by-step of a technique.
    Input: a short English search term (e.g. 'Helio Gracie', 'ADCC').
    Returns the article title and a short summary."""
    if USE_DUMMY:
        return "Helio Gracie (dummy): Brazilian martial artist, co-founder of Gracie Jiu-Jitsu."
    try:
        return _wiki_lookup(query.lower().strip())
    except Exception as e:
        return f"Wikipedia API error: {e}"


# ---------- Base de conhecimento (FAISS) ----------


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import (
        HuggingFaceEmbeddings,
    )  # import tardio: torch é pesado

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def load_technique_index():
    from langchain_community.vectorstores import FAISS

    # índice gerado por este projeto (pickle confiável)
    return FAISS.load_local(
        str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
    )


@tool
def search_technique(query: str) -> str:
    """Search the curated BJJ technique knowledge base (steps, common mistakes, defenses, safety).
    Use it to teach HOW to execute, defend or escape a technique.
    Input: the technique and position in English, e.g. 'armlock closed guard'.
    Returns the most relevant sections with their source sheet."""
    if USE_DUMMY:
        return "[armlock_closed_guard.md / Steps] (dummy) Control one arm, pivot, leg over the head, raise the hips."
    try:
        docs = load_technique_index().similarity_search(query, k=3)
    except Exception as e:
        return f"Knowledge base error: {e}"
    if not docs:
        return "No technique sheet found."
    return "\n\n".join(
        f"[{d.metadata['source']} / {d.metadata.get('section', '')}]\n{d.page_content}"
        for d in docs
    )


# ---------- Hardcoded ----------

IBJJF_POINTS = {
    "takedown": 2,
    "sweep": 2,
    "knee on belly": 2,
    "guard pass": 3,
    "mount": 4,
    "back control": 4,
}


@tool
def ibjjf_points(action: str) -> str:
    """Return how many points an action scores under IBJJF rules.
    Use it for scoring questions like 'how many points is a guard pass?'.
    Input: the action in English, one of: takedown, sweep, knee on belly, guard pass, mount, back control.
    """
    key = action.lower().strip()
    if key in IBJJF_POINTS:
        return f"{key}: {IBJJF_POINTS[key]} points (IBJJF)"
    return f"Unknown action '{action}'. Valid: {', '.join(IBJJF_POINTS)}"


# ---------- Text-to-query ----------


@tool
def query_adcc(question: str) -> str:
    """Answer statistical questions about ADCC matches 1998-2022 (submissions, winners, counts, averages)
    by querying a SQL database. Use it for numbers, rankings and competition history, NOT to teach techniques.
    Input: the full question in natural language, e.g. 'most common submission in 2022'.
    Returns the answer and the SQL used."""
    if USE_DUMMY:
        return "(dummy) RNC is the most common submission in ADCC (99 wins).\nSQL: SELECT submission, COUNT(*) ..."
    try:
        from src.text_to_sql import (
            answer_question,
        )  # import tardio: LLM/DB só carregam se a tool for usada

        r = answer_question(question)
    except Exception as e:
        return f"SQL tool error: {e}"
    return f"{r['answer']}\nSQL: {r['sql']}"
