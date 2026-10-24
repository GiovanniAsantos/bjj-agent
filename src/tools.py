import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import requests
from langchain_core.tools import tool

USE_DUMMY = os.getenv("USE_DUMMY", "false").lower() == "true"
WIKI_HEADERS = {"User-Agent": "bjj-agent/0.1 (Infnet MBA study project)"}

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index" / "techniques"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@tool
def search_video(query: str) -> str:
    """Search YouTube for a BJJ instructional video.
    Use ONLY when the student wants to see the technique in a video.
    Input: short English keywords (technique + position + bjj), e.g. 'kimura closed guard bjj'.
    Returns 'title — link' of the first result."""
    if USE_DUMMY:
        return "BJJ Mount Escape (dummy) — https://www.youtube.com/watch?v=dummy"
    try:
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 1,
            "key": os.getenv("YOUTUBE_API_KEY"),
        }
        r = requests.get(
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
    except Exception as e:
        return f"YouTube API error: {e}"


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for background on BJJ/MMA history, athletes, academies and general concepts.
    Do NOT use it to teach the step-by-step of a technique.
    Input: a short English search term (e.g. 'Helio Gracie', 'ADCC').
    Returns the article title and a short summary."""
    if USE_DUMMY:
        return "Helio Gracie (dummy): Brazilian martial artist, co-founder of Gracie Jiu-Jitsu."
    try:
        # 1) busca: termo -> chave do artigo
        r = requests.get(
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
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{key}",
            headers=WIKI_HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return f"{data['title']}: {data.get('extract', '')[:800]}"
    except Exception as e:
        return f"Wikipedia API error: {e}"


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings  # import tardio: torch é pesado
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def load_technique_index():
    from langchain_community.vectorstores import FAISS
    # índice gerado por este projeto (pickle confiável)
    return FAISS.load_local(str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True)


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
        f"[{d.metadata['source']} / {d.metadata.get('section', '')}]\n{d.page_content}" for d in docs
    )
