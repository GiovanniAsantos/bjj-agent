"""Interface web do Mestre (Fase 5.5): chat Streamlit sobre o agente verificado."""

import os
import uuid

from dotenv import load_dotenv

load_dotenv()  # antes de importar src: tools.py lê USE_DUMMY no import
from src.tools import load_technique_index
from src.verifier import get_nli
import streamlit as st

from src.agent import TOOL_CATEGORY, build_mestre_with_memory, get_session_history

VERDICT_LABEL = {
    "ok": "✅ sustentada pelas fichas",
    "pouco_sustentada": "🟡 pouco sustentada",
    "contradiz_fonte": "🔴 contradiz a fonte",
    "incoerente": "🔴 incoerente",
    "sem_fonte": "⚠️ sem fonte",
}


@st.cache_resource(show_spinner="Carregando o Mestre (FAISS + NLI)...")
def load_mestre():
    """Um runnable por processo; o histórico fica separado por session_id."""
    load_technique_index()  # embeddings + FAISS
    get_nli()  # mDeBERTa em CPU (~30s): paga no start, não na 1ª pergunta
    return build_mestre_with_memory()


def show_meta(meta: dict):
    """Painel de transparência: tools, categoria, veredito do verificador."""
    with st.expander(VERDICT_LABEL.get(meta["verdict"], meta["verdict"])):
        cats = sorted({TOOL_CATEGORY.get(t, "?") for t in meta["tools"]})
        st.write(f"**Tools:** {', '.join(meta['tools']) or 'nenhuma'}")
        st.write(f"**Categorias:** {', '.join(cats) or '-'}")
        rate = meta["support_rate"]
        st.write(f"**Suporte NLI:** {'-' if rate is None else f'{rate:.0%}'}")
        st.write(
            f"**Revisada:** {'sim' if meta['revised'] else 'não'} | **Tempo:** {meta['secs']}s"
        )


st.set_page_config(page_title="Mestre BJJ", page_icon="🥋")
st.title("🥋 Mestre — instrutor de Jiu-Jitsu")

if "session_id" not in st.session_state:
    st.session_state.session_id = f"web-{uuid.uuid4().hex[:8]}"
    st.session_state.messages = (
        []
    )  # só p/ redesenhar a tela; a memória do agente é o get_session_history

with st.sidebar:
    dummy = os.getenv("USE_DUMMY", "false").lower() == "true"
    st.caption(f"Modo: {'DUMMY (sem API)' if dummy else os.getenv('GROQ_MODEL')}")
    st.caption(f"Sessão: {st.session_state.session_id}")
    st.caption(
        f"Mensagens na memória: {len(get_session_history(st.session_state.session_id).messages)}"
    )
    if st.button("Nova conversa"):
        del st.session_state["session_id"]
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg:
            show_meta(msg["meta"])

if question := st.chat_input("Pergunte sobre técnica, regra, ADCC ou peça um vídeo"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("O Mestre está pensando..."):
            import time

            t0 = time.perf_counter()
            try:
                r = load_mestre().invoke(
                    {"input": question},
                    config={
                        "configurable": {"session_id": st.session_state.session_id}
                    },
                )
            except Exception as e:  # 429 do Groq, rede etc.: não derruba a UI
                st.error(f"Falha ao consultar o agente: {type(e).__name__}: {e}")
                st.stop()
        meta = {k: r[k] for k in ("tools", "verdict", "support_rate", "revised")}
        meta["secs"] = round(time.perf_counter() - t0, 1)
        st.markdown(r["output"])
        show_meta(meta)
    st.session_state.messages.append(
        {"role": "assistant", "content": r["output"], "meta": meta}
    )
