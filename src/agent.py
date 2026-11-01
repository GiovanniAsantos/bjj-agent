"""Agente final do Mestre (Fase 5.3–5.4): tools de 4 categorias + verificador NLI na saída."""

from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
import re
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.chains import build_llm
from src.tools import (
    ibjjf_points,
    query_adcc,
    search_technique,
    search_video,
    search_wikipedia,
)
from src.verifier import contexts_from_tool_output, verify

TOOLS = [search_technique, search_video, search_wikipedia, query_adcc, ibjjf_points]
TOOL_CATEGORY = {
    "ibjjf_points": "hardcoded",
    "search_technique": "semântica (RAG)",
    "search_video": "API externa",
    "search_wikipedia": "API externa",
    "query_adcc": "text-to-SQL",
}

AGENT_SYSTEM = """You are Mestre, a patient Brazilian Jiu-Jitsu black belt instructor. Always answer in Portuguese.

Tools (pick the one that fits; you may combine them):
- search_technique: HOW to execute, defend or escape a technique. Call it with the standard ENGLISH technique name
  + position + the part you need: 'steps' (how to do it) or 'defenses' (how to defend/escape it).
- search_video: only when the student asks for a video.
- search_wikipedia: history, athletes, academies.
- query_adcc: competition statistics (ADCC 1998-2022). Pass the full question.
- ibjjf_points: scoring under IBJJF rules.

Rules:
- ALWAYS call search_technique before describing any step of a technique, even when the student also asks for a video.
  Base the steps ONLY on its results and cite as (fonte: file.md).
- If the tools do not cover the question, say so. Never invent steps, numbers or links.
- If search_technique only MENTIONS the technique but does not teach it, say the knowledge base does not cover it
  and offer a video. Do not ask for the position in this case.
- If the question has no technique or position and the conversation gives no context, ask one clarifying question.
- Mention safety (tap early, release on tap) when teaching submissions.
- Keep answers short: at most 6 steps."""

REVISE_SYSTEM = """You fix a BJJ answer so it agrees with the sources. Answer in Portuguese.
Remove or correct the problematic sentences listed below. Keep everything else, keep citations.
Use ONLY the sources.

Sources:
{sources}

Problematic sentences:
{problems}"""


def build_agent_executor(llm=None, verbose: bool = False) -> AgentExecutor:
    llm = llm or build_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM),
            MessagesPlaceholder("chat_history", optional=True),  # usado no 5.4
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),  # tool calls + observações do loop
        ]
    )
    agent = create_openai_tools_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        verbose=verbose,
    )


def build_reviser(llm=None):
    prompt = ChatPromptTemplate.from_messages(
        [("system", REVISE_SYSTEM), ("human", "{answer}")]
    )
    return prompt | (llm or build_llm()) | StrOutputParser()


def teaches_steps(answer: str) -> bool:
    """Resposta com passo a passo numerado (1. / 2) ...) = está ensinando técnica."""
    return len(re.findall(r"^\s*\**\d+[.)]", answer, flags=re.M)) >= 2


def verified_answer(result: dict, reviser) -> dict:
    """Pós-processamento: NLI contra o que o RAG devolveu; revisa 1x se contradiz/incoerente."""
    steps = result.get("intermediate_steps", [])
    contexts = [
        c
        for action, obs in steps
        if action.tool == "search_technique"
        for c in contexts_from_tool_output(str(obs))
    ]
    answer = re.sub(r"【([^】]*)】", r" (fonte: \1)", result["output"])   # citação nativa do gpt-oss → formato do projeto
    check = verify(answer, contexts)
    revised = False
    if not contexts and teaches_steps(answer):
        check["verdict"] = "sem_fonte"
        answer += "\n\n⚠️ Estes passos não vieram da base de conhecimento; confirme com seu professor."

    if check["verdict"] in ("contradiz_fonte", "incoerente") and contexts:
        problems = [c["claim"] for c in check["claims"] if c["label"] == "contradiz"]
        problems += [f"{k['a']} <-> {k['b']}" for k in check["conflicts"]]
        answer = reviser.invoke(
            {
                "answer": answer,
                "sources": "\n".join(contexts),
                "problems": "\n".join(f"- {p}" for p in problems),
            }
        )
        check = verify(answer, contexts)
        revised = True
        if check["verdict"] in ("contradiz_fonte", "incoerente"):
            answer += "\n\n⚠️ O verificador ainda encontrou divergência com as fichas; confira com seu professor."

    if check["verdict"] == "pouco_sustentada":
        answer += (
            "\n\n⚠️ Parte desta resposta não está nas fichas da base de conhecimento."
        )
    return {
        "output": answer,
        "verdict": check["verdict"],
        "support_rate": check["support_rate"],
        "revised": revised,
        "tools": [action.tool for action, _ in steps],
        "check": check,
    }


def build_mestre(llm=None, verbose: bool = False):
    """Runnable final: {'input', 'chat_history'?} -> dict com 'output' verificado."""
    executor = build_agent_executor(llm, verbose)
    reviser = build_reviser(llm)
    return RunnableLambda(lambda x: verified_answer(executor.invoke(x), reviser))


_SESSIONS: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Um histórico por aluno/sessão (em memória; o Streamlit usa o mesmo session_id)."""
    return _SESSIONS.setdefault(session_id, InMemoryChatMessageHistory())


def build_mestre_with_memory(llm=None, verbose: bool = False):
    return RunnableWithMessageHistory(
        build_mestre(llm, verbose),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",  # = MessagesPlaceholder do prompt do agente
        output_messages_key="output",  # só o texto final vai para o histórico
    )
