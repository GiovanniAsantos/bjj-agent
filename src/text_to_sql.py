"""Text-to-SQL sobre o SQLite do ADCC (Fase 4.3)."""

import re
from functools import lru_cache
from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.chains import build_llm, SMALL_MODEL

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "bjj.sqlite"


class SQLQuery(BaseModel):
    reasoning: str = Field(
        description="Short reasoning: which columns and filters answer the question"
    )
    sql: str = Field(description="One SQLite SELECT query")


sql_parser = PydanticOutputParser(pydantic_object=SQLQuery)

COLUMN_DICT = """Column notes (table lutas = one ADCC match per row, 1998-2022):
- victory_method: 'SUBMISSION', 'POINTS', 'DECISION', 'INJURY', 'DESQUALIFICATION' (sic, misspelled)
- submission: free text, only when victory_method='SUBMISSION', else NULL. Ex: 'RNC' (rear naked choke),
  'Armbar', 'Guillotine', 'Heel hook', 'Inside heel hook', 'Outside heel hook', 'Triangle', 'Kimura'.
  Use LIKE '%heel hook%' to get all heel hook variants.
- submission_target: 'Arm', 'Leg', 'Neck', 'Other/Unknown', or NULL (not a submission)
- winner_points / loser_points: -1 means NO DATA (exclude with > -1 before averaging)
- female: 1 = women's division, 0 = men's
- absolute: 1 = open weight (absolute) division; then weight_class is NULL
- weight_class: code 0-4 (lightest to heaviest), NULL when absolute
- importance: round importance code 0-5 (higher = later round)
- winner_name / loser_name: athlete full name"""

FEW_SHOT = """Examples:
Q: Most common submission ever?
SQL: SELECT submission, COUNT(*) AS n FROM lutas WHERE submission IS NOT NULL GROUP BY submission ORDER BY n DESC LIMIT 5
Q: How many women's matches ended by submission?
SQL: SELECT COUNT(*) FROM lutas WHERE female = 1 AND victory_method = 'SUBMISSION' LIMIT 1
Q: Average winner points in 2019, when available?
SQL: SELECT AVG(winner_points) FROM lutas WHERE year = 2019 AND winner_points > -1 LIMIT 1"""

SQL_SYSTEM = (
    "You write SQLite queries for an ADCC (submission grappling) match database.\n"
    "Schema:\n{schema}\n\n{column_dict}\n\n{few_shot}\n\n"
    "Rules: a single SELECT only (no ;, no writes). Always end the query with a LIMIT clause "
    "(LIMIT 1 for a single aggregate like COUNT/AVG, LIMIT 5 for rankings so ties are visible, "
    "LIMIT 20 otherwise). "
    "Use only the columns above. Athlete names: use LIKE '%name%'.\n\n{format_instructions}"
)

ANSWER_SYSTEM = (
    "Você responde perguntas sobre estatísticas do ADCC. Use SOMENTE o resultado SQL abaixo. "
    "Responda em português, em 1-2 frases. Se houver empate no topo, cite todos os empatados. "
    "Se o resultado estiver vazio, diga que não há dados.\n\n"
    "SQL: {sql}\nResultado: {rows}"
)

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum)\b", re.I
)


def validate_sql(sql: str) -> str:
    """2ª camada de segurança: só um SELECT, sem comandos de escrita."""
    s = sql.strip().rstrip(";").strip()
    if not s.lower().startswith(("select", "with")):
        raise ValueError("only SELECT queries are allowed")
    if ";" in s:
        raise ValueError("multiple statements are not allowed")
    if FORBIDDEN.search(s):
        raise ValueError(f"forbidden keyword: {FORBIDDEN.search(s).group()}")
    return s


@lru_cache(maxsize=1)
def get_db():
    from langchain_community.utilities import SQLDatabase

    # 1ª camada de segurança: conexão read-only
    return SQLDatabase.from_uri(f"sqlite:///file:{DB_PATH.as_posix()}?mode=ro&uri=true")


@lru_cache(maxsize=1)
def build_sql_chains():
    llm = build_llm(SMALL_MODEL)
    sql_prompt = ChatPromptTemplate.from_messages(
        [("system", SQL_SYSTEM), ("human", "{question}")]
    ).partial(
        schema=get_db().get_table_info(),
        column_dict=COLUMN_DICT,
        few_shot=FEW_SHOT,
        format_instructions=sql_parser.get_format_instructions(),
    )
    answer_prompt = ChatPromptTemplate.from_messages(
        [("system", ANSWER_SYSTEM), ("human", "{question}")]
    )
    return sql_prompt | llm | sql_parser, answer_prompt | llm | StrOutputParser()


def run_sql_question(question: str, max_attempts: int = 2) -> dict:
    """Gera SQL, valida e executa. Se falhar, devolve o erro ao LLM e tenta de novo."""
    sql_gen_chain, _ = build_sql_chains()
    prompt_q, sql, error = question, "", ""
    for attempt in range(1, max_attempts + 1):
        sql = sql_gen_chain.invoke({"question": prompt_q}).sql
        try:
            rows = get_db().run(validate_sql(sql))
            return {"sql": sql, "rows": rows, "attempts": attempt, "error": None}
        except Exception as e:
            error = str(e).split("\n")[0][:200]
            prompt_q = (
                f"{question}\n\nYour previous query failed.\n"
                f"Query: {sql}\nError: {error}\nWrite a corrected query."
            )
    return {"sql": sql, "rows": None, "attempts": attempt, "error": error}


def answer_question(question: str) -> dict:
    r = run_sql_question(question)
    if r["rows"] is None:
        return {**r, "answer": f"Não consegui consultar o banco ({r['error']})."}
    _, answer_chain = build_sql_chains()
    return {**r, "answer": answer_chain.invoke({"question": question, **r})}
