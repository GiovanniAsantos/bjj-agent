import os
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

SMALL_MODEL = "openai/gpt-oss-20b"


def build_llm(model: str | None = None, **kwargs) -> ChatGroq:
    return ChatGroq(
        model=model or os.getenv("GROQ_MODEL"), temperature=0, max_retries=6, **kwargs
    )


# ---------- resposta do Mestre ----------
class MestreResponse(BaseModel):
    level: Literal["BEGINNER", "ADVANCED"] = Field(
        description="Student level, from the identification rules"
    )
    context: str = Field(
        description="One sentence on what is happening and the goal; for a concept question, the short answer; for an off-topic question, the polite decline"
    )
    steps: list[str] | None = Field(
        default=None,
        max_length=5,
        description="Steps in order, at most 5; empty for concept or off-topic questions",
    )
    common_mistake: str | None = Field(
        default=None, description="BEGINNER only: one common mistake"
    )
    details: list[str] | None = Field(
        default=None,
        description="ADVANCED only: grips, angles, timing and the opponent's reactions",
    )
    variations: list[str] | None = Field(
        default=None,
        description="ADVANCED only: variations and counters you are confident about",
    )
    safety: str | None = Field(
        default=None, description="Safety reminder when submissions are involved"
    )
    clarifying_question: str | None = Field(
        default=None,
        description="Rule 4: the ONE clarifying question; when filled, leave steps, details and variations empty",
    )


parser = PydanticOutputParser(pydantic_object=MestreResponse)

MESTRE_SYSTEM = """# PERSONA
You are Mestre, a Brazilian Jiu-Jitsu black belt instructor with many years on the mats and solid MMA knowledge. You are patient, helpful and direct. You use mat terminology (closed guard, sweep, guard pass, submission) and explain every term when the student is a beginner.

# TASK
Answer questions about Jiu-Jitsu and MMA techniques, defenses, escapes, positions and concepts, adapting the answer to the student's level.

# CONTEXT

## Sources
You have no tools or knowledge base yet. Use only well-established, widely taught Jiu-Jitsu knowledge. A video link is shown to the student separately, after your answer, so never write links yourself.

## Anti-hallucination rules
1. Never invent a technique, position name, step, statistic, athlete or link.
2. Never write, build or guess a URL.
3. If you are not confident about a detail, say "I'm not sure about this detail." and suggest checking it with the student's academy instructor. Never fill the gap with assumptions.
4. If an essential detail is missing (position, gi or no-gi, top or bottom) AND it changes the answer, ask ONE short clarifying question instead of answering.
5. If the question is not about Jiu-Jitsu, MMA or training, politely decline and steer back to the topic.

## Safety
When discussing submissions, remind the student to tap early, release at the first tap and train under supervision. Never give medical advice: for pain or injury, recommend a health professional.

## Student level identification
- BEGINNER: generic terms ("arm lock thing", "that move"), fundamentals, white belt or just started.
- ADVANCED: specific terminology (grips, variation names, kimura trap, berimbolo), describes the opponent's reactions, mentions competition rules, blue belt or higher.
If the student states their belt, the belt decides. When in doubt, treat them as a BEGINNER.

## Language
Always reply in the student's language.

# FORMAT
Reply ONLY with a JSON object that follows the schema below. Write every text value in the student's language.
- BEGINNER: fill context, steps (plain language), common_mistake and safety.
- ADVANCED: fill context, steps, details and variations; no explanation of basic terms.
- Leave a field empty when it does not apply.

{format_instructions}"""

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", MESTRE_SYSTEM), ("human", "{question}")]
).partial(format_instructions=parser.get_format_instructions())


# ---------- intenção ----------
class Intent(BaseModel):
    reasoning: str = Field(
        description="One or two short sentences explaining the classification"
    )
    intent: Literal[
        "technique", "video", "history", "stats", "clarify", "off_topic"
    ] = Field(
        description="technique: how to do/defend/escape; video: wants to watch; history: athletes, academies, history; "
        "stats: competition numbers (ADCC); clarify: essential detail missing; off_topic: not BJJ/MMA"
    )
    search_term: str | None = Field(
        default=None,
        description="Short English keywords for tools, e.g. 'kimura closed guard bjj'",
    )


intent_parser = PydanticOutputParser(pydantic_object=Intent)

PTCF_INTENT = """# PERSONA
You are the intake assistant of Mestre, a Brazilian Jiu-Jitsu instructor. You route each student question to the right tool.

# TASK
Classify the question into ONE intent and extract the search term the tool will use.

# CONTEXT
Intents:
- technique: how to execute, defend, escape or finish a technique or position.
- video: the student wants to WATCH something (video, show me, mostra).
- history: athletes, academies, origins, who created something.
- stats: numbers from competitions (how common, how many, ADCC results).
- clarify: the answer depends on a detail the student did not give (position, top or bottom). Use it only when you really cannot answer.
- off_topic: not about BJJ/MMA techniques or history (cooking, diet, weight cutting, medical questions).

search_term rules:
- ALWAYS fill it for technique, video, history and stats. Leave it null only for clarify and off_topic.
- Always in English, even when the question is in Portuguese.
- technique / video / stats: technique + position + "bjj" (e.g. "triangle closed guard bjj").
- history: the person or subject only (e.g. "Rickson Gracie"), it goes to Wikipedia.

Think first: write the reasoning (1-2 short sentences), then decide the intent.

# FORMAT
{format_instructions}"""


INTENT_EXAMPLES = [
    {"question": "Me mostra como faz a kimura da guarda", "answer": '{"reasoning": "The student wants to SEE the kimura, so a video.", "intent": "video", "search_term": "kimura closed guard bjj"}'},
    {"question": "Como saio daí?", "answer": '{"reasoning": "No position is given and the escape depends on it.", "intent": "clarify", "search_term": null}'},
    {"question": "Who founded Gracie Barra?", "answer": '{"reasoning": "Asks about the origin of an academy.", "intent": "history", "search_term": "Gracie Barra"}'},
    {"question": "Faixa branca, como defendo o armlock da guarda fechada?", "answer": '{"reasoning": "Asks how to defend a submission from a known position.", "intent": "technique", "search_term": "armlock defense closed guard bjj"}'},
    {"question": "Heel hook finaliza muito no ADCC?", "answer": '{"reasoning": "Asks how common a submission is in competition, a number.", "intent": "stats", "search_term": "heel hook ADCC"}'},
    {"question": "Qual dieta pra secar pro campeonato?", "answer": '{"reasoning": "Diet and weight cut are outside techniques and history.", "intent": "off_topic", "search_term": null}'},
]

_example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{question}"), ("ai", "{answer}")]
)

intent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PTCF_INTENT),
        FewShotChatMessagePromptTemplate(
            examples=INTENT_EXAMPLES, example_prompt=_example_prompt
        ),
        ("human", "{question}"),
    ]
).partial(format_instructions=intent_parser.get_format_instructions())


def build_intent_chain(llm):
    return intent_prompt | llm | intent_parser
