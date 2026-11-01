"""Verificador de fidelidade (NLI) e de coerência lógica (Fase 5.1–5.2)."""

import re
from functools import lru_cache
from itertools import combinations

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
LABELS = ["entailment", "neutral", "contradiction"]  # ordem do id2label do modelo
SUPPORT_T = 0.5  # entailment mínimo p/ considerar sustentada
CONTRA_T = 0.8  # contradição só com alta confiança (evita falso alarme)
# palavras que não indicam assunto (comuns em qualquer instrução de BJJ)
STOP = {
    "oponente",
    "para",
    "durante",
    "antes",
    "depois",
    "sobre",
    "você",
    "dele",
    "dela",
    "quando",
    "muito",
    "sempre",
    "nunca",
    "isso",
    "esse",
    "essa",
}
# frases que não são afirmações técnicas: link/vídeo e avisos de segurança (exigidos pelo prompt)
NON_TECHNICAL = re.compile(r"https?://|\bv[ií]deo\b|^⚠️|seguran[çc]a:", re.I)



@lru_cache(maxsize=1)
def get_nli():
    import torch
    from sentence_transformers import CrossEncoder

    # float32: o default fp16 é lento em CPU
    return CrossEncoder(NLI_MODEL, model_kwargs={"dtype": torch.float32})


def nli(pairs: list[tuple[str, str]]) -> list[dict]:
    """(premissa, hipótese) -> probabilidades por rótulo."""
    probs = get_nli().predict(pairs, apply_softmax=True)
    return [dict(zip(LABELS, map(float, p))) for p in probs]


def split_claims(text: str, min_len: int = 25) -> list[str]:
    """Quebra a resposta em afirmações (frases/itens), sem citações e markdown."""
    text = re.sub(r"\(fonte:[^)]*\)", "", text)
    text = re.sub(r"【[^】]*】", "", text)        # citação no formato nativo do gpt-oss
    text = re.sub(r"[*#>`_]", "", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    claims = [re.sub(r"^\s*(?:[-•]|\d+[.)])\s*", "", p).strip() for p in parts]
    claims = [re.sub(r"^[^–:]{3,40}\s[–:-]\s", "", c) for c in claims]   # tira "Título – " do item
    return [c for c in claims if len(c) >= min_len]


def contexts_from_tool_output(text: str) -> list[str]:
    """Saída do search_technique ('[ficha / seção]\\nconteúdo' separados por linha em branco) -> conteúdos."""
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return [
        c.split("\n", 1)[1] if c.startswith("[") and "\n" in c else c for c in chunks
    ]


def verify_against_sources(answer: str, contexts: list[str]) -> list[dict]:
    """5.1: cada afirmação × cada FRASE dos trechos recuperados (premissa curta = NLI mais confiável)."""
    premises = [p for ctx in contexts for p in split_claims(ctx, min_len=15)]
    results = []
    for claim in [c for c in split_claims(answer) if not NON_TECHNICAL.search(c)]:
        scores = nli([(p, claim) for p in premises])
        ent = max(s["entailment"] for s in scores)
        con = max(s["contradiction"] for s in scores)
        label = (
            "sustentada"
            if ent >= SUPPORT_T
            else "contradiz" if con >= CONTRA_T else "neutra"
        )
        results.append(
            {
                "claim": claim,
                "label": label,
                "entailment": round(ent, 3),
                "contradiction": round(con, 3),
            }
        )
    return results


def _topic_words(text: str) -> set[str]:
    return set(re.findall(r"\w{4,}", text.lower())) - STOP


def same_topic(a: str, b: str, min_overlap: float = 0.5) -> bool:
    wa, wb = _topic_words(a), _topic_words(b)
    return len(wa & wb) / max(1, min(len(wa), len(wb))) >= min_overlap


def check_coherence(answer: str, max_claims: int = 10) -> list[dict]:
    """5.2: a resposta não pode se contradizer. NLI só entre afirmações sobre o mesmo assunto."""
    claims = split_claims(answer)[:max_claims]
    pairs = [(a, b) for a, b in combinations(claims, 2) if same_topic(a, b)]
    if not pairs:
        return []
    scores = nli(pairs + [(b, a) for a, b in pairs])
    n = len(pairs)
    conflicts = []
    for i, (a, b) in enumerate(pairs):
        p = max(scores[i]["contradiction"], scores[i + n]["contradiction"])
        if p >= CONTRA_T:
            conflicts.append({"a": a, "b": b, "contradiction": round(p, 3)})
    return conflicts


def verify(answer: str, contexts: list[str]) -> dict:
    claims = verify_against_sources(answer, contexts) if contexts else []
    conflicts = check_coherence(answer)
    n = len(claims)
    support = sum(c["label"] == "sustentada" for c in claims) / n if n else None
    if any(c["label"] == "contradiz" for c in claims):
        verdict = "contradiz_fonte"
    elif conflicts:
        verdict = "incoerente"
    elif support is not None and support < 0.5:
        verdict = "pouco_sustentada"
    else:
        verdict = "ok"
    return {
        "verdict": verdict,
        "support_rate": support,
        "claims": claims,
        "conflicts": conflicts,
    }
