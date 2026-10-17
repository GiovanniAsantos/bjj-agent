# Relatório da Fase 1 — MVP Chain

> Sprint 1 (15–18/10/2026). Rubrica 1.1 a 1.6 e 2.1.
> Artefatos: [`notebooks/00_setup.ipynb`](../notebooks/00_setup.ipynb), [`notebooks/01_mvp_chain.ipynb`](../notebooks/01_mvp_chain.ipynb), [`docs/01_arquitetura.md`](01_arquitetura.md), [`docs/02_orquestradores.md`](02_orquestradores.md).

## 1. Objetivo da fase

Sair de uma chamada crua ao LLM para uma **chain reutilizável com saída estruturada**: o Mestre recebe a pergunta do aluno, identifica o nível, responde em um formato validado e indica um vídeo coerente com a técnica — sem ferramentas nem memória ainda.

## 2. Mapa da rubrica

| Item | Onde está | Resumo |
| --- | --- | --- |
| 1.1 Anatomia do agente | `01_arquitetura.md` §2 | LLM, system message (PTCF), memória, ferramentas, base de conhecimento, saída estruturada |
| 1.2 Reativo / deliberativo / híbrido | `01_arquitetura.md` §3 | Hoje reativo com fluxo fixo; alvo híbrido |
| 1.3 Maturidade agêntica | `01_arquitetura.md` §4 | Nível 0 → 1 nesta fase; alvo 4 |
| 1.4 Decomposição hierárquica | `01_arquitetura.md` §5 | Árvore em 5 etapas, marcando o que a fase 1 já cobre |
| 1.5 Memória (rascunho) + composição | `01_arquitetura.md` §6; notebook células 6–8 | Curto/longo prazo + `StudentProfile`; mesma chain em `@chain` e LCEL |
| 1.6 Justificativa da composição | Este relatório §5 | LCEL como padrão, `@chain` onde houver decisão |
| 2.1 Orquestradores | `02_orquestradores.md` | LangChain + LangGraph; CrewAI/AutoGen não compensam neste escopo |

## 3. O que foi construído

### 3.1 Ambiente (00_setup)

- Python **3.13** em `.venv`, dependências fixas em `requirements.txt`. O Python 3.15 instalado no sistema foi descartado: o `uuid-utils` (dependência do `langchain-core`) não tinha wheel para ele e a instalação travava compilando em Rust.
- `.env` com `GROQ_API_KEY`, `GROQ_MODEL=openai/gpt-oss-120b`, LangSmith e YouTube; `.env .example` sem segredos.
- Testes de conectividade: Groq, YouTube Data API v3, Wikipedia REST e carga do CSV do ADCC no SQLite (`data/db/bjj.sqlite`, tabela `lutas`).

### 3.2 Chain do Mestre (01_mvp_chain)

| Célula | Conteúdo | Conceito da rubrica |
| --- | --- | --- |
| 1 | `llm`, schema `MestreResponse`, `PydanticOutputParser`, `JsonOutputParser`, `chat_prompt` (PTCF), `search_prompt`, chains | `ChatPromptTemplate`, `PromptTemplate`, parsers |
| 2 | `search_video(query)` | Integração com API externa (base da futura ferramenta) |
| 3 | `chain_json.stream(...)` | `stream` + `JsonOutputParser` |
| 4 | `chain.batch(...)` + vídeo por pergunta | `batch` |
| 5 | Tempo `batch` × `invoke` | `invoke` × `batch` |
| 6–7 | `mestre_lcel` × `mestre_imperative` + grafos | LCEL × `@chain` |
| 8 | Resposta + vídeo: `@chain` sequencial × `RunnableParallel` | Justificativa da composição |

## 4. Decisões de projeto

| Decisão | Motivo |
| --- | --- |
| `openai/gpt-oss-120b` via Groq, `temperature=0` | Maior modelo de chat disponível no free tier; bom em JSON; temperatura 0 para respostas técnicas reproduzíveis |
| `max_retries=6` | O free tier limita a 8000 tokens/min; o cliente espera e tenta de novo no 429 |
| System prompt em PTCF, em inglês | Estrutura clara (Persona, Task, Context, Format); regras anti-alucinação e de segurança concentradas em um lugar |
| Prompt da fase 1 **sem ferramentas** | Com as ferramentas citadas no prompt e nenhuma ligada ao modelo, o Groq recusava a chamada (seção 6) |
| Vídeo buscado por código, não pelo LLM | O modelo nunca escreve URL (regra anti-alucinação 2); a busca vira ferramenta na fase 2 |
| Um schema único (`MestreResponse`) com campos opcionais | Mais simples que dois modelos; `level` decide quais campos são preenchidos |
| Regra de nível conservadora (*when in doubt → BEGINNER*) | Explicar demais é menos arriscado que pular fundamentos de segurança |
| Nomes de variáveis e campos em inglês | Consistência com o prompt; reduziu respostas em português para perguntas em inglês |

## 5. Composição: `@chain` × LCEL (1.5 e 1.6)

A mesma chain foi escrita das duas formas:

```python
mestre_lcel = chat_prompt | llm | parser          # declarativo

@chain_decorator
def mestre_imperative(inputs):                     # imperativo
    messages = chat_prompt.invoke(inputs)
    ai_message = llm.invoke(messages)
    return parser.invoke(ai_message)
```

As duas devolvem o mesmo `MestreResponse` e aceitam `invoke`, `batch` e `stream`. A diferença está no que o LangChain consegue **enxergar**:

| | LCEL | `@chain` |
| --- | --- | --- |
| Tipo | `RunnableSequence` | `RunnableLambda` |
| Grafo (`get_graph()`) | `Prompt → ChatGroq → Parser`: a **ordem real** das etapas | As três peças ligadas direto à entrada e à saída: sabe **quais** peças existem, não **a ordem** |
| Paralelismo | Declarado (`RunnableParallel`) | Manual |
| Lógica arbitrária (`if`, laços, tratamento de erro) | Desajeitada | Natural |

Caso com duas tarefas independentes (resposta + vídeo), mediana de 3 rodadas com 20 s de pausa:

| | Mediana | Rodadas |
| --- | --- | --- |
| `@chain` (sequencial) | 2,9 s | 3,8 / 2,8 / 2,9 |
| LCEL (`RunnableParallel`) | **1,7 s** | 1,7 / 1,7 / 1,9 |

**Escolha:** LCEL como padrão, porque o fluxo do Mestre é uma sequência fixa de etapas: menos código, paralelismo sem threads (~40% mais rápido no caso medido) e cada etapa visível no LangSmith. `@chain` fica reservado para pontos com decisão — por exemplo, não buscar vídeo quando o Mestre devolve `clarifying_question`. A partir da fase 2, as decisões passam para o agente (ReAct sobre LangGraph, ver `02_orquestradores.md`).

## 6. Resultados e achados

| Achado | Evidência | Consequência |
| --- | --- | --- |
| Prompt e código desalinhados | Prompt mandando chamar `search_technique` sem ferramenta ligada → `400 tool_use_failed`; depois respostas vazias (o modelo "chamava" a ferramenta dentro do raciocínio) | Prompt sem ferramentas na fase 1; versão com ferramentas guardada para a fase 2 |
| Validação funciona | O modelo devolveu `null` em `details`/`variations` → `OutputParserException` | Listas como `list[str] \| None`; o parser falha alto em vez de passar dado errado |
| Nível do aluno | "white belt" → BEGINNER; "brown belt" → ADVANCED com `details` e `variations`; sem faixa, "De La Riva" → BEGINNER | Regra funciona; perfil de longo prazo (fase 3) evita inferir a cada pergunta |
| Fora do tema | "Qual a melhor receita de bolo?" → recusa educada, só `level` + `context` | Regra 5 funciona |
| Erros técnicos sem base | Kimura: *"segure o pulso direito dele com a sua mão direita"*; leglock: *"posição superior (mount, side control ou X-guard)"* — X-guard é guarda por baixo | Principal justificativa do RAG (fase 4) e da verificação (fase 5) |
| Busca do vídeo | Frase inteira + 1 resultado trouxe vídeo de triângulo para pergunta de kimura; com `search_chain` (`PromptTemplate`), *"mount escape technique bjj"* → *The First 3 Mount Escapes You Need To Know in BJJ* | Palavras-chave geradas pelo LLM antes da busca |
| `batch` × `invoke` | 2 perguntas: batch 2,3–2,7 s × sequencial 4,0–5,6 s | `batch` ≈ tempo de uma chamada; limitado pelo rate limit |
| Stream | Mesmo código: 1 pedaço em uma execução, 258 em outra | Depende do provedor; interface precisa de indicador de "pensando" |
| Medição de latência | Sem pausa, o LCEL paralelo mediu 10,3 s (pior que o sequencial) por causa da espera no 429 | Medir com repetição, pausa e mediana |
| Tamanho do prompt | ~1.080 → ~690 tokens, mantendo as regras anti-alucinação | Mais chamadas por minuto no free tier |

## 7. Limitações

- **Sem base de conhecimento:** o conteúdo técnico vem só do modelo e tem erros que um faixa-roxa percebe; as regras anti-alucinação reduzem, mas não eliminam.
- **Idioma inconsistente:** parte das perguntas em inglês ainda recebe resposta em português.
- **Vídeo não verificado:** o título é coerente com a busca, mas ninguém confere se o vídeo mostra a mesma variação explicada.
- **Fluxo fixo:** toda pergunta busca vídeo, mesmo quando o Mestre só pede esclarecimento.
- **Free tier:** 8000 tokens/min limita batch grande e medições seguidas.

## 8. Próximos passos (Fase 2)

1. Transformar `search_video` em ferramenta e criar `search_wikipedia`; ligar ao modelo e montar o agente ReAct.
2. Restaurar no prompt as instruções de ferramentas (versão guardada na fase 1), agora com ferramentas reais.
3. Usar `@chain` (ou a decisão do agente) para pular o vídeo quando houver `clarifying_question`.
4. Investigar o idioma das respostas.
5. Atualizar o README: Python 3.13 (hoje diz 3.12) e nome do arquivo `.env.example` (hoje `.env .example`, com espaço).
