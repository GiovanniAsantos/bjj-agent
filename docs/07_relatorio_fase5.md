# Relatório da Fase 5 — Verificação, agente integrado, interface e avaliação

> Sprint 4 e semana de folga (01/11 a 04/11/2026). Rubrica 5.1 a 5.6.
> Artefatos: [`notebooks/05_agent_eval.ipynb`](../notebooks/05_agent_eval.ipynb), [`src/verifier.py`](../src/verifier.py), [`src/agent.py`](../src/agent.py), [`src/chains.py`](../src/chains.py) (`DummyChatModel`), [`app.py`](../app.py) (Streamlit), [`cli.py`](../cli.py), dataset `bjj-mestre-eval` no LangSmith.

## 1. Objetivo da fase

Juntar tudo num **único agente** que escolhe entre ferramentas de categorias diferentes, lembra da conversa, **confere a própria resposta** contra a evidência antes de entregar e é **medido** de forma sistemática. É a passagem do nível 3 para o nível 4 da escala de maturidade (`01_arquitetura.md` §4): agente autoverificável e avaliado.

## 2. Mapa da rubrica

| Item | Onde está | Resumo |
| --- | --- | --- |
| 5.1 Verificador NLI | `src/verifier.py` (`verify_against_sources`); `05_agent_eval` §5.1 | Cross-encoder mDeBERTa-v3 XNLI multilíngue; cada frase da resposta × cada frase das fichas → sustentada / neutra / contradiz |
| 5.2 Coerência lógica | `src/verifier.py` (`check_coherence`); `05_agent_eval` §5.2 | NLI entre pares de afirmações da própria resposta, só entre pares sobre o mesmo assunto |
| 5.3 `create_openai_tools_agent` + `AgentExecutor` | `src/agent.py`; `05_agent_eval` §5.3 | 5 tools em 4 categorias (hardcoded, RAG, API externa, text-to-SQL) + verificação na saída com 1 revisão |
| 5.4 `RunnableWithMessageHistory` | `src/agent.py` (`build_mestre_with_memory`); `05_agent_eval` §5.4 | Histórico por `session_id`; teste de 3 turnos com referências implícitas + isolamento entre sessões |
| 5.5 Interface | `app.py` (Streamlit), `cli.py` | Chat com painel de transparência (tools, categorias, veredito, suporte NLI, tempo) e uma sessão por aba |
| 5.6 Avaliação no LangSmith | `05_agent_eval` §5.6; experimentos `mestre-v1` e `mestre-v2` | Dataset de 8 perguntas; avaliadores de trajetória (recall/precisão de tools), qualidade (LLM-as-judge) e verificação (NLI) |

## 3. Verificador de fidelidade (5.1)

### 3.1 Modelo e decisão

| Escolha | Valor | Motivo |
| --- | --- | --- |
| Modelo | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | A resposta sai em **português** e as fichas estão em **inglês**: precisa de NLI multilíngue (cross-lingual) |
| Tipo | Cross-encoder (`sentence_transformers.CrossEncoder`) | Lê premissa e hipótese juntas; é mais preciso que comparar embeddings para decidir implicação × contradição |
| Execução | CPU, `float32`, carregado 1× (`lru_cache`) | O padrão fp16 é lento em CPU; ~200 MB, sem custo de API |
| Rótulos | `entailment` / `neutral` / `contradiction` | Na ordem do `id2label` do modelo |

Regras de decisão por afirmação:

```python
SUPPORT_T = 0.5   # entailment mínimo -> "sustentada"
CONTRA_T  = 0.8   # contradição só com alta confiança -> "contradiz" (evita falso alarme)
```

O limiar de contradição é **assimétrico** de propósito: uma contradição dispara a revisão da resposta (custo de +1 chamada de LLM), então só conta quando o modelo tem muita certeza. O que não é sustentado nem contradito fica `neutra`.

### 3.2 Fluxo

```
resposta ──split_claims──▶ afirmações (sem citação, sem markdown, sem "Título –")
                              │   filtro NON_TECHNICAL: links, "vídeo", avisos de segurança
fichas  ──split_claims──▶ premissas (frases curtas)
                              ▼
           NLI(premissa, afirmação) para todos os pares → max entailment / max contradiction
                              ▼
             veredito: contradiz_fonte > incoerente > pouco_sustentada (< 50% sustentadas) > ok
```

### 3.3 Teste controlado

Quatro respostas escritas à mão sobre o mesmo contexto real (3 trechos de `armlock_closed_guard.md` recuperados pelo `search_technique`):

| Resposta | O que simula | Veredito | Suporte |
| --- | --- | --- | --- |
| fiel | Passos corretos, parafraseados em PT | `ok` | 1,00 |
| contradiz_fonte | "Aplique a chave rápido e com força, sem esperar o oponente bater" | `contradiz_fonte` (con = 1,00) | 0,50 |
| alucinacao | "O armlock foi criado por Helio Gracie em 1930" | `pouco_sustentada` (ambas neutras) | 0,00 |
| incoerente | "Eleve o quadril…" + "Nunca eleve o quadril…" | `contradiz_fonte` + conflito interno p = 0,999 | 0,50 |

Os 4 casos foram classificados corretamente em 13,4 s de CPU (incluindo a carga do modelo). A alucinação fica `neutra`, não `contradiz`: a ficha não fala de história, então não tem como contradizer. Por isso o veredito `pouco_sustentada` existe e gera um aviso ao aluno.

### 3.4 Achado v1: premissa longa e pares sem relação

Na primeira versão, a premissa era o **chunk inteiro** da ficha. A resposta "fiel" saiu `contradiz_fonte` (entailment 0,18 no passo 1): o NLI perde precisão com premissas longas que misturam vários passos. A correção foi quebrar as fichas em **frases** e usar o máximo entre elas (a afirmação precisa ser sustentada por *alguma* frase).

## 4. Coerência lógica (5.2)

A resposta não pode se contradizer. A checagem roda NLI nos dois sentidos (a→b e b→a) entre pares de afirmações e marca conflito com contradição ≥ 0,8.

**Achado v1:** comparar todos os pares gerou conflito falso entre passos diferentes da mesma técnica ("Segure a manga" × "Passe a perna", p = 0,95). Para o NLI, duas ações diferentes parecem incompatíveis. A correção foi filtrar por assunto: só entram pares com ≥ 50% de palavras de conteúdo em comum (`same_topic`, com stopwords do domínio como "oponente", "durante", "sempre"). No teste controlado, o caso "incoerente" continuou detectado (p = 0,999) e a resposta "fiel" passou sem conflito.

## 5. Agente integrado (5.3)

### 5.1 Ferramentas

| Tool | Categoria | Origem |
| --- | --- | --- |
| `search_technique` | Semântica (RAG/FAISS) | Fase 2, índice reaproveitado na Fase 4 |
| `search_video` | API externa (YouTube) | Fase 2 |
| `search_wikipedia` | API externa (Wikipedia) | Fase 2 |
| `query_adcc` | Text-to-SQL (SQLite) | Fase 4 |
| `ibjjf_points` | Hardcoded | Fase 4 |

### 5.2 Montagem

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", AGENT_SYSTEM),
    MessagesPlaceholder("chat_history", optional=True),   # preenchido pelo 5.4
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),              # tool calls + observações
])
agent = create_openai_tools_agent(llm, TOOLS, prompt)
executor = AgentExecutor(agent=agent, tools=TOOLS, max_iterations=5,
                         handle_parsing_errors=True, return_intermediate_steps=True)
mestre = RunnableLambda(lambda x: verified_answer(executor.invoke(x), reviser))
```

- **`create_openai_tools_agent`** usa o protocolo de *tool calling* da OpenAI, que o Groq implementa. O `gpt-oss-120b` funciona bem nesse modo (ao contrário do ReAct em texto, Fase 2).
- **`return_intermediate_steps=True`** é o que permite ao verificador ler o que o `search_technique` devolveu e à avaliação extrair a trajetória.
- **`max_iterations=5`** limita o custo por pergunta (orçamento de 8000 tokens/min e 200k tokens/dia do Groq).

### 5.3 Verificação na saída (`verified_answer`)

1. Junta os trechos devolvidos pelo `search_technique` nos `intermediate_steps`.
2. Normaliza a citação nativa do gpt-oss (`【…】`) para o formato do projeto (`(fonte: ficha.md)`).
3. Roda `verify(resposta, trechos)`.
4. Se o veredito for `contradiz_fonte` ou `incoerente` e houver fonte: o **revisor** (prompt + LLM) reescreve só as frases problemáticas usando apenas as fontes. A resposta revisada é verificada de novo. Se ainda divergir, sai com aviso.
5. Se a resposta ensina passos numerados **sem** ter consultado o RAG: veredito `sem_fonte` e aviso "confirme com seu professor".
6. `pouco_sustentada`: aviso de que parte da resposta não está nas fichas.

Só **uma** revisão, por decisão: um loop de revisão até passar no NLI poderia gastar o orçamento do Groq e "otimizar para o verificador" em vez de para o aluno.

### 5.4 Resultados (uma pergunta por categoria)

| Pergunta | Tools | Categorias | Veredito | Revisada | Tempo |
| --- | --- | --- | --- | --- | --- |
| Como faço a kimura na guarda fechada? | `search_technique` | RAG | `ok` (suporte 0,83) | sim | 15,8 s |
| Quantos pontos vale uma passagem de guarda? | `ibjjf_points` | hardcoded | `ok` | não | 1,1 s |
| Qual a finalização mais comum da história do ADCC? | `query_adcc` | text-to-SQL | `ok` (RNC, 99) | não | 2,5 s |
| Me mostra um vídeo de raspagem de tesoura e me explica o primeiro passo | `search_technique` + `search_video` | RAG + API | `ok` (suporte 0,67) | não | 6,7 s |

O agente escolheu a tool certa nos 4 casos e combinou duas categorias no último.

### 5.5 Achados

- **Formato do gpt-oss poluindo o NLI.** Na v1, a resposta da kimura era fiel à ficha, mas o verificador marcou `contradiz_fonte` e disparou uma revisão desnecessária. Causa: os itens vinham como "**Título** – texto" com citação `【…】` colada, e o NLI comparava o título e a citação. A correção foi limpar os dois no `split_claims`.
- **Pergunta mista ignorava o RAG.** Em "Me mostra um vídeo… e me explica o primeiro passo", a v1 chamou só `search_video` e explicou o passo de memória. Sem trecho recuperado, o verificador não tinha o que checar e deu `ok`. Ou seja, **o verificador só protege o que passa pelo RAG**. Correções: regra "ALWAYS call search_technique before describing any step" no prompt e o veredito `sem_fonte` como rede de segurança em código.
- **Seção certa na busca.** No triângulo, a query sem "steps" recuperou a seção *Defenses*, e o agente ensinou a defesa como se fosse o ataque. A correção foi instruir o agente a incluir a parte desejada na query (`steps` para executar, `defenses` para defender/escapar).
- **Afirmações de segurança não são técnicas.** "Libere imediatamente ao sentir o tap" é exigido pelo prompt, mas não está em toda ficha, então derrubava o suporte. Avisos de segurança, links e menções a vídeo saem da verificação (`NON_TECHNICAL`).

## 6. Memória entre turnos (5.4)

```python
RunnableWithMessageHistory(
    build_mestre(llm),
    get_session_history,                 # dict session_id -> InMemoryChatMessageHistory
    input_messages_key="input",
    history_messages_key="chat_history", # = MessagesPlaceholder do prompt do agente
    output_messages_key="output",        # só o texto final verificado entra no histórico
)
```

`output_messages_key="output"` é importante: o runnable devolve um `dict` com veredito, tools e detalhes do NLI, mas só a resposta final vai para o histórico. Isso evita encher o contexto de metadados.

**Teste (3 turnos + sessão de controle):**

| Sessão | Pergunta | Tools | Comportamento |
| --- | --- | --- | --- |
| aluno-1 | Sou faixa branca e treino com kimono. Como faço a kimura na guarda fechada? | `search_technique` | Passo a passo da kimura (gi) |
| aluno-1 | E se ele defender puxando o braço? | `search_technique` | Entendeu "ele" = oponente e o contexto da kimura na guarda fechada |
| aluno-1 | Me mostra um vídeo **disso** | `search_technique` + `search_video` | Vídeo *Kimura From Closed Guard For White Belts* |
| aluno-2 | Me mostra um vídeo disso | — | Sessão nova, sem contexto: "Qual técnica ou posição você gostaria de ver em vídeo?" |

O histórico do aluno-1 terminou com 6 mensagens (3 pares) e a sessão aluno-2 ficou isolada. A pergunta de esclarecimento na sessão nova mostra que a referência foi resolvida pela memória, não por acaso.

**Relação com a Fase 3.** A Fase 3 construiu memória semântica explícita (`StudentProfile`, `FightState`) por extração de entidades. Aqui o agente final usa o histórico bruto, e o próprio LLM resolve as referências. É mais simples e suficiente para conversas curtas. O perfil persistente entre sessões ficou como próximo passo (§11).

## 7. Interface (5.5)

Duas interfaces sobre o **mesmo** `build_mestre_with_memory()`:

| | `app.py` (Streamlit) | `cli.py` (terminal) |
| --- | --- | --- |
| Comando | `python -m streamlit run app.py` | `python cli.py` |
| Sessão | `session_id` por aba (`web-<uuid>`), botão "Nova conversa" | `cli-<uuid>`, comando `nova` |
| Transparência | Expander por resposta: tools, categorias, veredito, suporte NLI, revisada, tempo | Linha de status com os mesmos campos |
| Erro (429, rede) | `st.error` sem derrubar a UI | Mensagem e segue o loop |

Decisões no Streamlit:

- **`@st.cache_resource` com aquecimento.** O FAISS (embeddings) e o mDeBERTa (~30 s em CPU) são carregados no start, não na primeira pergunta. O agente é criado uma vez por processo, e o histórico fica separado por `session_id`.
- **Duas memórias com papéis diferentes.** `st.session_state.messages` só redesenha a tela. A memória que o agente lê é a do `get_session_history`. A barra lateral mostra quantas mensagens o agente tem na memória da sessão.
- **`load_dotenv()` antes de importar `src`.** `tools.py` lê `USE_DUMMY` no import, então a ordem importa.

Testado em modo real: as 4 categorias de tool, memória entre turnos e isolamento entre duas abas.

## 8. Avaliação no LangSmith (5.6)

### 8.1 Dataset e avaliadores

Dataset `bjj-mestre-eval`, 8 exemplos cobrindo todas as categorias de tool, uma pergunta combinada (RAG + vídeo) e uma fora da base (berimbolo). Cada exemplo guarda `expected_tools` e uma resposta de referência.

| Avaliador | Tipo | Pergunta que responde |
| --- | --- | --- |
| `tool_recall` | Trajetória | O agente chamou todas as tools esperadas? |
| `tool_precision` | Trajetória | Das tools chamadas, quantas eram esperadas? (chamada sem tool esperada vazia = 1) |
| `quality` | LLM-as-judge (1–5 → 0–1) | Responde à pergunta, coerente com a referência, sem invenção, em português |
| `verified` | Verificador NLI | Veredito final `ok`? |

O juiz usa o `gpt-oss-20b` com `PydanticOutputParser` (`Grade{reasoning, score}`) e fallback para o 120b, porque o 20b às vezes devolve vazio.

### 8.2 Resultados

| Experimento | tool_recall | tool_precision | quality | verified |
| --- | --- | --- | --- | --- |
| mestre-v1 | 1,00 | 0,88 | 0,79 (1 NaN) | 0,62 |
| **mestre-v2** | **1,00** | **0,88** | **0,84** (sem NaN) | **0,62** |

Mudanças da v1 para a v2: seção na query (`steps`/`defenses`), regra "só mencionada ≠ ensinada", citação normalizada e juiz com fallback.

Por exemplo (v2):

| Pergunta | Tools | Veredito | recall | precision | quality | verified |
| --- | --- | --- | --- | --- | --- | --- |
| Quantos pontos vale uma montada? | `ibjjf_points` | ok | 1 | 1 | 1,00 | 1 |
| Me ensina a passar a guarda De La Riva e me mostra um vídeo | `search_technique`, `search_video` | ok | 1 | 1 | 1,00 | 1 |
| Como escapo da montada? | `search_technique` | pouco_sustentada | 1 | 1 | 0,50 | 0 |
| Como faço o berimbolo? | `search_technique`, `search_video` | pouco_sustentada | 1 | 0 | 0,25 | 0 |
| Como faço o triângulo da guarda fechada? | `search_technique` | contradiz_fonte | 1 | 1 | 1,00 | 0 |
| Quantas lutas terminaram em finalização em 2022 no ADCC? | `query_adcc` | ok | 1 | 1 | 1,00 | 1 |
| Quem foi Helio Gracie? | `search_wikipedia` | ok | 1 | 1 | 1,00 | 1 |
| Me mostra um vídeo de kimura | `search_video` | ok | 1 | 1 | 1,00 | 1 |

### 8.3 Análise das métricas

- **Roteamento resolvido.** O recall de 1,00 nas duas versões mostra que o agente sempre chama a tool necessária, inclusive na pergunta combinada. A seleção de ferramenta deixou de ser gargalo.
- **A precisão de 0,88 vem de um único exemplo, e o problema está no dataset.** No berimbolo, o esperado era `[]`, mas a regra do próprio `AGENT_SYSTEM` manda consultar `search_technique` antes de afirmar qualquer coisa sobre uma técnica e oferecer vídeo quando a base não cobre. O agente seguiu a política. O gabarito deveria ser `["search_technique"]` (e opcionalmente `search_video`). Isso mostra que **o gabarito de trajetória precisa acompanhar a política do agente**.
- **Qualidade e verificação medem coisas diferentes.** No triângulo, o juiz deu 1,00 e o NLI marcou `contradiz_fonte`. Na fuga da montada, o juiz deu 0,50 e o NLI `pouco_sustentada`. As duas métricas discordam em 2 de 8 casos. O NLI compara frases imperativas em português com fichas em inglês e erra para o lado do alarme (§10). O juiz não vê as fichas, só a referência curta. Nenhuma das duas sozinha basta. Por isso o projeto mantém as duas e trata o NLI como **alarme conservador**, não como nota.
- **`verified` = 0,62 parado entre v1 e v2.** As correções da v2 melhoraram a qualidade, mas os 3 casos não verificados são todos de RAG, onde o verificador atua. Os 5 casos `ok` incluem 4 sem fonte do RAG (IBJJF, ADCC, Wikipedia, vídeo), onde o verificador não tem o que checar. **A métrica superestima a fidelidade fora do RAG.** Essa é a principal limitação da fase (§10).
- **NaN do juiz.** Na v1, o `gpt-oss-20b` devolveu vazio no De La Riva → `OutputParserException` → NaN, o que distorce a média. O fallback para o 120b eliminou o NaN. Juiz pequeno é barato, mas precisa de rede de segurança.

## 9. Modo dummy (sem chaves)

`USE_DUMMY=true` faz o projeto rodar sem nenhuma chave de API. A rede só é necessária no primeiro download dos modelos locais do Hugging Face (embeddings e NLI):

- **Tools** (`src/tools.py`): cada tool devolve uma resposta fixa marcada com `(dummy)`.
- **LLM** (`src/chains.py`, `DummyChatModel`, devolvido por `build_llm()`): um `BaseChatModel` que simula o loop de tool calling. Na 1ª chamada, devolve um `tool_call` para `search_technique` com a pergunta. Depois da observação, devolve uma resposta final que cita o trecho. Sem tools ligadas, devolve um texto fixo. O `AgentExecutor`, o verificador NLI e o `RunnableWithMessageHistory` rodam de verdade.

Teste no notebook: 2 turnos na sessão `prof` → `search_technique` chamada nos dois, histórico com 4 mensagens. O Streamlit mostra "Modo: DUMMY (sem API)" na barra lateral.

## 10. Limitações

- **O verificador só cobre o `search_technique`.** Respostas de `search_wikipedia`, `query_adcc`, `ibjjf_points` e `search_video` não passam pela checagem de fidelidade (só pela de coerência). Um fato histórico errado vindo da Wikipedia sai com veredito `ok`. Estender a checagem exigiria usar as observações dessas tools como premissa. No caso do SQL, o número já vem do banco e o risco é menor.
- **NLI ruidoso em português imperativo vs. inglês descritivo.** Frases como "Mantenha o cotovelo a 90°" comparadas com "Keep the elbow at 90 degrees" podem sair com contradição alta (no teste da kimura, várias afirmações sustentadas tinham con > 0,9 ao mesmo tempo que ent > 0,6). Os limiares (0,5 / 0,8) foram calibrados em poucos casos.
- **Uma revisão só**, que nem sempre resolve: a resposta pode sair com o aviso de divergência.
- **Histórico em memória de processo.** O `InMemoryChatMessageHistory` some ao reiniciar o Streamlit e não é compartilhado entre processos. Para produção: `SQLChatMessageHistory` (SQLite) ou checkpointer do LangGraph.
- **APIs legadas por escolha.** `create_openai_tools_agent`, `AgentExecutor` e `RunnableWithMessageHistory` são exigidos pela rubrica e vêm de `langchain_classic` / estão marcados como *deprecated* no LangChain 1.x (o notebook mostra os `LangChainDeprecationWarning`). A alternativa atual é `create_agent` + checkpointer do LangGraph, já usada nas Fases 2 e 4 (`create_agent` + `MemorySaver`). As duas versões foram mantidas para comparação.
- **Dataset pequeno (8 exemplos)** e juiz `gpt-oss-20b`: suficiente para comparar versões e expor os achados, não para estimar uma taxa de acerto geral.
- **Limites do Groq free tier** (8000 tokens/min, 200k tokens/dia no 120b) obrigam a usar `time.sleep` entre perguntas e limitam o tamanho das avaliações.

## 11. Estado final e próximos passos

**Maturidade: nível 4.** O agente escolhe entre 5 tools de 4 categorias, mantém memória por sessão, verifica a resposta contra a evidência com NLI e coerência, revisa uma vez quando há divergência, avisa o aluno quando não há fonte e é avaliado no LangSmith com métricas de trajetória, qualidade e verificação.

Próximos passos:

1. Corrigir o gabarito do berimbolo no dataset (`expected_tools = ["search_technique"]`) e ampliar o dataset para ~30 perguntas.
2. Verificar também as observações de `search_wikipedia` (premissa = extrato da Wikipedia).
3. Seção `## Aliases` nas fichas + reindexação (correção sem custo do §7 da Fase 4).
4. Histórico persistente (`SQLChatMessageHistory`) e `StudentProfile` da Fase 3 entre sessões.
5. Migrar o agente final para `create_agent` + checkpointer do LangGraph.
