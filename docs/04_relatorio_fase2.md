# Relatório da Fase 2 — Agente ReAct, chains e confiabilidade

> Sprint 2 (20–25/10/2026). Rubrica 2.2 a 2.6 (2.1 foi entregue na fase 1, em `02_orquestradores.md`).
> Artefatos: [`notebooks/02_react_agent.ipynb`](../notebooks/02_react_agent.ipynb), [`notebooks/02b_chains_reliability.ipynb`](../notebooks/02b_chains_reliability.ipynb), [`src/tools.py`](../src/tools.py), [`data/kb/`](../data/kb/), [`data/index/techniques/`](../data/index/techniques/).

## 1. Objetivo da fase

Sair do fluxo fixo da fase 1 (toda pergunta → mesma chain → vídeo sempre) para um Mestre que **decide**: escolhe ferramentas, roteia a pergunta pela intenção, consulta uma base de conhecimento própria e continua de pé quando o LLM ou o parser falham.

## 2. Mapa da rubrica

| Item | Onde está | Resumo |
| --- | --- | --- |
| 2.2 Ferramentas + agente ReAct | `02_react_agent` células 1–11 | `search_video` e `search_wikipedia` como `@tool`; agente com function calling (`create_agent` + `MemorySaver`) e ReAct legado (`initialize_agent`, `ZERO_SHOT_REACT_DESCRIPTION`) v1 → v2 |
| 2.3 Chains sequencial, paralela e router | `02b` §1–3 | `RunnablePassthrough.assign` → ação; `RunnableParallel`; `RunnableBranch` × `MultiPromptChain` |
| 2.4 Integração com APIs externas | `src/tools.py`; `02_react_agent` célula 2 | YouTube Data API v3 e Wikipedia REST, com timeout, erro devolvido como texto e modo `USE_DUMMY` |
| 2.5 PTCF + few-shot + chain-of-thought | `02b` §4 | Classificador de intenção v1 (zero-shot) × v2 (PTCF + 6 exemplos + campo `reasoning`) |
| 2.6 Confiabilidade | `02b` §5–8 | `OutputFixingParser` × `RetryWithErrorOutputParser`, `with_retry`/`with_fallbacks`; memórias buffer/summary/token; tool FAISS com índice em disco; callback `MestreMonitor` |

## 3. O que foi construído

### 3.1 Ferramentas (`src/tools.py`)

| Tool | Fonte | Quando o agente deve usar | Falha |
| --- | --- | --- | --- |
| `search_video` | YouTube Data API v3, 1 resultado | Só quando o aluno quer **ver** a técnica | `"YouTube API error: ..."` / `"No video found."` |
| `search_wikipedia` | Wikipedia REST (busca → resumo, 800 caracteres) | História, atletas, academias; nunca passo a passo | `"Wikipedia API error: ..."` |
| `search_technique` | FAISS local (`data/index/techniques/`), top-3 | Como executar, defender ou escapar | `"Knowledge base error: ..."` |

As três devolvem **texto, nunca exceção**: o agente lê o erro como observação e decide (responder "não tenho certeza" em vez de quebrar). Com `USE_DUMMY=true` devolvem respostas fixas, para rodar sem chaves. As docstrings são o "manual" que o LLM lê para escolher a tool, por isso dizem quando usar, quando **não** usar e o formato da entrada.

### 3.2 Agentes (`02_react_agent`)

| Agente | Como chama a tool | Resultado (3 perguntas: vídeo, Helio Gracie, conceito/memória) |
| --- | --- | --- |
| `create_agent` (function calling) + `MemorySaver` | JSON nativo da API | 3/3: `search_video('mount escape bjj')`, `search_wikipedia('Helio Gracie')`, e "Sobre qual pessoa eu te perguntei antes?" respondido pela memória da thread, sem tool |
| ReAct v1 — prompt padrão + `gpt-oss-120b` | Texto `Action:` / `Action Input:` | Falhou: pulou a tool de vídeo e inventou URL (`watch?v=example1`); biografia sem rótulo `Final Answer:` → loop até `iteration limit`; `**Final Answer:**` em markdown → parser pegou só `**` |
| ReAct v2 — prompt do Mestre + `gpt-oss-120b` | Texto | Continuou pulando tools; com "you MUST call search_video" emitiu tool call nativo sem tools ligadas → `400 tool_use_failed` |
| ReAct v2 — prompt do Mestre + `qwen/qwen3.8-27b` | Texto | 3/3: segue o protocolo Thought/Action/Observation |

O v2 trocou três peças do `initialize_agent`: `prefix` com as regras anti-alucinação, `format_instructions` sem markdown nos rótulos e `handle_parsing_errors` com mensagem instrutiva em vez de `True`.

### 3.3 Chains (`02b` §1–3)

- **Sequencial:** `RunnablePassthrough.assign(info=intent_chain) | RunnableLambda(act_on_intent)`. O passo 1 classifica e o passo 2 só busca vídeo quando `intent == "video"`. Isso resolve a limitação da fase 1 ("toda pergunta busca vídeo").
- **Paralela:** `RunnableParallel(info=..., answer=...)`. Classificar e responder são independentes: **1,8 s × 2,6 s** sequencial (mesma pergunta, 15 s de pausa entre as medições).
- **Router:** `RunnableBranch` com 5 destinos (technique → `MestreResponse`, history → Wikipedia + prompt de evidência, video → `search_video`, stats → placeholder da fase 4, fallback → recusa ou pergunta de esclarecimento).

### 3.4 Base de conhecimento e índice

6 fichas em `data/kb/` (armlock e triângulo da guarda fechada, raspagem de tesoura, upa, kimura, passagem da De La Riva), com seções fixas: Position, Goal, Steps, Common mistakes, Safety. `MarkdownHeaderTextSplitter` gera 36 pedaços, indexados com `paraphrase-multilingual-MiniLM-L12-v2` em FAISS (9 s em CPU) e salvos em disco. `search_technique` carrega com `FAISS.load_local` + `lru_cache`, e a fase 4 reaproveita o índice sem reindexar.

## 4. Decisões de projeto

| Decisão | Motivo |
| --- | --- |
| `gpt-oss-120b` como modelo principal; `qwen3.8-27b` **só** no ReAct legado | O gpt-oss é modelo de raciocínio treinado para function calling: 3/3 no `create_agent`, mas não segue o protocolo de texto do ReAct (pula tool, inventa URL, emite tool call nativo → `400`). O qwen segue o texto. Trocar o modelo do agente legado foi mais barato que lutar contra o formato. |
| `create_agent` (LangGraph) para os próximos passos; `initialize_agent` só como requisito | `initialize_agent` está deprecated (warning na célula 6), faz parse frágil de texto e não tem checkpoint; o `create_agent` usa JSON nativo e memória por thread. Coerente com `02_orquestradores.md`. |
| `RunnableBranch` em vez de `MultiPromptChain` | Decisão tipada (Pydantic) e condição em Python; roteia para qualquer Runnable (tool, chain, função), não só para prompt; saída validada. O `MultiPromptChain` respondeu de memória, sem fonte, e a kimura veio em markdown livre, sem `safety`. |
| Classificador v2 em PTCF + few-shot | Define cada intenção e quando usar `clarify`/`off_topic`; os exemplos fixam o JSON e o `search_term` em inglês (§5). |
| CoT curto (`reasoning` antes de `intent`) | O gpt-oss já raciocina internamente (`reasoning_content`). O campo serve para **auditar** a decisão, não para "pensar passo a passo", e fica curto para gastar menos tokens. |
| `RetryWithErrorOutputParser` > `OutputFixingParser` | O fixing só vê a saída quebrada: inventou passos em uma execução e trocou o conteúdo por uma recusa em outra. O retry reenvia o prompt original e fechou o JSON truncado sem inventar (igual nas 2 execuções). Fixing só para erro puramente de formato. |
| `with_retry` + `with_fallbacks` com resposta segura | Com modelo inexistente, a chain devolve um `MestreResponse` válido ("I'm not sure...") em vez de exceção: a interface nunca quebra. |
| Memória: janela curta + perfil separado (fase 3) | O buffer cresce linear e estoura os 8000 tokens/min; o summary custa 1 chamada por turno e só compensa em conversa longa; o token buffer é barato, mas esqueceu faixa, peso e kimono (§5). |
| Embedding multilíngue local | Perguntas em PT e fichas em EN; roda em CPU, custo zero. |
| Prefixar `técnica - seção` no texto de cada pedaço | O splitter tira os títulos e um pedaço "Common mistakes" não dizia de qual técnica era: ficha certa no topo **2/6 → 6/6**. |
| `allow_dangerous_deserialization=True` | O `index.pkl` é pickle. É seguro só porque o índice é gerado pelo próprio projeto; nunca carregar índice de fonte desconhecida. |
| Callback local **e** LangSmith | O callback é uma regra que roda junto com a chain (orçamento de tokens, alerta, erro antes da exceção); o LangSmith fica com trace e avaliação (fase 5). |

## 5. Resultados e achados

### 5.1 Classificador de intenção (8 casos ambíguos, fora dos exemplos do few-shot)

| Classificador | Intenção certa | `search_term` coerente |
| --- | --- | --- |
| v1 zero-shot | 6/8 | 6/8 |
| v2 PTCF + few-shot + CoT | **8/8** | **8/8** |

O v1 errou os dois casos realmente ambíguos, ambos como `technique`: *"Como saio daqui? Ele tá me amassando"* (sem posição → `clarify`) e *"Como perder 5kg pra competir no sábado?"* (dieta → `off_topic`). O v1 também deixava `search_term=None` às vezes, e aí o router mandava a frase em português para a Wikipedia em inglês: *"Quem foi Helio Gracie?"* trouxe o artigo do **Leandro Lo**. O prompt de história respondeu "não tenho certeza" em vez de inventar. Com o v2, a evidência veio certa.

### 5.2 Parser e fallback

| Mecanismo | Entrada | Resultado |
| --- | --- | --- |
| `PydanticOutputParser` puro | `level="INTERMEDIATE"` + texto após o JSON | `OutputParserException` (falha alto) |
| `OutputFixingParser` | mesma | JSON válido, conteúdo não confiável (inventou passos / virou recusa) |
| `RetryWithErrorOutputParser` | JSON truncado + prompt original | Fechou o JSON mantendo só o que existia |
| `with_retry` + `with_fallbacks` | modelo inexistente | `MestreResponse` seguro, sem exceção |

### 5.3 Memória (mesma conversa de 4 turnos)

| Memória | Tokens no histórico | O que guardou | Custo extra |
| --- | --- | --- | --- |
| Buffer | 195 | tudo, literal | 0 |
| Summary | 191–196 | tudo, resumido (em inglês) | 1 chamada de LLM por turno |
| Token (limite 120) | 99 | só os 2 últimos turnos | 0 |

No follow-up *"E se ele defender a raspagem?"*, buffer e token resolveram a referência (raspagem de tesoura).

### 5.4 Recuperação e agente RAG

| Índice | Ficha certa no top-1 (6 perguntas PT/EN) |
| --- | --- |
| Pedaço = só o texto da seção | 2/6 (5 erradas apontando para `dlr_guard_pass.md`) |
| Pedaço = `técnica - seção` + texto | **6/6** |

Com o índice corrigido, o agente chamou `search_technique('triangle choke closed guard defense')` uma vez e baseou a defesa do triângulo na ficha.

### 5.5 Custo observado (`MestreMonitor`)

| Fluxo | Chamadas de LLM | Tokens |
| --- | --- | --- |
| Router, ramo history (classificar + responder) | 2 | ~1.610 |
| Router, ramo video (classificar + tool) | 1 | ~1.250 |
| Agente RAG com índice **antigo** | 4+ (`search_technique` repetida) | ~2.950 até o `429` |

O agente com o índice antigo repetia a busca procurando a ficha certa e bateu o **limite diário** do Groq (200.000 tokens/dia no gpt-oss-120b). O `on_llm_error` registrou o erro antes da exceção. Uma execução completa do 02b custa ~60 mil tokens.

## 6. Limitações

- **Evidências pendentes de reexecução:** a célula do agente RAG + callback (02b §7–8) ainda mostra a execução com o índice antigo, e a célula 11 do `02_react_agent` (ReAct v2 com gpt-oss × qwen) está sem saída salva. Os resultados descritos em §3.2 vêm da execução registrada no diagnóstico do notebook; rodar de novo quando o limite diário liberar.
- **Base pequena:** 6 fichas. Pergunta fora delas (ex.: berimbolo) depende do prompt para o agente admitir que não sabe; casos difíceis serão testados na fase 4.
- **Wikipedia rasa:** o resumo de 800 caracteres não traz detalhes como o criador do berimbolo; o Mestre responde "não tenho certeza" (certo, mas pouco útil).
- **Dois modelos no projeto:** o ReAct legado depende do `qwen3.8-27b`, e a lista de modelos do Groq muda (não há mais llama-3.x nesta conta).
- **Orçamento do Groq:** 8000 tokens/min e 200.000/dia obrigam `time.sleep` entre chamadas e limitam medições repetidas.
- **Stats sem dados:** o ramo `stats` do router ainda é placeholder.

## 7. Próximos passos (Fase 3)

1. Memória de curto prazo com janela (`trim_messages`) e resumo (`ConversationSummaryBufferMemory`), e memória episódica vetorial.
2. Perfil do aluno (`StudentProfile`) como memória semântica de longo prazo, alimentado por extração de entidades com Pydantic.
3. Reaproveitar o classificador v2 como reconhecimento de intenção no chatbot multi-turno.
4. Trace de uma conversa multi-turno salvo no LangSmith como evidência.
5. Fase 4: `query_adcc` (text-to-SQL) no lugar do placeholder `stats`.
