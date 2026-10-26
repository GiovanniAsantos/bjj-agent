# Relatório da Fase 3 — Memória, intenção e entidades

> Sprint 3 (26/10/2026). Rubrica 3.1 a 3.6.
> Artefatos: [`notebooks/03_memory.ipynb`](../notebooks/03_memory.ipynb), [`src/chains.py`](../src/chains.py), [`docs/evidencias/fase3_multiturno.json`](evidencias/fase3_multiturno.json), [trace público no LangSmith](https://smith.langchain.com/public/eda2d228-1b63-4177-a318-4d28cb6b6fe9/r).

## 1. Objetivo da fase

Fazer o Mestre **lembrar**: entender perguntas de acompanhamento ("e se ele defender isso?"), não perguntar de novo o que o aluno já disse (faixa, kimono) e nunca esquecer um fato crítico (lesão), mesmo em conversa longa e dentro do limite de 8000 tokens/min do Groq.

## 2. Mapa da rubrica

| Item | Onde está | Resumo |
| --- | --- | --- |
| 3.1 Taxonomia CoALA | Este relatório §3 | Trabalho, episódica, semântica e procedural mapeadas em objetos do projeto |
| 3.2 Janela, sumarização e busca vetorial | `03_memory` §3.2 | `trim_messages`, `ConversationSummaryBufferMemory`, `VectorStoreRetrieverMemory` (FAISS) × buffer, numa conversa de 8 turnos com fato crítico no turno 1 |
| 3.3 LangMem × Mem0 × Letta | Este relatório §5 | Comparação conceitual; decisão por implementação própria leve |
| 3.4 Chatbot com `ConversationBufferMemory` | `03_memory` §3.4 | Mestre com histórico via `MessagesPlaceholder`; mesma pergunta com e sem memória |
| 3.5 Intenção + entidades | `03_memory` §3.5 | `intent_chain` (v2 da fase 2) + `entity_chain` em paralelo; reescrita da pergunta; `FightState` e `StudentProfile` |
| 3.6 Teste multi-turno salvo | `03_memory` §3.6; trace + JSON | 6 turnos num único trace (`@traceable`), transcript em `docs/evidencias/` |

## 3. Taxonomia CoALA aplicada ao Mestre (3.1)

O CoALA (*Cognitive Architectures for Language Agents*, Sumers et al., 2023) separa a memória de um agente em **memória de trabalho** (o que está no contexto agora) e três memórias de **longo prazo**: episódica (experiências passadas), semântica (fatos sobre o mundo e o usuário) e procedural (como agir: código, prompts, políticas). O mapeamento no projeto:

| Memória CoALA | No BJJ Agent | Implementação | Escrita | Leitura |
| --- | --- | --- | --- | --- |
| **Trabalho** | Situação da luta em discussão: posição, técnica, ação do oponente; janela curta da conversa | `FightState` (Pydantic) + `trim_messages` (300 tokens) sobre o buffer | A cada turno, pelo merge das entidades extraídas | Entra no system do prompt de resposta e no prompt de reescrita |
| **Episódica** | Turnos passados da conversa, recuperáveis por similaridade | `VectorStoreRetrieverMemory` com FAISS em memória (§4.2) | `save_context` a cada turno | Busca pelos *k* turnos mais parecidos com a pergunta atual |
| **Semântica** | (a) Perfil do aluno: faixa, gi/no-gi, lesões, técnicas estudadas; (b) conhecimento técnico de BJJ | (a) `StudentProfile`; (b) fichas `data/kb/` + índice FAISS (`search_technique`, fase 2) | (a) Merge das entidades: campos sobrescrevem, listas acumulam; (b) indexação offline | (a) Sempre no prompt; (b) sob demanda pela tool |
| **Procedural** | Como o Mestre se comporta e decide | `MESTRE_SYSTEM` (PTCF + regras anti-alucinação), `PTCF_INTENT` + few-shot, regra de merge em código, roteamento por intenção | Pelo desenvolvedor (versões v1 → v2 medidas) | Sempre (system prompt e código) |

Duas decisões saem dessa taxonomia:

1. **Fato de perfil é memória semântica, não episódica.** A lesão no joelho não pode depender de "cair na janela" ou de "ser parecida com a pergunta". O §4.2 mostra as duas falhando exatamente nisso.
2. **Quem escreve na memória semântica é o código, não o LLM.** O extrator só propõe entidades da última mensagem ("never copy fields from the state"); a regra de merge decide o que fica. Isso evita que o modelo reafirme ou invente um fato antigo.

## 4. O que foi construído

### 4.1 Chatbot com `ConversationBufferMemory` (3.4)

```python
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
mestre_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", MESTRE_SYSTEM), MessagesPlaceholder("history"), ("human", "{question}"),
])
# por turno: load_memory_variables → invoke → save_context(input, to_text(resp))
```

`to_text` guarda só `context` + `steps` do `MestreResponse` (não o JSON inteiro) para economizar tokens a cada turno.

| Pergunta 3 ("E se ele defender isso?") | Resposta |
| --- | --- |
| **Com memória** | Continua o assunto do turno 2: libertar o braço preso mantendo a guarda fechada |
| **Sem memória** | `clarifying_question`: "Qual movimento ou posição você está tentando executar… gi ou no-gi, cima ou baixo?" |

O aluno disse "treino com kimono" só no turno 1 e o Mestre não perguntou de novo: o buffer carregou o fato. Custo: **542 tokens** de histórico após 3 turnos, crescendo linearmente.

### 4.2 Janela, resumo e busca vetorial (3.2)

Conversa roteirizada de 8 turnos (sem custo de LLM para encher as memórias); o turno 1 diz *"tenho uma lesão no joelho esquerdo"*. Pergunta final: *"Que raspagem da guarda aberta é segura pra mim?"*.

| Memória | Tokens injetados | Lesão no contexto | Resposta considera a lesão | Custo extra |
| --- | --- | --- | --- | --- |
| Buffer | 288 | sim | sim | 0 |
| Janela (`trim_messages`, 150 tokens) | 146 | **não** | **não** | 0 |
| Resumo + janela (`ConversationSummaryBufferMemory`, 150) | 262 | sim (resumo, em inglês) | sim | chamadas ao `gpt-oss-20b` a cada corte |
| Vetorial (`VectorStoreRetrieverMemory`, k=2) | 77 | **não** | **não** | 0 (embedding local) |

- A **janela** cortou o turno 1: comportamento esperado.
- A **vetorial** recuperou os turnos 8 e 3 (guarda aberta, postura): a busca encontra o **assunto** da pergunta, não a **restrição** do aluno.
- O **resumo** guardou a lesão, mas economizou pouco (262 × 288) porque saiu longo e em inglês; compensa só em conversa bem maior.
- A primeira medição usava `'joelho' in texto` e deu **falso positivo** em todas as linhas ("trave a perna atrás do joelho", "traga os joelhos ao peito"). O teste foi refeito com termos de lesão (`lesão`, `joelho esquerdo`, `injur`, `left-knee`, incluindo o hífen U+2011 que o 20b usou).

**Conclusão:** nenhuma memória de conversa sozinha resolve fato de perfil com custo baixo → perfil separado (§4.3).

### 4.3 Intenção, entidades e estado (3.5)

Pipeline por turno:

```
pergunta ─▶ condense_chain (20b) ─▶ pergunta autônoma ─┬─▶ intent_chain (120b, v2 da fase 2) ─┐
              ▲ histórico + estado                      └─▶ entity_chain (20b) ─────────────────┤ RunnableParallel
                                                                                               ▼
                                         update_memory ─▶ FightState + StudentProfile
                                                                                               ▼
                       intent == video ─▶ search_video (sem LLM)  |  senão ─▶ answer_chain (120b) com perfil + estado + janela
```

- `BJJEntities`: `position`, `technique`, `opponent_action`, `belt`, `gi`, `injuries`, todos em inglês.
- **Merge:** campo novo não-nulo sobrescreve; listas acumulam sem repetir; técnica citada entra em `studied_techniques`.
- **Prompt de resposta:** perfil + estado + janela de 300 tokens + *"NEVER suggest techniques that load an injured body part"*.
- **Vídeo sem 120b:** quando a intenção é `video`, a resposta é a própria busca, o que poupa ~1.500 tokens e elimina a contradição "não posso fornecer links" (§6).

Evolução da intenção nas 4 perguntas reais da conversa de teste (turnos 2 a 5):

| Versão | Acertos | O que mudou |
| --- | --- | --- |
| Classificador recebe a frase crua | 1/4 | "E se ele levantar?", "Que raspagem é segura?" → `clarify`; "Me mostra um vídeo disso" → `off_topic`, **sem vídeo** |
| + reescrita da pergunta (`condense_chain`) | 3/4 | "vídeo disso" → "vídeo do hip-bump sweep da guarda fechada" → `video` + link |
| + "sempre cite posição e técnica do estado" no prompt de reescrita | **4/4** | "E se ele levantar?" → "E se o adversário levantar e ficar em pé na guarda fechada?" → `technique` |

O padrão é o mesmo do `create_history_aware_retriever` do LangChain: a memória precisa chegar à etapa de **entender** a pergunta, não só à de responder. A fase 4 reaproveita a reescrita na busca.

### 4.4 Teste multi-turno salvo (3.6)

`run_conversation` com `@traceable(name="fase3-multiturno")` agrupa a conversa inteira em **um** trace (sem o decorator seriam ~18 traces soltos).

| # | Aluno | Reescrita | Intent | Resultado |
| --- | --- | --- | --- | --- |
| 1 | Sou faixa azul, treino com kimono e tenho lesão no joelho esquerdo. | (igual) | off_topic | Perfil: `blue`, `gi`, `left knee`; Mestre pergunta o que treinar "levando em conta a lesão" |
| 2 | Estou na guarda fechada e ele prendeu meu braço no chão. | Como escapar quando estou na guarda fechada e o oponente prendeu meu braço no chão? | technique | Estado: `closed guard`; resposta cita o joelho esquerdo |
| 3 | E se ele levantar e ficar em pé? | E se o adversário levantar e ficar em pé na guarda fechada? | technique | `opponent_action` atualizado; posição mantida |
| 4 | Que raspagem é segura pra mim agora? | Qual raspagem é segura… na guarda fechada enquanto o oponente fica em pé? | technique | Raspagem "sem sobrecarregar o joelho esquerdo" |
| 5 | Me mostra um vídeo disso. | Você pode me mostrar um vídeo da raspagem segura na guarda fechada? | video | Link do YouTube, sem chamada ao 120b |
| 6 | O que você já sabe sobre mim? | (igual) | off_topic | "Faixa azul, treina com kimono, lesão no joelho esquerdo…" |

- **Trace:** `run_id 01a17a98-9cbe-7d00-8edb-a97d0807332a`, 17 runs filhos, **27.143 tokens** no total → [link público](https://smith.langchain.com/public/eda2d228-1b63-4177-a318-4d28cb6b6fe9/r).
- **Transcript:** `docs/evidencias/fase3_multiturno.json`, com pergunta, reescrita, intenção, entidades, estado, perfil e resposta de cada turno.

## 5. LangMem × Mem0 × Letta (3.3)

| | **LangMem** (LangChain) | **Mem0** | **Letta** (ex-MemGPT) |
| --- | --- | --- | --- |
| O que é | SDK de memória de longo prazo para agentes LangGraph | Camada de memória plugável (open source + plataforma gerenciada) | Plataforma/servidor de agentes com estado |
| Modelo de memória | Semântica (fatos/coleções), episódica (exemplos) e procedural (otimização do prompt), sobre o `BaseStore` do LangGraph | Fatos extraídos das conversas, com escopo por usuário/agente/sessão; atualiza e deduplica fatos conflitantes | Hierarquia inspirada em SO: *core memory* (blocos no contexto, editáveis pelo agente), *recall* (histórico pesquisável) e *archival* (vetorial) |
| Quem escreve | Ferramentas de memória que o agente chama durante a conversa ou um gerenciador em segundo plano | Pipeline do Mem0 (LLM extrai e consolida) a cada `add` | O próprio agente, via ferramentas de auto-edição |
| Armazenamento | Store do LangGraph (memória, Postgres…) | Banco vetorial (+ grafo opcional) | Banco do servidor Letta |
| Encaixe no projeto | Alto: mesmo ecossistema (LangChain/LangGraph) | Médio: API própria ao lado do LangChain | Baixo: substitui o orquestrador; o agente passa a viver no servidor Letta |
| Custo/operação | Biblioteca; usa o LLM do projeto | OSS precisa de LLM + embeddings + banco; nuvem é paga acima do plano gratuito | Servidor a mais para operar |
| Risco com Groq free tier | Chamadas extras de extração/otimização | Extração por LLM a cada mensagem | Raciocínio de memória a cada passo do agente |

**Decisão: implementação própria leve, no padrão do LangMem.** O projeto já tem o que esses frameworks automatizam, em versão pequena e auditável: extração estruturada (`BJJEntities`), regra de merge em código, perfil semântico (`StudentProfile`) e memória episódica vetorial (FAISS). Motivos:

1. **Orçamento:** Mem0 e Letta adicionam chamadas de LLM por mensagem; com 8000 tokens/min e 200 mil tokens/dia, cada chamada conta (o teste do §4.4 já custou 27 mil).
2. **Controle:** o merge em código garante que lesão e faixa não sejam reescritas pelo modelo; nos três frameworks quem consolida é o LLM.
3. **Escopo:** Letta exigiria migrar o agente para outro runtime; Mem0, um serviço a mais. Para um único aluno por sessão, Pydantic + FAISS/SQLite basta.
4. **Caminho de evolução:** se o projeto crescer (vários alunos, perfis persistentes), o passo natural é o **LangMem** sobre o `Store` do LangGraph, o mesmo ecossistema do `create_agent` + `MemorySaver` da fase 2.

## 6. Resultados e achados

| Achado | Evidência | Consequência |
| --- | --- | --- |
| Memória resolve referência e evita repetir pergunta | 3.4: com memória continua o assunto; sem memória pede posição e gi | Buffer é a base, mas cresce linear (542 tokens em 3 turnos) |
| Janela e vetorial perdem fato de perfil | 3.2: lesão ausente no contexto e na resposta | Perfil vira memória semântica separada |
| Teste por palavra-chave engana | `'joelho'` casou com passos de técnica | Métrica com termos específicos; validar à mão |
| Classificador sem contexto erra follow-up | 1/4 sem reescrita → 4/4 com reescrita + regra de posição | `condense_chain` antes de intenção e entidades |
| Prompt e código desalinhados (de novo) | "Não posso fornecer links" enquanto o código entregava o link | Intenção `video` não passa pelo 120b |
| Memória lembra, mas o raciocínio pode aplicar errado | 3.2 (resumo): sugeriu gancho de De La Riva **com a perna esquerda**, a lesionada; 3.5: hip-bump sweep com o oponente **em pé** (técnica de oponente ajoelhado) | A validar pelo autor (faixa-roxa); reforça a verificação contra evidência da fase 5 |
| Erros técnicos sem base continuam | "guarda **fechurada**"; em execuções anteriores, "bridge" para soltar o braço na guarda fechada e "shrimp + kimura para escapar" | Justificam o RAG (fase 4) |
| Custo por turno | ~3 chamadas (reescrita 20b, intenção 120b ∥ entidades 20b, resposta 120b); 27 mil tokens em 6 turnos | Extração e reescrita no 20b preservam o limite diário do 120b |

## 7. Limitações

- **Estado não deduz transições:** com o oponente em pé, `position` continua `closed guard`; o merge só registra o que é dito. Possível causa da sugestão de hip-bump no turno 4.
- **Técnica vaga vira entidade:** no turno 5, `technique = "raspagem segura"`, porque o Mestre não nomeou a raspagem e o `to_text` não guarda nome de técnica → vídeo genérico.
- **Sem intenção para "atualizar perfil" ou meta-perguntas:** turnos 1 e 6 saem como `off_topic`, embora o Mestre responda bem.
- **Perfil não persiste entre sessões:** vive no kernel. Persistência em SQLite (ou `Store` do LangGraph) fica para a fase 5.
- **Título do YouTube com HTML escapado** (`&amp;`): corrigir com `html.unescape` em `search_video` (fase 4.2).
- **APIs legadas:** `ConversationBufferMemory`, `ConversationSummaryBufferMemory` e `VectorStoreRetrieverMemory` são deprecated (warnings no notebook); usadas por exigência da rubrica. A janela já usa a API atual (`trim_messages`).

## 8. Próximos passos (Fase 4)

1. `query_adcc` (text-to-SQL sobre `data/db/bjj.sqlite`) no lugar do placeholder `stats`.
2. Agente de recuperação com o índice FAISS existente e casos difíceis (gíria PT, fora da base, ambígua).
3. Reaproveitar a reescrita de pergunta como transformação de consulta (multi-query/HyDE) na busca.
4. Corrigir o `&amp;` do YouTube e documentar a tipologia das ferramentas.
