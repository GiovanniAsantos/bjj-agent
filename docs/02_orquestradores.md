# Orquestradores: LangChain × LangGraph × CrewAI × AutoGen

> Fase 1 — rubrica 2.1. Contexto: escolher o orquestrador do BJJ Agent para as fases 2 a 5.

## 1. O que um orquestrador resolve

Um agente é mais do que uma chamada ao LLM: é preciso montar prompts, chamar ferramentas, guardar estado entre turnos, decidir o próximo passo e registrar o que aconteceu. O orquestrador é a camada que organiza isso. Os quatro avaliados partem de ideias diferentes:

| Orquestrador | Ideia central | Unidade de composição |
| --- | --- | --- |
| **LangChain** | Componentes padronizados (modelos, prompts, parsers, ferramentas, retrievers) encadeados | `Runnable` + LCEL (`\|`) |
| **LangGraph** | Fluxo do agente como **grafo com estado**: nós, arestas condicionais e ciclos | `StateGraph` (nós + estado tipado) |
| **CrewAI** | Equipe de agentes com **papéis** (pesquisador, redator...) que dividem tarefas | `Agent` + `Task` + `Crew` |
| **AutoGen** (Microsoft) | Agentes que **conversam entre si** para resolver um problema | Agentes + conversa (chat em grupo) |

## 2. Comparação

| Critério | LangChain | LangGraph | CrewAI | AutoGen |
| --- | --- | --- | --- | --- |
| **Fluxo de controle** | Sequencial/paralelo (LCEL); ciclos só via agente pronto | Explícito: arestas condicionais, ciclos, retomada | Processo sequencial ou hierárquico entre agentes | Emerge da conversa entre agentes |
| **Estado e memória** | Histórico de mensagens, memória por integração | Estado tipado central + checkpoints persistentes por conversa | Memória por agente/equipe | Histórico da conversa entre agentes |
| **Multiagente** | Não é o foco | Possível (subgrafos, supervisor) | Foco principal | Foco principal |
| **Ferramentas e integrações** | Maior ecossistema (Groq, FAISS, SQL, YouTube, Wikipedia) | Usa os mesmos componentes do LangChain | Integra ferramentas próprias e do LangChain | Ferramentas como funções Python |
| **Observabilidade** | LangSmith nativo | LangSmith nativo | Integrações de terceiros | Integrações próprias / OpenTelemetry |
| **Controle × abstração** | Alto controle nas chains | Controle máximo do fluxo | Alta abstração (papéis e tarefas) | Alta abstração (conversa) |
| **Custo em tokens** | Baixo (uma chamada por etapa) | Baixo a médio (depende dos ciclos) | Alto (vários agentes conversando) | Alto (várias rodadas de conversa) |
| **Curva de aprendizado** | Baixa | Média (pensar em grafo e estado) | Baixa | Média |

## 3. Relação entre LangChain e LangGraph

Não são concorrentes diretos. O LangChain fornece os **componentes** (modelo, prompt, parser, ferramentas) e o LangGraph fornece o **fluxo com estado** que os conecta quando há decisões e ciclos. Na versão 1.x do LangChain usada no projeto (`langchain==1.4.6`, `langgraph==1.2.15`), o agente pronto do LangChain já roda sobre o LangGraph. Na prática, o projeto usa os dois em camadas: LCEL para as chains fixas (fase 1) e LangGraph quando o agente precisar decidir, repetir e guardar estado (fases 2, 3 e 5).

## 4. Escolha para o BJJ Agent

**Escolha: LangChain (componentes + LCEL), com LangGraph para o fluxo do agente.**

| Requisito do projeto | Por que LangChain/LangGraph atende | Por que CrewAI/AutoGen não compensam |
| --- | --- | --- |
| Um agente (Mestre) com várias ferramentas | ReAct com ferramentas é o caso central | O problema não pede vários agentes com papéis distintos |
| Groq free tier: 8000 tokens/min | Cada etapa é uma chamada controlada; batch e paralelismo medidos na fase 1 | Agentes conversando multiplicam as chamadas e estouram o limite (o 429 já aparece com ~10 chamadas seguidas) |
| RAG com FAISS, Text-to-SQL no SQLite, YouTube e Wikipedia | Integrações prontas no ecossistema | Exigiriam adaptar ou reaproveitar componentes do LangChain de qualquer forma |
| Memória curta e longa (fase 3) | Estado tipado e checkpoints por conversa no LangGraph | Memória menos explícita e mais difícil de inspecionar |
| Verificação da resposta (fase 5): se não sustentada, refazer | Aresta condicional + ciclo no grafo ("verificar → refazer ou entregar") | O ciclo fica implícito na conversa, sem garantia de parada |
| Avaliação com LangSmith (fase 5) | Integração nativa, trace por etapa | Depende de integração adicional |
| Continuidade com a fase 1 | `chat_prompt`, `MestreResponse` e `search_video` são reaproveitados como estão | Exigiria reescrever |

**Quando CrewAI ou AutoGen fariam sentido:** se o projeto evoluísse para vários especialistas independentes — por exemplo, um agente "analista de luta" revisando vídeos, outro "preparador físico" e um "coordenador" combinando as visões — e houvesse orçamento de tokens para as conversas entre eles.

## 5. Riscos da escolha

- **LangGraph tem curva de aprendizado:** pensar em nós, estado e arestas é diferente de chains lineares. Mitigação: começar pelo agente pronto e só descer para o grafo explícito quando precisar da verificação com ciclo (fase 5).
- **Evolução rápida das APIs:** o LangChain mudou bastante entre versões. Mitigação: versões fixas no `requirements.txt`.
