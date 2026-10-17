# Arquitetura do BJJ Agent

> Fase 1 — rubrica 1.1 a 1.4 e 1.5 (parcial). Implementação correspondente: [`notebooks/01_mvp_chain.ipynb`](../notebooks/01_mvp_chain.ipynb).

## 1. Problema e escopo

Praticantes de jiu-jitsu fazem perguntas como *"como saio de um armlock na guarda fechada?"*, *"e se ele trocar para o triângulo?"* ou *"o armlock é comum no ADCC?"*. Um LLM sozinho responde com fluência, mas comete erros técnicos (lado da pegada, ordem dos passos) que podem levar a treino errado ou a lesões — a seção 7 mostra exemplos reais observados na fase 1.

O BJJ Agent — persona **Mestre** — responde essas perguntas com **base em evidência**: consulta fichas técnicas revisadas, dados de competição e fontes externas, adapta a resposta ao nível do aluno e confere a própria resposta antes de entregar.

**Fora do escopo:** substituir professor, diagnóstico de lesão, planejamento de treino físico.

## 2. Anatomia do agente (1.1)

### 2.1 Visão alvo

```
                    ┌──────────────────────────────┐
  Pergunta  ──────▶ │  System message: Mestre      │
                    │  (PTCF + regras anti-        │
                    │  alucinação e segurança)     │
                    ├──────────────────────────────┤
                    │  LLM: openai/gpt-oss-120b    │ ◀──▶ Memória curto prazo (sessão)
                    │  (Groq) — raciocínio e       │ ◀──▶ Memória longo prazo (perfil)
                    │  escolha de ferramenta       │
                    └──────────────┬───────────────┘
         ┌──────────────┬──────────┼─────────────┬──────────────┐
         ▼              ▼          ▼             ▼              ▼
  search_technique  query_adcc  search_video  search_wikipedia  Verificador
   (RAG / FAISS)    (SQLite)    (YouTube)     (Wikipedia)       (NLI)
```

| Componente | Implementação | Papel no agente |
| --- | --- | --- |
| **LLM** | `openai/gpt-oss-120b` via Groq, `temperature=0`, `max_retries=6` | Interpreta a pergunta, identifica o nível do aluno, redige a resposta e, a partir da fase 2, decide quando chamar ferramenta. É um modelo de raciocínio: o "pensamento" vem separado em `reasoning_content`, por isso o prompt não pede *"reason step by step"*. Temperatura 0 para respostas técnicas reproduzíveis; `max_retries` absorve o erro 429 do free tier (8000 tokens/min). |
| **System message** | `ChatPromptTemplate` (`chat_prompt`), papel `system`, estruturado em **PTCF** | **Persona:** Mestre, faixa-preta paciente e direto, que explica termos para iniciantes. **Task:** responder sobre técnicas, defesas, saídas e posições adaptando ao nível. **Context:** fontes permitidas, 5 regras anti-alucinação, segurança (*tap early*, sem conselho médico), identificação de nível (BEGINNER/ADVANCED) e idioma. **Format:** JSON do schema `MestreResponse` via `{format_instructions}`. |
| **Memória** | Fase 3 (rascunho na seção 6) | Curto prazo: contexto da conversa atual (posição, técnica discutida). Longo prazo: perfil do aluno (faixa, peso, jogo preferido, lesões). |
| **Ferramentas** | `search_video` já existe como função (fase 1); demais nas fases 2 e 4 | `search_technique` (RAG), `query_adcc` (Text-to-SQL), `search_video` (YouTube), `search_wikipedia`. Cada uma com descrição clara para o LLM saber quando usar. |
| **Base de conhecimento** | `data/kb/` (Markdown) → índice FAISS em `data/index/` | Fichas técnicas curadas e revisadas por faixa-roxa. É a fonte de verdade para passos e erros comuns: o LLM redige, a base fundamenta. |
| **Saída estruturada** | `PydanticOutputParser` com o schema `MestreResponse` | Contrato entre etapas: o código consome um objeto validado, não texto livre (detalhe na seção 2.3). |

### 2.2 O que já está implementado (fase 1)

Dois fluxos fixos, montados com LCEL (`|`):

```
Resposta:  {question} ─▶ chat_prompt ─▶ llm ─▶ PydanticOutputParser ─▶ MestreResponse
Vídeo:     {question} ─▶ search_prompt ─▶ llm ─▶ StrOutputParser ─▶ query ─▶ search_video() ─▶ (title, link)
```

| Peça | Tipo | Para que |
| --- | --- | --- |
| `chat_prompt` | `ChatPromptTemplate` (system + human `{question}`) | Conversa com persona e regras; `.partial()` injeta `format_instructions` uma vez só |
| `search_prompt` | `PromptTemplate` (texto único) | Tarefa curta e pontual: transformar a pergunta em busca do YouTube (técnica + posição + bjj, até 6 palavras) |
| `chain` | `chat_prompt \| llm \| parser` | Resposta validada (`invoke` e `batch`) |
| `chain_json` | `chat_prompt \| llm \| json_parser` | Mesmo schema, retorno em `dict`, usado para `stream` |
| `search_chain` | `search_prompt \| llm \| StrOutputParser()` | Gera a query de busca |
| `search_video(query)` | função Python (YouTube Data API v3) | Devolve `(title, link)` ou `(None, None)` — nunca um link inventado |

O link do vídeo é buscado por **código** e mostrado ao lado da resposta: o LLM não escreve URLs (regra anti-alucinação 2).

### 2.3 Schema `MestreResponse`

| Campo | Tipo | Quem preenche |
| --- | --- | --- |
| `level` | `Literal["BEGINNER", "ADVANCED"]` | Sempre |
| `context` | `str` | Sempre: situação e objetivo; resposta curta em pergunta conceitual; recusa em pergunta fora do tema |
| `steps` | `list[str] \| None`, máx. 5 | BEGINNER e ADVANCED |
| `common_mistake` | `str \| None` | Só BEGINNER |
| `details` | `list[str] \| None` | Só ADVANCED: pegadas, ângulos, timing, reação do oponente |
| `variations` | `list[str] \| None` | Só ADVANCED, quando houver confiança |
| `safety` | `str \| None` | Quando envolver finalização |
| `clarifying_question` | `str \| None` | Regra 4: a única pergunta de esclarecimento |

`JsonOutputParser` vs `PydanticOutputParser`:

| | `JsonOutputParser` | `PydanticOutputParser` |
| --- | --- | --- |
| Retorno | `dict` | objeto `MestreResponse` |
| Valida tipos e valores | não | sim (`Literal`, `max_length`, tipos) |
| Stream parcial | sim (dict crescendo) | não (precisa do JSON completo) |
| Uso no projeto | `stream` na interface | contrato entre etapas do agente |

## 3. Classificação: reativo, deliberativo ou híbrido (1.2)

| Tipo | Como decide | Exemplo no domínio |
| --- | --- | --- |
| Reativo | Estímulo → resposta, sem plano nem estado interno | *"O que é um armlock?"* → resposta direta |
| Deliberativo | Monta um plano, avalia alternativas, usa modelo do mundo | *"Monte minha sequência de ataques da guarda fechada considerando que sou leve"* |
| **Híbrido** | Camada reativa para o simples, deliberativa quando precisa | — |

**Hoje (fase 1): reativo com fluxo fixo.** Toda pergunta segue o mesmo caminho (prompt → LLM → parser), e o vídeo é buscado sempre, por código, sem decisão do agente.

**Alvo: híbrido.**

- **Camada reativa:** perguntas conceituais ou de definição respondidas direto pelo LLM com a system message, sem ferramenta. Rápido e barato — importante com o limite de 8000 tokens/min do Groq.
- **Camada deliberativa:** perguntas que exigem evidência ou várias fontes passam pelo ciclo ReAct (pensar → agir → observar), com várias chamadas de ferramenta e verificação no fim. Ex.: *"o armlock é comum no ADCC e como eu defendo?"* → `query_adcc` + `search_technique` + síntese. O próprio agente decide quando chamar `search_video`.
- **Por que não só reativo:** sem consulta à base, o modelo erra detalhe técnico mesmo com regras anti-alucinação (seção 7).
- **Por que não só deliberativo:** planejar para toda pergunta desperdiça tokens e latência sem ganho em perguntas simples.

## 4. Maturidade agêntica: atual → alvo (1.3)

Escala usada:

| Nível | Descrição |
| --- | --- |
| 0 | LLM puro: prompt solto, resposta em texto livre |
| 1 | Chain: prompt template + saída estruturada, fluxo fixo |
| 2 | Uso de ferramentas: o LLM escolhe **uma** ferramenta por pergunta (roteamento) |
| 3 | Agente ReAct: várias etapas, várias ferramentas, memória entre turnos |
| 4 | Agente autoverificável: checa a própria resposta contra evidência, é avaliado sistematicamente |

| | Nível | Evidência |
| --- | --- | --- |
| **Início (00_setup)** | 0 | `llm.invoke("What is the diference between an armlock and a triangle?")` com string crua |
| **Atual (fim da fase 1)** | 1 | `chat_prompt` (PTCF) + `PydanticOutputParser` (`MestreResponse`), `search_chain` com `PromptTemplate`, `invoke`/`batch`/`stream`; vídeo por código fixo |
| **Alvo (fase 5)** | 4 | ReAct + memória + RAG/SQL + verificação NLI + avaliação no LangSmith |

Caminho: fase 2 → nível 2/3 (ferramentas + ReAct; `search_video` vira ferramenta), fase 3 → nível 3 (memória), fase 4 → nível 3 completo (RAG e SQL), fase 5 → nível 4 (verificação e avaliação).

## 5. Decomposição hierárquica (1.4)

Objetivo raiz quebrado em subobjetivos até ações executáveis (cada folha = uma chamada de LLM ou ferramenta). A coluna da direita mostra o que já existe.

```
0. Responder a pergunta de BJJ com resposta fundamentada
├── 1. Entender a pergunta
│   ├── 1.1 Classificar intenção (técnica | estatística | vídeo | conceito | fora do escopo)
│   ├── 1.2 Identificar nível do aluno (BEGINNER | ADVANCED)          ← fase 1: regra no prompt → level
│   ├── 1.3 Extrair entidades (técnica, posição, atleta, ano, peso)    ← fase 1: parcial, em search_chain
│   └── 1.4 Resolver referências com a memória ("e se ELE trocar PARA O TRIÂNGULO?")
├── 2. Reunir evidência (só intenções que precisam)
│   ├── 2.1 Técnica → search_technique (RAG na base)
│   ├── 2.2 Estatística → query_adcc (gerar SQL → executar → ler resultado)
│   ├── 2.3 Contexto → search_wikipedia
│   └── 2.4 Vídeo → search_video                                       ← fase 1: código fixo
├── 3. Montar a resposta
│   ├── 3.1 Sintetizar evidência adaptando ao nível
│   └── 3.2 Estruturar em MestreResponse                               ← fase 1: PydanticOutputParser
├── 4. Verificar
│   ├── 4.1 Checar afirmações contra a evidência (NLI)
│   └── 4.2 Se não sustentada → refazer 2/3 ou declarar incerteza
└── 5. Atualizar memória
    ├── 5.1 Curto prazo: salvar turno e posição/técnica em discussão
    └── 5.2 Longo prazo: atualizar perfil se o aluno revelou faixa, peso, lesão
```

Exemplo — *"I'm a white belt. Give me 3 ways to pass a De La Riva guard"*: 1.1 = técnica, 1.2 = BEGINNER (faixa declarada), 1.3 = {técnica: passagem, posição: De La Riva} → 2.1 → 2.4 → 3 → 4 → 5.1.
Exemplo — *"O armlock é comum no ADCC?"*: 1.1 = estatística → 2.2 → 3 → 4.

## 6. Rascunho da memória (1.5 — parcial)

| | Curto prazo | Longo prazo |
| --- | --- | --- |
| **O que guarda** | Turnos da conversa atual; estado da "luta" em discussão (posição, quem está por cima, técnica atual) | Perfil do aluno: faixa, peso/categoria, jogo preferido (guardeiro/passador), lesões, técnicas já estudadas |
| **Para que** | Perguntas de acompanhamento: *"e se ele trocar para o triângulo?"* precisa saber que ainda é guarda fechada com braço preso | Personalizar: `level` vem do perfil em vez de ser inferido a cada pergunta; evitar técnica que force joelho lesionado |
| **Duração** | Sessão | Entre sessões |
| **Implementação prevista** | Histórico de mensagens no `chat_prompt` (`MessagesPlaceholder`), com janela/resumo para não estourar tokens | Store persistente por usuário (SQLite/JSON) + extração de fatos com saída estruturada (Pydantic) |
| **Risco** | Cada turno já gasta ~1.000–1.250 tokens; histórico longo estoura 8000 tokens/min | Guardar fato errado e propagar; dados sensíveis (lesões) → guardar só o necessário |

Esboço do schema de longo prazo:

```python
class StudentProfile(BaseModel):
    belt: Literal["white", "blue", "purple", "brown", "black"] | None = None
    weight_kg: float | None = None
    style: Literal["guard_player", "passer", "balanced"] | None = None
    injuries: list[str] = []
    studied_techniques: list[str] = []
```

Detalhamento e implementação na fase 3.

## 7. Evidências e achados da fase 1

| Achado | O que aconteceu | Consequência no projeto |
| --- | --- | --- |
| **Prompt e código desalinhados** | O prompt mandava chamar `search_technique`/`search_video` sem ferramentas ligadas → Groq `400 tool_use_failed`, depois respostas vazias (o modelo "chamava" a ferramenta dentro do raciocínio) | Prompt da fase 1 sem ferramentas; a versão com ferramentas volta na fase 2 junto com `bind_tools` |
| **Validação do Pydantic** | O modelo devolveu `null` em `details`/`variations` onde o schema esperava lista → `OutputParserException` | Campos opcionais como `list[str] \| None`; o parser **falha alto** em vez de passar dado errado adiante |
| **Erros técnicos sem base** | Kimura: *"segure o pulso direito dele com a sua mão direita"*; leglock: *"estabeleça a posição superior (mount, side control ou X-guard)"* — X-guard é guarda por baixo | Justifica RAG (fase 4) e verificação NLI (fase 5) |
| **Identificação de nível** | Sem faixa declarada, "pass a De La Riva guard" virou BEGINNER (regra *when in doubt*); com "white belt"/"brown belt" o nível veio certo | Regra conservadora mantida; perfil de longo prazo (seção 6) resolve |
| **Busca do vídeo** | Frase inteira + 1 resultado trouxe vídeo de triângulo para pergunta de kimura | `search_chain` (`PromptTemplate`) gera palavras-chave → vídeos coerentes (ex.: *"mount escape technique bjj"* → *The First 3 Mount Escapes You Need To Know in BJJ*) |
| **Idioma** | Perguntas em inglês respondidas em português; com os campos do schema renomeados para inglês, parte passou a vir em inglês | Ainda em aberto: algo no prompt ainda puxa o português |
| **Stream** | Com o mesmo código e o mesmo prompt, o Groq entregou o JSON final ora em **1 bloco**, ora em **258 pedaços**; o raciocínio sempre vem aos pedaços | `stream` depende do provedor e varia entre execuções; na interface, mostrar indicador de "pensando" |
| **Tamanho do prompt** | Prompt reduzido de ~1.080 para ~690 tokens (fora as `format_instructions`), mantendo as regras anti-alucinação | Cabe mais chamadas por minuto no free tier |

### `batch` × `invoke`

Mesmas 2 perguntas (white belt / brown belt), `max_concurrency=2`, 2 rodadas:

| Rodada | `batch` | `invoke` sequencial |
| --- | --- | --- |
| 1 | 2,3 s | 4,0 s |
| 2 | 2,7 s | 5,6 s |

O `batch` fica perto do tempo de **uma** chamada (as duas rodam em paralelo); o sequencial soma as duas. O limite é o rate limit do Groq: com muitas perguntas, `max_concurrency` baixo + `max_retries` evitam o 429.
