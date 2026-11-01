# Relatório da Fase 4 — Ferramentas, APIs, Text-to-SQL e RAG

> Sprints 3–4 (29/10 e 01/11/2026). Rubrica 4.1 a 4.5.
> Artefatos: [`notebooks/04_tools_rag.ipynb`](../notebooks/04_tools_rag.ipynb), [`src/tools.py`](../src/tools.py), [`src/text_to_sql.py`](../src/text_to_sql.py).
>

## 1. Objetivo da fase

Dar ao Mestre **ferramentas de natureza diferente** e saber quando cada uma é a certa: regra fixa (pontuação IBJJF), conhecimento curado (fichas de técnica via FAISS), mundo externo (YouTube, Wikipedia) e dados estruturados (resultados do ADCC em SQLite). Cada tool precisa ser segura, barata e devolver ao agente uma observação legível mesmo quando falha.

## 2. Mapa da rubrica

| Item | Onde está | Resumo |
| --- | --- | --- |
| 4.1 Tipologia das ferramentas | `04_tools_rag` §4.1; este relatório §3 | Nova tool hardcoded `ibjjf_points`; tabela hardcoded × semântica × API × text-to-query |
| 4.2 APIs por tipo de uso | `04_tools_rag` §4.2; este relatório §4 | Busca, consulta, local; retry com backoff, cache, timeout, dummy; ausência deliberada de tools de escrita |
| 4.3 Text-to-SQL sobre SQLite | `04_tools_rag` §4.3; `src/text_to_sql.py`; `@tool query_adcc` | Geração estruturada (`SQLQuery`), segurança em 2 camadas, retry com erro, avaliação 6/6 contra gabarito |
| 4.4 Agente de recuperação | `04_tools_rag` §4.4; este relatório §6 | Índice da Fase 2 via `load_local`; diagnóstico do retrieval em 10 casos difíceis; agente `create_agent` com reescrita da query |
| 4.5 Falha real + correção | `04_tools_rag` §4.5; este relatório §7 | Fichas sem aliases + chunk "hub"; multi-query v1 (8/10) → v2 120b + RRF (10/10) |

## 3. Tipologia das ferramentas (4.1)

| Tipo | Tool | Quando usar | Custo | Determinismo | Risco |
| --- | --- | --- | --- | --- | --- |
| Hardcoded | `ibjjf_points` | Regra fixa e pequena (pontuação) | Zero, sem rede | Total | Desatualizar se a regra mudar |
| Semântica (RAG) | `search_technique` | "Como fazer/defender X" na KB curada | Embedding local (CPU) | Alto (mesmo índice → mesmo top-k) | Chunk errado se o vocabulário não casar (§7) |
| API externa | `search_video`, `search_wikipedia` | Vídeo / fato histórico fora da KB | Rede + quota | Baixo (resultado muda no tempo) | Timeout, quota, conteúdo não curado |
| Text-to-query | `query_adcc` | Estatística sobre dados estruturados | 1–2 chamadas LLM + SQL | Médio (o LLM gera o SQL) | SQL errado ou perigoso → read-only + validação |

**Por que uma tool hardcoded.** Pontuação IBJJF é conhecimento pequeno, estável e que precisa ser exato. Deixar o LLM "lembrar" ou buscar num RAG abre espaço para número alucinado. A tool é um `dict` com 6 ações:

```python
IBJJF_POINTS = {"takedown": 2, "sweep": 2, "knee on belly": 2,
                "guard pass": 3, "mount": 4, "back control": 4}
```

Entrada inválida não lança exceção: devolve `"Unknown action 'berimbolo'. Valid: takedown, sweep, ..."`. O agente recebe uma observação útil e pode se corrigir. A docstring lista as chaves válidas, porque o LLM só enxerga a descrição, não o `dict`.

**Fronteiras entre tools.** As docstrings delimitam o escopo de cada uma para o roteamento do agente: `search_wikipedia` diz "Do NOT use it to teach the step-by-step of a technique"; `query_adcc` diz "NOT to teach techniques". Assim "estatística de kimura" vai para o SQL e "como fazer kimura" vai para o RAG.

## 4. Integração de APIs por tipo de uso (4.2)

| Uso | Tool | Fonte | Timeout | Erro | Retry | Cache | Dummy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Busca (descoberta) | `search_video` | YouTube Data API v3 (quota 10.000/dia, 100 por busca) | 10 s | String → agente segue | 2× em 429/5xx | `lru_cache` 128 | Sim |
| Consulta (lookup factual) | `search_wikipedia` | Wikipedia REST (sem chave, exige User-Agent) | 10 s | String | 2× em 429/5xx | `lru_cache` 128 | Sim |
| Local (conhecimento curado) | `search_technique` | FAISS em disco | — | String | — | Índice carregado 1× | Sim |
| Local (dados estruturados) | `query_adcc` | SQLite read-only | — | String + retry com erro no prompt | — | Conexão e prompt 1× | Sim |
| Ação / escrita | nenhuma | — | — | — | — | — | — |

**Sem tools de escrita, por decisão.** O Mestre é tutor: não agenda, não posta, não paga. Isso elimina a necessidade de confirmação humana e de idempotência nesta fase.

### 4.1 Retry só para falhas transitórias

```python
_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])))
```

Repete apenas o que pode melhorar com o tempo (rate limit, erro de servidor), com backoff de 0,5 s e 1 s. Um 400/403 (chave inválida, requisição errada) falha na hora, porque repetir não resolve.

### 4.2 Cache sem cachear erro

O `lru_cache` guarda tudo o que a função retorna. Se ficasse direto na tool, que converte exceção em string, um `"YouTube API error"` ficaria preso no cache até reiniciar o processo. A solução foi separar duas funções:

- `_youtube_lookup` / `_wiki_lookup` (com `@lru_cache`): só fazem HTTP. Em falha, **lançam exceção**, e exceção não é cacheada.
- `search_video` / `search_wikipedia` (tools, sem cache): `USE_DUMMY`, normalização da chave (`lower().strip()`) e `try/except` que converte o erro em string para o agente.

Medição no notebook:

| Chamada | Tempo | Origem |
| --- | --- | --- |
| `search_wikipedia("Helio Gracie")` #1 | 0,998 s | Rede |
| `search_wikipedia("Helio Gracie")` #2 | 0,001 s | Cache (`hits=1, misses=1`) |

No YouTube o ganho principal é quota: cada busca repetida economiza 100 das 10.000 unidades diárias.

## 5. Text-to-SQL sobre SQLite (4.3)

**Dados:** tabela `lutas` (1.028 lutas do ADCC, 1998–2022), com várias armadilhas para um gerador ingênuo:

- `victory_method` contém o typo `DESQUALIFICATION`;
- `submission` é texto livre (`Heel hook`, `Inside heel hook`, `Outside heel hook` são a mesma família);
- `winner_points = -1` significa "sem dado" (660 de 1.028 linhas);
- `weight_class` é código 0–4 e fica `NULL` no absoluto.

### 5.1 Fluxo

```
pergunta → sql_prompt (schema + dicionário de colunas + few-shot + regras) → gpt-oss-20b
         → SQLQuery {reasoning, sql} → validate_sql → db.run (read-only)
         → (erro? devolve SQL + erro ao LLM, 1 nova tentativa)
         → answer_prompt (usa SOMENTE o resultado) → resposta em português + SQL usado
```

- **`SQLQuery` com `reasoning` antes de `sql`.** O modelo explicita colunas e filtros antes de escrever a query (chain-of-thought estruturado). O campo também fica como evidência.
- **Dicionário de colunas (`COLUMN_DICT`).** `get_table_info()` mostra só tipos e 3 linhas de amostra. Sem o dicionário, o modelo não teria como saber do `-1`, do typo ou das variações de heel hook.
- **Resposta separada da geração.** O LLM que redige a resposta vê só o resultado SQL e é instruído a não usar mais nada, o que reduz número inventado.
- **Modelo pequeno.** O SQL é uma tarefa curta e estruturada. O `gpt-oss-20b` preserva o orçamento diário do 120b (200k tokens/dia).

### 5.2 Segurança em duas camadas

| Camada | Mecanismo | Evidência |
| --- | --- | --- |
| 1. Conexão | `sqlite:///file:...?mode=ro&uri=true` | `DELETE FROM lutas WHERE 1=0` → `OperationalError: attempt to write a readonly database` |
| 2. Validação | `validate_sql`: só um `SELECT`/`WITH`, sem `;`, sem `INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/ATTACH/DETACH/PRAGMA/VACUUM` | `DROP TABLE lutas` → bloqueado; `SELECT 1; DELETE FROM lutas` → bloqueado; `PRAGMA table_info(lutas)` → bloqueado |

A camada 1 sozinha já impede escrita. A camada 2 rejeita **antes** de executar, com uma mensagem clara que alimenta o retry, e bloqueia também leituras indesejadas (`PRAGMA` expõe metadados). O caminho do banco é absoluto (`DB_PATH` relativo a `src/`), então a tool funciona igual no notebook, no Streamlit ou em script.

### 5.3 Avaliação contra gabarito

Seis perguntas, cada uma mirando uma armadilha, com gabarito calculado por SQL escrito à mão numa conexão `sqlite3` independente do LLM:

| Pergunta | Armadilha | Gabarito |
| --- | --- | --- |
| Finalizações em 2022 | Filtro por ano + método | 40 |
| Atleta com mais vitórias | Agregação por nome | Alexandre Ribeiro (30) |
| Vitórias por heel hook (todas as variações) | Grafias diferentes → `LIKE` | 52 |
| Finalização mais comum | `NULL` em `submission` | RNC (99) |
| Vitórias de Marcelo Garcia | Nome → `LIKE` | 27 |
| Média de pontos do vencedor com placar | `-1` = sem dado | 4,05 (sem o filtro daria 0,81) |

| Versão | Acerto | O que mudou |
| --- | --- | --- |
| v1 | 3/6 | Baseline |
| v1 + avaliador corrigido | 5/6 | A média estava certa (4,0489); o avaliador comparava com `ROUND(...,2)` por substring. Passou a comparar número com tolerância 0,01 |
| v2 | **6/6, todos na 1ª tentativa** | Regra "sempre terminar com LIMIT" (§5.4) |

### 5.4 Achado 1: SQL truncado pelo gpt-oss-20b

Na v1, heel hook e Marcelo Garcia falharam com `sqlite3.OperationalError: unrecognized token: "'%heel hook%"`. A saída crua do modelo:

```json
{"reasoning": "... filter with LIKE '%Marcelo Garcia%' and count the rows.",
 "sql": "SELECT COUNT(*) FROM lutas WHERE winner_name LIKE '%Marcelo Garcia%"}
```

O raciocínio estava correto, e até cita a string com as duas aspas. Mas quando o SQL **termina** numa string, o modelo engole a aspa simples que vem colada na aspa dupla de fechamento do JSON (`'"`). É um viés de tokenização, e **determinístico**:

- o retry com o erro no prompt **não corrigiu** o heel hook: o modelo repetiu o padrão nas 2 tentativas;
- o Marcelo Garcia passou só na 2ª tentativa, por variação, não por correção.

| Variante testada | Marcelo Garcia | Heel hook |
| --- | --- | --- |
| gpt-oss-20b original | `...Garcia%` ❌ | `...hook%` ❌ |
| gpt-oss-120b | ✅ | ✅ |
| **gpt-oss-20b + "sempre terminar com LIMIT"** | `...Garcia%' LIMIT 1` ✅ | `...hook%' LIMIT 1` ✅ |

A correção adotada foi a 3ª: regra no system prompt e few-shot com `LIMIT` em todos os exemplos. Se a string nunca é o último token, o bug não tem onde acontecer, e o modelo continua o barato.

**Lição:** retry com mensagem de erro serve para erro *lógico* (coluna errada, typo de valor), não para viés sistemático do modelo. Esse tipo de falha se resolve mudando o formato da saída ou trocando o modelo.

### 5.5 Achado 2: a correção escondeu um empate

Empacotada como `@tool query_adcc`, a pergunta "Qual a finalização mais comum entre as mulheres?" respondeu **"Toe hold, com 3 ocorrências"**. Conferindo no banco:

```
[('Toe hold', 3), ('RNC', 3)]
```

Era empate. O `LIMIT 1` introduzido no §5.4 escolheu um dos dois de forma arbitrária, e a resposta afirmou um único vencedor. Correção em duas pontas:

- **Geração:** regra "LIMIT 1 para agregado único (COUNT/AVG), LIMIT 5 para rankings, para que empates fiquem visíveis, LIMIT 20 nos demais"; few-shot de ranking com `LIMIT 5`.
- **Resposta:** "Se houver empate no topo, cite todos os empatados".

Resultado: *"As finalizações mais comuns entre as mulheres são Toe hold e RNC, cada uma ocorrendo 3 vezes."* (SQL com `LIMIT 5`).

**Lição:** uma restrição de formato tem efeito semântico, e só foi pega porque a resposta foi conferida contra o banco. Isso reforça a necessidade dos avaliadores da Fase 5.

### 5.6 Empacotamento

`src/text_to_sql.py` concentra a lógica: `SQLQuery`, `COLUMN_DICT`, `FEW_SHOT`, `validate_sql`, `get_db` e `build_sql_chains` (ambos com `lru_cache`, montados uma vez), `run_sql_question` (retry) e `answer_question`. A tool em `src/tools.py` é fina:

- `USE_DUMMY` devolve uma resposta fixa (testado);
- import tardio de `text_to_sql`, para que o LLM e o banco só carreguem se a tool for usada;
- `try/except` → string;
- retorno = resposta + SQL usado. A query fica visível ao agente e no trace, o que dá transparência sobre como o número foi obtido.

## 6. Agente de recuperação (4.4)

O índice da Fase 2 (`data/index/techniques/`, 36 chunks = 6 fichas × seções Position, Goal, Steps, Common mistakes, Defenses, Safety) é reaproveitado com `FAISS.load_local` via `load_technique_index()`, sem reindexar. Embedding: `paraphrase-multilingual-MiniLM-L12-v2`, local em CPU.

### 6.1 Retrieval isolado em casos difíceis

Antes do agente, o retrieval foi medido sozinho (`similarity_search_with_score`, k = 3, distância L2: menor = mais parecido). Assim, quando a resposta final erra, dá para separar a falha da busca da falha do LLM. As fichas estão só em inglês e não têm aliases.

| Query | Tipo | Esperado | Top-1 | Dist. | Top-1 ok | Top-3 ok |
| --- | --- | --- | --- | --- | --- | --- |
| armlock closed guard | Controle (EN exato) | armlock | armlock / Safety | 12,6 | ✅ | ✅ |
| chave de braço na guarda fechada | Gíria PT | armlock | armlock / Safety | 12,0 | ✅ | ✅ |
| juji gatame from guard | Nome japonês | armlock | kimura / Steps | 15,8 | ❌ | ❌ |
| raspagem de tesoura | PT | scissor_sweep | scissor_sweep / Common mistakes | 13,8 | ✅ | ✅ |
| ude garami | Nome japonês da kimura | kimura | dlr / Common mistakes | 14,8 | ❌ | ❌ |
| como sair de baixo da montada | PT descritivo | mount_escape_upa | dlr / Common mistakes | 12,4 | ❌ | ✅ |
| passar a guarda de la riva | PT + nome próprio | dlr_guard_pass | dlr / Common mistakes | 9,7 | ✅ | ✅ |
| trinagle chok | Erro de digitação | triangle | triangle / Goal | 15,8 | ✅ | ✅ |
| berimbolo | Fora da KB | — | dlr / Common mistakes | 18,9 | — | — |
| como saio daí? | Ambígua | — | dlr / Common mistakes | 15,6 | — | — |

**Top-1 5/8, top-3 6/8.** Observações:

- O embedding multilíngue resolve bem o português comum ("chave de braço", "raspagem de tesoura") e até o erro de digitação. Falha nos **nomes japoneses**, que não têm nenhuma palavra em comum com as fichas.
- **Chunk "hub":** `dlr_guard_pass / Common mistakes` foi top-1 em 5 das 10 queries. É curto e feito só de palavras genéricas do esporte (sweep, back, pass, guard, hips), por isso fica "perto de tudo" no espaço vetorial.
- **Distância não serve como limiar de abstenção.** O caso fora da KB (18,9) é pior que todos os acertos (máx. 15,8), mas "como saio daí?" (15,6) cai no meio deles.
- O "berimbolo" casou com o DLR porque a ficha **cita** a técnica ("the bottom player sweeps or takes the back (berimbolo)") sem ensiná-la. A KB menciona, mas não cobre.

### 6.2 Agente

```python
rag_agent = create_agent(llm, [search_technique], system_prompt=RAG_SYSTEM)
```

`create_agent` (LangChain 1.x, function calling nativo, gpt-oss-120b). O `RAG_SYSTEM` define 4 regras:

1. reescrever a pergunta como nome padrão da técnica em inglês antes de buscar;
2. responder só com as seções retornadas, citando a ficha;
3. admitir quando a KB não cobre o assunto;
4. pedir esclarecimento se não houver técnica nem posição.

`run_rag_agent` extrai a **trajetória**: o argumento de cada `tool_call` e as fichas presentes nos `ToolMessage`.

| Pergunta | Query enviada à tool | Fichas | Comportamento |
| --- | --- | --- | --- |
| Como faço chave de braço na guarda fechada? | `armlock closed guard` | armlock | ✅ Passo a passo citando a ficha e a seção |
| Como faço um juji gatame saindo da guarda? | `armbar closed guard` | armlock | ✅ Reescrita resolveu o caso que o retrieval puro errava |
| Me ensina o ude garami | — (não buscou) | — | ⚠️ Reconheceu "(kimura)" mas pediu a posição: a regra 4 foi aplicada demais |
| Como faço o berimbolo? | — (não buscou) | — | ✅ no resultado ("a base não inclui"), mas decidiu pela lista de fichas do prompt, não pelo retrieval |
| Como saio daí? | — | — | ✅ Pergunta de esclarecimento |

Dois achados:

- **A reescrita pelo agente funciona, mas tem viés de avaliação.** `juji gatame → armbar` e `ude garami → kimura` estavam como exemplos no próprio prompt, então o agente acerta esses casos porque foi ensinado. Essa correção não generaliza por construção, e é por isso que o §7 usa casos inéditos.
- **Descrever a KB no prompt deixa o agente pular o retrieval**, nos dois sentidos: recusa sem buscar (berimbolo) e pede esclarecimento sem buscar (ude garami). Isso é útil para economizar chamadas, mas torna o comportamento dependente do texto do prompt e não da base real. Se uma ficha nova for adicionada sem atualizar o prompt, o agente pode recusar o que sabe.

## 7. Falha real nos documentos-fonte + correção (4.5)

### 7.1 Diagnóstico

Falha escolhida: `ude garami → dlr_guard_pass / Common mistakes` e `juji gatame → kimura`.

**Causa nos documentos-fonte:** as fichas só usam o nome inglês (*Kimura from Closed Guard*, *Armlock from Closed Guard*). Não há aliases japoneses (ude garami, juji gatame, sankaku jime) nem em português (americana invertida, chave de braço). Sem sobreposição de vocabulário, o embedding cai no vizinho mais genérico, que é o chunk hub do §6.1.

Para um teste justo, 2 casos **inéditos**, fora de qualquer prompt, foram adicionados: `sankaku jime` (triângulo em japonês) e `fuga da montada com ponte` (upa em português). Avaliação com o mesmo `eval_retrieval` nos 10 casos dentro da KB.

### 7.2 Correção v1: multi-query (gpt-oss-20b, união pelo menor score)

Um LLM pequeno gera 3 reformulações em inglês (nome padrão + posição). Busca-se com a original + as 3 e une-se pelo melhor score de cada chunk. É o mesmo algoritmo do `MultiQueryRetriever` (`langchain_classic`), mas implementado à mão para manter os scores, já que o componente pronto devolve só a união dos documentos e não permitiria a tabela comparável.

Resultado: **top-1 6/10 → 8/10**. Resolveu `juji gatame` e o inédito `sankaku jime`. Duas falhas residuais, com causas diferentes:

- **`ude garami`: o modelo pequeno não conhece o termo.** As variações geradas foram `Arm Entanglement guard / side control / mount`, uma tradução literal do japonês, não o nome da técnica. O mesmo prompt no gpt-oss-120b gerou `Kimura from guard / side control / mount`. **O multi-query herda o conhecimento de domínio do LLM que reformula.**
- **`como sair de baixo da montada`: a fusão por menor score é frágil.** A query original continua no conjunto e traz o hub com distância 12,4. Basta um resultado ruim com score baixo para dominar o ranking. Mesmo com o 120b, no `ude garami` o kimura (14,3) venceu o hub (14,8) por apenas 0,5.

### 7.3 Correção v2: multi-query (gpt-oss-120b) + Reciprocal Rank Fusion

```python
for v in variants:
    for rank, (doc, dist) in enumerate(index.similarity_search_with_score(v, k=k)):
        entry = fused.setdefault(key, [doc, dist, 0.0])
        entry[2] += 1 / (60 + rank)          # RRF (Cormack et al., 2009)
ranked = sorted(fused.values(), key=lambda e: -e[2])[:k]
```

O RRF ignora a escala da distância e soma pontos por **concordância**: um chunk que aparece nas listas de várias reformulações sobe. O hub só aparece na lista da query original e é neutralizado.

| Query | Antes | v1 (20b, min) | v2 (120b, RRF) | Top-1 v2 |
| --- | --- | --- | --- | --- |
| armlock closed guard | ✅ | ✅ | ✅ | armlock / Safety |
| chave de braço na guarda fechada | ✅ | ✅ | ✅ | armlock / Safety |
| juji gatame from guard | ❌ | ✅ | ✅ | armlock / Steps |
| raspagem de tesoura | ✅ | ✅ | ✅ | scissor_sweep / Defenses |
| ude garami | ❌ | ❌ | ✅ | kimura / Common mistakes |
| como sair de baixo da montada | ❌ | ❌ | ✅ | mount_escape_upa / Steps |
| passar a guarda de la riva | ✅ | ✅ | ✅ | dlr / Common mistakes |
| trinagle chok | ✅ | ✅ | ✅ | triangle / Common mistakes |
| **sankaku jime** (inédito) | ❌ | ✅ | ✅ | triangle / Common mistakes |
| **fuga da montada com ponte** (inédito) | ✅ | ✅ | ✅ | mount_escape_upa / Defenses |
| **Top-1 / Top-3** | **6/10 · 7/10** | **8/10 · 9/10** | **10/10 · 10/10** | |

Os casos fora da KB (berimbolo, "como saio daí?") continuam recuperando *alguma* ficha, porque retrieval sempre devolve k resultados. A abstenção é responsabilidade do agente (regra 3) e do verificador NLI da Fase 5.

### 7.4 Trade-offs e alternativas

| Abordagem | Custo por busca | Generaliza para termo novo? | Observação |
| --- | --- | --- | --- |
| Reescrita pelo agente (§6.2) | 0 extra (já há a chamada do agente) | Só o que o modelo sabe; viés se os exemplos forem os testes | Sem controle sobre a query |
| Multi-query v1 (20b) | +1 chamada pequena | Parcial | Falha com jargão que o modelo pequeno não conhece |
| **Multi-query v2 (120b + RRF)** | +1 chamada (~400 tokens) e latência | Sim, nos casos testados | Adotada |
| HyDE | +1 chamada, saída longa | Sim | Melhor para perguntas vagas com resposta longa. Aqui o problema é de vocabulário (nome da técnica), que queries curtas atacam diretamente |
| **Corrigir a fonte:** seção `## Aliases` em cada ficha + reindexar | 0 em tempo de execução | Só os aliases listados | Ataca a causa raiz; complementar ao multi-query |

Precisão de seção: o top-1 da v2 acerta a **ficha**, mas a seção nem sempre é a mais útil (ex.: "Common mistakes" para uma pergunta "como fazer"). Como a tool devolve k = 3 seções, o agente costuma receber também "Steps". Um re-ranking por seção conforme a intenção (aprender → Steps; defender → Defenses) fica como melhoria futura.

## 8. Limitações

- **Significado de `weight_class` e `importance`** foi inferido ("0–4 leve→pesado", "0–5 fase mais avançada") e ainda não confirmado na documentação do dataset no Kaggle. Uma pergunta por categoria de peso pode receber resposta com rótulo errado.
- **Avaliação SQL pequena** (6 perguntas + 2 de uso). Suficiente para expor os dois achados, não para estimar taxa de acerto geral.
- **Retry não resolve falhas sistemáticas** (§5.4); depende de o prompt evitar o padrão.
- **Cache em memória de processo** (`lru_cache`): some ao reiniciar e não é compartilhado entre usuários do Streamlit.
- **Conjunto de retrieval pequeno** (10 casos na KB, 2 fora): mostra o mecanismo da falha e da correção, não estima recall geral. Os 10/10 da v2 não garantem robustez a jargão que nem o 120b conheça.
- **Multi-query v2 não está no `search_technique` de produção:** custaria +1 chamada do 120b por busca, somada à chamada do agente. Na Fase 5 o agente continua reescrevendo a query; a seção `## Aliases` nas fichas é a correção sem custo pendente.
- **O agente pula o retrieval** quando o prompt descreve a KB (§6.2); a lista de fichas no prompt precisa acompanhar a base.

## 9. Próximos passos (Fase 5)

- Verificador NLI (cross-encoder multilíngue) comparando cada frase da resposta com as seções recuperadas, mais checagem de coerência entre as próprias frases.
- `create_openai_tools_agent` + `AgentExecutor` com as 4 categorias de tool (hardcoded, RAG, API, text-to-SQL) e `RunnableWithMessageHistory`.
- Avaliação no LangSmith: qualidade da resposta e trajetória (tools esperadas × executadas), reaproveitando a extração de trajetória do `run_rag_agent`.
