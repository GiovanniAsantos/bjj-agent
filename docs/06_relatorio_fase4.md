# Relatório da Fase 4 — Ferramentas, APIs, Text-to-SQL e RAG

> Sprint 3 (29–31/10/2026). Rubrica 4.1 a 4.5.
> Artefatos: [`notebooks/04_tools_rag.ipynb`](../notebooks/04_tools_rag.ipynb), [`src/tools.py`](../src/tools.py), [`src/text_to_sql.py`](../src/text_to_sql.py).
>
> **Status:** §1–§5 cobrem 4.1–4.3 (29/10). §6–§7 (4.4, 4.5) serão preenchidas após o agente de recuperação.

## 1. Objetivo da fase

Dar ao Mestre **ferramentas de natureza diferente** e saber quando cada uma é a certa: regra fixa (pontuação IBJJF), conhecimento curado (fichas de técnica via FAISS), mundo externo (YouTube, Wikipedia) e dados estruturados (resultados do ADCC em SQLite). Cada tool precisa ser segura, barata e devolver ao agente uma observação legível mesmo quando falha.

## 2. Mapa da rubrica

| Item | Onde está | Resumo |
| --- | --- | --- |
| 4.1 Tipologia das ferramentas | `04_tools_rag` §4.1; este relatório §3 | Nova tool hardcoded `ibjjf_points`; tabela hardcoded × semântica × API × text-to-query |
| 4.2 APIs por tipo de uso | `04_tools_rag` §4.2; este relatório §4 | Busca, consulta, local; retry com backoff, cache, timeout, dummy; ausência deliberada de tools de escrita |
| 4.3 Text-to-SQL sobre SQLite | `04_tools_rag` §4.3; `src/text_to_sql.py`; `@tool query_adcc` | Geração estruturada (`SQLQuery`), segurança em 2 camadas, retry com erro, avaliação 6/6 contra gabarito |
| 4.4 Agente de recuperação | *a preencher* | Índice FAISS existente (`load_local`), casos difíceis |
| 4.5 Falha real + correção | *a preencher* | Diagnóstico nos documentos-fonte, multi-query / HyDE |

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

*A preencher.*

## 7. Falha real nos documentos-fonte + correção (4.5)

*A preencher.*

## 8. Limitações (parcial)

- **Significado de `weight_class` e `importance`** foi inferido ("0–4 leve→pesado", "0–5 fase mais avançada") e ainda não confirmado na documentação do dataset no Kaggle. Uma pergunta por categoria de peso pode receber resposta com rótulo errado.
- **Avaliação SQL pequena** (6 perguntas + 2 de uso). Suficiente para expor os dois achados, não para estimar taxa de acerto geral.
- **Retry não resolve falhas sistemáticas** (§5.4); depende de o prompt evitar o padrão.
- **Cache em memória de processo** (`lru_cache`): some ao reiniciar e não é compartilhado entre usuários do Streamlit.
