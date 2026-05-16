# SHL Assessment Recommender — Approach Document

**Candidate submission for SHL Labs AI Intern Role**

---

## 1. Problem Decomposition

The task breaks into four independent sub-problems:

1. **Data acquisition** — Scrape the SHL catalog (Individual Test Solutions only) into a structured JSON store at build time, so the runtime service never makes live HTTP calls for catalog data.
2. **Retrieval** — Given a user's hiring intent, find the most relevant assessments quickly (<5s) using semantic similarity.
3. **Dialogue management** — Decide whether to clarify, recommend, refine, compare, or refuse given the current conversation state.
4. **Schema enforcement** — Guarantee every response conforms to the exact evaluator schema and that every URL is a real catalog URL (zero hallucination tolerance on URLs).

---

## 2. Architecture

```
POST /chat
    │
    ▼
Intent Extraction (LLM call #1, temp=0)
    │  → job_role, job_level, test_types_wanted, is_vague,
    │    is_comparison_request, is_refinement, search_query
    │
    ▼
Catalog Search  ──────────────────────────────────────┐
  FAISS IndexFlatIP + sentence-transformers            │
  (all-MiniLM-L6-v2, cosine similarity)               │
  Hard filters: test_type, job_level, remote_only      │
  Returns top-15 candidates                           │
    │                                                  │
    ▼                                                  │
Response Generation (LLM call #2, temp=0.2)   ◄───────┘
  System prompt + catalog context injected
  LLM picks 1-10 from the provided list
    │
    ▼
Validation layer
  Every URL cross-checked against catalog
  Hallucinated recommendations dropped silently
  Pydantic schema enforced
    │
    ▼
ChatResponse { reply, recommendations, end_of_conversation }
```

**Two LLM calls per turn** is a deliberate design choice. The first call (intent extraction, zero temperature) is structured and deterministic — it drives retrieval without relying on the LLM's priors about SHL products. The second call generates the actual reply grounded in retrieved context. This separation makes hallucination much harder: the LLM can only recommend what was provided in the prompt.

---

## 3. Catalog Scraping

The catalog at `shl.com/solutions/products/product-catalog/` is JavaScript-rendered. The scraper (`scrape_catalog.py`) uses paginated HTTP requests with `type=1` to filter Individual Test Solutions, then fetches each product's detail page for description, job levels, and language availability. Rate limiting (0.4-0.5s delay) is applied to stay polite. The scraper is run once at build/deploy time and its output (`data/shl_catalog.json`) is baked into the container.

---

## 4. Retrieval Strategy

**Embedding model**: `all-MiniLM-L6-v2` (22M params, ~80ms encode latency for a query). Chosen because it runs on CPU, has no API cost, and is well-calibrated for semantic similarity on short texts.

**Index**: FAISS `IndexFlatIP` (exact inner product = cosine after L2 normalization). Flat index is fine for ~400 items — approximate indices add complexity with no latency benefit at this scale.

**Query construction**: The intent extractor produces a `search_query` string (e.g. "Java developer mid-level stakeholder communication skills") optimized for retrieval rather than using the raw user message. This improves recall significantly over raw-message search.

**Hard filters**: Applied before vector search — test type letters (A/B/C/D/E/K/P/S), job level, and remote_testing flag. Filtering before scoring avoids polluting the top-K with irrelevant results.

**Fallback**: If `sentence-transformers` or `faiss` is unavailable (e.g. cold import during tests), a TF-style keyword scorer is used transparently.

---

## 5. Prompt Design

**System prompt** encodes four explicit behaviors (CLARIFY/RECOMMEND/REFINE/COMPARE) with clear trigger conditions: a recommendation requires at least two of {role, level, domain, purpose}. The "enough context" rule prevents premature recommendations on vague queries.

**Catalog injection**: The top-15 retrieved assessments are injected as a structured block in the user turn immediately before the model generates its reply. This grounds the LLM completely — it has no incentive to recall product names from training weights because a better answer is right there in context.

**Output format**: Strict JSON schema is enforced in the prompt. A post-processing validation layer then re-checks every recommendation URL against the catalog regardless of LLM compliance.

---

## 6. Evaluation

Three evaluation layers:

| Layer | What it checks | Location |
|---|---|---|
| Unit probes | Schema, refusal, vague query behavior, dedup | `evaluate.py --probes-only` |
| Trace replay | Multi-turn recall@10, type coverage | `evaluate.py --traces-only` |
| Live stress test | Timeout compliance, concurrent load | Manual / locust |

**What didn't work initially**: Using the raw user message as the search query gave poor recall for indirect phrasing ("someone who works with numbers" didn't surface numerical reasoning tests). Moving to LLM-extracted `search_query` fixed this. A single LLM call that both retrieved and generated was tried first — it hallucinated URLs even with strict prompting. Separating retrieval from generation eliminated the problem.

---

## 7. Tech Stack Justification

| Component | Choice | Why |
|---|---|---|
| API | FastAPI + Pydantic v2 | Async, typed, fast — schema validation is built-in |
| LLM | Gemini 2.0 Flash (default) | Free tier, 1M token context, ~1s latency on simple turns |
| Embeddings | all-MiniLM-L6-v2 | Free, CPU-runnable, 384-dim, excellent on short HR text |
| Vector store | FAISS in-memory | Zero infra, ~5ms search on 400 items, persists to disk |
| Scraping | requests + BeautifulSoup4 | Catalog is server-rendered HTML with pagination |
| Deployment | Render (free tier) | Free HTTPS, Docker support, accepts `render.yaml` |

**AI tools used**: Claude (Anthropic) used for prompt drafting iteration and code review. All design decisions were made and can be defended independently.

---

## 8. Known Limitations and Future Work

- The FAISS index must be rebuilt when the catalog changes. A lightweight nightly scrape job and index rebuild would keep it fresh.
- The intent extractor adds ~800ms latency per turn. Caching common intents or using a smaller classification model would help.
- Recall@10 is limited by catalog scrape completeness — if the scraper misses pages, retrieval quality degrades. The scraper includes 3-retry logic and empty-page detection to mitigate this.
