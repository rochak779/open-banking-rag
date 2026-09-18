# UK Open Banking RAG — Design

**Status:** Approved for v1 build. Written 2026-09-18. Revised 2026-09-18 (FCA Handbook deferred to v2; hosted demo added to v1; generation LLM switched from Claude to Gemini).

## Purpose

A retrieval-augmented system that answers questions about UK Open Banking by combining official regulation text with the Open Banking API specification — including cross-cutting questions that need both (e.g. "does the Payment Initiation API's consent flow satisfy PSD2 SCA requirements?").

Two goals, with different bars:

1. **Upwork portfolio case study** (primary, near-term). Demonstrates RAG design, groundedness evaluation, and financial-services domain depth — differentiated from generic "chatbot on my PDFs" demos. Low risk: a labeled snapshot with a clear "as of" date and an explicit "not legal/compliance advice" disclaimer is sufficient.
2. **Commercial product for fintechs** (possible, longer-term, not yet validated). This raises the bar significantly — wrong compliance guidance has real consequences for a paying buyer. Treat this as a question to answer *after* v1 ships and gets real feedback, not something to architect for now. Do not let this ambition inflate v1 scope.

## Scope

- **v1** — UK regulations (Payment Services Regulations 2017 + retained PSD2/SCA-RTS text from legislation.gov.uk) + the Open Banking Limited (OBL) Read/Write API Specification. This is the only phase currently designed in detail below.
- **v2** — Add the FCA Handbook (relevant modules) and 3-5 major ASPSPs' own developer API docs (e.g. Barclays, HSBC, Monzo, Starling) as new collections, reusing the same ingestion pipeline.
- **v3** — Add PSP-specific documentation the same way.

Do not attempt all 40+ ASPSPs or a fully general "every possible detail" scope in v1 — it isn't demoable in a reasonable timeframe and isn't necessary to prove the concept.

**Why the FCA Handbook moved to v2:** it was the only v1 source behind a registration + Terms & Conditions gate, and the least load-bearing of the three. The SCA and consent requirements that make the cross-cutting questions interesting live in PSR 2017 and the retained SCA-RTS, both freely available on legislation.gov.uk. Deferring it removes the only external blocker from v1 without weakening the demo.

## Architecture

Segmented vector collections with query routing, not a single flat index and not a full agentic tool-use system:

```
Query -> Router (regulation vs spec vs both)
  -> Regulation collection (PSR 2017, retained PSD2 / SCA-RTS clauses)
  -> Spec collection (OBL API endpoints, schemas)
-> Merge + rerank (v2+, skip for v1) -> Answer with citations
```

Rationale: regulation clauses and API endpoint definitions have very different structure and granularity, so mixing them into one index hurts retrieval quality. Separate collections also give a clean seam to add ASPSP/PSP collections in v2/v3 without redesigning the router.

The router itself should be a single cheap LLM call (or keyword/embedding heuristic) — not a framework. Two routing targets don't justify LangChain/LlamaIndex-level infrastructure.

## Why an LLM is needed

RAG's value here is generation, not just retrieval:
- Synthesizing an answer across both a regulation clause and an API spec section into one coherent explanation (pure retrieval would return both chunks separately, unexplained).
- Explicitly saying "I don't know" when nothing relevant is retrieved, instead of returning the closest-but-wrong chunk.
- Groundedness/correctness evaluation — the actual differentiator for the portfolio case study — requires something to evaluate, which requires generation to exist in the first place.

## Sources & acquisition (v1)

Both v1 sources are free, ungated, and require no scraping — direct downloads and structured API calls only. No scraping tool and no account registration is needed for v1.

| Source | Content | Access method | Cost / gating |
|---|---|---|---|
| **OBL Read/Write API Specification** | OpenAPI 3.0 YAML/JSON for Account Info, Payment Initiation, Confirmation of Funds, VRP, Events | Direct download / `git clone` from `github.com/OpenBankingUK/read-write-api-specs`, files under `/dist/openapi/`. Pick a tagged release (e.g. `v4.0.0`) for a stable snapshot. | Free, MIT-style "Open Licence" (attribution required, no account needed) |
| **Payment Services Regulations 2017 (PSR) / retained PSD2 text** | UK regulation text, structured by regulation/section | `requests` calls to legislation.gov.uk's structured API — append `/data.xml` to any provision URL (e.g. `legislation.gov.uk/uksi/2017/752/data.xml`) for full XML with Dublin Core metadata. Walk the table of contents programmatically. | Free, fully public, no account needed |

**Deferred to v2 — FCA Handbook.** Access is via the FCA's Handbook API (`handbook.fca.org.uk/handbook-api`), free but gated behind account registration and Terms & Conditions acceptance. Before v2 ingestion starts, register the account and specifically review the T&Cs for any restriction on storing or re-serving extracted content in a third-party tool. Nothing in v1 depends on this.

**v2/v3 note:** Individual ASPSP and PSP developer portals won't be uniformly structured like OBL's repo (mix of Swagger/Redoc pages, PDFs, plain HTML). When that phase starts, use **Firecrawl** rather than direct downloads — it handles JS-rendered doc sites and is purpose-built for turning bulk web documentation into clean content for a knowledge base.

## Tech stack (v1)

| Layer | Choice | Why |
|---|---|---|
| Language | Python | Standard for RAG; every library below is Python-first |
| Generation LLM | Gemini `gemini-3.8-flash` via the `google-genai` SDK | Flash-tier is the right fit for synthesis over supplied context, and Google AI Studio's free tier removes the hosted demo's spend risk entirely. Note this costs the original Claude-native positioning argument — a deliberate trade |
| Embeddings | Voyage AI (`voyage-3`, A/B'd against `voyage-law-2`) | Retained after the Gemini switch on its own merits, not vendor alignment: `voyage-law-2` is purpose-built for legal text, and measuring it against `voyage-3` is a publishable case-study result rather than an assumption |
| Vector store | Chroma (embedded, local) | Zero infra for a two-collection, few-thousand-chunk system. Swap for pgvector/Pinecone only if v2/v3 or commercialization introduces real concurrent load |
| Parsing/chunking | Hand-rolled (`pyyaml` for OpenAPI, `lxml` for legislation XML) — not LangChain/LlamaIndex | Both sources are already well-structured; a generic text-splitter would throw that structure away. Chunk by endpoint/schema (spec) and by regulation/section (legislation), carrying citation metadata. Also a stronger technical story than "called a framework's default splitter" |
| Query routing | Keyword rules first, then a single `gemini-3.5-flash-lite` call | Two routing targets don't need a framework, and most questions route on an unambiguous keyword with no API call at all |
| Reranking | Skipped in v1 | Two small collections don't need it; revisit with Voyage rerank if v2's added collections make merge quality a problem |
| Vector similarity search | Handled internally by Chroma (HNSW) | FAISS would provide the same nearest-neighbor search Chroma already does internally — no reason to add it as a separate piece at this scale |
| Eval harness | Four hand-written scorers, no framework | Ragas would pull LangChain into a deliberately framework-free project, and three of the four metrics must judge against this system's own citation format. Refusal quality isn't a stock metric anywhere. "Built the harness" is also the stronger case-study claim |
| UI | Streamlit | Matches the existing Upwork case-study plan; fastest path to a demoable chat + citations + eval dashboard |
| Metadata/eval storage | SQLite | No infra needed at this scale |
| Hosting | Streamlit Community Cloud (free) | Gives the Upwork case study a clickable live demo. Deploys from the GitHub repo; secrets injected via its secrets manager, never committed |
| Demo cost control | Stay on the Gemini free tier + in-app per-session query limit + pre-cached example answers | On the free tier the exposure is rate-limit quota rather than cash, so the failure mode is a throttled demo, not a bill. The per-session cap protects the demo's availability when it matters |
| Dependency management | `uv` | Fast, single-tool venv + lockfile; Streamlit Cloud reads the exported `requirements.txt` |
| Eval judging | `gemini-3.1-pro-preview` | A stronger model than the generator, but still Gemini judging Gemini — see the risk below |

## Evaluation

A golden set of ~30-50 questions spanning pure-regulation, pure-spec, and cross-cutting questions. Score retrieval relevance, groundedness, answer correctness, and refusal quality. This eval harness and its before/after results are themselves a required artifact for the portfolio case study, not an optional extra.

## Risks to keep in view

- **Liability/trust**: this is compliance-adjacent content. Ship with a clear "not legal/compliance advice" disclaimer and an "as of" freshness date from day one.
- **Staleness**: regulations, the Handbook, and the API spec all change independently. A one-time snapshot is fine for a portfolio demo; it is not fine if this becomes a real reference tool or product without a refresh process.
- **Self-judging eval**: groundedness and correctness are scored by a Gemini model grading Gemini output, which grades generously. Accepted to keep the stack single-vendor. Mitigated by judging with a stronger model than the generator, and by leading with refusal accuracy and routing accuracy — neither of which involves model judgement. This must be declared in the case study rather than left for a reviewer to find.
- **Public demo availability**: the hosted demo runs on your own Gemini and Voyage keys. On the free tier the risk is quota exhaustion, not spend — the demo breaks rather than bills. The per-session query cap and pre-cached examples exist to keep it working when someone who matters clicks the link.
- **Scope creep**: hold the line at "regs + core spec" for v1 even if the commercial-product ambition tempts adding more sources early.

## Next steps

See the implementation plan for the detailed build sequence. At a high level:

1. Sign up for the required accounts and keys (Google AI Studio, Voyage AI, GitHub, Streamlit Community Cloud).
2. Set up the project repo and Python environment.
3. Build the ingestion pipeline for the OBL spec + legislation.gov.uk — no blocker, can start immediately.
4. Build the two Chroma collections + router + generation + citation grounding.
5. Build the golden eval set and harness.
6. Build the Streamlit UI + eval dashboard.
7. Add spend caps and rate limits, then deploy the hosted demo.
8. Package as a case study: architecture diagram, before/after eval numbers, write-up.
