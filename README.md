# UK Open Banking RAG

Ask questions about UK Open Banking and get answers grounded in the law (Payment Services Regulations 2017, SCA-RTS) and the Open Banking Read/Write API specification. Every claim is cited with a deep link to the source provision or endpoint, and the system says so when the sources don't cover the question.

**Live demo:** https://openbanking.streamlit.app/ (free hosting: the first visit after a quiet spell can take a minute to wake up)

> This tool is a technical demonstration. It is not legal, regulatory or compliance advice. Answers are drawn from a snapshot taken 2026-09-18 and may be out of date. SCA-RTS citations are the retained EU text published on legislation.gov.uk, which does not include the FCA's later amendments. Always check the cited source.

## How well it works

Measured on a 40-question golden set (four bands of 10), written from the source text. The full method and caveats are in the [case study](docs/case-study.md).

| | Result |
|---|---|
| Unanswerable questions correctly declined | **10/10** in every run, including a trap question with a related but wrong article retrievable |
| Routing to the right source(s), answerable questions | 21/30 → **26/30** after fixing the router (cross-cutting: 4/10 → 8/10) |
| Correct answer-or-decline decision, answerable questions | 21/30 |
| Groundedness / correctness (LLM-judged, relative) | 1.00 / 0.54 |
| Embedding A/B | `voyage-4-lite` beat the legal-domain `voyage-law-2` on retrieval relevance (0.55 vs 0.41) at equal correctness and a sixth of the price |

The weak spot is stated, not hidden: questions that bridge a regulation and an endpoint are still often declined (4/10 correct decisions). The case study explains why and what would come next.

## Quickstart

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/), plus API keys for Google Gemini and Voyage AI (both have free tiers).

```bash
uv sync
cp .env.example .env          # then fill in GEMINI_API_KEY and VOYAGE_API_KEY
uv run python -m obrag.index.build      # fetches the pinned sources and builds the index
uv run streamlit run src/obrag/app/streamlit_app.py
```

The repo already includes a built index (`data/chroma`), so the build step is only needed to rebuild it. On the Gemini free tier, set `OBRAG_GENERATION_MODEL=gemini-3.5-flash-lite` in `.env`: the default `gemini-3.8-flash` allows 20 requests a day.

Run the tests with `uv run pytest`, and the evaluation with `uv run python -m obrag.evaluation.runner --label <name>` (set `OBRAG_JUDGE_MODEL=gemma-4-31b-it` on the free tier).

## More

- [Architecture](docs/architecture.md): pipeline diagram, chunking, models
- [Case study](docs/case-study.md): the problem, the eval, the numbers, what did not work, limits

## Scope

- Sources: the Payment Services Regulations 2017, the SCA-RTS (EU) 2018/389 as retained, and the Open Banking Limited Read/Write API specification v4.0.1 (tag `v4.0.1-Update-1`), pinned on 2026-09-18.
- Not included: the FCA Handbook, the FCA's amended SCA-RTS, bank-specific documentation, payment scheme rules.

## Source attribution

- The Payment Services Regulations 2017: Crown copyright, from [legislation.gov.uk](https://www.legislation.gov.uk/uksi/2017/752), reused under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
- Commission Delegated Regulation (EU) 2018/389: as published by [legislation.gov.uk](https://www.legislation.gov.uk/eur/2018/389); © European Union.
- Open Banking Read/Write API specifications: © Open Banking Limited, from [OpenBankingUK/read-write-api-specs](https://github.com/OpenBankingUK/read-write-api-specs), under the [Open Banking Open Licence](https://www.openbanking.org.uk/open-licence/).
