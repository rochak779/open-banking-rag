# Grounded Q&A over UK Open Banking: law and API spec in one answer

## The problem

Building on UK Open Banking means reading two very different bodies of text side by side: the law (the Payment Services Regulations 2017 and the SCA-RTS) and the Open Banking Limited Read/Write API specification. The useful questions sit across both: *does the payment consent flow satisfy strong customer authentication?* The answer has to cite the regulation and the endpoint, and it has to say "I don't know" when the sources don't cover the question. In a compliance-adjacent tool, a confident answer with the wrong citation is worse than no answer.

## Why generic RAG fails here

The standard recipe (split text into fixed windows, embed, retrieve, generate) breaks on this material in specific, measurable ways:

- **The structure is the citation.** A regulation's number and an endpoint's method and path are what make an answer checkable. Fixed-window chunks cut through both. This system chunks one provision or one API operation per chunk, and takes each citation link from the source's own identifier (CLML `DocumentURI`, OpenAPI method and path).
- **The obvious parser is silently wrong.** Written the way the XML *looks*, the legislation parser dropped **35 amended regulations**, including regulations 6, 20 and 22. Amended provisions carry a footnote marker in front of their number, so the number read as blank. It also cited schedule paragraphs as the regulation with the same number, and indexed quoted text from other Acts as PSR provisions. On the API side, every parameter and response in the OBL spec is a `$ref`, so a direct parser produced "None (in None, optional)". Links built from the spec's version number returned 404, because the tag is `v4.0.1-Update-1`, not `v4.0.1`. None of these failures raise an error. They only show up if you check the output against the source.
- **The vocabulary gap is real.** Practitioners say "SCA", "TPP", "VRP". The law says "strong customer authentication" and "account information service provider"; the spec says "VRP" but never "variable recurring payment". For the headline question, "Does the Payment Initiation API consent flow satisfy SCA requirements?", the regulation that defines when SCA applies (PSR regulation 100) ranked **27th** by embedding similarity. It ranks first when the question spells out "strong customer authentication".

## Architecture

See [architecture.md](architecture.md). In short: two collections (regulation, spec), a small-model router whose keyword rules can only widen a search to both collections, a similarity floor that lets off-topic questions reach zero results, an equal share of the top 6 for each routed collection, and a generator that must cite numbered sources or reply `INSUFFICIENT_CONTEXT`. No RAG framework: each stage is a short, tested Python module.

## How it was evaluated

A golden set of **40 questions in four bands of 10**: regulation-only, spec-only, cross-cutting (must bridge a named regulation and a named endpoint), and unanswerable. Every expected answer point was written from the indexed source text, not from memory. The unanswerable band is plausible on purpose: Faster Payments limits, PSD3, the FCA Consumer Duty, commercial VRP pricing, and a trap question about the FCA's article 10A, where a related but wrong article is retrievable. Some questions deliberately probe known weaknesses (abbreviations, industry terms, an oversized definitions chunk, a field-level schema question), so the eval measures those weaknesses instead of hiding them.

Four scores per question:

| Metric | How it is scored | Trust it as |
|---|---|---|
| Refusal accuracy | truth table: answered vs refused, against whether the question is answerable | absolute |
| Routing accuracy | set comparison of the collections chosen vs expected | absolute |
| Retrieval relevance | LLM judge: fraction of retrieved sources that could help answer | relative |
| Groundedness | LLM judge: are the answer's claims supported by its sources? | relative |
| Correctness | LLM judge: fraction of expected points the answer makes | relative |

The judge is Gemma 4 31B, a different model family from the Gemini generator. Before it graded anything real, it was checked on six cases with known answers: a supported answer, a fabricated one, a half-fabricated one, answers covering 1 of 3 and 3 of 3 expected points, and a half-relevant source set. It scored all six as expected (1.0, 0.0, 0.5, 0.33, 1.0, 0.5).

## Results

Three runs over the same 40 questions, all with `gemini-3.5-flash-lite` generating and Gemma 4 31B judging. Every number below comes from `summarise()` over `data/eval.db`.

**It declines what it cannot answer: 10 of 10 unanswerable questions were refused in every run.** That includes the trap question about the FCA's article 10A. A related but wrong article (article 10) is retrievable, and the system still declined rather than answering from it.

### Before and after: fixing the router

Run 1 is the baseline. Run 3 changes one thing: keyword rules may only widen a search to both collections, never narrow it to one.

| Band | Routing correct | Refusal decision correct | Retrieval relevance | Groundedness | Correctness |
|---|---|---|---|---|---|
| Regulation-only (10) | 8 → 9 | 9 → 9 | 0.67 → 0.52 | 0.80 → 1.00 | 0.82 → 0.80 |
| Spec-only (10) | 9 → 9 | 8 → 8 | 0.60 → 0.60 | 0.99 → 1.00 | 0.60 → 0.57 |
| Cross-cutting (10) | **4 → 8** | 3 → 4 | 0.39 → 0.53 | 1.00 → 1.00 | 0.13 → 0.23 |
| **All answerable (30)** | **21 → 26** | 20 → 21 | 0.55 → 0.55 | 0.93 → 1.00 | 0.52 → 0.54 |

- **Routing is the measured win.** In the baseline, the keyword shortcut routed 4 of 9 questions correctly against 17 of 21 for the model router, and caused 5 of the 6 cross-cutting misses: "under the PSRs … which API endpoints?" matched a regulation keyword and never searched the spec. With keywords only allowed to widen the search, cross-cutting routing went from 4/10 to 8/10.
- **Cross-cutting answers are still the weak spot.** Better routing turned one refusal into a full answer (cross-03, correctness 0 → 1.0, citing PSR regulation 68 and `POST /funds-confirmations` side by side). But three other questions are now routed correctly and still declined. The generator only answers when a source links the two sides explicitly, and it refuses otherwise. That is the next change to measure, not something the routing fix could reach.
- **Read the small differences as noise.** Each band is 10 questions, so one question moves a band score by 0.1, and run-to-run variance was not measured. The regulation-only relevance drop (0.67 → 0.52) came with identical refusal decisions and one more correct route; it is not evidence of a regression.
- **Groundedness is near 1.0 almost everywhere.** The generator is told to use only the numbered sources, and it mostly does. One defect found by reading, not by the metric: cross-09 was answered with no citation markers at all, which the groundedness judge does not penalise. Since fixed: an answer with no valid citations is now declined, with the retrieved sources still shown.

### Embedding comparison: the legal model lost

Same pipeline, different embeddings (run 1 against run 2, answerable questions):

| | `voyage-4-lite` | `voyage-law-2` |
|---|---|---|
| Retrieval relevance | **0.55** | 0.41 |
| Groundedness | 0.93 | 0.99 |
| Correctness | 0.52 | 0.51 |
| Refusal decision correct | 20/30 | 20/30 |
| Price per token | 1× | 6× |

The legal-domain model retrieved worse sources (0.41 against 0.55, and 0.31 against 0.60 on spec-only questions: the API spec is not legal text) for the same correctness, at six times the price. Its higher groundedness is real, but the judge only checks answers against the sources they cite, so it cannot show whether better sources existed. `voyage-4-lite` stays.

### Scores that could not be measured

A judge call failing (Gemma server errors or its 16K tokens-per-minute limit) leaves a gap rather than a guess: 4, 7 and 2 missing scores in runs 1, 2 and 3, out of 120 judged values per run. Groundedness in runs 1 and 2 was re-judged after a bug where the judge numbered sources differently from the answer's citations. That bug scored a verbatim, correctly cited definition 0.0.

## What did not work, and what changed

Each of these was found by measuring, not by reasoning about the design:

- **The router never answered "both".** With the first prompt, all three mixed test questions were routed to the law only, silently dropping the API half. Rewriting the prompt to say when "both" applies fixed it: 12 of 12 labelled questions routed correctly. That was a small hand-labelled set; the golden set measures it properly.
- **The planned similarity floor filtered nothing.** At 0.2, nonsense questions (capital of France, banana bread) still retrieved sources: their top scores reached 0.244. Real questions started at 0.312. The floor is now 0.28, and off-topic questions get zero results and a refusal without a model call.
- **One collection crowded out the other.** On the headline question, a plain merge by score gave the generator 5 near-identical consent endpoints and 1 regulation, so the model correctly refused. Each routed collection now gets an equal share of the top 6.
- **Expanding abbreviations was tried and dropped.** Appending "strong customer authentication" to "SCA" moved regulation 100 from rank 27 to 5, and "VRP" from 30 to 4. But it left "TPP" unchanged and pushed "What is an AIS?" from rank 1 to 4. A mixed result, so it was not adopted; the golden set's abbreviation probes measure the gap instead.
- **Free-tier limits shaped the model choices.** `gemini-3.8-flash` allows 20 requests a day on the free tier, and the planned judge, `gemini-3.1-pro-preview`, allows none. Generation moved to `gemini-3.5-flash-lite` (500/day), and judging to Gemma.

## Limits

- **A snapshot, not live law.** Sources were pinned on 2026-09-18. Nothing updates automatically.
- **The SCA-RTS text is the retained EU version.** The index holds the SCA-RTS as published on legislation.gov.uk, without the FCA's amendments: for example, article 10A from PS21/19 is missing, and contactless limits are stated in euros. legislation.gov.uk also marks the regulation for revocation under FSMA 2023. The disclaimer says so, and SCA-RTS provisions are kept out of the answerable golden-set bands. Replacing it with the FCA's version is v2 work.
- **No FCA Handbook.** It is the one gated source, deferred to v2.
- **Operation-level spec chunks only.** Request and response schemas are named but their fields are not indexed, so field-level questions (such as what an empty `ExpirationDateTime` means) are expected to miss.
- **The judge is not independent.** Gemma is a different model family from the Gemini generator, which helps, but it is the same vendor and a smaller model than the planned judge. Treat groundedness, correctness and relevance as relative measures for comparing runs, not absolute claims. Refusal and routing accuracy have no model in them.
- **Small, self-written golden set.** 40 questions, written by the person who built the system, so a question the author did not think of is not covered. Per-band numbers rest on 10 questions each.
- **Not compliance advice.** Every answer carries that disclaimer.

## What v2 would add

- The FCA Handbook and the FCA's current SCA-RTS, replacing the retained EU text.
- Schema-field chunks for the API spec.
- Hybrid (keyword + vector) retrieval, aimed at the abbreviation gap the golden set measures.
- A generator prompt that bridges a regulation and an endpoint when both are retrieved but no source states the link, measured on the cross-cutting band.
- A run-to-run variance measurement, so small score differences can be read with confidence.
