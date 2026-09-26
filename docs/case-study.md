# Grounded Q&A over UK Open Banking: law and API spec in one answer

> **Draft.** The results section is waiting for the baseline and `voyage-law-2` eval runs. Every number elsewhere comes from a measurement taken while building the system; the source is noted next to each.

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

<!-- RESULTS PENDING: fill from `summarise()` output for run 1 (baseline, voyage-4-lite) and run 2 (voyage-law-2). Lead with refusal accuracy on the unanswerable band, then routing, then the judged scores per band, then the embedding A/B. Every number must trace to a row in data/eval.db. -->

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
