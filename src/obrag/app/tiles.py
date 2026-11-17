"""The tiles drawn as HTML, and the small shaping steps that feed them.

Streamlit lays out the widgets; everything here is a pure function that returns
data or an HTML string, so the escaping and the numbers on screen can be tested
without a browser. All text that came from a visitor, the model or the index is
escaped before it reaches a tile.
"""

import re
from html import escape

from obrag.models import Answer

REPO_URL = "https://github.com/rochak779/open-banking-rag"

COLLECTION_LABELS = {"regulation": "Law", "spec": "API spec"}

BAND_LABELS = {
    "overall": "All questions",
    "answerable": "Answerable",
    "regulation_only": "Law only",
    "spec_only": "API spec only",
    "cross_cutting": "Law and API",
    "unanswerable": "Out of scope",
}

# Column order for the score grid. "Right call" is the answer-or-decline decision.
METRIC_LABELS = {
    "refusal_accuracy": "Right call",
    "correctness": "Correct",
    "groundedness": "Grounded",
    "retrieval_relevance": "Relevant sources",
    "routing_accuracy": "Routing",
}

_QUESTION_BANDS = ["regulation_only", "spec_only", "cross_cutting", "unanswerable"]

# st.html strips inline SVG, so icons and the mark are drawn in styles.css and
# placed with a class. One stroke weight and one 24px grid for every icon.
def icon(name: str) -> str:
    return f'<i class="ob-icon ob-icon--{name}" aria-hidden="true"></i>'


MARK = '<i class="ob-mark" aria-hidden="true"></i>'

# The mark is the product's own device: a citation marker, "[1]", cut down to
# its brackets and a dot. Streamlit takes an SVG string as the browser-tab icon.
FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
    '<rect width="40" height="40" rx="12" fill="#CFFE32"/>'
    '<path d="M16 11h-4.5v18H16M24 11h4.5v18H24" fill="none" stroke="#0B0F12" '
    'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="20" cy="20" r="2.6" fill="#0B0F12"/></svg>'
)


def source_rows(answer: Answer) -> list[dict]:
    """The cited sources in the order the answer numbers them."""
    by_citation = {r.chunk.citation: r.chunk for r in answer.retrieved}
    rows = []
    for index, citation in enumerate(answer.citations, start=1):
        chunk = by_citation.get(citation)
        rows.append({
            "index": index,
            "citation": citation,
            "url": chunk.source_url if chunk else "",
            "collection": chunk.collection if chunk else "",
        })
    return rows


def decision_counts(summary: dict) -> list[dict]:
    """Right answer-or-decline decisions per question type, as whole questions."""
    counts = []
    for band in _QUESTION_BANDS:
        scores = summary.get(band)
        if not scores or scores.get("refusal_accuracy") is None:
            continue
        counts.append({
            "band": band,
            "label": BAND_LABELS[band],
            "correct": round(scores["refusal_accuracy"] * scores["n"]),
            "n": scores["n"],
        })
    return counts


def topbar_html() -> str:
    links = [
        ("Case study", f"{REPO_URL}/blob/main/docs/case-study.md"),
        ("Architecture", f"{REPO_URL}/blob/main/docs/architecture.md"),
        ("GitHub", REPO_URL),
    ]
    nav = "".join(
        f'<a href="{escape(url)}" target="_blank" rel="noopener">{escape(label)}{icon("arrow")}</a>'
        for label, url in links
    )
    return (
        '<header class="ob-topbar">'
        f'<div class="ob-brand">{MARK}<span>UK Open Banking RAG</span></div>'
        f'<nav class="ob-nav" aria-label="Project links">{nav}</nav>'
        "</header>"
    )


def hero_html() -> str:
    return (
        '<div class="ob-hero">'
        "<h1>Open Banking answers you can check.</h1>"
        "<p>Ask about the UK payment rules or the Open Banking API. Every answer "
        "cites the regulation or endpoint it came from, or says the sources "
        "don't cover it.</p>"
        '<ul class="ob-cluster" aria-label="Sources searched">'
        '<li class="ob-chip ob-chip--outline">PSRs 2017</li>'
        f'<li class="ob-dot ob-dot--lime">{icon("law")}</li>'
        '<li class="ob-chip ob-chip--paper"><i></i>SCA-RTS</li>'
        f'<li class="ob-dot ob-dot--ring">{icon("api")}</li>'
        '<li class="ob-chip ob-chip--paper"><i></i>Read/Write API v4.0.1</li>'
        "</ul>"
        "</div>"
    )


def disclaimer_html(text: str) -> str:
    # Keep ISO dates on one line; the browser would otherwise break at a hyphen.
    body = re.sub(r"\d{4}-\d{2}-\d{2}", r'<span class="ob-nowrap">\g<0></span>', escape(text))
    return f'<p class="ob-notice">{icon("info")}<span>{body}</span></p>'


def sources_html(rows: list[dict]) -> str:
    if not rows:
        return (
            '<p class="ob-empty">Nothing was cited. The passages the search found '
            "are listed below so you can check them yourself.</p>"
        )
    items = []
    for row in rows:
        citation = escape(row["citation"])
        title = (
            f'<a href="{escape(row["url"])}" target="_blank" rel="noopener">{citation}{icon("arrow")}</a>'
            if row["url"] else f"<span>{citation}</span>"
        )
        kind = COLLECTION_LABELS.get(row["collection"], "")
        tag = f'<em class="ob-tag">{escape(kind)}</em>' if kind else ""
        items.append(f'<li><b>{row["index"]}</b><div>{title}{tag}</div></li>')
    return f'<ol class="ob-sources">{"".join(items)}</ol>'


def passage_html(citation: str, collection: str, score: float, text: str) -> str:
    kind = COLLECTION_LABELS.get(collection, collection)
    return (
        '<article class="ob-passage">'
        f"<header><strong>{escape(citation)}</strong>"
        f'<em class="ob-tag">{escape(kind)}</em>'
        f'<span class="ob-score">match {score:.3f}</span></header>'
        f"<p>{escape(text)}</p></article>"
    )


def decision_chart_html(counts: list[dict]) -> str:
    """Vertical bars, one per question type, each labelled with its own count."""
    correct = sum(c["correct"] for c in counts)
    total = sum(c["n"] for c in counts)
    bars = []
    for c in counts:
        share = c["correct"] / c["n"] if c["n"] else 0
        full = " ob-bar--full" if c["correct"] == c["n"] else ""
        bars.append(
            f'<li class="ob-bar{full}" title="{escape(c["label"])}: {c["correct"]} of {c["n"]}">'
            f'<span class="ob-bar__value">{c["correct"]}/{c["n"]}</span>'
            f'<span class="ob-bar__track"><span class="ob-bar__fill" style="height:{share:.0%}"></span></span>'
            f'<span class="ob-bar__label">{escape(c["label"])}</span></li>'
        )
    return (
        '<section class="ob-tile ob-tile--paper ob-chart">'
        "<h3>Right call on test questions</h3>"
        f'<p class="ob-figure">{correct}<small>of {total}</small></p>'
        f'<ul class="ob-bars">{"".join(bars)}</ul>'
        '<p class="ob-foot">Answered when the sources covered it, declined when they did not.</p>'
        "</section>"
    )


def stats_html(counts: list[dict], run_label: str, snapshot_date: str) -> str:
    declined = next((c for c in counts if c["band"] == "unanswerable"), None)
    declined_tile = ""
    if declined:
        units = "".join(
            f'<li class="{"is-right" if i < declined["correct"] else ""}"></li>'
            for i in range(declined["n"])
        )
        declined_tile = (
            '<section class="ob-tile ob-tile--paper ob-stat">'
            f'<p class="ob-figure">{declined["correct"]}<small>of {declined["n"]}</small></p>'
            "<h3>out-of-scope questions declined, not guessed</h3>"
            f'<ul class="ob-units" aria-hidden="true">{units}</ul>'
            f'<p class="ob-foot">Latest evaluation run: {escape(run_label)}</p>'
            "</section>"
        )
    sources_tile = (
        '<section class="ob-tile ob-tile--lime ob-pinned">'
        "<h3>Three sources, one snapshot</h3>"
        "<ul>"
        "<li>Payment Services Regulations 2017</li>"
        "<li>SCA-RTS (EU) 2018/389, as retained</li>"
        "<li>Open Banking Read/Write API v4.0.1</li>"
        "</ul>"
        f'<p class="ob-foot">Pinned on {escape(snapshot_date)}. Nothing updates on its own.</p>'
        "</section>"
    )
    chart = decision_chart_html(counts) if counts else ""
    return f'<div class="ob-bento">{declined_tile}{sources_tile}{chart}</div>'


def _cell(value: float | None) -> str:
    if value is None:
        return '<td class="ob-grid__none"><span>not scored</span></td>'
    return (
        f'<td><span class="ob-grid__value">{value:.2f}</span>'
        f'<span class="ob-grid__track"><span style="width:{value:.0%}"></span></span></td>'
    )


def score_grid_html(rows: list[dict], with_routing: bool) -> str:
    """One row per question type, one bar per score, every bar on the same 0-1 scale."""
    metrics = [m for m in METRIC_LABELS if with_routing or m != "routing_accuracy"]
    head = "".join(f'<th scope="col">{escape(METRIC_LABELS[m])}</th>' for m in metrics)
    body = []
    for row in rows:
        label = escape(BAND_LABELS.get(row["band"], row["band"]))
        cells = "".join(_cell(row.get(m)) for m in metrics)
        body.append(
            f'<tr><th scope="row">{label}<small>{row.get("n", "")} questions</small></th>{cells}</tr>'
        )
    return (
        '<div class="ob-grid-wrap"><table class="ob-grid">'
        f'<thead><tr><th scope="col"><span class="ob-sr">Question type</span></th>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def config_chips_html(config: dict, with_routing: bool) -> str:
    chips = [
        ("Answers", config.get("generation_model")),
        ("Search", config.get("embedding_model")),
        ("Judge", config.get("judge_model")),
        ("Passages per question", config.get("top_k")),
        ("Router", "on" if with_routing else "off"),
    ]
    items = "".join(
        f"<li><span>{escape(name)}</span>{escape(str(value))}</li>"
        for name, value in chips if value is not None
    )
    return f'<ul class="ob-config">{items}</ul>'
