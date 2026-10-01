"""Streamlit front end: ask questions, see citations, read the eval numbers.

Display decisions live in format_answer, summary_rows and obrag.app.limits so they can
be tested without launching a browser; everything below them is layout.
"""

from dataclasses import replace

import streamlit as st

from obrag.app.limits import QueryBudget, cache_answer, cached_answer, resolve_answer
from obrag.config import DISCLAIMER, SNAPSHOT_DATE, load_settings
from obrag.evaluation.runner import summarise
from obrag.evaluation.storage import EvalStore
from obrag.models import Answer
from obrag.query.pipeline import ask

# Chosen from the golden-set runs: each answered (or, for the last, refused)
# correctly with full marks, so the pre-cached answers are the system at its best
# on each band. The original headline SCA question is declined (see the case study).
EXAMPLES = [
    "What must a funds confirmation for a card-based payment contain under the PSRs, "
    "and how does the Confirmation of Funds API return it?",
    "How much can a payer be made to pay for an unauthorised payment made with a lost or stolen card?",
    "What happens if a PISP sends the same x-idempotency-key twice when creating a domestic payment?",
    "What is Barclays' rate limit on the accounts endpoint?",
]

# One router and one generator call per question, each against its own 500/day
# free quota shared by all visitors.
MAX_QUERIES_PER_SESSION = 10


def format_answer(answer: Answer) -> str:
    if answer.refused:
        return f"**No grounded answer.**\n\n{answer.text}"
    lines = [answer.text, "", "**Sources**"]
    by_citation = {r.chunk.citation: r.chunk.source_url for r in answer.retrieved}
    for index, citation in enumerate(answer.citations, start=1):
        url = by_citation.get(citation, "")
        lines.append(f"{index}. [{citation}]({url})" if url else f"{index}. {citation}")
    return "\n".join(lines)


def summary_rows(summary: dict) -> list[dict]:
    ordered = ["overall"] + [band for band in summary if band != "overall"]
    return [{"band": band, **summary[band]} for band in ordered if band in summary]



def render_ask_tab(settings) -> None:
    st.caption(f"Snapshot: {SNAPSHOT_DATE}")
    with st.form("ask"):
        question = st.text_input("Ask a question about UK Open Banking")
        submitted = st.form_submit_button("Ask")
    st.caption("Try: " + " · ".join(f"_{e}_" for e in EXAMPLES))

    if submitted and question.strip():
        st.session_state["question"] = question
    question = st.session_state.get("question")
    if not question:
        return

    budget = QueryBudget(max_queries=MAX_QUERIES_PER_SESSION)
    session_answers = st.session_state.setdefault("answers", {})
    with st.spinner("Routing, retrieving and answering..."):
        answer, source = resolve_answer(
            question, session_answers, st.session_state, budget,
            lambda q: ask(q, settings=settings),
        )
    if answer is None:
        st.error(
            f"This demo allows {MAX_QUERIES_PER_SESSION} questions per session to stay within "
            "its free API quota. The example questions above are still available — they are "
            "pre-cached. Reload the page to start a new session."
        )
        return

    st.markdown(f"**Q:** {question}")
    st.markdown(format_answer(answer))
    if source == "example":
        st.caption("Cached example answer — no API call made.")
    else:
        st.caption(f"{budget.remaining(st.session_state)} questions remaining this session.")

    with st.expander(f"Retrieved chunks ({len(answer.retrieved)})"):
        for item in answer.retrieved:
            st.markdown(f"**{item.chunk.citation}** · {item.chunk.collection} · {item.score:.3f}")
            st.text(item.chunk.text[:1200])


def render_eval_tab(settings) -> None:
    store = EvalStore(settings)
    runs = store.runs()
    if not runs:
        st.info("No evaluation runs recorded yet. Run `python -m obrag.evaluation.runner --label baseline`.")
        return
    labels = {f"{r['id']}: {r['label']} ({r['created_at']})": r["id"] for r in runs}
    chosen = st.selectbox("Run", list(labels))
    run_id = labels[chosen]
    results = store.results(run_id)
    st.dataframe(summary_rows(summarise(results)))
    with st.expander(f"Per-question results ({len(results)})"):
        st.dataframe(results)


def main() -> None:
    st.set_page_config(page_title="UK Open Banking RAG", page_icon="🏦", layout="wide")
    st.title("UK Open Banking RAG")
    st.warning(DISCLAIMER)

    # No Gemini retries: a visitor on an exhausted free-tier quota should see the
    # "unavailable" message in seconds, not after the SDK waits out 429s.
    settings = replace(load_settings(), gemini_retries=0)
    # The example cache is module-level, so it is shared by every session in this
    # process: a cold start costs one ask() per example, not every visitor.
    if "obrag_examples_cached" not in st.session_state:
        with st.spinner("Warming up the example answers..."):
            for example in EXAMPLES:
                if cached_answer(example) is None:
                    cache_answer(example, ask(example, settings=settings))
        st.session_state["obrag_examples_cached"] = True

    ask_tab, eval_tab = st.tabs(["Ask", "Evaluation"])
    with ask_tab:
        render_ask_tab(settings)
    with eval_tab:
        render_eval_tab(settings)


if __name__ == "__main__":
    main()
