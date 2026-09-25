"""Streamlit front end: ask questions, see citations, read the eval numbers.

Display decisions live in format_answer, summary_rows and answer_for so they can
be tested without launching a browser; everything below them is layout.
"""

from collections.abc import Callable

import streamlit as st

from obrag.config import DISCLAIMER, SNAPSHOT_DATE, load_settings
from obrag.evaluation.runner import summarise
from obrag.evaluation.storage import EvalStore
from obrag.models import Answer
from obrag.query.pipeline import ask

EXAMPLES = [
    "Does the Payment Initiation API's consent flow satisfy SCA requirements?",
    "When must strong customer authentication be applied?",
    "What does POST /domestic-payment-consents return on success?",
    "What is Barclays' rate limit on the accounts endpoint?",
]


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


def answer_for(question: str, cache: dict, ask_fn: Callable[[str], Answer]) -> Answer:
    """Ask once per distinct question and reuse the answer on every rerun.

    Streamlit reruns the whole script on any click, including switching tabs,
    and each fresh ask() spends two calls of the free-tier daily quota.
    """
    key = " ".join(question.split()).lower()
    if key not in cache:
        cache[key] = ask_fn(question)
    return cache[key]


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

    cache = st.session_state.setdefault("answers", {})
    with st.spinner("Routing, retrieving and answering..."):
        answer = answer_for(question, cache, lambda q: ask(q, settings=settings))
    st.markdown(f"**Q:** {question}")
    st.markdown(format_answer(answer))

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

    settings = load_settings()
    ask_tab, eval_tab = st.tabs(["Ask", "Evaluation"])
    with ask_tab:
        render_ask_tab(settings)
    with eval_tab:
        render_eval_tab(settings)


if __name__ == "__main__":
    main()
