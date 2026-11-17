"""Streamlit front end: ask questions, see citations, read the eval numbers.

Display decisions live in summary_rows, obrag.app.tiles and obrag.app.limits so
they can be tested without launching a browser; everything below them is layout.
The look comes from .streamlit/config.toml (theme) and styles.css (tiles).
"""

from dataclasses import replace
from html import escape
from pathlib import Path

import streamlit as st

from obrag.app import tiles
from obrag.app.examples import EXAMPLES, load_examples
from obrag.app.limits import QueryBudget, cache_answer, resolve_answer
from obrag.config import DISCLAIMER, SNAPSHOT_DATE, load_settings
from obrag.evaluation.runner import summarise
from obrag.evaluation.storage import EvalStore
from obrag.query.pipeline import ask

# One generator call per question, against a 500/day free quota shared by all
# visitors.
MAX_QUERIES_PER_SESSION = 10

STYLES = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")


def summary_rows(summary: dict) -> list[dict]:
    ordered = ["overall"] + [band for band in summary if band != "overall"]
    return [{"band": band, **summary[band]} for band in ordered if band in summary]


def used_router(run: dict) -> bool:
    # Runs saved before the use_router flag existed all called the router.
    return run["config"].get("use_router", True)


def drop_routing(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k not in ("routing_accuracy", "routing_correct")} for row in rows]


def _pick_example(example: str) -> None:
    st.session_state["question"] = example


def render_ask_tab(settings, runs: list[dict]) -> None:
    hero, ask_column = st.columns([5, 7], gap="medium")
    with hero, st.container(key="hero_tile", height="stretch"):
        st.html(tiles.hero_html())
    with ask_column, st.container(key="ask_tile", height="stretch"):
        st.html("<h2>Ask about UK Open Banking</h2>")
        with st.form("ask", border=False):
            row = st.container(horizontal=True, vertical_alignment="center")
            question = row.text_input(
                "Ask a question about UK Open Banking",
                placeholder="Ask a question about UK Open Banking",
                label_visibility="collapsed",
            )
            submitted = row.form_submit_button("Ask", type="primary")
        st.html('<p class="ob-try">Or open a saved example</p>')
        with st.container(key="examples"):
            for index, example in enumerate(EXAMPLES):
                st.button(example, key=f"example_{index}", on_click=_pick_example,
                          args=(example,), width="stretch")

    st.html(tiles.disclaimer_html(DISCLAIMER))

    if submitted and question.strip():
        st.session_state["question"] = question
    question = st.session_state.get("question")
    if question:
        render_answer(question, settings)

    counts = []
    if runs:
        latest = runs[0]
        counts = tiles.decision_counts(summarise(EvalStore(settings).results(latest["id"])))
    st.html(tiles.stats_html(counts, runs[0]["label"] if runs else "", SNAPSHOT_DATE))


def render_answer(question: str, settings) -> None:
    budget = QueryBudget(max_queries=MAX_QUERIES_PER_SESSION)
    session_answers = st.session_state.setdefault("answers", {})
    with st.spinner("Searching the law and the API spec, then answering..."):
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

    answer_column, sources_column = st.columns([7, 5], gap="medium")
    with answer_column, st.container(key="answer_tile", height="stretch"):
        st.html(f"<h2>{escape(question)}</h2>")
        if answer.refused:
            st.html('<p class="ob-badge"><i></i>No grounded answer</p>')
        st.markdown(answer.text)
        note = (
            "Saved example answer. No API call was made." if source == "example"
            else f"{budget.remaining(st.session_state)} questions remaining this session."
        )
        st.html(f'<p class="ob-foot">{note}</p>')
    with sources_column, st.container(key="sources_tile", height="stretch"):
        st.html("<h3>Sources</h3>" + tiles.sources_html(tiles.source_rows(answer)))
        with st.expander(f"Retrieved passages ({len(answer.retrieved)})"):
            st.html('<div class="ob-passages">' + "".join(
                tiles.passage_html(item.chunk.citation, item.chunk.collection, item.score, item.chunk.text[:1200])
                for item in answer.retrieved
            ) + "</div>")


def render_eval_tab(settings, runs: list[dict]) -> None:
    if not runs:
        st.info("No evaluation runs recorded yet. Run `python -m obrag.evaluation.runner --label baseline`.")
        return
    store = EvalStore(settings)
    picker, scores = st.columns([4, 8], gap="medium")
    with picker, st.container(key="run_tile", height="stretch"):
        st.html(
            "<h2>Measured, run by run</h2>"
            "<p>The same 40 hand-written questions are asked after every change: "
            "10 on the law, 10 on the API spec, 10 that need both and 10 the sources "
            "cannot answer.</p>"
        )
        labels = {f"{r['id']}: {r['label']} ({r['created_at']})": r for r in runs}
        run = labels[st.selectbox("Run", list(labels))]
        routed = used_router(run)
        st.html(tiles.config_chips_html(run["config"], routed))

    results = store.results(run["id"])
    rows = summary_rows(summarise(results))
    if not routed:
        rows, results = drop_routing(rows), drop_routing(results)
    routing_note = "" if routed else (
        " Routing not scored: this run searched both collections for every question."
    )
    with scores, st.container(key="scores_tile", height="stretch"):
        st.html(
            "<h3>Scores by question type</h3>"
            + tiles.score_grid_html(rows, routed)
            + '<p class="ob-foot">Right call is a plain check: answered when the sources '
            "covered it, declined when they did not. Correct, Grounded and Relevant sources "
            f"are scored from 0 to 1 by a second model, so use them to compare runs.{routing_note}</p>"
        )

    with st.container(key="tables_tile"):
        with st.expander("Summary table"):
            st.dataframe(rows)
        with st.expander(f"Per-question results ({len(results)})"):
            st.dataframe(results)


def main() -> None:
    st.set_page_config(page_title="UK Open Banking RAG", page_icon=tiles.FAVICON, layout="wide")
    st.html(f"<style>{STYLES}</style>")
    st.html(tiles.topbar_html())

    # No Gemini retries: a visitor on an exhausted free-tier quota should see the
    # "unavailable" message in seconds, not after the SDK waits out 429s.
    settings = replace(load_settings(), gemini_retries=0)
    if "obrag_examples_cached" not in st.session_state:
        for example, answer in load_examples().items():
            cache_answer(example, answer)
        st.session_state["obrag_examples_cached"] = True

    runs = EvalStore(settings).runs()
    ask_tab, eval_tab = st.tabs(["Ask", "Evaluation"])
    with ask_tab:
        render_ask_tab(settings, runs)
    with eval_tab:
        render_eval_tab(settings, runs)


if __name__ == "__main__":
    main()
