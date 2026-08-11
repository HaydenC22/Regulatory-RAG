import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Regulatory RAG + Compliance Agent", layout="wide")
st.title("Regulatory RAG + Compliance Agent")
st.caption(
    "Citation-grounded Q&A and licensing-checklist agent over public Singapore MAS "
    "regulatory documents. Synthetic/public data only — not legal advice."
)

chat_tab, checklist_tab, eval_tab = st.tabs(["Chat", "Compliance Checklist", "Evaluation"])


with chat_tab:
    st.subheader("Ask a question about MAS regulation")
    question = st.text_input(
        "Question", placeholder="e.g. What does Notice PSN02 require for customer due diligence?"
    )
    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving and generating..."):
            try:
                resp = httpx.post(f"{API_BASE_URL}/api/v1/query", json={"question": question}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                st.error(f"Request failed: {exc}")
                data = None

        if data:
            if data["refused"]:
                st.warning(data["answer"])
            else:
                st.success(data["answer"])
            st.caption(f"Confidence: {data['confidence']} · Latency: {data['latency_ms']:.0f} ms")

            if data["citations"]:
                st.markdown("**Citations**")
                for c in data["citations"]:
                    with st.expander(f"{c['doc_id']} §{c.get('paragraph_id') or 'n/a'}"):
                        st.write(c["quote"])

            with st.expander("Retrieved chunks (debug)"):
                for chunk in data["retrieved_chunks"]:
                    st.markdown(
                        f"**{chunk['doc_id']} §{chunk.get('paragraph_id') or 'n/a'}** — {chunk['doc_title']}"
                    )


with checklist_tab:
    st.subheader("Licensing / obligation checklist")
    business_description = st.text_area(
        "Business description",
        placeholder="e.g. We operate a cross-border stablecoin payment service from Singapore.",
    )
    if st.button("Generate checklist", type="primary") and business_description:
        with st.spinner("Classifying activities and building checklist..."):
            try:
                resp = httpx.post(
                    f"{API_BASE_URL}/api/v1/checklist",
                    json={"business_description": business_description},
                    timeout=120,
                )
                resp.raise_for_status()
                checklist = resp.json()
            except httpx.HTTPError as exc:
                st.error(f"Request failed: {exc}")
                checklist = None

        if checklist:
            activities = ", ".join(checklist["classified_activities"]) or "none detected"
            st.markdown(f"**Classified activities:** {activities}")
            st.markdown(f"**Overall confidence:** {checklist['overall_confidence']}")

            for item in checklist["items"]:
                icon = {"yes": "✅", "no": "⬜", "uncertain": "❓"}[item["applicable"]]
                with st.expander(f"{icon} {item['requirement']} ({item['confidence']} confidence)"):
                    st.write(item["rationale"])
                    for c in item["citations"]:
                        st.caption(f"{c['doc_id']} §{c.get('paragraph_id') or 'n/a'}: {c['quote']}")

            if checklist["flagged_for_manual_review"]:
                st.warning("Flagged for manual review: " + ", ".join(checklist["flagged_for_manual_review"]))


with eval_tab:
    st.subheader("Evaluation results")
    try:
        resp = httpx.get(f"{API_BASE_URL}/api/v1/eval/latest", timeout=30)
        resp.raise_for_status()
        runs = resp.json()
    except httpx.HTTPError as exc:
        st.error(f"Could not load eval results: {exc}")
        runs = []

    if not runs:
        st.info("No eval runs recorded yet. Run `make eval` after ingestion to populate this tab.")
    else:
        import pandas as pd

        df = pd.DataFrame(runs).set_index("config_name")
        st.dataframe(df)

    st.markdown("See `docs/eval/report_final.md` for the full baseline-vs-final writeup.")
