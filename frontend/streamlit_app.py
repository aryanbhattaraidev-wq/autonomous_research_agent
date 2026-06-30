"""
frontend/streamlit_app.py
Streamlit UI for the Autonomous Research Agent.
Communicates with the FastAPI backend at http://localhost:8000.
"""
from __future__ import annotations
import os
import time
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")
PAGE_TITLE = "🔬 Research Agent"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
  code, pre { font-family: 'JetBrains Mono', monospace; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #0f0f17; color: #e0e0f0; }
  [data-testid="stSidebar"] * { color: #e0e0f0 !important; }

  /* Cards */
  .ra-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
  }
  .ra-tag {
    display: inline-block;
    background: #2d2d5e;
    color: #a0c0ff;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin: 2px 3px;
  }
  .ra-source {
    font-size: 0.8rem;
    color: #7090c0;
    word-break: break-all;
  }
  .ra-badge-done   { color: #50e090; }
  .ra-badge-error  { color: #ff6060; }
  .ra-step { display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0; }
  .ra-step .dot { width: 8px; height: 8px; border-radius: 50%; background: #5060ff; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────

def api_post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the backend. Make sure the FastAPI server is running on port 8000.")
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def api_get(path: str) -> dict | list | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def badge(ok: bool) -> str:
    return '<span class="ra-badge-done">✔ done</span>' if ok else '<span class="ra-badge-error">✘ error</span>'


# ── Session State ─────────────────────────────────────────────
if "research_result" not in st.session_state:
    st.session_state.research_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_topic" not in st.session_state:
    st.session_state.active_topic = ""


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 Research Agent")
    st.markdown("---")

    # Saved notes
    st.markdown("### 📚 Saved Notes")
    saved = api_get("/notes") or []
    if saved:
        for note_name in saved:
            if st.button(f"📄 {note_name}", key=f"note_{note_name}", use_container_width=True):
                st.session_state.active_topic = note_name
                # Load the note markdown
                md = api_get(f"/notes/{note_name}")
                if md:
                    st.session_state.research_result = {"_loaded_note": md, "topic": note_name}
    else:
        st.caption("No notes saved yet. Run a research session to create some.")

    st.markdown("---")

    # Memory topics
    st.markdown("### 🧠 Memory (Vector DB)")
    mem_topics = api_get("/memory/topics") or []
    if mem_topics:
        for t in mem_topics:
            st.markdown(f'<div class="ra-tag">{t}</div>', unsafe_allow_html=True)
    else:
        st.caption("Vector DB is empty.")

    st.markdown("---")
    st.markdown(
        "<small>Built with LangGraph · FastAPI · Qdrant · Streamlit</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════
st.markdown("# 🔬 Autonomous Research Agent")
st.markdown("Enter a topic and let the AI plan, search, summarise, and take notes — automatically.")

tab_research, tab_notes, tab_chat = st.tabs(["🚀 New Research", "📋 Report & Notes", "💬 Follow-up Q&A"])


# ── TAB 1 : New Research ──────────────────────────────────────
with tab_research:
    with st.form("research_form"):
        topic_input = st.text_area(
            "Research topic or question",
            placeholder="e.g.  Explain diffusion models and recent image generation techniques",
            height=100,
        )
        col1, col2 = st.columns([3, 1])
        with col2:
            submitted = st.form_submit_button("🔍 Research", use_container_width=True, type="primary")

    if submitted and topic_input.strip():
        topic = topic_input.strip()
        st.session_state.active_topic = topic
        st.session_state.chat_history = []

        progress_bar = st.progress(0, text="Initialising …")
        status_box = st.empty()

        steps = [
            (15, "🗺️  Planning research …"),
            (35, "🌐  Searching the web …"),
            (60, "📖  Summarising findings …"),
            (80, "📝  Writing notes …"),
            (95, "🧠  Storing to memory …"),
        ]
        for pct, msg in steps:
            progress_bar.progress(pct, text=msg)
            status_box.info(msg)
            time.sleep(0.3)

        with st.spinner("Running full research pipeline …"):
            result = api_post("/research", {"topic": topic})

        progress_bar.progress(100, text="✅ Complete!")
        status_box.empty()

        if result:
            st.session_state.research_result = result
            st.success(f"Research complete for **{topic}**! See the Report & Notes tab.")
            st.balloons()
        else:
            st.error("Research failed. Check that the backend is running and your API keys are set.")

    elif submitted:
        st.warning("Please enter a topic first.")


# ── TAB 2 : Report & Notes ────────────────────────────────────
with tab_notes:
    res = st.session_state.research_result

    if res is None:
        st.info("Run a research session first to see results here.")
    elif "_loaded_note" in res:
        # Loaded from sidebar
        st.markdown(f"## 📄 {res['topic']}")
        st.markdown(res["_loaded_note"])
    else:
        topic = res.get("topic", "")
        status = res.get("status", "")
        st.markdown(f"## {topic}")
        st.markdown(
            f'Status: {badge(status == "done")} &nbsp;|&nbsp; '
            f'<small>{len(res.get("summaries", []))} summaries · '
            f'{len(res.get("flashcards", []))} flashcards · '
            f'{len(res.get("sources", []))} sources</small>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Key concepts
        concepts = res.get("key_concepts", [])
        if concepts:
            st.markdown("### 🔑 Key Concepts")
            tags_html = "".join(f'<span class="ra-tag">{c}</span>' for c in concepts)
            st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown("")

        # Report / note markdown
        report = res.get("report") or res.get("notes_markdown", "")
        if report:
            with st.expander("📋 Full Research Report", expanded=True):
                st.markdown(report)

        # Flashcards
        flashcards = res.get("flashcards", [])
        if flashcards:
            with st.expander(f"🃏 Flashcards ({len(flashcards)})"):
                for i, fc in enumerate(flashcards, 1):
                    q = fc.get("question", "")
                    a = fc.get("answer", "")
                    with st.container():
                        st.markdown(f'<div class="ra-card"><b>Q{i}:</b> {q}</div>', unsafe_allow_html=True)
                        with st.expander("Show answer"):
                            st.write(a)

        # Summaries breakdown
        summaries = res.get("summaries", [])
        if summaries:
            with st.expander(f"📑 Summaries ({len(summaries)})"):
                for s in summaries:
                    st.markdown(f"#### {s.get('topic', 'Summary')}")
                    st.write(s.get("summary", ""))
                    ideas = s.get("key_ideas", [])
                    if ideas:
                        st.markdown("**Key ideas:** " + " · ".join(ideas))
                    terms = s.get("important_terms", [])
                    if terms:
                        st.markdown("**Terms:** " + " · ".join(terms))
                    st.markdown("---")

        # Sources
        sources = res.get("sources", [])
        if sources:
            with st.expander(f"🔗 Sources ({len(sources)})"):
                for src in sources:
                    st.markdown(f'<div class="ra-source">🔗 <a href="{src}" target="_blank">{src}</a></div>', unsafe_allow_html=True)

        # Download note
        notes_md = res.get("notes_markdown", report)
        if notes_md:
            st.download_button(
                "⬇️ Download Note (.md)",
                data=notes_md,
                file_name=f"{topic.lower().replace(' ', '_')[:40]}.md",
                mime="text/markdown",
            )


# ── TAB 3 : Follow-up Q&A ─────────────────────────────────────
with tab_chat:
    active_topic = st.session_state.active_topic
    if not active_topic:
        st.info("Run a research session first, then ask follow-up questions here.")
    else:
        st.markdown(f"### 💬 Ask about: **{active_topic}**")
        st.caption("Your questions are answered using the stored research notes (RAG).")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources used"):
                        for src in msg["sources"]:
                            st.markdown(f'<span class="ra-source">{src}</span>', unsafe_allow_html=True)

        # Chat input
        user_q = st.chat_input("Ask a follow-up question …")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.write(user_q)

            with st.chat_message("assistant"):
                with st.spinner("Thinking …"):
                    result = api_post("/followup", {
                        "topic": active_topic,
                        "question": user_q,
                        "conversation_history": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat_history
                        ],
                    })
                if result:
                    answer = result.get("answer", "Sorry, I could not generate an answer.")
                    sources = result.get("sources_used", [])
                    st.write(answer)
                    if sources:
                        with st.expander("Sources used"):
                            for src in sources:
                                st.markdown(f'<span class="ra-source">{src}</span>', unsafe_allow_html=True)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    st.error("Failed to get an answer. Is the backend running?")