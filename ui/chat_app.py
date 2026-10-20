"""Streamlit chat UI for the RAG agent. Run with:

    uv run streamlit run ui/chat_app.py
"""
import os
import sys
import uuid
from pathlib import Path

# `streamlit run ui/chat_app.py` sets sys.path[0] to this file's own
# directory (ui/), not the project root - `import agent` fails without this,
# regardless of the cwd the command was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # local dev: reads .env, a no-op if it doesn't exist

try:
    # Streamlit Community Cloud: secrets set via the app's Secrets UI land
    # in st.secrets (TOML), not necessarily os.environ - bridge them so the
    # rest of this app (and everything in agent/, ingestion/) can keep
    # reading os.environ either way. Raises StreamlitSecretNotFoundError
    # locally when no .streamlit/secrets.toml exists, which is expected
    # and fine since load_dotenv() already covered that case above.
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except Exception:
    pass

from agent.config import AgentConfig
from agent.graph import build_agent
from agent.results import classify_answer_source, extract_sources

st.set_page_config(page_title="AI / Software-Dev Research Assistant", page_icon="📄")

st.markdown(
    """
    <style>
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
            "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 760px;
        padding-top: 3.5rem;
        padding-bottom: 7rem;
    }

    [data-testid="stHeading"] h1 {
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #1d1d1f;
    }

    [data-testid="stCaptionContainer"] {
        color: #6e6e73 !important;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 18px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03), 0 4px 14px rgba(0, 0, 0, 0.035);
    }

    [data-testid="stChatMessageContent"] p {
        color: #1d1d1f;
        line-height: 1.55;
    }

    [data-testid="stChatInput"] {
        border-radius: 26px;
        border: 1px solid rgba(0, 0, 0, 0.08);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        background-color: #ffffff;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #0071e3;
        box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.15);
    }

    [data-testid="stChatInputTextArea"] {
        color: #1d1d1f;
    }

    [data-testid="stChatInputSubmitButton"] {
        background-color: #0071e3 !important;
        border-radius: 999px !important;
    }

    [data-testid="stExpander"] {
        border-radius: 14px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        background-color: #ffffff;
    }

    [data-testid="stChatMessageAvatarUser"] {
        background-color: #1d1d1f !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #0071e3 !important;
    }

    a, a:visited {
        color: #0071e3;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }

    .source-badge {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 0.6rem;
    }
    .source-badge-corpus {
        background-color: rgba(0, 113, 227, 0.1);
        color: #0071e3;
    }
    .source-badge-web {
        background-color: rgba(255, 149, 0, 0.12);
        color: #c26a00;
    }
    .source-badge-none {
        background-color: rgba(0, 0, 0, 0.06);
        color: #6e6e73;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI / Software-Dev Research Assistant")
st.caption(
    "Grounded in a curated arXiv corpus of AI/software-engineering papers, "
    "with a live web search fallback when the corpus doesn't cover a question."
)

if "agent" not in st.session_state:
    st.session_state.agent = build_agent(AgentConfig())
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # [{"role", "content", "sources"}]
if "message_count" not in st.session_state:
    # Tracks how many messages the agent's checkpointed state held after the
    # last turn, so a new turn's tool-call results can be isolated from the
    # full accumulated thread history (result["messages"] is cumulative,
    # not just this turn's delta).
    st.session_state.message_count = 0


def _start_new_chat() -> None:
    # New thread_id is enough to give the checkpointer a blank slate for
    # this conversation - no need to rebuild the agent/LLM connection.
    st.session_state.history = []
    st.session_state.message_count = 0
    st.session_state.thread_id = str(uuid.uuid4())


with st.sidebar:
    st.button("🆕  New Chat", on_click=_start_new_chat, use_container_width=True)


def _render_badge(badge: dict | None) -> None:
    if not badge:
        return
    st.markdown(
        f'<span class="source-badge source-badge-{badge["kind"]}">{badge["icon"]} {badge["label"]}</span>',
        unsafe_allow_html=True,
    )


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            if source["kind"] == "corpus":
                st.markdown(f"📄 [{source['title']} ({source['arxiv_id']})]({source['url']})")
            else:
                st.markdown(f"🌐 [{source['title']}]({source['url']})")


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        _render_badge(turn.get("badge"))
        st.markdown(turn["content"])
        _render_sources(turn.get("sources") or [])

question = st.chat_input("Ask about AI or software-development research...")
if question:
    st.session_state.history.append({"role": "user", "content": question, "sources": None, "badge": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )

        new_messages = result["messages"][st.session_state.message_count :]
        st.session_state.message_count = len(result["messages"])

        answer = result["messages"][-1].content
        sources = extract_sources(new_messages)
        badge = classify_answer_source(new_messages)

        _render_badge(badge)
        st.markdown(answer)
        _render_sources(sources)

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "sources": sources, "badge": badge}
    )
