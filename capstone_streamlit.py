"""
Run with:
    streamlit run capstone_streamlit.py

Requirements (env):
    GROQ_API_KEY=gsk_...
"""

import uuid
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Legal Document Assistant",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CACHED RESOURCES

@st.cache_resource(show_spinner="Loading Legal Assistant ...")
def load_agent():
    from agent import graph, run_query as _run_query
    return graph, _run_query


_graph, run_query = load_agent()

# CUSTOM CSS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --navy:      #0b1426;
    --navy-mid:  #111d35;
    --navy-card: #162040;
    --navy-border: #1e2d50;
    --blue:      #2563eb;
    --blue-light: #3b82f6;
    --gold:      #f59e0b;
    --gold-dim:  #d97706;
    --text-primary:   #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted:     #64748b;
    --green:  #10b981;
    --yellow: #f59e0b;
    --red:    #ef4444;
    --cyan:   #06b6d4;
}

/* ── Global reset ── */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .block-container {
    background-color: #0b1426 !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* ── Remove Streamlit default padding ── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 860px !important;
}

/* ── ALL text elements ── */
p, span, div, li, label, small, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #f1f5f9 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d1a30 !important;
    border-right: 1px solid #1e2d50 !important;
}
[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] small {
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
}

/* ── Sidebar divider ── */
[data-testid="stSidebar"] hr {
    border-color: #1e2d50 !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #162040 !important;
    color: #cbd5e1 !important;
    border: 1px solid #1e2d50 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background-color: #1e3a6e !important;
    color: #f1f5f9 !important;
    border-color: #2563eb !important;
}
.stButton > button[kind="secondary"] {
    background-color: #1e2d50 !important;
    color: #94a3b8 !important;
}
.stButton > button[kind="secondary"]:hover {
    background-color: #2563eb !important;
    color: #fff !important;
}

/* ── Metrics ── */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="metric-container"] {
    color: #f1f5f9 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #60a5fa !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div {
    background-color: #1e2d50 !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #2563eb, #06b6d4) !important;
}
[data-testid="stProgressBar"] + div {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background-color: #111d35 !important;
    border: 1px solid #1e2d50 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: #f1f5f9 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background-color: #1e3a6e !important;
    border-radius: 12px !important;
    border: 1px solid #2563eb !important;
    padding: 12px 16px !important;
    margin: 6px 0 !important;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background-color: #111d35 !important;
    border-radius: 12px !important;
    border: 1px solid #1e2d50 !important;
    padding: 12px 16px !important;
    margin: 6px 0 !important;
}

/* Avatar icons */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    background-color: #162040 !important;
    border: 1px solid #1e2d50 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background-color: #111d35 !important;
    border: 1px solid #1e2d50 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #f1f5f9 !important;
    caret-color: #2563eb !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #475569 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.15) !important;
}

/* ── Spinner text ── */
[data-testid="stSpinner"] p {
    color: #94a3b8 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0b1426; }
::-webkit-scrollbar-thumb { background: #1e2d50; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #2563eb; }

/* ── Custom components ── */
.legal-header {
    background: linear-gradient(135deg, #0f2460 0%, #1a3a8f 50%, #1e40af 100%);
    color: white;
    padding: 22px 28px;
    border-radius: 12px;
    margin-bottom: 18px;
    border: 1px solid #2563eb;
    box-shadow: 0 4px 24px rgba(37,99,235,0.2);
}
.legal-header h1 {
    margin: 0;
    font-size: 1.7rem;
    font-weight: 700;
    color: #fff !important;
    letter-spacing: -0.02em;
}
.legal-header p {
    margin: 5px 0 0 0;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.75) !important;
}

.disclaimer {
    background: rgba(245,158,11,0.08);
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.83rem;
    color: #fcd34d !important;
    margin-bottom: 18px;
}
.disclaimer b { color: #fde68a !important; }

.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-right: 5px;
    font-family: 'Inter', sans-serif;
}
.badge-green  { background: rgba(16,185,129,0.15); color: #34d399 !important; border: 1px solid rgba(16,185,129,0.3); }
.badge-yellow { background: rgba(245,158,11,0.15); color: #fbbf24 !important; border: 1px solid rgba(245,158,11,0.3); }
.badge-red    { background: rgba(239,68,68,0.15);  color: #f87171 !important; border: 1px solid rgba(239,68,68,0.3); }
.badge-blue   { background: rgba(6,182,212,0.15);  color: #22d3ee !important; border: 1px solid rgba(6,182,212,0.3); }

.route-chip {
    background: rgba(37,99,235,0.12);
    color: #60a5fa !important;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid rgba(37,99,235,0.25);
}

.welcome-box {
    background: linear-gradient(135deg, #111d35, #162040);
    border: 1px solid #1e2d50;
    border-radius: 12px;
    padding: 28px 32px;
    margin: 12px 0 24px;
}
.welcome-box h3 {
    color: #f1f5f9 !important;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 10px;
}
.welcome-box p {
    color: #94a3b8 !important;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 16px;
}
.welcome-box ul {
    list-style: none;
    padding: 0;
    margin: 0;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}
.welcome-box ul li {
    background: rgba(37,99,235,0.08);
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: 8px;
    padding: 9px 14px;
    font-size: 0.82rem;
    color: #93c5fd !important;
    cursor: default;
}

.topic-pill {
    display: inline-block;
    background: rgba(37,99,235,0.08);
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.75rem;
    color: #93c5fd !important;
    margin: 3px 2px;
}
</style>
""", unsafe_allow_html=True)


# SESSION STATE INITIALISATION

def _init_session():
    defaults = {
        "thread_id":    str(uuid.uuid4()),
        "chat_history": [],
        "conv_history": [],
        "total_queries": 0,
        "scores":       [],
        "pending_q":    None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


# SIDEBAR

with st.sidebar:
    st.markdown("## Legal Assistant")
    st.divider()

    st.markdown("### Session Stats")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Queries", st.session_state.total_queries)
    with col_b:
        scores = st.session_state.scores
        avg_s  = sum(scores) / len(scores) if scores else 0.0
        st.metric("Avg Score", f"{avg_s:.2f}" if scores else "-")

    if scores:
        st.progress(avg_s, text=f"Quality: {avg_s:.0%}")

    st.divider()

    st.markdown("### Knowledge Base")
    topics = [
        "Contract Law Basics",
        "Non-Disclosure Agreements",
        "Employment Agreements",
        "Legal Terminology",
        "Intellectual Property",
        "Data Protection (GDPR/CCPA)",
        "Compliance Policies",
        "Legal Document Structure",
        "Common Legal Clauses",
        "Liability & Indemnity",
    ]
    topic_html = "".join(f"<span class='topic-pill'>{t}</span>" for t in topics)
    st.markdown(topic_html, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Sample Questions")
    samples = [
        "What is an NDA?",
        "Explain force majeure",
        "What is GDPR?",
        "What are IP types?",
        "What is indemnification?",
        "What day is today?",
    ]
    for q in samples:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.pending_q = q
            st.rerun()

    st.divider()

    if st.button("New Conversation", use_container_width=True, type="secondary"):
        st.session_state.thread_id    = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.conv_history = []
        st.session_state.total_queries = 0
        st.session_state.scores        = []
        st.session_state.pending_q     = None
        st.rerun()

    st.divider()
    st.markdown(
        "<small style='color:#475569'>Session: <code style='color:#60a5fa'>{}</code></small>".format(
            st.session_state.thread_id[:12] + "..."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        "<small style='color:#475569'>Informational use only. Not legal advice.</small>",
        unsafe_allow_html=True,
    )


# MAIN HEADER

st.markdown("""
<div class='legal-header'>
    <h1>Legal Document Assistant</h1>
    <p>Legal information for paralegals and junior lawyers &nbsp;·&nbsp; Powered by RAG + Agentic AI</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='disclaimer'>
    <b>Notice:</b> Responses are based on a curated legal knowledge base.
    This system does <b>NOT</b> provide legal advice.
    Always consult a qualified attorney for your specific situation.
</div>
""", unsafe_allow_html=True)


# HELPERS

def _score_badge(score: float) -> str:
    if score >= 0.80:
        cls, label = "badge-green",  "GOOD"
    elif score >= 0.60:
        cls, label = "badge-yellow", "FAIR"
    else:
        cls, label = "badge-red",    "LOW"
    return f"<span class='badge {cls}'>{label} {score:.2f}</span>"


def _route_chip(route: str) -> str:
    return f"<span class='route-chip'>{route}</span>"


# CHAT HISTORY DISPLAY

if not st.session_state.chat_history:
    st.markdown("""
<div class='welcome-box'>
    <h3>Welcome to Legal Document Assistant</h3>
    <p>Ask questions about legal concepts, document types, clauses, and more.
    The assistant searches a curated legal knowledge base and evaluates its own answers before responding.</p>
    <ul>
        <li>What is a non-disclosure agreement?</li>
        <li>Explain elements of a valid contract</li>
        <li>What rights do data subjects have under GDPR?</li>
        <li>Difference between void and voidable contracts?</li>
    </ul>
</div>
""", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

            meta = msg.get("meta", {})
            if meta:
                score   = meta.get("score",       0.0)
                route   = meta.get("route",        "-")
                retries = meta.get("eval_retries", 0)
                sources = meta.get("sources",      [])

                chips = _score_badge(score) + "&nbsp;&nbsp;" + _route_chip(route)
                if retries:
                    chips += f"&nbsp;&nbsp;<span class='badge badge-blue'>{retries} retry</span>"
                st.markdown(chips, unsafe_allow_html=True)

                with st.expander("Response Details", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Quality Score", f"{score:.2f}")
                    c2.metric("Route",         route)
                    c3.metric("Retries",       retries)
                    if sources:
                        st.markdown("**Sources retrieved:**")
                        for s in sources:
                            st.markdown(f"- {s}")


# INPUT HANDLING

pending = st.session_state.pop("pending_q", None)
user_input = st.chat_input("Ask a legal question ...", key="chat_input")
active_question = user_input or pending

if active_question:
    active_question = active_question.strip()

    with st.chat_message("user"):
        st.markdown(active_question)

    st.session_state.chat_history.append({"role": "user", "content": active_question})

    with st.chat_message("assistant"):
        with st.spinner("Searching legal knowledge base ..."):
            result = run_query(
                question             = active_question,
                thread_id            = st.session_state.thread_id,
                conversation_history = list(st.session_state.conv_history),
            )

        answer  = result["answer"]
        score   = result["score"]
        route   = result["route"]
        retries = result["eval_retries"]
        sources = result["sources"]

        st.markdown(answer)

        chips = _score_badge(score) + "&nbsp;&nbsp;" + _route_chip(route)
        if retries:
            chips += f"&nbsp;&nbsp;<span class='badge badge-blue'>{retries} retry</span>"
        st.markdown(chips, unsafe_allow_html=True)

        with st.expander("Response Details", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Quality Score", f"{score:.2f}")
            c2.metric("Route",         route)
            c3.metric("Retries",       retries)
            if sources:
                st.markdown("**Sources retrieved:**")
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": answer,
        "meta": {
            "score":        score,
            "route":        route,
            "eval_retries": retries,
            "sources":      sources,
        },
    })

    st.session_state.conv_history = result.get("messages", [])
    st.session_state.total_queries += 1
    st.session_state.scores.append(score)

    st.rerun()
