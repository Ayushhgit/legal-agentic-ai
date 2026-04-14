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
    """
    Import the agent module and cache the compiled graph.
    @st.cache_resource ensures the knowledge base is built exactly once,
    even across multiple user sessions.
    """
    from agent import graph, run_query as _run_query
    return graph, _run_query


# Load agent (cached)
_graph, run_query = load_agent()

# CUSTOM CSS

st.markdown("""
<style>
/* General layout */
.stApp { background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif; }

/* Header banner */
.legal-header {
    background: linear-gradient(135deg, #0d1b4b 0%, #1a3a8f 60%, #2d5be3 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 14px rgba(13,27,75,0.35);
}
.legal-header h1 { margin: 0; font-size: 2rem; }
.legal-header p  { margin: 4px 0 0 0; font-size: 0.95rem; opacity: 0.85; }

/* Disclaimer box */
.disclaimer {
    background: #fff8e1;
    border-left: 5px solid #f9a825;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.86rem;
    margin-bottom: 16px;
}

/* Metric badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-green  { background:#d4edda; color:#155724; }
.badge-yellow { background:#fff3cd; color:#856404; }
.badge-red    { background:#f8d7da; color:#721c24; }
.badge-blue   { background:#cce5ff; color:#004085; }

/* Route chip */
.route-chip {
    background: #e8eaf6;
    color: #283593;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: monospace;
}

/* Sidebar */
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)


# SESSION STATE INITIALISATION

def _init_session():
    defaults = {
        "thread_id":    str(uuid.uuid4()),
        "chat_history": [],     # list of {role, content, meta?}
        "conv_history": [],     # raw messages list passed to agent
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

    # Session metrics
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

    # Available topics
    st.markdown("### Knowledge Base Topics")
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
    for t in topics:
        st.markdown(f"<small>- {t}</small>", unsafe_allow_html=True)

    st.divider()

    # Quick-action sample questions
    st.markdown("### Try a Sample Question")
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

    # New conversation
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
        "<small>Session ID: <code>{}</code></small>".format(
            st.session_state.thread_id[:12] + "..."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        "<small><b>Disclaimer:</b> Informational responses only. Not legal advice.</small>",
        unsafe_allow_html=True,
    )


# MAIN HEADER

st.markdown("""
<div class='legal-header'>
    <h1>Legal Document Assistant</h1>
    <p>Legal information for paralegals and junior lawyers</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='disclaimer'>
    <b>Important Notice:</b> This system provides informational responses based
    on a curated legal knowledge base. It does <b>NOT</b> constitute legal advice.
    Always consult a qualified attorney for advice specific to your situation.
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
### Welcome!

This assistant helps you quickly understand legal documents and concepts.
Start by typing a question below or selecting a sample from the sidebar.

**Example questions:**
- *What is a non-disclosure agreement?*
- *Explain the elements of a valid contract.*
- *What rights do data subjects have under GDPR?*
- *What is the difference between void and voidable contracts?*
""")

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

# Consume pending question (set by sidebar button)
pending = st.session_state.pop("pending_q", None)

# Chat input widget
user_input = st.chat_input("Ask a legal question ...", key="chat_input")

# Direct input takes precedence over sidebar button
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
