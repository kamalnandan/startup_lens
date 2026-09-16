"""
StartupLens V2
Colorful Streamlit UI for querying the Neo4j startup knowledge graph.
"""

import html
import time

import requests
import streamlit as st
from app_config import get_required_setting, get_setting


st.set_page_config(
    page_title="StartupLens",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Configuration ──────────────────────────────────────────────────────────────
ACCESS_CODE = get_required_setting("ACCESS_CODE")
ADMIN_CODE = get_required_setting("ADMIN_CODE")

#####
# ── Access Control ─────────────────────────────────────────────────────────────

def check_access():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.is_admin = False

    if not st.session_state.authenticated:
        st.markdown(
            """
            <div style="max-width: 400px; margin: 100px auto; text-align: center;">
                <h1>🔭 StartupLens</h1>
                <p style="color: #666;">YC Startup Intelligence</p>
                <p style="color: #888; font-size: 0.9rem;">Enter your access code to continue</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            code = st.text_input("Access code", type="password", placeholder="Enter access code")
            if st.button("Continue", type="primary", use_container_width=True):
                if code == ADMIN_CODE:
                    st.session_state.authenticated = True
                    st.session_state.is_admin = True
                    st.rerun()
                elif code == ACCESS_CODE:
                    st.session_state.authenticated = True
                    st.session_state.is_admin = False
                    st.rerun()
                else:
                    st.error("Invalid access code. DM Kamal on LinkedIn for access.")
        st.stop()

check_access()

#####



API_URL = get_required_setting("STARTUPLENS_API_URL").rstrip("/")
API_KEY = get_required_setting("STARTUPLENS_API_KEY")

# Two answer engines read two different graphs, and which one spoke changes
# what an answer is worth. Configuring both makes the choice visible in the
# app rather than a restart away; configuring one keeps the app exactly as it
# was, so nothing here can break a single-engine deployment.
ASSERTION_URL = get_setting("STARTUPLENS_API_URL_V2", "").rstrip("/")
ASSERTION_KEY = get_setting("STARTUPLENS_API_KEY_V2", "") or API_KEY

CLASSIC_ENGINE = "Classic"
ASSERTION_ENGINE = "Assertion"

ENGINES = {CLASSIC_ENGINE: (API_URL, API_KEY)}
if ASSERTION_URL:
    # Listed first so it is the default. The engine that shows its sources is
    # the one worth reaching for by accident.
    ENGINES = {ASSERTION_ENGINE: (ASSERTION_URL, ASSERTION_KEY), **ENGINES}


def active_engine() -> tuple:
    """The engine the sidebar has selected, as (name, url, key)."""
    name = st.session_state.get("engine") or next(iter(ENGINES))
    url, key = ENGINES[name]
    return name, url, key

# st.sidebar.info(f"Connecting to: {API_URL}")

# try:
#     r = requests.get(f"{API_URL}/health", timeout=5)
#     st.sidebar.success(f"Health check: {r.status_code}")
#     st.sidebar.json(r.json())
# except Exception as e:
#     st.sidebar.error(f"Connection error: {type(e).__name__}: {e}")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --primary: #6366f1;
            --secondary: #06b6d4;
            --success: #10b981;
            --ink: #172033;
            --muted: #64748b;
            --surface: rgba(255, 255, 255, 0.92);
        }

        html, body, [class*="css"] {
            font-family: "Inter", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 5%, rgba(99, 102, 241, 0.14), transparent 28%),
                radial-gradient(circle at 92% 12%, rgba(6, 182, 212, 0.12), transparent 26%),
                #f6f8fc;
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #10172a 0%, #172554 100%);
        }

        [data-testid="stSidebar"] * {
            color: #eef2ff;
        }

        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: #67e8f9;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.5rem 2.75rem;
            margin-bottom: 1.25rem;
            border-radius: 24px;
            color: white;
            background: linear-gradient(120deg, #4338ca 0%, #6366f1 48%, #0891b2 100%);
            box-shadow: 0 18px 45px rgba(67, 56, 202, 0.22);
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -70px;
            top: -100px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.12);
        }

        .hero-kicker {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            opacity: 0.82;
        }

        .hero h1 {
            margin: 0.35rem 0;
            font-size: clamp(2.2rem, 5vw, 3.8rem);
            line-height: 1.05;
            font-weight: 800;
        }

        .hero p {
            max-width: 720px;
            margin: 0;
            font-size: 1.05rem;
            line-height: 1.6;
            color: #e0e7ff;
        }

        .section-title {
            margin: 1.4rem 0 0.7rem;
            font-size: 1.1rem;
            font-weight: 800;
            color: var(--ink);
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.5rem 0 1.3rem;
        }

        .metric-card {
            min-height: 105px;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 18px;
            background: var(--surface);
            box-shadow: 0 8px 24px rgba(30, 41, 59, 0.07);
        }

        .metric-card .icon {
            font-size: 1.35rem;
        }

        .metric-card .value {
            margin-top: 0.35rem;
            font-size: 1.65rem;
            font-weight: 800;
            color: var(--ink);
        }

        .metric-card .label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--muted);
        }

        .result-card {
            margin-top: 0.7rem;
            padding: 1.4rem 1.5rem;
            border: 1px solid rgba(99, 102, 241, 0.18);
            border-left: 5px solid var(--primary);
            border-radius: 18px;
            background: var(--surface);
            box-shadow: 0 12px 30px rgba(30, 41, 59, 0.08);
        }

        .badge {
            display: inline-block;
            margin: 0 0.4rem 0.4rem 0;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            color: #3730a3;
            background: #e0e7ff;
        }

        .badge.cyan {
            color: #155e75;
            background: #cffafe;
        }

        .badge.green {
            color: #065f46;
            background: #d1fae5;
        }

        div[data-testid="stTextArea"] textarea {
            border: 1px solid #c7d2fe;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.95);
            box-shadow: 0 8px 24px rgba(30, 41, 59, 0.06);
            font-size: 1rem;
        }

        div.stButton > button {
            border: 1px solid #c7d2fe;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.9);
            color: #3730a3;
            font-weight: 650;
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            border-color: var(--primary);
            color: var(--primary);
            box-shadow: 0 7px 18px rgba(99, 102, 241, 0.16);
            transform: translateY(-1px);
        }

        div.stButton > button[kind="primary"] {
            border: none;
            color: white;
            background: linear-gradient(90deg, #4f46e5, #0891b2);
        }

        @media (max-width: 900px) {
            .metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .hero {
                padding: 2rem 1.5rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


EXAMPLE_QUERIES = {
    "🎯 YC Research": [
        "Which YC batches produced the most fintech companies?",
        "What industries are trending in the most recent YC batches?",
        "How many YC companies are Active vs Dead vs Acquired?",
    ],
    "🌍 Founders": [
        "Which YC founders from India built fintech companies?",
        "Which countries outside the US produce the most YC companies?",
        "Who are the most connected founders across YC?",
    ],
    "💰 Acquisitions & Funding": [
        "Which YC companies were acquired and by whom?",
        "Which companies raised Series A funding?",
        "Which investors have the largest YC portfolios?",
    ],
    "📈 Patterns": [
        "Which industries have the most failed (Dead) YC startups?",
        "Which technologies are most used by YC companies?",
        "What is the breakdown of YC companies by stage?",
    ],
}

METRIC_STYLES = {
    "Company": ("🏢", "Companies"),
    "Founder": ("👤", "Founders"),
    "Investor": ("💰", "Investors"),
    "Industry": ("🧭", "Industries"),
}


if "history" not in st.session_state:
    st.session_state.history = []
if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def select_query(query: str) -> None:
    st.session_state.query_text = query


def clear_query() -> None:
    st.session_state.query_text = ""
    st.session_state.last_result = None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_graph_stats(base_url: str) -> dict:
    """Counts from whichever engine is selected.

    The url is a parameter rather than a constant so the cache is keyed by
    engine. Sharing one cache entry would show the previous engine's counts
    after a switch, which is the most quietly misleading thing this page
    could do.
    """
    response = requests.get(f"{base_url}/status", timeout=5)
    response.raise_for_status()
    payload = response.json()
    if "graph_stats" in payload:
        return payload["graph_stats"]
    # The assertion API labels its nodes V2Company rather than Company so the
    # two graphs can never be confused for one another. The sidebar counts the
    # same things either way, so strip the prefix rather than special-case it.
    return {name[2:] if name.startswith("V2") else name: count
            for name, count in payload.get("counts", {}).items()}


def metric_cards(stats: dict) -> str:
    cards = []
    for node_type, (icon, label) in METRIC_STYLES.items():
        value = stats.get(node_type, 0)
        cards.append(
            '<div class="metric-card">'
            f'<div class="icon">{icon}</div>'
            f'<div class="value">{html.escape(f"{value:,}")}</div>'
            f'<div class="label">{label}</div>'
            "</div>"
        )
    return '<div class="metric-grid">' + "".join(cards) + "</div>"


def render_sources(parts: list) -> None:
    """Show the sentence behind every fact in the answer.

    The citation is the product. An answer a reader cannot check is worth
    about as much as a guess, so the evidence is one click away rather than
    hidden behind an admin flag.
    """
    cited = [part for part in parts if part.get("rows")]
    if not cited:
        return
    total = sum(len(part["rows"]) for part in cited)
    with st.expander(f"📎 Sources ({total:,} supporting facts)", expanded=False):
        for part in cited:
            st.caption(part["question"])
            for row in part["rows"][:25]:
                evidence = row.get("evidence")
                values = ", ".join(str(value) for key, value in row.items()
                                   if key != "evidence" and value is not None)
                st.markdown(f"- **{values}**")
                if evidence:
                    st.markdown(f"  > {evidence}")


def render_result(result: dict) -> None:
    method = str(result.get("method", "unknown")).title()
    count = int(result.get("result_count", 0))
    duration = float(result.get("duration", 0))

    st.markdown('<div class="section-title">💬 Intelligence Brief</div>', unsafe_allow_html=True)
    # The engine is named on the answer itself. A toggle in the sidebar can be
    # left on the wrong setting, and an answer whose origin you have to
    # remember is an answer you cannot trust later.
    engine = result.get("engine")
    engine_badge = (f'<span class="badge">{html.escape(engine)} engine</span>'
                    if engine else "")
    st.markdown(
        '<div class="result-card">'
        f'{engine_badge}'
        f'<span class="badge">{html.escape(method)} search</span>'
        f'<span class="badge cyan">{count:,} records</span>'
        f'<span class="badge green">{duration:.2f} seconds</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    answer = result.get("answer")
    if answer:
        st.markdown(answer)
    else:
        # Not an error. The assertion API returns no answer when the graph
        # holds no supported fact, and saying so plainly is the point - the
        # alternative is a confident sentence nothing backs.
        st.info("No supported fact was found for this question, so no answer "
                "is given. Nothing here is inferred or filled in from "
                "elsewhere.")

    render_sources(result.get("parts", []))

    # if result.get("cypher"):
    #     with st.expander("🔧 View generated Cypher", expanded=False):
    #         st.code(result["cypher"], language="cypher")

    if result.get("cypher") and st.session_state.get("is_admin", False):
        with st.expander("🔧 View generated Cypher", expanded=False):
            st.code(result["cypher"], language="cypher")


with st.sidebar:
    st.markdown("## 🔭 StartupLens")
    st.caption("YC Startup Intelligence")
    st.divider()

    if len(ENGINES) > 1:
        st.markdown("### Answer engine")
        st.radio(
            "Answer engine",
            list(ENGINES),
            key="engine",
            label_visibility="collapsed",
            captions=["Grounded in extracted assertions, with sources"
                      if name == ASSERTION_ENGINE else "The original graph"
                      for name in ENGINES],
        )
        st.divider()

    engine_name, engine_url, engine_key = active_engine()

    st.markdown("### Live graph")
    try:
        sidebar_stats = fetch_graph_stats(engine_url)
        st.success("API connected")
        for node_type in ("Company", "Founder", "Investor", "Industry"):
            if node_type in sidebar_stats:
                st.metric(node_type, f"{sidebar_stats[node_type]:,}")
    except requests.RequestException:
        sidebar_stats = {}
        st.warning("API is currently offline")

    st.divider()
    st.markdown("### Recent questions")
    if st.session_state.history:
        for index, item in enumerate(reversed(st.session_state.history[-5:])):
            st.button(
                f"↩ {item['query'][:32]}...",
                key=f"sidebar_history_{index}",
                on_click=select_query,
                args=(item["query"],),
                use_container_width=True,
            )
    else:
        st.caption("Your recent questions will appear here.")

    st.divider()
    st.caption("Powered by Neo4j + Azure OpenAI")


st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">YC Startup Intelligence · 5,700+ Companies · 20 Years of Data</div>
        <h1>Research Y Combinator<br>like never before.</h1>
        <p>
            Explore every YC batch, founder, investor, and industry trend
            through an AI-powered knowledge graph. Ask complex questions,
            get instant answers.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    graph_stats = fetch_graph_stats(active_engine()[1])
except requests.RequestException:
    graph_stats = {}

st.markdown(metric_cards(graph_stats), unsafe_allow_html=True)

st.markdown('<div class="section-title">✨ Questions founders actually ask</div>', unsafe_allow_html=True)
tabs = st.tabs(list(EXAMPLE_QUERIES))
for tab, questions in zip(tabs, EXAMPLE_QUERIES.values()):
    with tab:
        columns = st.columns(2)
        for index, question in enumerate(questions):
            with columns[index % 2]:
                st.button(
                    question,
                    key=f"example_{question}",
                    on_click=select_query,
                    args=(question,),
                    use_container_width=True,
                )

st.markdown('<div class="section-title">🔎 Ask StartupLens</div>', unsafe_allow_html=True)
query = st.text_area(
"Ask about YC companies, founders, investors, batches, or trends",
placeholder="For example: What industries are trending in recent YC batches?",
    height=105,
    key="query_text",
)

search_column, clear_column, spacer = st.columns([1.2, 1, 5])
with search_column:
    search_clicked = st.button(
        "Search graph",
        type="primary",
        use_container_width=True,
    )
with clear_column:
    st.button(
        "Clear",
        on_click=clear_query,
        use_container_width=True,
    )

if search_clicked:
    if not query.strip():
        st.warning("Enter a question before searching.")
    else:
        with st.spinner("Exploring the startup knowledge graph..."):
            started_at = time.time()
            try:
                # Forward end-user IP to FastAPI
                user_ip = st.context.headers.get("X-Forwarded-For", "unknown").split(",")[0].strip()
                engine_name, engine_url, engine_key = active_engine()
                response = requests.post(
                    f"{engine_url}/query",
                    json={"query": query.strip()},
                    headers={"X-Real-IP": user_ip, "X-API-Key": engine_key},
                    timeout=120,
                )
                response.raise_for_status()
                payload = response.json()
                rows = sum(len(part.get("rows", []))
                           for part in payload.get("parts", []))
                result = {
                    "query": query.strip(),
                    "answer": payload["answer"],
                    "method": payload.get("method", payload.get("status", "unknown")),
                    "cypher": payload.get("cypher"),
                    "result_count": payload.get("result_count", rows),
                    "parts": payload.get("parts", []),
                    "status": payload.get("status", ""),
                    "engine": engine_name,
                    "duration": round(time.time() - started_at, 2),
                }
                st.session_state.last_result = result
                st.session_state.history.append(result)
            except requests.ConnectionError:
                st.error("Cannot connect to the API. Make sure the FastAPI server is running.")
            except requests.Timeout:
                st.error("The query timed out. Try a more specific question.")
            except requests.RequestException:
                st.error("The API could not complete the query.")
            except (KeyError, ValueError):
                st.error("The API returned an unexpected response.")

if st.session_state.last_result:
    render_result(st.session_state.last_result)

if len(st.session_state.history) > 1:
    st.markdown('<div class="section-title">🕐 Previous research</div>', unsafe_allow_html=True)
    for item in reversed(st.session_state.history[:-1]):
        with st.expander(item["query"], expanded=False):
            st.markdown(item["answer"] or "_No supported fact was found._")
            detail_columns = st.columns(3)
            detail_columns[0].caption(f"Engine: {item.get('engine', 'unknown')}")
            detail_columns[1].caption(f"Records: {item['result_count']:,}")
            detail_columns[2].caption(f"Time: {item['duration']:.2f}s")
            # if item.get("cypher"):
            #     st.code(item["cypher"], language="cypher")
            if item.get("cypher") and st.session_state.get("is_admin", False):
                st.code(item["cypher"], language="cypher")
