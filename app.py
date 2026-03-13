"""
app.py — CreditAI · Professional Fintech Credit Risk Dashboard.

Run with:  streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from model import load_model, predict

# ─────────────────────────────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CreditAI — Risk Assessor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
#  CSS — refined dark fintech theme
# ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Font ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

/* ── App background ────────────────────────────────────── */
.stApp {
    background: linear-gradient(168deg, #0b1120 0%, #111a2e 40%, #0f1629 100%);
}
.block-container { padding: 1.4rem 2rem 2rem; max-width: 1200px; }

/* ── Sidebar ───────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #080d19;
    border-right: 1px solid rgba(255,255,255,0.04);
}
section[data-testid="stSidebar"] * { color: #b0bec5 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; font-weight: 700 !important; }

/* ── Metric cards ──────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #131b2e;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    transition: box-shadow 0.25s ease;
}
div[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.45);
}
div[data-testid="stMetric"] label {
    color: #5f7285 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e8ecf1 !important;
    font-weight: 700 !important;
    font-size: 1.35rem !important;
}

/* ── Cards / containers ────────────────────────────────── */
.card {
    background: #131b2e;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 24px 26px;
    box-shadow: 0 2px 14px rgba(0,0,0,0.25);
}
.card h4 {
    margin: 0 0 14px;
    font-size: 0.82rem;
    font-weight: 700;
    color: #5f7285;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}

/* ── Tabs ──────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.01em;
}

/* ── Button ────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    box-shadow: 0 3px 14px rgba(37,99,235,0.3);
    transition: all 0.25s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(37,99,235,0.45);
}

/* ── Risk badges ───────────────────────────────────────── */
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 7px 22px; border-radius: 30px;
    font-weight: 700; font-size: 0.82rem;
    letter-spacing: 0.07em; text-transform: uppercase;
}
.badge-low    { background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.25); }
.badge-medium { background: rgba(245,158,11,0.12); color: #f59e0b; border: 1px solid rgba(245,158,11,0.25); }
.badge-high   { background: rgba(239,68,68,0.12);  color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }

/* ── About cards ───────────────────────────────────────── */
.about-card {
    background: #131b2e;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 26px 28px;
    box-shadow: 0 2px 14px rgba(0,0,0,0.25);
    height: 100%;
}
.about-card h3 {
    margin: 0 0 12px; font-size: 1rem; font-weight: 700; color: #e2e8f0;
}
.about-card p, .about-card li {
    color: #8899a6; font-size: 0.86rem; line-height: 1.75;
}
.about-card b { color: #cbd5e1; }

/* ── Divider / footer ──────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.05) !important; margin: 1rem 0 !important; }
.footer { text-align: center; color: #3e4c5e; font-size: 0.74rem; padding: 12px 0 4px; letter-spacing: 0.04em; }

/* ── Hero ──────────────────────────────────────────────── */
.hero { text-align: center; padding: 0.4rem 0 0.2rem; }
.hero h1 {
    font-size: 1.8rem; font-weight: 800; color: #e8ecf1;
    margin-bottom: 2px;
}
.hero p { color: #5f7285; font-size: 0.85rem; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def get_model():
    return load_model()

try:
    bundle = get_model()
    model_loaded = True
except FileNotFoundError as exc:
    model_loaded = False
    model_error = str(exc)

# ─────────────────────────────────────────────────────────────────────
#  Plotly helpers (dark theme, minimal chrome)
# ─────────────────────────────────────────────────────────────────────
_PLOT_BG = "rgba(0,0,0,0)"
_GRID    = "rgba(255,255,255,0.04)"
_TICK    = dict(color="#5f7285", size=10)
_FONT   = dict(family="Inter, sans-serif")


def _score_color(score: int) -> str:
    if score >= 750: return "#10b981"
    if score >= 650: return "#f59e0b"
    return "#ef4444"


def make_gauge(score: int) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(size=48, color="#e8ecf1", family="Inter"), suffix=""),
        title=dict(text="CREDIT SCORE", font=dict(size=11, color="#5f7285")),
        gauge=dict(
            axis=dict(range=[300, 900], tickwidth=1, tickcolor="#1e293b",
                      tickfont=_TICK),
            bar=dict(color=_score_color(score), thickness=0.28),
            bgcolor="#1a2235",
            borderwidth=0,
            steps=[
                dict(range=[300, 600], color="rgba(239,68,68,0.06)"),
                dict(range=[600, 650], color="rgba(245,158,11,0.06)"),
                dict(range=[650, 700], color="rgba(245,158,11,0.04)"),
                dict(range=[700, 750], color="rgba(16,185,129,0.04)"),
                dict(range=[750, 900], color="rgba(16,185,129,0.08)"),
            ],
            threshold=dict(line=dict(color="#3b82f6", width=2), thickness=0.75, value=650),
        ),
    ))
    fig.update_layout(
        height=240, margin=dict(l=28, r=28, t=36, b=8),
        paper_bgcolor=_PLOT_BG, font=_FONT,
    )
    return fig


def make_prob_bar(prob: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[""], x=[1 - prob], name="Repayment", orientation="h",
        marker=dict(color="#10b981"), text=[f"{(1-prob)*100:.1f}%"],
        textposition="inside", textfont=dict(color="white", size=12, family="Inter"),
    ))
    fig.add_trace(go.Bar(
        y=[""], x=[prob], name="Default", orientation="h",
        marker=dict(color="#ef4444"), text=[f"{prob*100:.1f}%"],
        textposition="inside", textfont=dict(color="white", size=12, family="Inter"),
    ))
    fig.update_layout(
        barmode="stack", height=64,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=_PLOT_BG, plot_bgcolor=_PLOT_BG,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5,
                    font=dict(color="#8899a6", size=10)),
        xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False),
    )
    return fig


def make_radar(metrics: dict) -> go.Figure:
    labels = list(metrics.keys())
    vals   = list(metrics.values())
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(59,130,246,0.1)",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=6, color="#60a5fa"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=_PLOT_BG,
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=_GRID,
                            tickfont=dict(color="#5f7285", size=8)),
            angularaxis=dict(tickfont=dict(color="#b0bec5", size=11), gridcolor=_GRID),
        ),
        height=310, margin=dict(l=56, r=56, t=24, b=24),
        paper_bgcolor=_PLOT_BG, showlegend=False, font=_FONT,
    )
    return fig


def make_fi_chart(fi: dict) -> go.Figure:
    df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
    df = df.sort_values("Importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["Importance"], y=df["Feature"], orientation="h",
        marker=dict(color=df["Importance"],
                    colorscale=[[0, "#1e3a5f"], [0.5, "#2563eb"], [1, "#60a5fa"]]),
        texttemplate="%{x:.4f}", textposition="outside",
        textfont=dict(color="#5f7285", size=9),
    ))
    fig.update_layout(
        height=max(320, len(df) * 26),
        margin=dict(l=8, r=56, t=8, b=8),
        paper_bgcolor=_PLOT_BG, plot_bgcolor=_PLOT_BG,
        xaxis=dict(title="", tickfont=_TICK, gridcolor=_GRID, zeroline=False),
        yaxis=dict(tickfont=dict(color="#b0bec5", size=10)),
        font=_FONT,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="hero">'
    '<h1>🏦 CreditAI</h1>'
    '<p>AI-Powered Credit Risk Assessment · XGBoost · CIBIL-Aligned 300 – 900</p>'
    '</div>', unsafe_allow_html=True,
)
st.markdown("---")

if not model_loaded:
    st.error(f"⚠️ **Model not found.** Run `python train_model.py` first.\n\n```\n{model_error}\n```")
    st.stop()

# ─────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📋 Applicant Profile")
    st.caption("Complete all fields, then click **Assess Credit Risk**.")
    st.markdown("---")

    st.markdown("### 💰 Financials")
    income        = st.number_input("Annual Income (₹)", 0, 100_000_000, 200_000, 10_000, format="%d")
    credit_amount = st.number_input("Credit Amount (₹)", 0, 100_000_000, 500_000, 10_000, format="%d")
    annuity       = st.number_input("Annuity Amount (₹)", 0, 10_000_000, 25_000, 1_000, format="%d")
    goods_price   = st.number_input("Goods Price (₹)", 0, 100_000_000, 450_000, 10_000, format="%d")

    st.markdown("---")
    st.markdown("### 👤 Personal")
    age                = st.slider("Age (years)", 18, 80, 35)
    employment_years   = st.slider("Employment (years)", 0, 45, 5)
    registration_years = st.slider("Registration (years)", 0, 50, 10)

    st.markdown("---")
    st.markdown("### 📑 Profile")
    own_car       = st.selectbox("Owns Car?", ["No", "Yes"])
    own_realty    = st.selectbox("Owns Real Estate?", ["No", "Yes"])
    family_status = st.selectbox("Family Status", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"])
    housing       = st.selectbox("Housing", ["House / apartment", "Rented apartment", "With parents", "Municipal apartment", "Office apartment", "Co-op apartment"])
    children       = st.number_input("Children", 0, 20, 0)
    family_members = st.number_input("Family Members", 1, 20, 2)

    st.markdown("---")
    st.markdown("### 📞 Scores & Contact")
    phone_change_years = st.slider("Phone Change (yrs ago)", 0, 20, 2)
    id_publish_years   = st.slider("ID Document Age (yrs)", 0, 30, 5)
    ext_source_2 = st.slider("External Score 2", 0.0, 1.0, 0.5, 0.01)
    ext_source_3 = st.slider("External Score 3", 0.0, 1.0, 0.5, 0.01)

# ─────────────────────────────────────────────────────────────────────
#  Feature builder
# ─────────────────────────────────────────────────────────────────────

def build_features() -> dict:
    f = {
        "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit_amount,
        "AMT_ANNUITY": annuity, "AMT_GOODS_PRICE": goods_price,
        "AGE_YEARS": age, "EMPLOYMENT_YEARS": employment_years,
        "REGISTRATION_YEARS": registration_years,
        "PHONE_CHANGE_YEARS": phone_change_years,
        "ID_PUBLISH_YEARS": id_publish_years,
        "EXT_SOURCE_2": ext_source_2, "EXT_SOURCE_3": ext_source_3,
        "CNT_CHILDREN": children, "CNT_FAM_MEMBERS": family_members,
        "FEAT_credit_income_ratio":  credit_amount / (income + 1),
        "FEAT_annuity_income_ratio": annuity / (income + 1),
        "FEAT_loan_to_goods":        credit_amount / (goods_price + 1),
        "FEAT_employment_age_ratio": employment_years / (age + 1),
        "FLAG_OWN_CAR": 1 if own_car == "Yes" else 0,
        "FLAG_OWN_REALTY": 1 if own_realty == "Yes" else 0,
    }
    if   age <= 25: f["FEAT_age_bucket"] = 0
    elif age <= 35: f["FEAT_age_bucket"] = 1
    elif age <= 50: f["FEAT_age_bucket"] = 2
    elif age <= 65: f["FEAT_age_bucket"] = 3
    else:           f["FEAT_age_bucket"] = 4
    return f

# ─────────────────────────────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────────────────────────────

tab_assess, tab_insights, tab_about = st.tabs([
    "🔍  Risk Assessment", "📊  Model Insights", "ℹ️  About",
])

# ═══════════════════════════════════════════════════════════════════
#  TAB 1 — Risk Assessment
# ═══════════════════════════════════════════════════════════════════

with tab_assess:
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        clicked = st.button("🔍  Assess Credit Risk", use_container_width=True)

    if clicked:
        result = predict(build_features(), bundle)
        prob  = result["default_probability"]
        score = result["credit_score"]
        cat   = result["risk_category"]

        st.markdown("---")

        # ── Metric row (4 equal columns) ──────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            icon = "✅" if result["decision"] == "Approve" else "❌"
            st.metric("Decision", f"{icon}  {result['decision']}")
        with c2:
            st.metric("Default Probability", f"{prob*100:.2f}%")
        with c3:
            st.metric("Credit Score", f"{score} / 900")
        with c4:
            st.metric("Risk Band", result["risk_band"])

        st.markdown("")

        # ── Gauge + probability (2 equal columns) ─────────
        left, right = st.columns(2)

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(score), use_container_width=True,
                           config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h4>Default Probability</h4>', unsafe_allow_html=True)
            st.plotly_chart(make_prob_bar(prob), use_container_width=True,
                           config={"displayModeBar": False})

            # Risk badge
            badge_cls  = f"badge-{cat.lower()}"
            badge_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(cat, "⚪")
            st.markdown(
                f'<div style="text-align:center;margin:18px 0 6px;">'
                f'<span class="badge {badge_cls}">{badge_icon} {cat} Risk</span></div>',
                unsafe_allow_html=True,
            )

            # Interpretation
            interp = {
                "Excellent":  "Very low risk — strong approval candidate.",
                "Good":       "Low risk — likely to be approved.",
                "Fair":       "Moderate risk — may need additional review.",
                "Poor":       "Elevated risk — consider with caution.",
                "Very Poor":  "High risk of default.",
            }.get(result["risk_band"], "")
            st.markdown(
                f'<p style="text-align:center;color:#5f7285;font-size:0.84rem;margin:0;">{interp}</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Detailed report ────────────────────────────────
        with st.expander("📄  Detailed Report"):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"""
| Metric | Value |
|:-------|------:|
| Credit Score | **{score}** / 900 |
| Risk Band | {result['risk_band']} |
| Risk Category | {badge_icon} {cat} |
| Default Prob. | {prob*100:.2f}% |
| Decision | {result['decision']} |
""")
            with d2:
                st.markdown("""
| Range | Band |
|:------|:-----|
| 750 – 900 | Excellent |
| 700 – 749 | Good |
| 650 – 699 | Fair |
| 600 – 649 | Poor |
| 300 – 599 | Very Poor |
""")

    else:
        st.info("👈 Enter applicant details in the sidebar, then click **Assess Credit Risk**.")

# ═══════════════════════════════════════════════════════════════════
#  TAB 2 — Model Insights
# ═══════════════════════════════════════════════════════════════════

with tab_insights:
    metrics = bundle.get("metrics")
    fi      = bundle.get("feature_importances")

    if not metrics and not fi:
        st.warning("Re-train the model with the updated `train_model.py` to enable this section.")
    else:
        # ── Top metric cards ───────────────────────────────
        if metrics:
            mc = st.columns(5)
            labels = ["AUC-ROC", "F1-Score", "Precision", "Recall", "Accuracy"]
            keys   = ["auc", "f1", "precision", "recall", "accuracy"]
            for col, lbl, key in zip(mc, labels, keys):
                with col:
                    st.metric(lbl, f"{metrics.get(key, 0):.4f}")

        st.markdown("")

        # ── Two-column chart grid (aligned top) ────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="card"><h4>Performance Radar</h4>', unsafe_allow_html=True)
            if metrics:
                display_m = {
                    "AUC": metrics["auc"], "F1": metrics["f1"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"], "Accuracy": metrics["accuracy"],
                }
                st.plotly_chart(make_radar(display_m), use_container_width=True,
                               config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="card"><h4>Feature Importance (Top 20)</h4>', unsafe_allow_html=True)
            if fi:
                st.plotly_chart(make_fi_chart(fi), use_container_width=True,
                               config={"displayModeBar": False})
            else:
                st.info("Not available — re-train to populate.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Model config row ──────────────────────────────
        st.markdown("")
        cfg = st.columns(4)
        with cfg[0]: st.metric("Algorithm", "XGBoost")
        with cfg[1]: st.metric("Features", str(len(bundle.get("feature_cols", []))))
        with cfg[2]: st.metric("Threshold", f"{bundle.get('threshold', 0.5):.4f}")
        with cfg[3]: st.metric("Score Scale", "300 – 900")

# ═══════════════════════════════════════════════════════════════════
#  TAB 3 — About
# ═══════════════════════════════════════════════════════════════════

with tab_about:
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
<div class="about-card">
<h3>🏦 How Credit Scoring Works</h3>
<p>Traditional scores rely on established credit history — excluding
<b>billions of unbanked individuals</b>. CreditAI uses <b>alternative data</b>:</p>
<ul>
<li><b>Income & Employment</b> — stability and earning capacity</li>
<li><b>Loan Burden Ratios</b> — credit-to-income, annuity-to-income</li>
<li><b>Demographic Signals</b> — age, family, housing type</li>
<li><b>External Scores</b> — normalised partner-data signals</li>
<li><b>Document Activity</b> — ID age, phone-change frequency</li>
</ul>
<p>Output: a <b>300 – 900 credit score</b> (CIBIL-aligned).
Applicants scoring <b>≥ 650</b> are recommended for approval.</p>
</div>
""", unsafe_allow_html=True)

    with a2:
        st.markdown("""
<div class="about-card">
<h3>🤖 What the AI Evaluates</h3>
<p>An <b>XGBoost</b> ensemble trained on <b>307,511 applicants</b>
(Home Credit Default Risk, Kaggle).</p>
<p><b>Pipeline:</b></p>
<ul>
<li><b>Preprocessing</b> — drop cols with &gt;55% missing, encode categoricals, median-impute</li>
<li><b>Feature Engineering</b> — credit/income ratio, annuity ratio, loan-to-goods, age bands, employment stability</li>
<li><b>Training</b> — 1 000 trees, depth 6, lr 0.02, <code>scale_pos_weight=11</code></li>
<li><b>Threshold Tuning</b> — F1-optimal boundary via precision-recall curve</li>
</ul>
<p><b>Compliance:</b> Protected attributes excluded. Designed for EU AI Act Art. 13 and RBI Fair Practices.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
<div class="about-card" style="text-align:center;">
<h3 style="margin-bottom:16px;">Risk Categories</h3>
<span class="badge badge-low" style="margin:4px 8px;">🟢 Low — &lt;20% default probability</span>
<span class="badge badge-medium" style="margin:4px 8px;">🟡 Medium — 20%–50%</span>
<span class="badge badge-high" style="margin:4px 8px;">🔴 High — ≥50%</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div class="footer">'
    'CreditAI · AI-Powered Alternate Credit Scoring · XGBoost + Streamlit · CIBIL 300 – 900'
    '</div>', unsafe_allow_html=True,
)
