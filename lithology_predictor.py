"""
Subsurface Lithology Predictor
AI-Powered Well Log Interpretation for Oil and Gas
Author: Bose Abubakre, PhD MBA
Portfolio: Oil and Gas AI | github.com/boseabubakre

Data: FORCE 2020 Lithofacies Competition Dataset (public domain)
      118 North Sea wells - Norwegian Continental Shelf
      Zenodo DOI: 10.5281/zenodo.4351155

Features: GR, RHOB, NPHI, DTC, RSHA, RMED, RDEP, CALI, SP, ROPA
Target: 12 lithology classes (Sandstone, Shale, Limestone, Dolomite, etc.)

Run: streamlit run lithology_predictor.py
Install: pip install streamlit pandas numpy plotly scikit-learn xgboost
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lithology Predictor",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #ffffff;
    color: #111827;
  }
  .main, .stApp { background-color: #ffffff; }

  [data-testid="metric-container"] {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 12px 16px;
  }
  [data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #ffffff;
    letter-spacing: 0.06em;
  }
  [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    color: #f0a500;
    font-weight: 500;
  }
  [data-testid="stSidebar"] {
    background-color: #f0f4ff;
    border-right: 1px solid #3a7bd5;
  }
  h1, h2 { color: #111827; }
  h2 { font-size: 15px !important; border-bottom: 1px solid #bfdbfe; padding-bottom: 6px; }
  hr { border: none; border-top: 1px solid #bfdbfe; margin: 8px 0; }
  .info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 3px solid #93c5fd;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 13px;
    color: #1e3a8a;
    margin: 8px 0;
  }
  .lith-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    margin: 2px;
  }
  #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── LITHOLOGY DEFINITIONS ─────────────────────────────────────────────────────
LITH_MAP = {
    30000: "Sandstone",
    65030: "Sandstone/Shale",
    65000: "Shale",
    80000: "Marl",
    74000: "Dolomite",
    70000: "Limestone",
    70032: "Chalk",
    88000: "Halite",
    86000: "Anhydrite",
    99000: "Tuff",
    90000: "Coal",
    93000: "Basement",
}

LITH_COLORS = {
    "Sandstone":       "#f0a500",
    "Sandstone/Shale": "#c8a028",
    "Shale":           "#6b8c6b",
    "Marl":            "#8fbc8f",
    "Dolomite":        "#58a6ff",
    "Limestone":       "#a0c8ff",
    "Chalk":           "#e8f0ff",
    "Halite":          "#bc8cff",
    "Anhydrite":       "#ff7b7b",
    "Tuff":            "#ff9e44",
    "Coal":            "#444444",
    "Basement":        "#8b4513",
}

# ── DATA GENERATION ───────────────────────────────────────────────────────────
@st.cache_data
def generate_synthetic_well_data(n_samples=5000, n_wells=8, seed=42):
    """
    Generates geologically realistic synthetic well log data
    modeled after the FORCE 2020 North Sea dataset.

    In production: download real data from
    https://zenodo.org/records/4351155
    and load with: pd.read_csv('train.csv', sep=';')
    """
    np.random.seed(seed)
    records = []

    well_names = [f"WELL_{chr(65+i)}" for i in range(n_wells)]

    # Geological layer definitions: (lithology_code, thickness_range, log_params)
    layer_templates = [
        # (lith_code, gr_mean, gr_std, rhob_mean, rhob_std, nphi_mean, nphi_std, dtc_mean)
        (30000, 25,  12, 2.35, 0.08, 0.18, 0.05, 70),   # Sandstone - low GR, low NPHI
        (65000, 90,  18, 2.55, 0.06, 0.30, 0.06, 100),  # Shale - high GR, high NPHI
        (70000, 15,  8,  2.71, 0.04, 0.12, 0.04, 65),   # Limestone - low GR, low NPHI, high RHOB
        (74000, 20,  10, 2.85, 0.05, 0.10, 0.03, 55),   # Dolomite - low GR, very high RHOB
        (65030, 60,  20, 2.45, 0.08, 0.24, 0.06, 85),   # Sand/Shale - intermediate
        (80000, 40,  15, 2.60, 0.06, 0.25, 0.05, 90),   # Marl
        (70032, 10,  5,  2.68, 0.04, 0.30, 0.06, 75),   # Chalk
        (88000, 5,   3,  2.03, 0.04, 0.00, 0.01, 67),   # Halite - very low RHOB
    ]

    for well in well_names:
        depth = 1500.0
        max_depth = np.random.uniform(3000, 4500)

        # Pick 6-12 layers for this well
        n_layers = np.random.randint(6, 13)
        layers = np.random.choice(len(layer_templates), n_layers,
                                  p=[0.25, 0.30, 0.15, 0.08, 0.10, 0.05, 0.04, 0.03])

        for layer_idx in layers:
            lith_code, gr_m, gr_s, rhob_m, rhob_s, nphi_m, nphi_s, dtc_m = layer_templates[layer_idx]
            thickness = np.random.uniform(30, 300)
            n_pts = max(3, int(thickness / 0.5))

            depths = np.linspace(depth, depth + thickness, n_pts)

            # Realistic log responses with depth trend and noise
            depth_factor = (depth - 1500) / 3000
            gr    = np.clip(gr_m + gr_s * np.random.randn(n_pts) + depth_factor * 5, 0, 150)
            rhob  = np.clip(rhob_m + rhob_s * np.random.randn(n_pts) + depth_factor * 0.05, 1.8, 3.0)
            nphi  = np.clip(nphi_m + nphi_s * np.random.randn(n_pts) - depth_factor * 0.02, -0.05, 0.6)
            dtc   = np.clip(dtc_m + 5 * np.random.randn(n_pts) - depth_factor * 8, 40, 140)
            # Resistivity: high in hydrocarbons and carbonates, low in shales
            rmed  = np.exp(np.random.normal(
                2.5 if lith_code in [30000, 70000, 74000, 70032] else 0.8, 0.5, n_pts))
            rdep  = rmed * np.random.uniform(0.8, 1.3, n_pts)
            rsha  = rmed * np.random.uniform(0.5, 0.9, n_pts)
            cali  = 8.5 + np.random.uniform(-0.5, 2.0, n_pts) * (1.5 if lith_code == 65000 else 0.3)
            sp    = -gr * 0.4 + np.random.normal(0, 5, n_pts)

            for i in range(n_pts):
                records.append({
                    "WELL":     well,
                    "DEPTH_MD": round(depths[i], 1),
                    "GR":       round(float(gr[i]), 2),
                    "RHOB":     round(float(rhob[i]), 3),
                    "NPHI":     round(float(nphi[i]), 3),
                    "DTC":      round(float(dtc[i]), 2),
                    "RSHA":     round(float(rsha[i]), 3),
                    "RMED":     round(float(rmed[i]), 3),
                    "RDEP":     round(float(rdep[i]), 3),
                    "CALI":     round(float(cali[i]), 2),
                    "SP":       round(float(sp[i]), 2),
                    "LITHOLOGY_CODE": lith_code,
                    "LITHOLOGY": LITH_MAP[lith_code],
                })

            depth += thickness
            if depth > max_depth:
                break

    df = pd.DataFrame(records).head(n_samples)
    return df

@st.cache_resource
def train_model(df):
    """Train gradient boosting classifier on well log features."""
    features = ["GR", "RHOB", "NPHI", "DTC", "RSHA", "RMED", "RDEP", "CALI", "SP"]
    X = df[features].fillna(df[features].median())
    y = df["LITHOLOGY"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, subsample=0.8
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test.values).mean()

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    return model, X_test, y_test, y_pred, accuracy, importance

def chart_layout(**overrides):
    base = dict(
        paper_bgcolor="#f0f4ff",
        plot_bgcolor="#ffffff",
        font=dict(family="IBM Plex Mono", color="#1e3a8a", size=10),
        xaxis=dict(gridcolor="#dbeafe", linecolor="#dbeafe"),
        yaxis=dict(gridcolor="#dbeafe", linecolor="#dbeafe"),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(bgcolor="#f0f4ff", bordercolor="#dbeafe", borderwidth=1)
    )
    base.update(overrides)
    return base

# ── LOAD DATA AND TRAIN ───────────────────────────────────────────────────────
with st.spinner("Loading geological data and training model..."):
    df = generate_synthetic_well_data()
    model, X_test, y_test, y_pred, accuracy, importance = train_model(df)

# ── HEADER ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([4, 2])
with c1:
    st.markdown("## 🪨 Subsurface Lithology Predictor")
    st.markdown(
        '<div class="info-box">AI-powered lithology classification from wireline log data. '
        'Trained on synthetic North Sea well data modeled after the '
        '<b>FORCE 2020 Lithofacies Competition</b> dataset (118 wells, Norwegian Continental Shelf). '
        'Connect real LAS files from <b>zenodo.org/records/4351155</b> for production use.</div>',
        unsafe_allow_html=True)
with c2:
    st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
    st.metric("Model Accuracy", f"{accuracy:.1%}")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Controls")
    wells = df["WELL"].unique().tolist()
    selected_well = st.selectbox("Select Well", wells)
    depth_range = st.slider(
        "Depth Range (m)",
        float(df["DEPTH_MD"].min()),
        float(df["DEPTH_MD"].max()),
        (float(df["DEPTH_MD"].min()), float(df["DEPTH_MD"].min()) + 500)
    )
    show_prediction = st.toggle("Show AI Predictions", value=True)
    st.markdown("---")
    st.markdown("### Predict from Log Values")
    st.markdown('<div style="font-family:IBM Plex Mono;font-size:11px;color:#1e3a8a">Enter log readings:</div>',
                unsafe_allow_html=True)
    gr_in   = st.slider("GR (API)",    0.0,  150.0, 45.0)
    rhob_in = st.slider("RHOB (g/cc)", 1.8,  3.0,   2.45)
    nphi_in = st.slider("NPHI (v/v)",  -0.05, 0.6,  0.22)
    dtc_in  = st.slider("DTC (us/ft)", 40.0, 140.0, 80.0)
    rmed_in = st.slider("RMED (ohm)",  0.1,  100.0, 5.0)
    rdep_in = st.slider("RDEP (ohm)",  0.1,  100.0, 6.0)
    rsha_in = st.slider("RSHA (ohm)",  0.1,  50.0,  3.0)
    cali_in = st.slider("CALI (in)",   6.0,  16.0,  8.5)
    sp_in   = st.slider("SP (mV)",    -80.0, 20.0, -20.0)

    if st.button("🔮 Predict Lithology", type="primary"):
        input_vals = [[gr_in, rhob_in, nphi_in, dtc_in, rsha_in, rmed_in, rdep_in, cali_in, sp_in]]
        pred = model.predict(input_vals)[0]
        proba = model.predict_proba(input_vals)[0]
        classes = model.classes_
        conf = dict(zip(classes, proba))
        top3 = sorted(conf.items(), key=lambda x: x[1], reverse=True)[:3]
        color = LITH_COLORS.get(pred, "#ffffff")
        st.markdown(
            f'<div style="background:{color}22;border:2px solid {color};border-radius:8px;'
            f'padding:12px;text-align:center;margin-top:8px">'
            f'<div style="font-family:IBM Plex Mono;font-size:11px;color:#1e3a8a">PREDICTED</div>'
            f'<div style="font-family:IBM Plex Mono;font-size:18px;font-weight:500;color:{color}">'
            f'{pred}</div></div>',
            unsafe_allow_html=True)
        st.markdown("**Top 3 probabilities:**")
        for lith, prob in top3:
            c = LITH_COLORS.get(lith, "#aaa")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;margin:3px 0">'
                f'<span style="font-family:IBM Plex Mono;font-size:12px;color:{c}">{lith}</span>'
                f'<span style="font-family:IBM Plex Mono;font-size:12px;color:#111827">{prob:.1%}</span>'
                f'</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-family:IBM Plex Mono;font-size:10px;color:#2563eb'>
    Built by Bose Abubakre<br>
    PhD Geoscientist · MBA<br>
    github.com/boseabubakre
    </div>""", unsafe_allow_html=True)

# ── WELL LOG DISPLAY ──────────────────────────────────────────────────────────
well_df = df[df["WELL"] == selected_well].copy()
well_df = well_df[
    (well_df["DEPTH_MD"] >= depth_range[0]) &
    (well_df["DEPTH_MD"] <= depth_range[1])
].copy()

if show_prediction:
    features = ["GR", "RHOB", "NPHI", "DTC", "RSHA", "RMED", "RDEP", "CALI", "SP"]
    X_well = well_df[features].fillna(df[features].median())
    well_df["PREDICTED"] = model.predict(X_well)

st.markdown(f"#### Well Log Display — {selected_well}  ({depth_range[0]:.0f}–{depth_range[1]:.0f} m)")

# Five-track log plot
fig_logs = make_subplots(
    rows=1, cols=5,
    subplot_titles=["GR (API)", "RHOB / NPHI", "Resistivity", "DTC", "Lithology"],
    horizontal_spacing=0.03
)

depth = well_df["DEPTH_MD"].values

# Track 1: GR
fig_logs.add_trace(go.Scatter(
    x=well_df["GR"], y=depth, mode="lines", name="GR",
    line=dict(color="#f0a500", width=1.5),
    fill="tozerox", fillcolor="rgba(240,165,0,0.15)"
), row=1, col=1)

# Track 2: RHOB and NPHI
fig_logs.add_trace(go.Scatter(
    x=well_df["RHOB"], y=depth, mode="lines", name="RHOB",
    line=dict(color="#ff7b7b", width=1.5)
), row=1, col=2)
fig_logs.add_trace(go.Scatter(
    x=well_df["NPHI"], y=depth, mode="lines", name="NPHI",
    line=dict(color="#58a6ff", width=1.5, dash="dot")
), row=1, col=2)

# Track 3: Resistivity (log scale)
fig_logs.add_trace(go.Scatter(
    x=np.log10(well_df["RDEP"].clip(0.01)), y=depth, mode="lines", name="RDEP",
    line=dict(color="#3fb950", width=1.5)
), row=1, col=3)
fig_logs.add_trace(go.Scatter(
    x=np.log10(well_df["RMED"].clip(0.01)), y=depth, mode="lines", name="RMED",
    line=dict(color="#39d353", width=1, dash="dot")
), row=1, col=3)

# Track 4: DTC
fig_logs.add_trace(go.Scatter(
    x=well_df["DTC"], y=depth, mode="lines", name="DTC",
    line=dict(color="#bc8cff", width=1.5)
), row=1, col=4)

# Track 5: Lithology column
lith_col = "PREDICTED" if (show_prediction and "PREDICTED" in well_df.columns) else "LITHOLOGY"
liths = well_df[lith_col].values
for i in range(len(depth) - 1):
    lith = liths[i]
    color = LITH_COLORS.get(lith, "#888888")
    fig_logs.add_shape(
        type="rect",
        x0=0, x1=1,
        y0=depth[i], y1=depth[i+1],
        fillcolor=color,
        line=dict(width=0),
        row=1, col=5
    )

# Invisible scatter to get legend
unique_liths = well_df[lith_col].unique()
for lith in unique_liths:
    fig_logs.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        name=lith,
        marker=dict(color=LITH_COLORS.get(lith, "#888"), size=10, symbol="square"),
        showlegend=True
    ), row=1, col=5)

fig_logs.update_layout(
    **chart_layout(
        height=650,
        margin=dict(l=40, r=20, t=50, b=30),
        showlegend=True,
        hovermode="y unified",
        legend=dict(x=1.02, y=1, bgcolor="#f0f4ff", bordercolor="#dbeafe", borderwidth=1)
    )
)

# All y-axes inverted (depth increases downward)
for i in range(1, 6):
    fig_logs.update_yaxes(autorange="reversed", row=1, col=i)
    if i > 1:
        fig_logs.update_yaxes(showticklabels=False, row=1, col=i)

fig_logs.update_xaxes(title_text="", row=1, col=5, range=[0, 1],
                       showticklabels=False, showgrid=False)

st.plotly_chart(fig_logs, use_container_width=True)

# ── ROW 2: ACCURACY + FEATURE IMPORTANCE + LITHOLOGY DISTRIBUTION ────────────
st.markdown("---")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### Model Accuracy by Class")
    report = classification_report(y_test, y_pred, output_dict=True)
    classes_r = [k for k in report.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
    f1_scores = [report[k]["f1-score"] for k in classes_r]
    colors_r  = [LITH_COLORS.get(k, "#888") for k in classes_r]

    fig_acc = go.Figure(go.Bar(
        x=f1_scores, y=classes_r, orientation="h",
        marker_color=colors_r,
        text=[f"{s:.0%}" for s in f1_scores],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9, color="#1e3a8a")
    ))
    fig_acc.update_layout(**chart_layout(
        height=320,
        xaxis=dict(range=[0, 1.15], title="F1 Score", gridcolor="#dbeafe"),
        margin=dict(l=100, r=50, t=20, b=30)
    ))
    st.plotly_chart(fig_acc, use_container_width=True)

with c2:
    st.markdown("#### Feature Importance")
    fig_imp = go.Figure(go.Bar(
        x=importance["Importance"],
        y=importance["Feature"],
        orientation="h",
        marker_color="#58a6ff",
        marker_line_color="#3a7bd5",
        marker_line_width=1,
        text=[f"{v:.3f}" for v in importance["Importance"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9, color="#1e3a8a")
    ))
    fig_imp.update_layout(**chart_layout(
        height=320,
        xaxis=dict(title="Importance", gridcolor="#dbeafe"),
        margin=dict(l=60, r=60, t=20, b=30)
    ))
    st.plotly_chart(fig_imp, use_container_width=True)

with c3:
    st.markdown("#### Lithology Distribution")
    lith_counts = df["LITHOLOGY"].value_counts()
    colors_pie = [LITH_COLORS.get(l, "#888") for l in lith_counts.index]
    fig_pie = go.Figure(go.Pie(
        labels=lith_counts.index,
        values=lith_counts.values,
        marker_colors=colors_pie,
        hole=0.5,
        textinfo="label+percent",
        textfont=dict(family="IBM Plex Mono", size=9),
    ))
    fig_pie.update_layout(**chart_layout(
        height=320,
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10)
    ))
    st.plotly_chart(fig_pie, use_container_width=True)

# ── ROW 3: CROSS PLOTS ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Petrophysical Cross Plots — Lithology Discrimination")

cp1, cp2 = st.columns(2)

sample = df.sample(min(2000, len(df)), random_state=42)

with cp1:
    st.markdown("##### GR vs RHOB  (Shale indicator)")
    fig_cp1 = go.Figure()
    for lith in sample["LITHOLOGY"].unique():
        mask = sample["LITHOLOGY"] == lith
        fig_cp1.add_trace(go.Scatter(
            x=sample.loc[mask, "GR"],
            y=sample.loc[mask, "RHOB"],
            mode="markers",
            name=lith,
            marker=dict(color=LITH_COLORS.get(lith, "#888"), size=4, opacity=0.7)
        ))
    fig_cp1.update_layout(**chart_layout(
        height=320,
        xaxis=dict(title="GR (API)", gridcolor="#dbeafe"),
        yaxis=dict(title="RHOB (g/cc)", gridcolor="#dbeafe"),
        margin=dict(l=60, r=20, t=20, b=50)
    ))
    st.plotly_chart(fig_cp1, use_container_width=True)

with cp2:
    st.markdown("##### NPHI vs RHOB  (Lithology discriminator)")
    fig_cp2 = go.Figure()
    for lith in sample["LITHOLOGY"].unique():
        mask = sample["LITHOLOGY"] == lith
        fig_cp2.add_trace(go.Scatter(
            x=sample.loc[mask, "NPHI"],
            y=sample.loc[mask, "RHOB"],
            mode="markers",
            name=lith,
            marker=dict(color=LITH_COLORS.get(lith, "#888"), size=4, opacity=0.7),
            showlegend=False
        ))
    fig_cp2.update_layout(**chart_layout(
        height=320,
        xaxis=dict(title="NPHI (v/v)", gridcolor="#dbeafe"),
        yaxis=dict(title="RHOB (g/cc)", gridcolor="#dbeafe"),
        margin=dict(l=60, r=20, t=20, b=50)
    ))
    st.plotly_chart(fig_cp2, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='font-family:IBM Plex Mono;font-size:10px;color:#2563eb;text-align:center;padding:8px'>
Subsurface Lithology Predictor &nbsp;·&nbsp;
Built by Bose Abubakre, PhD MBA &nbsp;·&nbsp;
github.com/boseabubakre &nbsp;·&nbsp;
Data modeled after FORCE 2020 (zenodo.org/records/4351155) &nbsp;·&nbsp;
For portfolio and research use only
</div>
""", unsafe_allow_html=True)
