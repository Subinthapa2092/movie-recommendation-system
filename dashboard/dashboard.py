"""
CineMatch — Analytics Dashboard (Streamlit)
Run: streamlit run dashboard/dashboard.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
from src.persistence import load_all

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── dark-ish theme via CSS injection ──────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d0f14; }
  [data-testid="stHeader"]            { background: transparent; }
  .block-container                    { padding-top: 2rem; }
  h1, h2, h3                          { color: #e4e6f0; }
  .stMetric label                     { color: #6b7099 !important; font-size: 0.8rem; }
  .stMetric [data-testid="stMetricValue"] { color: #e8b14f; font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
PALETTE    = ["#e8b14f", "#7c6af5", "#5ce89a", "#e85c5c", "#4fc3e8", "#e86af5"]
BG         = "#0d0f14"
SURFACE    = "#14171f"


@st.cache_resource(show_spinner="Loading models…")
def load():
    return load_all(models_dir=MODELS_DIR)


m = load()

content = m["content"]
user_pos = m["user_pos"]
user_factors = m["user_factors"]
item_factors = m["item_factors"]
global_mean  = m["global_mean"]

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("## 🎬 CineMatch  Analytics Dashboard")
st.caption("Live metrics from your trained recommendation models.")
st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Movies Indexed",   f"{len(content):,}")
k2.metric("Known Users",      f"{len(user_pos):,}")
k3.metric("SVD Latent Factors", user_factors.shape[1])
k4.metric("TF-IDF Vocab",     f"{m['tfidf_matrix'].shape[1]:,}")
k5.metric("Global Mean Rating", f"{global_mean:.3f}")

st.divider()

# ── helpers ───────────────────────────────────────────────────────────────────
def styled_fig():
    fig, ax = plt.subplots(facecolor=BG)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color("#252940")
    ax.tick_params(colors="#6b7099")
    ax.xaxis.label.set_color("#6b7099")
    ax.yaxis.label.set_color("#6b7099")
    ax.title.set_color("#e4e6f0")
    return fig, ax

def styled_fig2():
    fig, axes = plt.subplots(1, 2, facecolor=BG, figsize=(12, 4))
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values(): spine.set_color("#252940")
        ax.tick_params(colors="#6b7099")
        ax.xaxis.label.set_color("#6b7099")
        ax.yaxis.label.set_color("#6b7099")
        ax.title.set_color("#e4e6f0")
    return fig, axes

# ── row 1: genre distribution + top movies ────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Genre Distribution")
    genres_flat = content["genre_list"].dropna().explode() if "genre_list" in content.columns else \
                  content["genre_str"].dropna().str.split().explode()
    genre_counts = genres_flat.value_counts().head(15)

    fig, ax = styled_fig()
    fig.set_size_inches(7, 4)
    bars = ax.barh(genre_counts.index[::-1], genre_counts.values[::-1],
                   color=PALETTE[0], edgecolor="none")
    ax.set_xlabel("Number of movies")
    ax.set_title("Top 15 Genres by Movie Count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with c2:
    st.subheader("Top Movies by Weighted Score")
    if "vote_average" in content.columns and "vote_count" in content.columns:
        C = content["vote_average"].mean()
        m_thresh = content["vote_count"].quantile(0.70)
        qual = content[content["vote_count"] >= m_thresh].copy()
        v, R = qual["vote_count"], qual["vote_average"]
        qual["ws"] = (v / (v + m_thresh)) * R + (m_thresh / (v + m_thresh)) * C
        top = qual.nlargest(12, "ws")[["title", "ws"]].dropna()

        fig, ax = styled_fig()
        fig.set_size_inches(7, 4)
        ax.barh(top["title"].values[::-1], top["ws"].values[::-1],
                color=PALETTE[1], edgecolor="none")
        ax.set_xlabel("Weighted Score")
        ax.set_title("Top 12 Movies by IMDB-style Score")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("vote_average / vote_count not in content table.")

# ── row 2: SVD variance + user activity ───────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.subheader("SVD Explained Variance")
    # Approximate per-component variance from singular values stored in item_factors norms
    component_norms = np.linalg.norm(item_factors, axis=1)
    total = component_norms.sum()
    var_ratio = component_norms / total
    cumvar    = np.cumsum(var_ratio)

    fig, ax = styled_fig()
    fig.set_size_inches(7, 4)
    ax.bar(range(1, len(var_ratio)+1), var_ratio * 100, color=PALETTE[2], alpha=0.7, label="Per component")
    ax2 = ax.twinx()
    ax2.plot(range(1, len(cumvar)+1), cumvar * 100, color=PALETTE[0], linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative %", color="#6b7099")
    ax2.tick_params(colors="#6b7099")
    ax2.set_facecolor(SURFACE)
    ax.set_xlabel("Component")
    ax.set_ylabel("% variance")
    ax.set_title("SVD Component Variance (item factors)")
    ax.set_xlim(0.5, len(var_ratio) + 0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with c4:
    st.subheader("User Factor Norms Distribution")
    norms = np.linalg.norm(user_factors, axis=1)

    fig, ax = styled_fig()
    fig.set_size_inches(7, 4)
    ax.hist(norms, bins=40, color=PALETTE[3], edgecolor="none", alpha=0.85)
    ax.axvline(norms.mean(), color=PALETTE[0], linestyle="--", linewidth=1.5,
               label=f"Mean {norms.mean():.2f}")
    ax.set_xlabel("L2 norm of user factor vector")
    ax.set_ylabel("Users")
    ax.set_title("Distribution of User Embedding Norms")
    ax.legend(facecolor=SURFACE, labelcolor="#6b7099", edgecolor="#252940")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── row 3: popularity distribution ────────────────────────────────────────────
st.subheader("Movie Popularity & Vote Average Distribution")
d1, d2 = st.columns(2)

with d1:
    if "popularity" in content.columns:
        pop = content["popularity"].dropna()
        pop = pop[pop < pop.quantile(0.98)]   # clip outliers
        fig, ax = styled_fig()
        fig.set_size_inches(6, 3.5)
        ax.hist(pop, bins=60, color=PALETTE[4], edgecolor="none", alpha=0.85)
        ax.set_xlabel("Popularity score")
        ax.set_ylabel("Movies")
        ax.set_title("Popularity Distribution (98th pct clip)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with d2:
    if "vote_average" in content.columns:
        va = content["vote_average"].dropna()
        va = va[va > 0]
        fig, ax = styled_fig()
        fig.set_size_inches(6, 3.5)
        ax.hist(va, bins=40, color=PALETTE[5], edgecolor="none", alpha=0.85)
        ax.axvline(va.mean(), color=PALETTE[0], linestyle="--", linewidth=1.5,
                   label=f"Mean {va.mean():.2f}")
        ax.set_xlabel("Vote Average")
        ax.set_ylabel("Movies")
        ax.set_title("Vote Average Distribution")
        ax.legend(facecolor=SURFACE, labelcolor="#6b7099", edgecolor="#252940")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── interactive: try a content-based query ────────────────────────────────────
st.divider()
st.subheader("Try it: Content-Based Lookup")
titles = sorted(content["title"].dropna().unique().tolist())
selected = st.selectbox("Pick a movie", titles, index=titles.index("The Dark Knight") if "The Dark Knight" in titles else 0)
n_recs = st.slider("How many recommendations?", 5, 20, 10)

if st.button("Find similar movies", type="primary"):
    from src.content_based import content_recommend
    df = content_recommend(selected, m["tfidf_matrix"], m["content"], m["indices"], n=n_recs)
    if df is not None:
        st.dataframe(
            df.rename(columns={"genre_str": "Genres", "similarity_score": "Similarity"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(f"'{selected}' not found in the index.")

st.caption("CineMatch | Built by Subin Thapa | Tribhuvan University BDS Project")