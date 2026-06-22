"""
Loads all pre-trained models from the models/ directory into app.state
so they are shared across all requests without re-loading.
Also loads links.csv from data/ for movieId <-> tmdbId mapping.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.persistence import load_all


def load_models_into_state(app):
    models_dir = os.environ.get("MODELS_DIR", "models")
    data_dir   = os.environ.get("DATA_DIR",   "data")

    m = load_all(models_dir=models_dir)

    # Content-based
    app.state.tfidf_matrix = m["tfidf_matrix"]
    app.state.content      = m["content"]
    app.state.indices      = m["indices"]

    # SVD collaborative
    app.state.user_factors     = m["user_factors"]
    app.state.item_factors     = m["item_factors"]
    app.state.user_pos         = m["user_pos"]
    app.state.movie_pos        = m["movie_pos"]
    app.state.movie_ids_lookup = m["movie_ids_lookup"]
    app.state.user_item_sparse = m["user_item_sparse"]
    app.state.global_mean      = m["global_mean"]

    # Classifiers
    app.state.lr           = m["lr"]
    app.state.dt           = m["dt"]
    app.state.nb           = m["nb"]
    app.state.feature_cols = m["feature_cols"]

    # Links table — real movieId <-> tmdbId mapping
    links_path = os.path.join(data_dir, "links.csv")
    links = pd.read_csv(links_path)
    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"]  = links["tmdbId"].astype(int)
    links["movieId"] = links["movieId"].astype(int)
    app.state.links = links

    # Movie list for autocomplete
    app.state.movie_titles = sorted(m["content"]["title"].dropna().unique().tolist())

    print(f"Models loaded — {len(app.state.content):,} movies, "
          f"{len(app.state.user_pos):,} users, "
          f"{len(links):,} links entries.")