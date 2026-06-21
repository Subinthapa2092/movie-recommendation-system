import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def hybrid_recommend(user_id, title,
                     tfidf_matrix, content, indices,
                     user_factors, item_factors,
                     user_pos, movie_ids_lookup,
                     global_mean, links,
                     n=10, w_content=0.6, w_collab=0.4):

    if title not in indices:
        print(f"'{title}' not found.")
        return None

    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    # Content similarity
    sim_row  = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    max_sim  = sim_row.max() + 1e-9
    sim_norm = sim_row / max_sim

    # Collaborative predictions
    if user_id in user_pos:
        u_idx         = user_pos[user_id]
        user_pred_arr = (user_factors[u_idx] @ item_factors) + global_mean
        user_pred     = pd.Series(user_pred_arr, index=movie_ids_lookup)
        user_pred_norm = user_pred / (user_pred.max() + 1e-9)
    else:
        user_pred_norm = None

    tmdb_to_movieid = links.dropna(subset=['tmdbId']).copy()
    tmdb_to_movieid['tmdbId'] = tmdb_to_movieid['tmdbId'].astype(int)
    tmdb_to_movieid = tmdb_to_movieid.drop_duplicates(subset=['tmdbId']) \
                                     .set_index('tmdbId')['movieId']

    content_ids      = content['id'].values
    mapped_movie_ids = pd.Series(content_ids).map(tmdb_to_movieid)

    if user_pred_norm is not None:
        col_scores = mapped_movie_ids.map(user_pred_norm).fillna(0).values
    else:
        col_scores = np.zeros(len(content))

    hybrid_scores       = w_content * sim_norm + w_collab * col_scores
    hybrid_scores[idx]  = -np.inf

    top_idx = np.argsort(hybrid_scores)[::-1][:n]
    result  = content[['title', 'genre_str']].iloc[top_idx].copy()
    result['hybrid_score'] = [round(s, 4) for s in hybrid_scores[top_idx]]
    return result.reset_index(drop=True)
