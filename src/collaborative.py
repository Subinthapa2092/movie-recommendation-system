import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_svd_model(ratings, links, movies, n_components=50):
    global_mean = ratings['rating'].mean()

    user_idx,  user_ids_lookup  = pd.factorize(ratings['userId'].values,  sort=True)
    movie_idx, movie_ids_lookup = pd.factorize(ratings['movieId'].values, sort=True)

    n_users  = len(user_ids_lookup)
    n_movies = len(movie_ids_lookup)

    centered_values  = ratings['rating'].values - global_mean
    user_item_sparse = csr_matrix(
        (centered_values, (user_idx, movie_idx)),
        shape=(n_users, n_movies)
    )

    user_pos  = {uid: i for i, uid in enumerate(user_ids_lookup)}
    movie_pos = {mid: i for i, mid in enumerate(movie_ids_lookup)}

    sparsity = 1 - (ratings.shape[0] / (n_users * n_movies))
    print(f'User-Item matrix shape : {user_item_sparse.shape}')
    print(f'Sparsity               : {sparsity:.2%}')

    svd          = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item_sparse)
    item_factors = svd.components_

    print(f'User factors shape     : {user_factors.shape}')
    print(f'Item factors shape     : {item_factors.shape}')
    print(f'Variance explained     : {svd.explained_variance_ratio_.sum():.2%}')

    # Evaluate
    valid_mask    = ratings['userId'].isin(user_pos) & ratings['movieId'].isin(movie_pos)
    valid_ratings = ratings[valid_mask]
    u_positions   = valid_ratings['userId'].map(user_pos).values
    m_positions   = valid_ratings['movieId'].map(movie_pos).values
    actual        = valid_ratings['rating'].values

    n          = len(actual)
    chunk_size = 500_000
    predicted_centered = np.empty(n, dtype=np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        u_chunk = u_positions[start:end]
        m_chunk = m_positions[start:end]
        predicted_centered[start:end] = np.einsum(
            'ij,ij->i',
            user_factors[u_chunk],
            item_factors[:, m_chunk].T
        )

    predicted_arr  = predicted_centered + global_mean
    base_preds_svd = np.full(len(actual), global_mean)

    base_rmse = mean_squared_error(actual, base_preds_svd) ** 0.5
    base_mae  = mean_absolute_error(actual, base_preds_svd)
    svd_rmse  = mean_squared_error(actual, predicted_arr)  ** 0.5
    svd_mae   = mean_absolute_error(actual, predicted_arr)

    print()
    print('=== SVD Collaborative Filtering ===')
    print(f'  Base Model  RMSE: {round(base_rmse, 4)}  MAE: {round(base_mae, 4)}')
    print(f'  SVD Model   RMSE: {round(svd_rmse,  4)}  MAE: {round(svd_mae,  4)}')
    print(f'  RMSE improved by {round(base_rmse - svd_rmse, 4)} ({round((base_rmse - svd_rmse) / base_rmse * 100, 1)}%)')
    print(f'  MAE  improved by {round(base_mae  - svd_mae,  4)} ({round((base_mae  - svd_mae)  / base_mae  * 100, 1)}%)')

    return (user_factors, item_factors, user_pos, movie_pos,
            movie_ids_lookup, user_item_sparse, global_mean,
            base_rmse, base_mae, svd_rmse, svd_mae)


def collaborative_recommend(user_id, user_factors, item_factors,
                             user_pos, movie_ids_lookup,
                             user_item_sparse, global_mean,
                             links, movies, n=10):
    if user_id not in user_pos:
        print(f'User {user_id} not found.')
        return None

    u_idx = user_pos[user_id]
    user_preds_arr = (user_factors[u_idx] @ item_factors) + global_mean
    user_preds     = pd.Series(user_preds_arr, index=movie_ids_lookup)

    already_rated_idx      = user_item_sparse[u_idx].nonzero()[1]
    already_rated_movie_ids = movie_ids_lookup[already_rated_idx]
    user_preds = user_preds.drop(labels=already_rated_movie_ids, errors='ignore')

    top_movie_ids = user_preds.sort_values(ascending=False).head(n).index.tolist()
    top_links     = links[links['movieId'].isin(top_movie_ids)].copy()
    top_links['tmdbId'] = pd.to_numeric(top_links['tmdbId'], errors='coerce').astype('Int64')
    result = top_links.merge(
        movies[['id', 'title', 'genre_str']],
        left_on='tmdbId', right_on='id', how='left'
    )[['movieId', 'title', 'genre_str']]
    return result.reset_index(drop=True)
