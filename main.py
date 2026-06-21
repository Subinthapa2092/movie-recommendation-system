import warnings
warnings.filterwarnings('ignore')

from src.data_loader    import load_data
from src.preprocessing  import clean_movies, parse_genres, parse_credits_keywords, merge_all
from src.eda            import run_eda, run_genre_avg_rating
from src.weighted_score import compute_weighted_score
from src.features       import build_features
from src.models         import (run_baseline, run_logistic, run_decision_tree,
                                run_naive_bayes, compare_models)
from src.content_based  import build_content_model, content_recommend
from src.collaborative  import build_svd_model, collaborative_recommend
from src.hybrid         import hybrid_recommend
from sklearn.metrics    import accuracy_score, mean_absolute_error, mean_squared_error


def main():
    # ── 1. Load ───────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 1 — Load Data')
    print('='*60)
    movies, ratings, links, credits, keywords = load_data(data_dir='data')

    # ── 2. Clean ──────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 2 — Clean Movies')
    print('='*60)
    movies = clean_movies(movies)

    # ── 3. Parse genres ───────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 3 — Parse Genres')
    print('='*60)
    movies = parse_genres(movies)

    # ── 4. Parse credits & keywords ───────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 4 — Parse Credits & Keywords')
    print('='*60)
    movies = parse_credits_keywords(movies, credits, keywords)

    # ── 5. Merge ──────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 5 — Merge Ratings + Links + Movies')
    print('='*60)
    df, links = merge_all(movies, ratings, links)

    # ── 6. EDA ────────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 6 — EDA')
    print('='*60)
    run_eda(df, ratings)
    run_genre_avg_rating(df)

    # ── 7. Weighted Score ─────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 7 — Weighted Score (IMDB Formula)')
    print('='*60)
    compute_weighted_score(movies)

    # ── 8. Feature Engineering ────────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 8 — Feature Engineering')
    print('='*60)
    X_train, X_test, y_train, y_test, feature_cols = build_features(df)

    # ── 9. Models ─────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 9 — Baseline')
    print('='*60)
    baseline_preds = run_baseline(X_train, X_test, y_train, y_test)

    print('\n' + '='*60)
    print('STEP 10 — Logistic Regression')
    print('='*60)
    lr, lr_preds, lr_train_preds = run_logistic(X_train, X_test, y_train, y_test)

    print('\n' + '='*60)
    print('STEP 11 — Decision Tree')
    print('='*60)
    dt, dt_preds, dt_train_preds = run_decision_tree(X_train, X_test, y_train, y_test, feature_cols)

    print('\n' + '='*60)
    print('STEP 12 — Naive Bayes')
    print('='*60)
    nb, nb_preds, nb_train_preds = run_naive_bayes(X_train, X_test, y_train, y_test)

    print('\n' + '='*60)
    print('STEP 13 — Model Comparison')
    print('='*60)
    compare_models(y_train, y_test,
                   baseline_preds,
                   lr_preds, lr_train_preds,
                   dt_preds, dt_train_preds,
                   nb_preds, nb_train_preds)

    # ── 10. Content-Based ─────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 14 — Content-Based Filtering')
    print('='*60)
    tfidf_matrix, content, indices = build_content_model(movies)

    print('\nContent-Based Recommendations for The Dark Knight:')
    print(content_recommend('The Dark Knight', tfidf_matrix, content, indices).to_string())
    print('\nContent-Based Recommendations for Toy Story:')
    print(content_recommend('Toy Story', tfidf_matrix, content, indices).to_string())

    # ── 11. SVD Collaborative ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 15 — SVD Collaborative Filtering')
    print('='*60)
    (user_factors, item_factors, user_pos, movie_pos,
     movie_ids_lookup, user_item_sparse, global_mean,
     base_rmse, base_mae, svd_rmse, svd_mae) = build_svd_model(ratings, links, movies)

    print('\nCollaborative Recommendations for User 1:')
    print(collaborative_recommend(1, user_factors, item_factors, user_pos,
                                  movie_ids_lookup, user_item_sparse, global_mean,
                                  links, movies).to_string())
    print('\nCollaborative Recommendations for User 5:')
    print(collaborative_recommend(5, user_factors, item_factors, user_pos,
                                  movie_ids_lookup, user_item_sparse, global_mean,
                                  links, movies).to_string())

    # ── 12. Hybrid ────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 16 — Hybrid Recommender')
    print('='*60)

    print('\nHybrid — User 1, The Dark Knight:')
    print(hybrid_recommend(1, 'The Dark Knight',
                           tfidf_matrix, content, indices,
                           user_factors, item_factors,
                           user_pos, movie_ids_lookup, global_mean, links).to_string())

    print('\nHybrid — User 5, Toy Story:')
    print(hybrid_recommend(5, 'Toy Story',
                           tfidf_matrix, content, indices,
                           user_factors, item_factors,
                           user_pos, movie_ids_lookup, global_mean, links).to_string())

    print('\nHybrid — User 10, Interstellar:')
    print(hybrid_recommend(10, 'Interstellar',
                           tfidf_matrix, content, indices,
                           user_factors, item_factors,
                           user_pos, movie_ids_lookup, global_mean, links).to_string())

    # ── Final Summary ─────────────────────────────────────────────────────────
    print('\n' + '-'*60)
    print('      MOVIE RECOMMENDATION SYSTEM — COMPLETE')
    print('-'*60)
    print()
    print('Dataset  : The Movies Dataset (rounakbanik — Kaggle)')
    print(f'Ratings  : {len(ratings):,}  |  Users: {ratings.userId.nunique()}  |  Movies: {ratings.movieId.nunique()}')
    print()
    print('--- Classification Models (Target: Liked = rating >= 4.0) ---')
    print(f'  Baseline             Accuracy: {accuracy_score(y_test, baseline_preds):.2%}')
    print(f'  Logistic Regression  Accuracy: {accuracy_score(y_test, lr_preds):.2%}  RMSE: {mean_squared_error(y_test, lr_preds)**0.5:.4f}  MAE: {mean_absolute_error(y_test, lr_preds):.4f}')
    print(f'  Decision Tree        Accuracy: {accuracy_score(y_test, dt_preds):.2%}  RMSE: {mean_squared_error(y_test, dt_preds)**0.5:.4f}  MAE: {mean_absolute_error(y_test, dt_preds):.4f}')
    print(f'  Naive Bayes          Accuracy: {accuracy_score(y_test, nb_preds):.2%}  RMSE: {mean_squared_error(y_test, nb_preds)**0.5:.4f}  MAE: {mean_absolute_error(y_test, nb_preds):.4f}')
    print()
    print('--- SVD Collaborative Filtering ---')
    print(f'  Base Model  RMSE: {round(base_rmse, 4)}  MAE: {round(base_mae, 4)}')
    print(f'  SVD Model   RMSE: {round(svd_rmse,  4)}  MAE: {round(svd_mae,  4)}')
    print()
    print('--- Recommendation System ---')
    print('  Content-Based  : TF-IDF + Cosine Similarity (genre+overview+cast+keywords)')
    print('  Collaborative  : SVD Matrix Factorization (50 latent factors)')
    print('  Hybrid         : 60% Content + 40% Collaborative')
    print()
    print('-'*60)
    print('PROJECT COMPLETE!')
    print('-'*60)


if __name__ == '__main__':
    main()
