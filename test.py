"""
test.py
Run with: python test.py
Tests all src modules using small mock data — no real CSVs needed.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PASS = '  [PASS]'
FAIL = '  [FAIL]'


# ── Mock Data ─────────────────────────────────────────────────────────────────

def make_mock_movies():
    return pd.DataFrame({
        'id'                : [1, 2, 3, 4, 5],
        'title'             : ['Toy Story', 'Jumanji', 'The Dark Knight', 'Interstellar', 'Inception'],
        'genres'            : [
            "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}]",
            "[{'id': 12, 'name': 'Adventure'}]",
            "[{'id': 28, 'name': 'Action'}, {'id': 80, 'name': 'Crime'}]",
            "[{'id': 878, 'name': 'Science Fiction'}]",
            "[{'id': 28, 'name': 'Action'}, {'id': 878, 'name': 'Science Fiction'}]",
        ],
        'overview'          : [
            'A cowboy doll is profoundly threatened when a new spaceman figure supplants him.',
            'A board game that releases wild jungle dangers.',
            'Batman fights the Joker in Gotham City.',
            'A team travels through a wormhole in space.',
            'A thief who steals secrets through dreams.',
        ],
        'vote_average'      : [7.7, 6.9, 9.0, 8.6, 8.8],
        'vote_count'        : [5000, 2000, 12000, 11000, 14000],
        'release_date'      : ['1995-10-30', '1995-12-15', '2008-07-18', '2014-11-07', '2010-07-16'],
        'runtime'           : [81, 104, 152, 169, 148],
        'original_language' : ['en', 'en', 'en', 'en', 'en'],
        'popularity'        : [21.9, 17.2, 35.1, 32.0, 29.0],
        'crew'              : [
            "[{'job': 'Director', 'name': 'John Lasseter'}]",
            "[{'job': 'Director', 'name': 'Joe Johnston'}]",
            "[{'job': 'Director', 'name': 'Christopher Nolan'}]",
            "[{'job': 'Director', 'name': 'Christopher Nolan'}]",
            "[{'job': 'Director', 'name': 'Christopher Nolan'}]",
        ],
        'cast'              : [
            "[{'name': 'Tom Hanks'}, {'name': 'Tim Allen'}]",
            "[{'name': 'Robin Williams'}]",
            "[{'name': 'Christian Bale'}, {'name': 'Heath Ledger'}]",
            "[{'name': 'Matthew McConaughey'}, {'name': 'Anne Hathaway'}]",
            "[{'name': 'Leonardo DiCaprio'}, {'name': 'Joseph Gordon-Levitt'}]",
        ],
        'keywords'          : [
            "[{'name': 'toy'}, {'name': 'friendship'}]",
            "[{'name': 'jungle'}, {'name': 'board game'}]",
            "[{'name': 'batman'}, {'name': 'joker'}]",
            "[{'name': 'space'}, {'name': 'wormhole'}]",
            "[{'name': 'dream'}, {'name': 'heist'}]",
        ],
    })


def make_mock_ratings():
    return pd.DataFrame({
        'userId'  : [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        'movieId' : [1, 2, 3, 1, 3, 2, 3, 4, 1, 4],
        'rating'  : [4.0, 3.5, 5.0, 2.0, 4.5, 3.0, 4.0, 5.0, 1.0, 4.0],
    })


def make_mock_links():
    return pd.DataFrame({
        'movieId' : [1, 2, 3, 4, 5],
        'tmdbId'  : [1.0, 2.0, 3.0, 4.0, 5.0],
    })


def make_mock_credits():
    movies = make_mock_movies()
    return pd.DataFrame({
        'id'   : movies['id'],
        'crew' : movies['crew'],
        'cast' : movies['cast'],
    })


def make_mock_keywords():
    movies = make_mock_movies()
    return pd.DataFrame({
        'id'       : movies['id'],
        'keywords' : movies['keywords'],
    })


# ── Test Functions ────────────────────────────────────────────────────────────

def test_clean_movies():
    print('\n[1] test_clean_movies')
    from src.preprocessing import clean_movies

    raw = make_mock_movies().drop(columns=['crew', 'cast', 'keywords'])
    raw['id'] = raw['id'].astype(str)       # simulate raw string ids
    raw.loc[0, 'vote_average'] = 'bad'      # simulate corrupt value

    try:
        result = clean_movies(raw)
        assert 'year' in result.columns,          'year column missing'
        assert result['vote_average'].dtype in [np.float64, float], 'vote_average not float'
        assert result['id'].dtype == int,         'id not int'
        print(PASS, f'shape={result.shape}, cols={result.columns.tolist()}')
    except Exception as e:
        print(FAIL, str(e))


def test_parse_genres():
    print('\n[2] test_parse_genres')
    from src.preprocessing import clean_movies, parse_genres

    raw = make_mock_movies().drop(columns=['crew', 'cast', 'keywords'])
    movies = clean_movies(raw)

    try:
        result = parse_genres(movies)
        assert 'genre_list' in result.columns, 'genre_list missing'
        assert 'genre_str'  in result.columns, 'genre_str missing'
        assert isinstance(result['genre_list'].iloc[0], list), 'genre_list not a list'
        print(PASS, f'sample genre_list: {result["genre_list"].iloc[0]}')
    except Exception as e:
        print(FAIL, str(e))


def test_parse_credits_keywords():
    print('\n[3] test_parse_credits_keywords')
    from src.preprocessing import clean_movies, parse_genres, parse_credits_keywords

    raw      = make_mock_movies().drop(columns=['crew', 'cast', 'keywords'])
    movies   = parse_genres(clean_movies(raw))
    credits  = make_mock_credits()
    keywords = make_mock_keywords()

    try:
        result = parse_credits_keywords(movies, credits, keywords)
        assert 'director'      in result.columns, 'director missing'
        assert 'cast_list'     in result.columns, 'cast_list missing'
        assert 'keyword_list'  in result.columns, 'keyword_list missing'
        print(PASS, f'director sample: {result["director"].iloc[2]}')
    except Exception as e:
        print(FAIL, str(e))


def test_merge_all():
    print('\n[4] test_merge_all')
    from src.preprocessing import clean_movies, parse_genres, parse_credits_keywords, merge_all

    raw      = make_mock_movies().drop(columns=['crew', 'cast', 'keywords'])
    movies   = parse_genres(clean_movies(raw))
    movies   = parse_credits_keywords(movies, make_mock_credits(), make_mock_keywords())
    ratings  = make_mock_ratings()
    links    = make_mock_links()

    try:
        df, links_out = merge_all(movies, ratings, links)
        assert 'rating'    in df.columns, 'rating missing'
        assert 'title'     in df.columns, 'title missing'
        assert 'genre_str' in df.columns, 'genre_str missing'
        assert len(df) > 0,               'merged df is empty'
        print(PASS, f'merged shape: {df.shape}')
    except Exception as e:
        print(FAIL, str(e))


def test_weighted_score():
    print('\n[5] test_weighted_score')
    from src.preprocessing  import clean_movies, parse_genres
    from src.weighted_score import compute_weighted_score

    raw    = make_mock_movies().drop(columns=['crew', 'cast', 'keywords'])
    movies = parse_genres(clean_movies(raw))

    try:
        result = compute_weighted_score(movies)
        assert 'weighted_score' in result.columns, 'weighted_score column missing'
        assert result['weighted_score'].iloc[0] >= result['weighted_score'].iloc[-1], 'not sorted'
        print(PASS, f'top movie: {result["title"].iloc[0]}  score: {round(result["weighted_score"].iloc[0], 4)}')
    except Exception as e:
        print(FAIL, str(e))


def test_build_features():
    print('\n[6] test_build_features')
    from src.preprocessing import clean_movies, parse_genres, parse_credits_keywords, merge_all
    from src.features      import build_features

    raw      = make_mock_movies().drop(columns=['crew', 'cast', 'keywords'])
    movies   = parse_genres(clean_movies(raw))
    movies   = parse_credits_keywords(movies, make_mock_credits(), make_mock_keywords())
    df, _    = merge_all(movies, make_mock_ratings(), make_mock_links())

    try:
        X_train, X_test, y_train, y_test, feature_cols = build_features(df)
        assert len(X_train) > 0,         'X_train is empty'
        assert len(feature_cols) > 0,    'feature_cols is empty'
        assert set(y_train.unique()).issubset({0, 1}), 'y_train has values other than 0/1'
        print(PASS, f'X_train={X_train.shape}  X_test={X_test.shape}  features={len(feature_cols)}')
    except Exception as e:
        print(FAIL, str(e))


def test_models():
    print('\n[7] test_models (baseline + LR + DT + NB)')
    from src.preprocessing import clean_movies, parse_genres, parse_credits_keywords, merge_all
    from src.features      import build_features
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    import numpy as np

    raw      = make_mock_movies().drop(columns=['crew', 'cast', 'keywords'])
    movies   = parse_genres(clean_movies(raw))
    movies   = parse_credits_keywords(movies, make_mock_credits(), make_mock_keywords())
    df, _    = merge_all(movies, make_mock_ratings(), make_mock_links())
    X_train, X_test, y_train, y_test, feature_cols = build_features(df)

    try:
        majority       = int(y_train.mode()[0])
        baseline_preds = np.full(len(y_test), majority)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)

        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        dt_preds = dt.predict(X_test)

        nb = GaussianNB()
        nb.fit(X_train, y_train)
        nb_preds = nb.predict(X_test)

        assert len(baseline_preds) == len(y_test), 'baseline preds length mismatch'
        assert len(lr_preds)       == len(y_test), 'LR preds length mismatch'
        assert len(dt_preds)       == len(y_test), 'DT preds length mismatch'
        assert len(nb_preds)       == len(y_test), 'NB preds length mismatch'
        print(PASS, f'all 4 models ran | LR acc={accuracy_score(y_test,lr_preds):.0%} DT acc={accuracy_score(y_test,dt_preds):.0%} NB acc={accuracy_score(y_test,nb_preds):.0%}')
    except Exception as e:
        print(FAIL, str(e))


def test_content_based():
    print('\n[8] test_content_based')
    from src.preprocessing import clean_movies, parse_genres, parse_credits_keywords
    from src.content_based import build_content_model, content_recommend

    raw      = make_mock_movies().drop(columns=['crew', 'cast', 'keywords'])
    movies   = parse_genres(clean_movies(raw))
    movies   = parse_credits_keywords(movies, make_mock_credits(), make_mock_keywords())

    try:
        tfidf_matrix, content, indices = build_content_model(movies)
        assert tfidf_matrix.shape[0] == len(content), 'matrix row count mismatch'

        result = content_recommend('Toy Story', tfidf_matrix, content, indices, n=3)
        assert result is not None,              'returned None'
        assert 'title' in result.columns,       'title column missing'
        assert 'similarity_score' in result.columns, 'similarity_score missing'
        assert len(result) <= 3,                'returned more than n results'
        print(PASS, f'recommendations for Toy Story:\n{result.to_string()}')
    except Exception as e:
        print(FAIL, str(e))


def test_content_based_unknown_title():
    print('\n[9] test_content_based (unknown title)')
    from src.preprocessing import clean_movies, parse_genres, parse_credits_keywords
    from src.content_based import build_content_model, content_recommend

    raw    = make_mock_movies().drop(columns=['crew', 'cast', 'keywords'])
    movies = parse_genres(clean_movies(raw))
    movies = parse_credits_keywords(movies, make_mock_credits(), make_mock_keywords())

    try:
        tfidf_matrix, content, indices = build_content_model(movies)
        result = content_recommend('NonExistentMovie123', tfidf_matrix, content, indices)
        assert result is None, 'should return None for unknown title'
        print(PASS, 'correctly returned None for unknown title')
    except Exception as e:
        print(FAIL, str(e))


def test_collaborative():
    print('\n[10] test_collaborative')
    from src.preprocessing  import clean_movies, parse_genres, parse_credits_keywords
    from src.collaborative  import build_svd_model, collaborative_recommend

    raw      = make_mock_movies().drop(columns=['crew', 'cast', 'keywords'])
    movies   = parse_genres(clean_movies(raw))
    movies   = parse_credits_keywords(movies, make_mock_credits(), make_mock_keywords())
    ratings  = make_mock_ratings()
    links    = make_mock_links()

    try:
        (user_factors, item_factors, user_pos, movie_pos,
         movie_ids_lookup, user_item_sparse, global_mean,
         base_rmse, base_mae, svd_rmse, svd_mae) = build_svd_model(ratings, links, movies, n_components=2)

        assert user_factors.shape[0] > 0,  'user_factors empty'
        assert svd_rmse <= base_rmse + 0.5,'SVD RMSE unreasonably high'

        result = collaborative_recommend(1, user_factors, item_factors, user_pos,
                                         movie_ids_lookup, user_item_sparse, global_mean,
                                         links, movies, n=3)
        print(PASS, f'SVD RMSE={round(svd_rmse,4)}  base RMSE={round(base_rmse,4)}')
        if result is not None:
            print('       collab recs for user 1:\n', result.to_string())
    except Exception as e:
        print(FAIL, str(e))


def test_hybrid():
    print('\n[11] test_hybrid')
    from src.preprocessing import clean_movies, parse_genres, parse_credits_keywords
    from src.content_based import build_content_model
    from src.collaborative import build_svd_model
    from src.hybrid        import hybrid_recommend

    raw      = make_mock_movies().drop(columns=['crew', 'cast', 'keywords'])
    movies   = parse_genres(clean_movies(raw))
    movies   = parse_credits_keywords(movies, make_mock_credits(), make_mock_keywords())
    ratings  = make_mock_ratings()
    links    = make_mock_links()

    tfidf_matrix, content, indices = build_content_model(movies)
    (user_factors, item_factors, user_pos, movie_pos,
     movie_ids_lookup, user_item_sparse, global_mean, *_) = build_svd_model(ratings, links, movies, n_components=2)

    try:
        result = hybrid_recommend(1, 'Toy Story',
                                  tfidf_matrix, content, indices,
                                  user_factors, item_factors,
                                  user_pos, movie_ids_lookup, global_mean, links, n=3)
        assert result is not None,              'returned None'
        assert 'hybrid_score' in result.columns,'hybrid_score missing'
        assert 'Toy Story' not in result['title'].values, 'query movie should be excluded'
        print(PASS, f'hybrid recs for user 1 + Toy Story:\n{result.to_string()}')
    except Exception as e:
        print(FAIL, str(e))


# ── Runner ────────────────────────────────────────────────────────────────────

def run_interactive():
    """Build models from mock data and let user test recommendations interactively."""
    print('\n' + '='*60)
    print('  INTERACTIVE MODE — type a movie and user ID')
    print('  Available movies: Toy Story, Jumanji, The Dark Knight,')
    print('                    Interstellar, Inception')
    print('  Available user IDs: 1, 2, 3, 4')
    print('  Type quit to exit')
    print('='*60)

    from src.preprocessing import clean_movies, parse_genres, parse_credits_keywords
    from src.content_based import build_content_model, content_recommend
    from src.collaborative import build_svd_model, collaborative_recommend
    from src.hybrid        import hybrid_recommend

    raw      = make_mock_movies().drop(columns=['crew', 'cast', 'keywords'])
    movies   = parse_genres(clean_movies(raw))
    movies   = parse_credits_keywords(movies, make_mock_credits(), make_mock_keywords())
    ratings  = make_mock_ratings()
    links    = make_mock_links()

    print('\nBuilding models...')
    tfidf_matrix, content, indices = build_content_model(movies)
    (user_factors, item_factors, user_pos, movie_pos,
     movie_ids_lookup, user_item_sparse, global_mean, *_) = build_svd_model(ratings, links, movies, n_components=2)
    print('Models ready!\n')

    while True:
        print('-'*40)
        title   = input('Enter movie title : ').strip()
        if title.lower() == 'quit':
            break
        user_id = input('Enter user ID     : ').strip()
        if user_id.lower() == 'quit':
            break

        try:
            user_id = int(user_id)
        except ValueError:
            print('User ID must be a number.')
            continue

        print('\n--- Content-Based Recommendations ---')
        cb = content_recommend(title, tfidf_matrix, content, indices, n=5)
        if cb is not None:
            print(cb.to_string(index=False))

        print('\n--- Collaborative Recommendations ---')
        collab = collaborative_recommend(user_id, user_factors, item_factors,
                                         user_pos, movie_ids_lookup,
                                         user_item_sparse, global_mean,
                                         links, movies, n=5)
        if collab is not None:
            print(collab.to_string(index=False))
        else:
            print(f'User {user_id} not found.')

        print('\n--- Hybrid Recommendations (60% content + 40% collab) ---')
        hybrid = hybrid_recommend(user_id, title,
                                  tfidf_matrix, content, indices,
                                  user_factors, item_factors,
                                  user_pos, movie_ids_lookup, global_mean, links, n=5)
        if hybrid is not None:
            print(hybrid.to_string(index=False))

        print()


if __name__ == '__main__':
    import sys

    if '--interactive' in sys.argv or '-i' in sys.argv:
        run_interactive()
    else:
        print('='*60)
        print('  MOVIE RECOMMENDATION SYSTEM — TESTS')
        print('='*60)

        test_clean_movies()
        test_parse_genres()
        test_parse_credits_keywords()
        test_merge_all()
        test_weighted_score()
        test_build_features()
        test_models()
        test_content_based()
        test_content_based_unknown_title()
        test_collaborative()
        test_hybrid()

        print('\n' + '='*60)
        print('  ALL TESTS DONE')
        print('='*60)
        print()
        print('Tip: run with -i flag for interactive mode:')
        print('     python test.py -i')
