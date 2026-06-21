import ast
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_genres(genre_str):
    try:
        return [g['name'] for g in ast.literal_eval(genre_str)]
    except Exception:
        return []


def _get_director(crew_str):
    try:
        for person in ast.literal_eval(crew_str):
            if person['job'] == 'Director':
                return person['name']
    except Exception:
        pass
    return ''


def _get_top_cast(cast_str, n=3):
    try:
        return [c['name'] for c in ast.literal_eval(cast_str)[:n]]
    except Exception:
        return []


def _get_keywords(kw_str):
    try:
        return [k['name'] for k in ast.literal_eval(kw_str)]
    except Exception:
        return []


# ── main functions ────────────────────────────────────────────────────────────

def clean_movies(movies):
    movies = movies[pd.to_numeric(movies['id'], errors='coerce').notnull()].copy()
    movies['id'] = movies['id'].astype(int)

    movies = movies[[
        'id', 'title', 'genres', 'overview',
        'vote_average', 'vote_count', 'release_date',
        'runtime', 'original_language', 'popularity'
    ]].copy()

    movies['vote_average'] = pd.to_numeric(movies['vote_average'], errors='coerce').fillna(0)
    movies['vote_count']   = pd.to_numeric(movies['vote_count'],   errors='coerce').fillna(0)
    movies['popularity']   = pd.to_numeric(movies['popularity'],   errors='coerce').fillna(0)
    movies['overview']     = movies['overview'].fillna('')
    movies['year']         = pd.to_datetime(movies['release_date'], errors='coerce').dt.year

    print('Movies cleaned shape:', movies.shape)
    print('Null check:')
    print(movies.isnull().sum())
    return movies


def parse_genres(movies):
    movies['genre_list'] = movies['genres'].apply(_parse_genres)
    movies['genre_str']  = movies['genre_list'].apply(lambda x: ' '.join(x))
    print('Genres parsed successfully!')
    print(movies[['title', 'genre_list']].head(3).to_string())
    return movies


def parse_credits_keywords(movies, credits, keywords):
    credits['id']  = credits['id'].astype(int)
    keywords['id'] = keywords['id'].astype(int)

    credits['director']      = credits['crew'].apply(_get_director)
    credits['cast_list']     = credits['cast'].apply(_get_top_cast)
    keywords['keyword_list'] = keywords['keywords'].apply(_get_keywords)

    movies = movies.merge(credits[['id', 'director', 'cast_list']], on='id', how='left')
    movies = movies.merge(keywords[['id', 'keyword_list']],         on='id', how='left')

    movies['director']     = movies['director'].fillna('')
    movies['cast_list']    = movies['cast_list'].apply(lambda x: x if isinstance(x, list) else [])
    movies['keyword_list'] = movies['keyword_list'].apply(lambda x: x if isinstance(x, list) else [])

    print('Credits & keywords merged!')
    print('Movies shape now:', movies.shape)
    print(movies[['title', 'director', 'cast_list']].head(2).to_string())
    return movies


def merge_all(movies, ratings, links):
    links = links.dropna(subset=['tmdbId']).copy()
    links['tmdbId'] = links['tmdbId'].astype(int)

    df = ratings.merge(links[['movieId', 'tmdbId']], on='movieId', how='inner')
    df = df.merge(
        movies[['id', 'title', 'genre_list', 'genre_str',
                'vote_average', 'vote_count', 'year', 'director', 'overview']],
        left_on='tmdbId', right_on='id', how='inner'
    )
    drop_cols = [c for c in ['id', 'timestamp'] if c in df.columns]
    df = df.drop(columns=drop_cols)

    print('Merged df shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print(df.head(3).to_string())
    return df, links
