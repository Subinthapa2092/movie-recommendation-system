def compute_weighted_score(movies):
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(0.70)

    print(f'Mean rating across all movies (C) : {round(C, 2)}')
    print(f'Minimum vote threshold (m)        : {round(m, 0)}')

    qualified = movies[movies['vote_count'] >= m].copy()
    v = qualified['vote_count']
    R = qualified['vote_average']
    qualified['weighted_score'] = (v / (v + m)) * R + (m / (v + m)) * C

    top = qualified.sort_values('weighted_score', ascending=False)
    print('\nTop 10 movies by weighted score:')
    print(top[['title', 'vote_count', 'vote_average', 'weighted_score']].head(10).to_string())
    return top
