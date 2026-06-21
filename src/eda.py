import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


def run_eda(df, ratings):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Rating Distribution
    ratings['rating'].value_counts().sort_index().plot(
        kind='bar', ax=axes[0, 0], color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Rating Distribution')
    axes[0, 0].set_xlabel('Rating')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=0)

    # 2. Top 10 Genre Frequency
    all_genres   = df['genre_list'].explode().dropna()
    genre_counts = Counter(all_genres)
    genre_df     = pd.DataFrame(genre_counts.items(), columns=['genre', 'count']) \
                     .sort_values('count', ascending=False).head(10)
    axes[0, 1].barh(genre_df['genre'], genre_df['count'], color='steelblue')
    axes[0, 1].set_title('Top 10 Genres')
    axes[0, 1].invert_yaxis()

    # 3. Ratings per User
    user_activity = ratings.groupby('userId')['rating'].count()
    axes[1, 0].hist(user_activity, bins=20, color='teal', edgecolor='black')
    axes[1, 0].set_title('Ratings per User')
    axes[1, 0].set_xlabel('Rating Count')
    axes[1, 0].set_ylabel('Users')

    # 4. Top 10 Most Rated Movies
    top_movies = df.groupby('title')['rating'].count().sort_values(ascending=False).head(10)
    axes[1, 1].barh(top_movies.index, top_movies.values, color='goldenrod')
    axes[1, 1].set_title('Top 10 Most Rated Movies')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.show()
    plt.close()

    print(f'Average rating     : {round(ratings["rating"].mean(), 2)}')
    print(f'Most common rating : {ratings["rating"].mode()[0]}')
    print(f'Avg ratings/user   : {round(user_activity.mean(), 1)}')
    print(f'Top genre          : {genre_df.iloc[0]["genre"]}')


def run_genre_avg_rating(df):
    sample_df = df.sample(n=min(500000, len(df)), random_state=42)

    genre_sum   = defaultdict(float)
    genre_count = defaultdict(int)

    for genres, rating in zip(sample_df['genre_list'], sample_df['rating']):
        if not isinstance(genres, list):
            continue
        if pd.isna(rating):
            continue
        for genre in genres:
            genre_sum[genre]   += rating
            genre_count[genre] += 1

    avg_ratings = {g: genre_sum[g] / genre_count[g] for g in genre_sum}
    avg_genre_rating = pd.DataFrame(avg_ratings.items(), columns=['Genre', 'Average Rating']) \
                         .sort_values('Average Rating', ascending=False).head(12)

    plt.figure(figsize=(10, 5))
    plt.bar(avg_genre_rating['Genre'], avg_genre_rating['Average Rating'])
    plt.title('Average Rating per Genre')
    plt.xlabel('Genre')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(avg_genre_rating)
