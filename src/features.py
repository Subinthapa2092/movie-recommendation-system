import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_features(df, max_rows=3_000_000, random_state=42):
    # Subsample BEFORE building the feature matrix — fitting classifiers like
    # LogisticRegression (lbfgs) internally upcasts X to float64, which roughly
    # doubles memory at the moment of fit() and is what crashes on 20M+ rows.
    # A few million rows is plenty for a binary "liked" classifier and keeps
    # everything downstream (scaling, fitting) comfortably in memory.
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)
        print(f'Subsampled to {max_rows:,} rows (from full dataset) for model training.')

    # Target: liked = 1 if rating >= 4.0
    liked = (df['rating'] >= 4.0).astype('int8')

    # Genre one-hot encoding — done once per unique title (tiny table),
    # never exploded or merged against the full ratings table.
    unique_genres = df[['title', 'genre_list']].drop_duplicates(subset='title')
    genre_dummies = (
        unique_genres.set_index('title')['genre_list']
        .explode()
        .str.get_dummies()
        .groupby(level=0)
        .sum()
        .astype('int8')
    )
    genre_cols = genre_dummies.columns.tolist()

    # Build the working feature frame column-by-column instead of merge()/copy(),
    # which avoids pandas duplicating the full original dataframe in memory.
    base_cols = [c for c in ['vote_average', 'vote_count', 'popularity'] if c in df.columns]

    feature_data = {}
    for col in base_cols:
        feature_data[col] = df[col].to_numpy(dtype='float32', copy=False)

    # title -> row position in genre_dummies, then take rows via numpy (fast, low-memory)
    title_to_pos = {t: i for i, t in enumerate(genre_dummies.index)}
    row_pos      = df['title'].map(title_to_pos)
    valid_mask   = row_pos.notna()
    row_pos_arr  = row_pos.fillna(0).astype('int64').to_numpy()

    genre_values = genre_dummies.to_numpy()[row_pos_arr]          # (n_rows, n_genres) int8
    genre_values[~valid_mask.to_numpy()] = 0                       # zero out unmatched titles

    for j, gcol in enumerate(genre_cols):
        feature_data[gcol] = genre_values[:, j]

    feature_cols = base_cols + genre_cols
    X = pd.DataFrame(feature_data, columns=feature_cols)
    y = liked

    del feature_data, genre_values, row_pos, row_pos_arr, valid_mask

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    del X, y

    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols).astype('float32')
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=feature_cols).astype('float32')

    print(f'X_train shape : {X_train.shape}')
    print(f'X_test shape  : {X_test.shape}')
    print(f'Liked ratio   : {round(y_train.mean(), 3)}')

    return X_train, X_test, y_train, y_test, feature_cols