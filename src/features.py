import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_features(df):
    df = df.copy()

    # Target: liked = 1 if rating >= 4.0
    df['liked'] = (df['rating'] >= 4.0).astype(int)

    # Genre one-hot encoding
    genre_dummies = df['genre_list'].explode().str.get_dummies().groupby(level=0).sum()
    genre_cols    = genre_dummies.columns.tolist()
    df = df.join(genre_dummies)

    base_cols    = [c for c in ['vote_average', 'vote_count', 'popularity'] if c in df.columns]
    feature_cols = base_cols + genre_cols
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols]
    y = df['liked']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=feature_cols)

    print(f'X_train shape : {X_train.shape}')
    print(f'X_test shape  : {X_test.shape}')
    print(f'Liked ratio   : {round(y.mean(), 3)}')

    return X_train, X_test, y_train, y_test, feature_cols

