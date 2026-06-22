import os
import pickle

import numpy as np
import scipy.sparse as sp


#  paths 

MODELS_DIR = 'models'


def _path(filename, models_dir=MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, filename)


#  classification models (.pkl)
def save_classifiers(lr, dt, nb, feature_cols, models_dir=MODELS_DIR):
    """Save the three trained classifiers + the feature column list."""
    with open(_path('logistic_regression.pkl', models_dir), 'wb') as f:
        pickle.dump(lr, f)
    with open(_path('decision_tree.pkl', models_dir), 'wb') as f:
        pickle.dump(dt, f)
    with open(_path('naive_bayes.pkl', models_dir), 'wb') as f:
        pickle.dump(nb, f)
    with open(_path('feature_cols.pkl', models_dir), 'wb') as f:
        pickle.dump(feature_cols, f)

    print('Saved classifiers:')
    print(f'  {_path("logistic_regression.pkl", models_dir)}')
    print(f'  {_path("decision_tree.pkl", models_dir)}')
    print(f'  {_path("naive_bayes.pkl", models_dir)}')
    print(f'  {_path("feature_cols.pkl", models_dir)}')


def load_classifiers(models_dir=MODELS_DIR):
    """Load the three trained classifiers + the feature column list."""
    with open(_path('logistic_regression.pkl', models_dir), 'rb') as f:
        lr = pickle.load(f)
    with open(_path('decision_tree.pkl', models_dir), 'rb') as f:
        dt = pickle.load(f)
    with open(_path('naive_bayes.pkl', models_dir), 'rb') as f:
        nb = pickle.load(f)
    with open(_path('feature_cols.pkl', models_dir), 'rb') as f:
        feature_cols = pickle.load(f)

    return lr, dt, nb, feature_cols


#  content-based model (.pkl) 

def save_content_model(tfidf_matrix, content, indices, models_dir=MODELS_DIR):
    """Save the TF-IDF matrix (sparse), content table, and title->index lookup."""
    sp.save_npz(_path('tfidf_matrix.npz', models_dir), tfidf_matrix)

    with open(_path('content_table.pkl', models_dir), 'wb') as f:
        pickle.dump(content, f)
    with open(_path('content_indices.pkl', models_dir), 'wb') as f:
        pickle.dump(indices, f)

    print('Saved content-based model:')
    print(f'  {_path("tfidf_matrix.npz", models_dir)}')
    print(f'  {_path("content_table.pkl", models_dir)}')
    print(f'  {_path("content_indices.pkl", models_dir)}')


def load_content_model(models_dir=MODELS_DIR):
    """Load the TF-IDF matrix (sparse), content table, and title->index lookup."""
    tfidf_matrix = sp.load_npz(_path('tfidf_matrix.npz', models_dir))

    with open(_path('content_table.pkl', models_dir), 'rb') as f:
        content = pickle.load(f)
    with open(_path('content_indices.pkl', models_dir), 'rb') as f:
        indices = pickle.load(f)

    return tfidf_matrix, content, indices


#  SVD collaborative model (.npz + .pkl) 

def save_svd_model(user_factors, item_factors, user_pos, movie_pos,
                    movie_ids_lookup, user_item_sparse, global_mean,
                    models_dir=MODELS_DIR):
    """Save SVD factors/sparse matrix as .npz, and the id<->index lookups as .pkl."""
    np.savez(
        _path('svd_factors.npz', models_dir),
        user_factors=user_factors,
        item_factors=item_factors,
        movie_ids_lookup=np.asarray(movie_ids_lookup),
        global_mean=global_mean,
    )
    sp.save_npz(_path('user_item_sparse.npz', models_dir), user_item_sparse)

    with open(_path('user_pos.pkl', models_dir), 'wb') as f:
        pickle.dump(user_pos, f)
    with open(_path('movie_pos.pkl', models_dir), 'wb') as f:
        pickle.dump(movie_pos, f)

    print('Saved SVD collaborative model:')
    print(f'  {_path("svd_factors.npz", models_dir)}')
    print(f'  {_path("user_item_sparse.npz", models_dir)}')
    print(f'  {_path("user_pos.pkl", models_dir)}')
    print(f'  {_path("movie_pos.pkl", models_dir)}')


def load_svd_model(models_dir=MODELS_DIR):
    """Load SVD factors/sparse matrix + id<->index lookups."""
    npz = np.load(_path('svd_factors.npz', models_dir), allow_pickle=True)
    user_factors = npz['user_factors']
    item_factors = npz['item_factors']
    movie_ids_lookup = npz['movie_ids_lookup']
    global_mean = float(npz['global_mean'])

    user_item_sparse = sp.load_npz(_path('user_item_sparse.npz', models_dir))

    with open(_path('user_pos.pkl', models_dir), 'rb') as f:
        user_pos = pickle.load(f)
    with open(_path('movie_pos.pkl', models_dir), 'rb') as f:
        movie_pos = pickle.load(f)

    return (user_factors, item_factors, user_pos, movie_pos,
            movie_ids_lookup, user_item_sparse, global_mean)


#  convenience: save / load everything in one call

def save_all(lr, dt, nb, feature_cols,
             tfidf_matrix, content, indices,
             user_factors, item_factors, user_pos, movie_pos,
             movie_ids_lookup, user_item_sparse, global_mean,
             models_dir=MODELS_DIR):
    save_classifiers(lr, dt, nb, feature_cols, models_dir)
    save_content_model(tfidf_matrix, content, indices, models_dir)
    save_svd_model(user_factors, item_factors, user_pos, movie_pos,
                   movie_ids_lookup, user_item_sparse, global_mean, models_dir)
    print('\nAll models saved to', os.path.abspath(models_dir))


def load_all(models_dir=MODELS_DIR):
    # Classifiers are optional — web app only needs recommenders
    try:
        lr, dt, nb, feature_cols = load_classifiers(models_dir)
    except Exception:
        lr = dt = nb = feature_cols = None

    tfidf_matrix, content, indices = load_content_model(models_dir)
    (user_factors, item_factors, user_pos, movie_pos,
     movie_ids_lookup, user_item_sparse, global_mean) = load_svd_model(models_dir)

    return {
        'lr': lr, 'dt': dt, 'nb': nb, 'feature_cols': feature_cols,
        'tfidf_matrix': tfidf_matrix, 'content': content, 'indices': indices,
        'user_factors': user_factors, 'item_factors': item_factors,
        'user_pos': user_pos, 'movie_pos': movie_pos,
        'movie_ids_lookup': movie_ids_lookup,
        'user_item_sparse': user_item_sparse, 'global_mean': global_mean,
    }
