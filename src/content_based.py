import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_content_model(movies):
    content = movies[['id', 'title', 'genre_str', 'overview',
                       'director', 'cast_list', 'keyword_list']].copy()
    content = content.dropna(subset=['title']).reset_index(drop=True)

    content['cast_str']    = content['cast_list'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else '')
    content['keyword_str'] = content['keyword_list'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else '')
    content['soup']        = (content['genre_str']    + ' ' +
                              content['overview']     + ' ' +
                              content['director']     + ' ' +
                              content['cast_str']     + ' ' +
                              content['keyword_str'])

    tfidf        = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf.fit_transform(content['soup'])
    indices      = pd.Series(content.index, index=content['title']).drop_duplicates()

    print('content rows        :', len(content))
    print('tfidf_matrix shape  :', tfidf_matrix.shape)
    print('Shapes match        :', len(content) == tfidf_matrix.shape[0])

    return tfidf_matrix, content, indices


def content_recommend(title, tfidf_matrix, content, indices, n=10):
    if title not in indices:
        print(f"'{title}' not found.")
        return None

    idx        = indices[title]
    sim_row    = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = sorted(enumerate(sim_row), key=lambda x: x[1], reverse=True)[1:n + 1]
    result     = content[['title', 'genre_str']].iloc[[i[0] for i in sim_scores]].copy()
    result['similarity_score'] = [round(s[1], 4) for s in sim_scores]
    return result.reset_index(drop=True)
