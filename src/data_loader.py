import pandas as pd


def load_data(data_dir='data'):
    movies   = pd.read_csv(f'{data_dir}/movies_metadata.csv', low_memory=False)
    ratings  = pd.read_csv(f'{data_dir}/ratings.csv')
    links    = pd.read_csv(f'{data_dir}/links.csv')
    credits  = pd.read_csv(f'{data_dir}/credits.csv')
    keywords = pd.read_csv(f'{data_dir}/keywords.csv')

    print('movies_metadata :', movies.shape)
    print('ratings         :', ratings.shape)
    print('links           :', links.shape)
    print('credits         :', credits.shape)
    print('keywords        :', keywords.shape)

    return movies, ratings, links, credits, keywords
