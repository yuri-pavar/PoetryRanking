
import pandas as pd


def preprocess_data_tf_idf(df, min_rating):
    df_mod = df.copy()
    df_mod.output_text = df_mod.output_text.str.lower()
    df_mod.rating = df_mod.rating.apply(lambda x: 0 if x <= min_rating else 1)

    return df_mod[['output_text', 'rating']]


def preprocess_data_bert(df, min_rating, spec_token):
    df_mod = df.copy()
    df_mod.output_text = df_mod.output_text.str.replace('\n', spec_token)
    df_mod.output_text = df_mod.output_text.str.lower()
    df_mod.rating = df_mod.rating.apply(lambda x: 0 if x <= min_rating else 1)
    df_mod.genre = df_mod.genre.str.lower()
    df_mod = df_mod.drop('url', axis=1)

    return df_mod


