#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import multiprocessing
from operator import itemgetter
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string
from nltk import word_tokenize
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ONLINE = True
TOY = False

if ONLINE:
    TRAIN_FILE = "../input/train.csv"
    RESOURCE_FILE = "../input/resources.csv"
    TEST_FILE = "../input/test.csv"
    WORKERS = 32

else:
    TRAIN_FILE = "train.csv"
    RESOURCE_FILE = "resources.csv"
    TEST_FILE = "test.csv"
    WORKERS = 2

def _apply_df(args):
    df, func, num, kwargs = args
    return num, df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, i, kwargs) for i, d in enumerate(np.array_split(df, workers))])
    pool.close()
    result = sorted(result, key=lambda x: x[0])
    return pd.concat([i[1] for i in result])


def tokenize(x):
    return word_tokenize(x)


def count_punctuation(tokens, punctuation_char):
    return len([token for token in tokens if token == punctuation_char])


def preprocess_df(df, workers):
    if __name__ == "__main__":
        dfr = pd.read_csv(RESOURCE_FILE)
        dfr['total'] = dfr['price'] * dfr['quantity']
        dfr['has_zero'] = dfr['price'].apply(lambda x: 1 if x == 0 else 0)
        dfr = dfr.groupby('id').agg('sum').reset_index()
        
        # merging essays
        df['student_description'] = df['project_essay_1']
        df.loc[df.project_essay_3.notnull(), 'student_description'] = df.loc[
                                                                          df.project_essay_3.notnull(), 'project_essay_1'] + \
                                                                      df.loc[
                                                                          df.project_essay_3.notnull(), 'project_essay_2']
        df['project_description'] = df['project_essay_2']

        df.loc[df.project_essay_3.notnull(), 'project_description'] = df.loc[
                                                                          df.project_essay_3.notnull(), 'project_essay_3'] + \
                                                                      df.loc[
                                                                          df.project_essay_3.notnull(), 'project_essay_4']

        df['project_subject_categories'] = df['project_subject_categories'].apply(lambda x: x.split(", "))
        df['project_subject_subcategories'] = df['project_subject_subcategories'].apply(lambda x: x.split(", "))
        df['teacher_prefix'] = df['teacher_prefix'].fillna('None')
        df = df.merge(dfr, how='inner', on='id')

        df['student_tokens'] = apply_by_multiprocessing(df['student_description'], tokenize, workers=workers)
        df['student_word_count'] = df['student_tokens'].apply(lambda x: len(x))
        df['student_unique_words'] = df['student_tokens'].apply(lambda x: len(set(x)))
        df['student_n_periods'] = df['student_tokens'].apply(lambda x: count_punctuation(x, '.'))
        df['student_n_commas'] = df['student_tokens'].apply(lambda x: count_punctuation(x, ','))
        df['student_n_questions'] = df['student_tokens'].apply(lambda x: count_punctuation(x, '?'))
        df['student_n_exclamations'] = df['student_tokens'].apply(lambda x: count_punctuation(x, '!'))
        df['student_word_len'] = df['student_tokens'].apply(lambda x: np.mean([len(token) for token in x]))

        del (df['student_tokens'])

        df['project_tokens'] = apply_by_multiprocessing(df['project_description'], tokenize, workers=workers)
        df['project_word_count'] = df['project_tokens'].apply(lambda x: len(x))
        df['project_unique_words'] = df['project_tokens'].apply(lambda x: len(set(x)))

        df['project_n_periods'] = df['project_tokens'].apply(lambda x: count_punctuation(x, '.'))
        df['project_n_commas'] = df['project_tokens'].apply(lambda x: count_punctuation(x, ','))
        df['project_n_questions'] = df['project_tokens'].apply(lambda x: count_punctuation(x, '?'))
        df['project_n_exclamations'] = df['project_tokens'].apply(lambda x: count_punctuation(x, '!'))
        df['project_word_len'] = df['project_tokens'].apply(lambda x: np.mean([len(token) for token in x]))
        del (df['project_tokens'])
        del (df['project_essay_1'])
        del (df['project_essay_2'])
        del (df['project_essay_3'])
        del (df['project_essay_4'])
        return df

if __name__ == "__main__":

    train = pd.read_csv(TRAIN_FILE)
    if TOY is True:
        train = train.sample(frac=.1)
    print("Read Train")

    train = preprocess_df(train, workers=WORKERS)

    print("Complete Preprocessing")
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

    raw_X = train
    raw_y = train['project_is_approved'].values

    X_train, X_test, y_train, y_test = train_test_split(raw_X, raw_y, test_size=0.2)

    mapper = DataFrameMapper([
        (['teacher_number_of_previously_posted_projects'], StandardScaler()),
        (['project_grade_category'], LabelBinarizer()),
        (['student_word_count', 'project_word_count', 'student_n_periods', 'student_n_commas', 'student_n_exclamations', 'student_n_periods', 'project_n_periods', 'project_n_commas',
         'project_n_questions', 'project_n_exclamations'], PCA(1)),
        (['total', 'price', 'quantity'], PCA(1)),
        (['has_zero'], StandardScaler()),
        ('student_description',
         [TfidfVectorizer(use_idf=True, ngram_range=(1, 2), stop_words='english'),
          NMF(n_components=20)]),
        ('project_description',
         [TfidfVectorizer(use_idf=True, ngram_range=(1, 2), stop_words='english'),
          NMF(n_components=20)]),
    ], sparse=True)

    X_train = mapper.fit_transform(X_train)
    X_test = mapper.transform(X_test)

    print("Mapping Complete")
    
    X_trainR, y_trainR = X_train, y_train

    xgb_params = {'eta': 0.2,
                  'max_depth': 10,
                  'max_delta_step': 6,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'objective': 'binary:logistic',
                  'eval_metric': 'auc'
                  }

    d_train = xgb.DMatrix(X_trainR, y_trainR)
    d_test = xgb.DMatrix(X_test, y_test)

    watchlist = [(d_train, 'train'), (d_test, 'valid')]
    model_xgb = xgb.train(xgb_params, d_train, 1000, watchlist, verbose_eval=5, early_stopping_rounds=25)
    
    df_test = pd.read_csv(TEST_FILE)
    df_test = preprocess_df(df_test, WORKERS)
    X_test_actual = mapper.transform(df_test)
    X_test_actual = xgb.DMatrix(X_test_actual)
    y_pred_actual = model_xgb.predict(X_test_actual)
    my_submission = pd.DataFrame({'id': df_test.id, 'project_is_approved': y_pred_actual})
    my_submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


my_submission

