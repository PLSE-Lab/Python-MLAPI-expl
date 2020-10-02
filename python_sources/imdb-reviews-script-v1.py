# Loading libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm_notebook
from collections import Counter
import string

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
import lightgbm as lgb

import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])
import en_core_web_sm
nlp_ents = en_core_web_sm.load()

PATH = '../input/aclimdb/aclImdb/'

# Reading data
def read_review(train_test, neg_pos, label):
    df = []
    file_path = PATH + train_test + '/' + neg_pos + '/'
    for file in tqdm_notebook(os.listdir(file_path)):
        with open(file_path + file, 'r') as f:
            df.append(f.read().replace('<br />',''))
    df = pd.DataFrame(df, columns=['review'])
    df['label'] = label
    
    return df

X_train = pd.concat([read_review('train', 'neg', 0), read_review('train', 'pos', 1)])
X_test = pd.concat([read_review('test', 'neg', 0), read_review('test', 'pos', 1)])
X_full = pd.concat([X_train,X_test])

# Custom features: Entities
ents_list = []
for review in tqdm_notebook(X_full['review']):
    ents_list.append(Counter([token.label for token in nlp_ents(review).ents]))
ents_list = pd.DataFrame.from_records(ents_list).fillna(0)
ents_list.columns = ['entity_%s' %s for s in ents_list.columns]
X_full.reset_index(inplace=True, drop=True)
X_full = pd.concat([X_full, ents_list], axis=1)

# Lemmatization
X_full_lem = []
for review in tqdm_notebook(X_full['review'].values):
    X_full_lem.append(" ".join([token.lemma_ for token in nlp(review)]))

# Custom features: Rates
X_full_rates = {}
for i,key in enumerate(['rate_%s' % s for s in range(11)]):
    X_full_rates[key] = []
    for review in X_full['review'].values:
        X_full_rates[key].append(review.count(str(i)+'/10'))
X_full_rates = pd.DataFrame(X_full_rates)

# Vectorization
vect = CountVectorizer(ngram_range=(1,3), max_features=100000)
X_train_vect = vect.fit_transform(X_full_lem[:25000])
X_test_vect = vect.transform(X_full_lem[25000:])

# Custom features: Countings
def custom_features(reviews, df_vect, df_rates, df):
    df_custom = hstack([
        df_vect,
        pd.DataFrame([len(review) for review in reviews]),
        pd.DataFrame([len(review.split()) for review in reviews]),
        pd.DataFrame([len([word for word in review.split() if word.istitle()]) for review in reviews]),
        pd.DataFrame([len([word for word in review.split() if word.isupper()]) for review in reviews]),
        pd.DataFrame([len([punct for punct in review if punct in string.punctuation]) for review in reviews]),
        df_rates,
        df.iloc[:,2:]
    ])
    return df_custom
X_train_custom = custom_features(X_full['review'].values[:25000], X_train_vect, X_full_rates.iloc[:25000,:], X_full.iloc[:25000,:])
X_test_custom = custom_features(X_full['review'].values[25000:], X_test_vect, X_full_rates.iloc[25000:,:], X_full.iloc[25000:,:])

# Logistic Regression
logit_clf = LogisticRegression(C=0.1, penalty='l2', random_state=19)
logit_clf.fit(X_train_custom, X_train['label'].values)
logit_pred = logit_clf.predict(X_test_custom)
logit_pred_proba = logit_clf.predict_proba(X_test_custom)

# LightGBM
lgb_params = {
    'objective':'binary',
    'metric':'binary_logloss',
    'lambda_l2':1,
    'colsample_bytree':0.95,
    'learning_rate':0.1,
    'max_depth':6,
    'num_leaves':85
}
X_train_lgb = lgb.Dataset(X_train_custom.astype('float32'), X_train['label'].values)
lgb_clf = lgb.train(lgb_params, X_train_lgb, num_boost_round=1000)
lgb_pred = lgb_clf.predict(X_test_custom.astype('float32'))

# Combining
test_pred = np.where((lgb_pred * 0.5 + logit_pred_proba[:,1] * 0.5) > 0.5, 1, 0)
acc = accuracy_score(X_test['label'].values, test_pred)
acc = pd.DataFrame({'accuracy':[acc]})
acc.to_csv('accuracy.csv', index=False)







