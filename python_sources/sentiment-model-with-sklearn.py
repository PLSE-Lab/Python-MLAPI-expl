# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
from time import time
import re, string
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB

from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer, TweetTokenizer

import matplotlib.pyplot as plt


class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return dict()

class TextCleaner(BaseEstimator, TransformerMixin):
    _numeric = re.compile('(\$)?\d+([\.,]\d+)*')# with decimals and so on figures
    _links = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')# links url
    _trans_table = str.maketrans({key: None for key in string.punctuation})
#    def __init__(self):
#        self.name='TextCleaner'
        
    def clean_text(self, tokens):
        for i, w in enumerate(tokens):
            if TextCleaner._numeric.match(w):
                tokens[i] = w.translate(TextCleaner._trans_table)
            elif TextCleaner._links.match(w):
                tokens[i] = '_LINK_'
            else:
                continue
        return tokens
            
    def transform(self, X):
        for i, item in enumerate(X):
            tokenized = item.split()
            X[i] = " ".join(self.clean_text(tokenized))

        return X
        
    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    def fit(self, X, y=None):
        self.transform(X)
        return self

def print_significant_features(pipeline=None, n=20):
    feature_names = pipeline.get_params()['vect'].get_feature_names()
    coefs=[]
    try:
        coefs = pipeline.get_params()['clf'].coef_
    except:
        coefs.append(pipeline.get_params()['clf'].feature_importances_)
    print("Total features: {}".format(len(coefs[0])))
    coefs_with_fns = sorted(zip(coefs[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_2, fn_2) in top:
        print("%.4f: %-16s" % (coef_2, str(fn_2)))

def plot_significant_features(pipeline=None, n=20):
    feature_names = pipeline.get_params()['vect'].get_feature_names()
    coefs=[]
    try:
        coefs = pipeline.get_params()['clf'].coef_
    except:
        coefs.append(pipeline.get_params()['clf'].feature_importances_)
    
    print("Total features: {}".format(len(coefs[0])))
    coefs_with_fns = sorted(zip(coefs[0], feature_names))
    top = coefs_with_fns[:-(n + 1):-1]
    
    y,X = zip(*top)

    plt.figure()
    plt.title("Top 20 most important features")
    plt.gcf().subplots_adjust(bottom=0.25)
    ax = plt.subplot(111)
    
    ax.bar(range(len(X)), y, color="r", align="center")
    ax.set_xticks(range(len(X)))
    ax.set_xlim(-1, len(X))
    ax.set_xticklabels(X,rotation='vertical')
    plt.savefig('sentiment_feature_importance.png')
    plt.close()

df = pd.read_csv('../input/Tweets.csv')

#p=df[df['text'].str.contains('0162389030167')]
#print([s for s in [t for t in p[:2]['text'].str.split('\s+')]])

#uncomment if you dont care about neutral tweets
#df = df.loc[df['airline_sentiment'].isin(['negative', 'positive'])]


X=df['text'].values

y=df['airline_sentiment']
#y=y.map({'negative':-1,'neutral':0, 'positive':1}).values #Score:0.7634365520206998

y=y.map({'negative':0,'neutral':1, 'positive':2}).values #Score:


stops = set(stopwords.words("english"))

pipe_params = {
    'clf__class_weight':'balanced'
    ,'clf__random_state':42
    ,'tfidf__norm': 'l2'
    ,'tfidf__use_idf': True
    ,'vect__max_df': .7
    ,'vect__max_features': 10000
    ,'vect__ngram_range': (1,2)
    ,'vect__strip_accents': 'unicode'
    ,'vect__stop_words': stops
}

pipeline = Pipeline([
    ('cleantxt', TextCleaner()),
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    #('clf', DecisionTreeClassifier()),#Score:0.6279615220025925
    ('clf', ExtraTreesClassifier(n_jobs=-1, n_estimators=500)),#Score:0.7634365520206998
    #('dense', DenseTransformer()),
    #('clf', GaussianNB())#Score:0.6259115628872863
])

pipeline.set_params(**pipe_params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
t0 = time()
#X_train = X_train[:3]
#y_train = y_train[:3]



pipeline.fit(X_train, y_train)
print("Training done in %0.3fs" % (time() - t0))
print()


scores = cross_val_score(pipeline, X_test, y_test)
score = scores.mean()
print("Score:{}".format(score))
#print_significant_features(pipeline=pipeline)
plot_significant_features(pipeline=pipeline)
print("Total done in %0.3fs" % (time() - t0))

