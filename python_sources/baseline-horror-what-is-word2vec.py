#!/usr/bin/env python
# coding: utf-8

# <h1>Introduction<h1>
# 
# Hello, everyone! I am going to try to predict probabilities using the magic of distributional semantics, and I propose a baseline solution based on Logistic Regression and Word2Vec. I hope this notebook will be helpful, and I will highly appreciate any critique or feedback. Feel free to write your thoughts at the comments section!

# In[ ]:


import numpy as np
import pandas as pd 
from subprocess import check_output
from gensim.models import Word2Vec

from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

alpha_tokenizer = RegexpTokenizer('[A-Za-z]\w+')
lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')


# In[ ]:


from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

author_mapping = {'EAP':0, 'HPL':1, 'MWS':2}
y_train = train['author'].map(author_mapping).values


# <h1>Pre-process the data<h1>
# 
# In order to use Word2Vec, you need to pre-process the data. It's very simple: you just need to split sentences to words (**tokenization**), bring the words to their basic form (**lemmatization**), and remove some very common words like articles or prepositions (**stop-word removal**). I'm using RegexpTokenizer, WordNetLemmatizer and NLTK stop word list. You could start experimenting already at this step and try to extend the stop word list or to use another lemmatizer! It will be interesting to know what will happen!

# In[ ]:


data = [[lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(sent) if word.lower() not in stop] for sent in train.text.values]


# ## Some initial experiemnts with simple vectorizers
# 
# I will write more about it very soon.

# In[ ]:


vectorizers = [ ('3-gram TF-IDF Vectorizer on words', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                ('3-gram Count Vectorizer on words', CountVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                ('3-gram Hashing Vectorizer on words', HashingVectorizer(ngram_range=(1, 5), analyzer='word', binary=False)),
                ('TF-IDF + SVD', Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                                 ('svd', TruncatedSVD(n_components=150)),
                                ])),
                ('TF-IDF + SVD + Normalizer', Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                                 ('svd', TruncatedSVD(n_components=150)),
                                 ('norm', Normalizer()),
                                ]))
              ]


# In[ ]:


estimators = [(KNeighborsClassifier(n_neighbors=3), 'K-Nearest Neighbors', 'yellow'),
              (SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False), 'Support Vector Machine', 'red'),
              (LogisticRegression(tol=1e-8, penalty='l2', C=0.1), 'Logistic Regression', 'green'),
              (MultinomialNB(), 'Naive Bayes', 'magenta'),
              (RandomForestClassifier(n_estimators=10, criterion='gini'), 'Random Forest', 'gray'),
              (None, 'XGBoost', 'pink')
]


# In[ ]:


params = {}
params['objective'] = 'multi:softprob'
params['eta'] = 0.1
params['max_depth'] = 3
params['silent'] = 1
params['num_class'] = 3
params['eval_metric'] = 'mlogloss'
params['min_child_weight'] = 1
params['subsample'] = 0.8
params['colsample_bytree'] = 0.3
params['seed'] = 0


# In[ ]:


y_train, y_test = train_test_split(train, test_size=0.3)
y_train = y_train['author'].map(author_mapping).values
y_test = y_test['author'].map(author_mapping).values

def compare():
    for vectorizer in vectorizers:
        print(vectorizer[0] + '\n')
        X = vectorizer[1].fit_transform(train.text.values)
        X_train, X_test = train_test_split(X, test_size=0.3)
        for estimator in estimators:
            if estimator[1] == 'XGBoost': 
                xgtrain = xgb.DMatrix(X_train, y_train)
                xgtest = xgb.DMatrix(X_test)
                model = xgb.train(params=list(params.items()), dtrain=xgtrain,  num_boost_round=40)
                predictions = model.predict(xgtest, ntree_limit=model.best_ntree_limit).argmax(axis=1)
            else:
                estimator[0].fit(X_train, y_train)
                predictions = estimator[0].predict(X_test)
            print(accuracy_score(predictions, y_test), estimator[1])


# ## Baseline Solution on TF-IDF + SVD + XGBoost

# In[ ]:


train = pd.read_csv('../input/train.csv')
X = vectorizers[4][1].fit_transform(np.hstack((train.text.values, test.text.values)))
X_train, X_test = train_test_split(X, test_size=8392)
xgtrain = xgb.DMatrix(X_train, y_train)
xgtest = xgb.DMatrix(X_test)
model = xgb.train(params=list(params.items()), dtrain=xgtrain,  num_boost_round=40)
probs = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
author = pd.DataFrame(probs)
final = pd.DataFrame()
final['id'] = test.id
final['EAP'] = author[0]
final['HPL'] = author[1]
final['MWS'] = author[2]
final.to_csv('submission.csv', sep=',',index=False)


# <h1>Distributional semantics<h1>
# 
# **Distributional semantic models** are frameworks that can represent words of natural language through real-valued vectors of fixed dimensions (the so-called **word embeddings**). The word "distributional" here is a reference to a distributional hypothesis that says that word semantics is distributed along all of its contexts. Such models able to capture various functional or topical relations between words through words context for for each word observed in a given corpus. Predicting words given their contexts (like **continuous bag-of-words** (CBOW) works) and  predicting the contexts from the words (like **continuous skip-gram** (SG) works) are two possible options of capturing the context, and this is how the distributional semantic model Word2Vec works. In short, with skip gram, you can create a lot more training instances from limited amount of data. We will set paramater sg to 1. It defines the training algorithm, and if sg=1, skip-gram is employed (and CBOW is employed otherwise).
# 
# About some other parameters:
# *min_count *= ignore all words with total frequency lower than this.
# *size* is the dimensionality of the feature vectors.
# *window* is the maximum distance between the current and predicted word within a sentence.

# In[ ]:


NUM_FEATURES = 150

model = Word2Vec(data, min_count=3, size=NUM_FEATURES, window=5, sg=1, alpha=1e-4, workers=4)


# Now we have 10852 words in our model, and we could try to find most similar words for some examples. Let's try the word "raven".

# In[ ]:


len(model.wv.vocab)


# In[ ]:


model.most_similar('raven')


# <h1>Compositional distributional semantics<h1>
# 
# We are able to represent each word in a form of a vector, but how to represent the whole sentence? Well ,semantics of sentences and phrases can be also captured as a composition of the word embeddings -- for instance, through **compositional distributional semantics** (CDS). CDS is a nominal notion of a method of capturing semantics of composed linguistic units like sentences and phrases by composing the distributional representations of the words that these units contain. The semantics of a whole sentence can be represented as a composition of words embeddings of the words constituting the sentence. An averaged unordered composition (or an arithmetic mean) is a one of the most popular methods of capturing semantics of a sentence since it is an effective solution despite its simplicity. Since one could claim that word embeddings are the building blocks of compositional representation, and while it has been shown that semantic relations can be mapped to translations in the learned vector space, the claim could be made for sentence representations of the embeddings.

# In[ ]:


def get_feature_vec(tokens, num_features, model):
    featureVec = np.zeros(shape=(1, num_features), dtype='float32')
    missed = 0
    for word in tokens:
        try:
            featureVec = np.add(featureVec, model[word])
        except KeyError:
            missed += 1
            pass
    if len(tokens) - missed == 0:
        return np.zeros(shape=(num_features), dtype='float32')
    return np.divide(featureVec, len(tokens) - missed).squeeze()


# In[ ]:


vectors = []
for i in train.text.values:
    vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))


# <h1>Training the classifier<h1>
# 
# We are representing the labels of the authors in a form of numeric class labels, and then we are ready to train the classifier. I picked Logistic Regression, but you could use another one.

# In[ ]:


estimator = LogisticRegression(C=1)
estimator.fit(np.array(vectors), y_train);


# <h1>Making predictions<h1>
# 
# And we are ready to make predictions! We will use the ability of the classifier to predict the probabilities of given classes.

# In[ ]:


test_vectors = []
for i in test.text.values:
    test_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))


# In[ ]:


probs = estimator.predict_proba(test_vectors)


# <h1>Submission<h1>
# 
# One final step: make a dataframe to submit our results!

# In[ ]:


author = pd.DataFrame(probs)

final = pd.DataFrame()
final['id'] = test.id
final['EAP'] = author[0]
final['HPL'] = author[1]
final['MWS'] = author[2]
# final.to_csv('submission.csv', sep=',',index=False)


# That's all for now! Thanks for reading this notebook. I'm glad if it helped you to learn something new. This is a very first version of this small tutorial, and I'm working hard to make it better. I plan to introduce some other methods of word vectors composing and to try to use some syntax features
# 
# Witch-ing you a spook-tacular Halloween! Do not let ghouls and spooks to ruin your models, and don't fear the curse of dimensionality!
