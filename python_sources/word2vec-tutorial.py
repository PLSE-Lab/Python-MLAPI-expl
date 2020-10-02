#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Importing Dataset

# In[ ]:


import pandas as pd
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
train = pd.read_csv("../input/nlp-getting-started/train.csv")


# In[ ]:


train.head()


# # Steps
# The problems seems to be of three steps:
# * Preprocessing of words
# * Word2Vec training on our dataset
# * Prediction using vector outputted by Word2Vec.

# # Step 1
# Text Preprocessing

# In[ ]:


import string
import re
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text



# In[ ]:


train['text'] = train['text'].apply(lambda x: clean_text(x))
test['text'] = test['text'].apply(lambda x: clean_text(x))

# Let's take a look at the updated text
train['text'].head()


# # Step 2 : Training Word2Vec on our dataset from scratch.

# Well there are two methods of implementation for Word2Vec on a given dataset.
# * Either train from scratch.
# * Or use google pretrained word2vec model.
# 

# In[ ]:


from gensim.models import Word2Vec, KeyedVectors
import nltk


# In[ ]:


# Tokenizing the training and the test set
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))
test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))
train['text'].head()


# In[ ]:


# Lets remove the stopwords as it does not seems of any meaning


# In[ ]:


from nltk.corpus import stopwords 
def remove_stopwords(text):
    """
    Removing stopwords belonging to english language
    
    """
    words = [w for w in text if w not in stopwords.words('english')]
    return words


# In[ ]:


train['text'] = train['text'].apply(lambda x : remove_stopwords(x))
test['text'] = test['text'].apply(lambda x : remove_stopwords(x))
train.head(10)


# let us concatenate both test and train column to get a larger corpus containing larger no of words.

# In[ ]:


test['target'] = 0


# In[ ]:


def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

train['text'] = train['text'].apply(lambda x : combine_text(x))
test['text'] = test['text'].apply(lambda x : combine_text(x))
train.head()


# In[ ]:


# Now we will make a corpus of words to start word2vec training.


# In[ ]:


df = pd.concat([train,test])


# In[ ]:


corpus = df['text'].values


# In[ ]:


corpus


# In[ ]:


Corpus_list = [nltk.word_tokenize(title) for title in corpus]


# In[ ]:


Corpus_list


# In[ ]:


model = Word2Vec(Corpus_list,min_count=1,size = 100)


# In[ ]:


model.most_similar('death')


# We can also used google pretrained saved model on newspaper as no of trained words much larger corpus and would give more accurate results then this small dataset

# Importing the pretrained vectors using this link
# 
# https://www.kaggle.com/umbertogriffo/googles-trained-word2vec-model-in-python

# In[ ]:


get_ipython().system('pwd')


# In[ ]:


path = "../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin"


# In[ ]:


import gensim
model = gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)


# Lets play around with this 300 sized vector space for all words.

# In[ ]:


w = model["hello"]
print(len(w))


# So we can see their quite some diffrences between the pretrained vector on our dataset and that on google newspaper dataset.

# In[ ]:


print(w)


# If you are very much intrested in this than follow this link
# 
# https://code.google.com/archive/p/word2vec/

# Now to represent the word we must convert the whole word to something small value of single numerical.So right now covert the 300 vector word to a single value by simple averaging defined by the class below.
# Also it seems that the pretrained vector is much better than the that trained on our small dataset.So we will carry on with this only.

# In[ ]:


class MeanEmbeddingVectorizer(object):

    def __init__(self, word_model):
        self.word_model = word_model
        self.vector_size = word_model.wv.vector_size

    def fit(self):  # comply with scikit-learn transformer requirement
        return self

    def transform(self, docs):  # comply with scikit-learn transformer requirement
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector

    def word_average(self, sent):
        """
        Compute average word vector for a single doc/sentence.


        :param sent: list of sentence tokens
        :return:
            mean: float of averaging word vectors
        """
        mean = []
        for word in sent:
            if word in self.word_model.wv.vocab:
                mean.append(self.word_model.wv.get_vector(word))

        if not mean:  # empty words
            # If a text is empty, return a vector of zeros.
            #logging.warning("cannot compute average owing to no vector for {}".format(sent))
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean


    def word_average_list(self, docs):
        """
        Compute average word vector for multiple docs, where docs had been tokenized.

        :param docs: list of sentence in list of separated tokens
        :return:
            array of average word vector in shape (len(docs),)
        """
        return np.vstack([self.word_average(sent) for sent in docs])


# In case you want to use different technique than averaging we can use tfidf using this technique.

# In[ ]:


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


# In[ ]:


mean_vec_tr = MeanEmbeddingVectorizer(model)
doc_vec = mean_vec_tr.transform(Corpus_list)


# In[ ]:


print('Shape of word-mean doc2vec...')
display(doc_vec.shape)


# In[ ]:


Corpus_train = train['text'].values


# In[ ]:


train_corpus = [nltk.word_tokenize(title) for title in Corpus_train]
doc_vec_1 = mean_vec_tr.transform(train_corpus)


# In[ ]:


len(train_corpus)


# In[ ]:


print('Shape of word-mean doc2vec...')
display(doc_vec_1.shape)


# In[ ]:


Corpus_test = test['text'].values
test_corpus = [nltk.word_tokenize(title) for title in Corpus_test]
doc_vec_2 = mean_vec_tr.transform(test_corpus)
print('Shape of word-mean doc2vec...')
display(doc_vec_2.shape)


# In[ ]:


X = doc_vec_1
y = train['target']


# # Step 3 : Training the model and predicting test set.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import model_selection
clf = LogisticRegression(C=1.0)
scores = model_selection.cross_val_score(clf,X,y, cv=5, scoring="f1")
scores


# In[ ]:


clf.fit(X,y)


# In[ ]:


X_vec_test = doc_vec_2


# In[ ]:


sample_submission_1 = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
sample_submission_1["target"] = clf.predict(X_vec_test)
sample_submission_1.to_csv("submission.csv", index=False)


# In[ ]:


# Using Advance Algorithms


# In[ ]:


from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline


# In[ ]:


from sklearn.model_selection import StratifiedKFold


# In[ ]:


# xgb_model = XGBClassifier()

# #brute force scan for all parameters, here are the tricks
# #usually max_depth is 6,7,8
# #learning rate is around 0.05, but small changes may make big diff
# #tuning min_child_weight subsample colsample_bytree can have 
# #much fun of fighting against overfit 
# #n_estimators is how many round of boosting
# #finally, ensemble xgboost with multiple seeds may reduce variance
# parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#               'objective':['binary:logistic'],
#               'learning_rate': [0.05,0.01,0.1], #so called `eta` value
#               'max_depth': [6,7,8,10],
#               'min_child_weight': [11],
#               'silent': [1],
#               'subsample': [0.8],
#               'colsample_bytree': [0.7,0.6,.5],
#               'n_estimators': [100,1000], #number of trees, change it to 1000 for better results
#               'missing':[-999],
#               'seed': [1337]}


# clf = GridSearchCV(xgb_model, parameters, n_jobs=5,  
#                    scoring='roc_auc',
#                    verbose=2, refit=True)

# clf.fit(X,y)

# #trust your CV!


# In[ ]:


# print("Best parameters set found on development set:")
# print()
# print(clf.best_params_)


# In[ ]:


clf = XGBClassifier(colsample_bytree=0.7, learning_rate= 0.05, max_depth= 8,
                    min_child_weight=11, missing= -999, n_estimators= 1000,
                    nthread= 4, objective='binary:logistic', seed=1337, silent=1, subsample=0.8)
scores = model_selection.cross_val_score(clf,X,y, cv=5, scoring="f1")
scores


# In[ ]:


clf.fit(X,y)


# In[ ]:


sample_submission_1 = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
sample_submission_1["target"] = clf.predict(X_vec_test)
sample_submission_1.to_csv("submission_3.csv", index=False)


# In[ ]:




