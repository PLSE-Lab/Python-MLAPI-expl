#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


spam_data = pd.read_csv('/kaggle/input/textdata/spam.csv')
spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(spam_data['text'],
                                                spam_data['target'],
                                                random_state=0)


# Percentage of document that are spam

# In[ ]:


len(spam_data[spam_data['target']==1])/len(spam_data['target'])*100


# Fiting the training data X_train using a Count Vectorizer

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vocab = CountVectorizer()    
vocab = vocab.fit(X_train).vocabulary_
    
#we want only the keys i.e. words.
vocab = [words for words in vocab.keys()]
    
#store the length in the seperate list.
len_vocab = [len(words) for words in vocab]
    
#use the index of the longest token.
vocab[np.argmax(len_vocab)]


# Fitting a multinomial Naive Bayes classifier model with smoothing alpha=0.1

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

cv = CountVectorizer().fit(X_train)
    
# Transform both X_train and X_test with the same CV object:
X_train_cv = cv.transform(X_train)
X_test_cv = cv.transform(X_test)
    
# Classifier for prediction:
clf = MultinomialNB(alpha=0.1)
clf.fit(X_train_cv, y_train)
preds_test = clf.predict(X_test_cv)
    
roc_auc_score(y_test, preds_test)


# Fitting and transforming the training data X_train using a Tfidf Vectorizer with default parameters.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer().fit(X_train)
feature_names = np.array(tfidf.get_feature_names())
    
X_train_tf = tfidf.transform(X_train)
    
max_tf_idfs = X_train_tf.max(0).toarray()[0] # Get largest tfidf values across all documents.
sorted_tf_idxs = max_tf_idfs.argsort() # Sorted indices
sorted_tf_idfs = max_tf_idfs[sorted_tf_idxs] # Sorted TFIDF values
    
# feature_names doesn't need to be sorted! You just access it with a list of sorted indices!
smallest_tf_idfs = pd.Series(sorted_tf_idfs[:20], index=feature_names[sorted_tf_idxs[:20]])                    
largest_tf_idfs = pd.Series(sorted_tf_idfs[-20:][::-1], index=feature_names[sorted_tf_idxs[-20:][::-1]])
    
(smallest_tf_idfs, largest_tf_idfs)


# Fitting and transforming the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than 3.
# 
# Then fitting a multinomial Naive Bayes classifier model with smoothing alpha=0.1 and compute the area under the curve (AUC) score using the transformed test data.

# In[ ]:


tf = TfidfVectorizer(min_df=3).fit(X_train)
X_train_tf = tf.transform(X_train)
X_test_tf = tf.transform(X_test)
clf = MultinomialNB(alpha=0.1)
clf.fit(X_train_tf, y_train)
pred = clf.predict(X_test_tf)
roc_auc_score(y_test, pred)


# The average length of documents (number of characters) for not spam and spam documents

# In[ ]:


len_spam = [len(x) for x in spam_data.loc[spam_data['target']==1, 'text']]
len_not_spam = [len(x) for x in spam_data.loc[spam_data['target']==0, 'text']]
(np.mean(len_not_spam), np.mean(len_spam))


# Combine new features into the training data:

# In[ ]:


from scipy.sparse import csr_matrix, hstack
def add_feature(X,feature_to_add):
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# Fitting and transforming the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than 5.
# 
# Using this document-term matrix and an additional feature, the length of document (number of characters), fitting a Support Vector Classification model with regularization C=10000. Then computing the area under the curve (AUC) score using the transformed test data.

# In[ ]:


from sklearn.svm import SVC

len_train = [len(x) for x in X_train]
len_test = [len(x) for x in X_test]
    
tf = TfidfVectorizer(min_df=5).fit(X_train)
X_train_tf = tf.transform(X_train)
X_test_tf = tf.transform(X_test)
    
X_train_tf = add_feature(X_train_tf, len_train)
X_test_tf = add_feature(X_test_tf, len_test)
    
clf = SVC(C=10000)
clf.fit(X_train_tf, y_train)
pred = clf.predict(X_test_tf)
    
roc_auc_score(y_test, pred)


# The average number of digits per document for not spam and spam documents.

# In[ ]:


dig_spam = [sum(char.isnumeric() for char in x) for x in spam_data.loc[spam_data['target']==1,'text']]
dig_not_spam = [sum(char.isnumeric() for char in x) for x in spam_data.loc[spam_data['target']==0,'text']]
(np.mean(dig_not_spam), np.mean(dig_spam))


# Fitting and transformig the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than 5 and using word n-grams from n=1 to n=3 (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# 
# the length of document (number of characters)
# number of digits per document
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.

# In[ ]:


from sklearn.linear_model import LogisticRegression

dig_train = [sum(char.isnumeric() for char in x) for x in X_train]
dig_test = [sum(char.isnumeric() for char in x) for x in X_test]
    
tf = TfidfVectorizer(min_df = 5, ngram_range = (1,3)).fit(X_train)
X_train_tf = tf.transform(X_train)
X_test_tf = tf.transform(X_test)
    
X_train_tf = add_feature(X_train_tf, dig_train)
X_test_tf = add_feature(X_test_tf, dig_test)
    
clf = LogisticRegression(C=100).fit(X_train_tf, y_train)
pred = clf.predict(X_test_tf)
    
roc_auc_score(y_test, pred)


# The average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents

# In[ ]:


(np.mean(spam_data.loc[spam_data['target']==0,'text'].str.count('\W')), 
np.mean(spam_data.loc[spam_data['target']==1,'text'].str.count('\W')))


# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than 5 and using character n-grams from n=2 to n=5.
# 
# To tell Count Vectorizer to use character n-grams pass in analyzer='char_wb' which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# 
# the length of document (number of characters)
# number of digits per document
# number of non-word characters (anything other than a letter, digit or underscore.)
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also find the 10 smallest and 10 largest coefficients from the model and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients: ['length_of_doc', 'digit_count', 'non_word_char_count']

# In[ ]:


len_train = [len(x) for x in X_train]
len_test = [len(x) for x in X_test]
dig_train = [sum(char.isnumeric() for char in x) for x in X_train]
dig_test = [sum(char.isnumeric() for char in x) for x in X_test]
    
# Not alpha numeric:
nan_train = X_train.str.count('\W')
nan_test = X_test.str.count('\W')
    
cv = CountVectorizer(min_df = 5, ngram_range=(2,5), analyzer='char_wb').fit(X_train)
X_train_cv = cv.transform(X_train)
X_test_cv = cv.transform(X_test)
    
X_train_cv = add_feature(X_train_cv, [len_train, dig_train, nan_train])
X_test_cv = add_feature(X_test_cv, [len_test, dig_test, nan_test])
    
clf = LogisticRegression(C=100).fit(X_train_cv, y_train)
pred = clf.predict(X_test_cv)
    
score = roc_auc_score(y_test, pred)
    
feature_names = np.array(cv.get_feature_names() + ['length_of_doc', 'digit_count', 'non_word_char_count'])
sorted_coef_index = clf.coef_[0].argsort()
small_coeffs = list(feature_names[sorted_coef_index[:10]])
large_coeffs = list(feature_names[sorted_coef_index[:-11:-1]])
    
(score, small_coeffs, large_coeffs)


# In[ ]:




