#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/train.csv")


# #### Head of Data

# In[ ]:


data.head(25)


# ### Class Distribution

# In[ ]:


data[['toxic', 'severe_toxic', 'obscene','threat', 'insult', 'identity_hate']].sum()


# In[ ]:


import sklearn


# In[ ]:


msk = np.random.rand(len(data)) < 0.8


# In[ ]:


train = data[msk]
test = data[~msk]


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train[['toxic', 'severe_toxic', 'obscene','threat', 'insult', 'identity_hate']].mean()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


count_vect = CountVectorizer()


# In[ ]:


X_train = count_vect.fit_transform(train[['comment_text']].as_matrix().reshape((-1, )))


# In[ ]:


X_train.shape


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[ ]:


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)


# In[ ]:


X_train_tfidf.shape


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


clf = MultinomialNB().fit(X_train_tfidf, train[['toxic']].as_matrix().reshape((-1,)))


# In[ ]:


X_test_tfidf = tfidf_transformer.transform(count_vect.transform(test[['comment_text']].as_matrix().reshape((-1,))))
X_test_tfidf.shape


# In[ ]:


y_ = clf.predict(X_test_tfidf)
y_prob = clf.predict_proba(X_test_tfidf)


# In[ ]:


y = test[['toxic']].as_matrix().reshape((-1,))
(y_ == y).astype('float').mean()


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# In[ ]:


print (classification_report(y, y_))
print(confusion_matrix(y, y_))
print(roc_auc_score(y, y_))
print(roc_auc_score(y, y_prob[:, 1]))


# In[ ]:


# Accuracy at predicting toxic language
# Sampling bias obviously is skewing the results
476/2552


# In[ ]:


# This is definitely hate speech
#print(test[['comment_text']][(y_ == 0) & (y == 1)].as_matrix()[0][0])


# In[ ]:


# try rebuilding classifier with weighted classes
y_train = train[['toxic']].as_matrix().reshape((-1,))
pos_cases = (y_train == 1)
num_pos_cases = pos_cases.sum()
idx_desc = np.argsort(-y_train) #sort indices high to low so positive cases are at the front
balance_samples = idx_desc[:2*num_pos_cases] #extract 2x the num pos cases so now 50/50 pos/neg
print(y_train[balance_samples].mean()) #confirm probability
y_train_bal = y_train[balance_samples]
X_train_tfidf_bal = X_train_tfidf[balance_samples]
print(y_train_bal.shape, X_train_tfidf_bal.shape)
#neg_cases = (y_train == 0)[:(pos_cases.sum())]
#clf = MultinomialNB(cl) 
#.fit(X_train_tfidf, y_train)


# In[ ]:


# Create classifier based on balanced classes
clf = MultinomialNB().fit(X_train_tfidf_bal, y_train_bal)
y_ = clf.predict(X_test_tfidf)
y_prob = clf.predict_proba(X_test_tfidf)
print (classification_report(y, y_))
print(confusion_matrix(y, y_))
print(roc_auc_score(y, y_prob[:, 1]))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# clf_rf = RandomForestClassifier(n_estimators=50)
# clf_rf.fit(X_train_tfidf_bal, y_train_bal)


# In[ ]:


# y_ = clf_rf.predict(X_test_tfidf)
# y_prob = clf_rf.predict_proba(X_test_tfidf)

# print (classification_report(y, y_))
# print(confusion_matrix(y, y_))
# print(roc_auc_score(y, y_prob[:,1]))


# In[ ]:




