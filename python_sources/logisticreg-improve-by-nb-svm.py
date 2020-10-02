#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import log_loss


# Inpire by this [kernel](http://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline)
# If you wanna read more - go to Jeremy Howard kernel
# 

# ## **Read data**

# In[ ]:


train = pd.read_csv('../input/train.csv.gz')
test = pd.read_csv('../input/test.csv.gz')


# In[ ]:


train.fillna('', inplace=True)
test.fillna('', inplace=True)


# In[ ]:


vectorizer = CountVectorizer()
labeler = LabelEncoder()
lb = LabelBinarizer()


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


X = vectorizer.fit_transform(train.name.values)
X_test = vectorizer.transform(test.name.values)


# In[ ]:


y = labeler.fit_transform(train.category)


# In[ ]:


labels = lb.fit_transform(y) # matrix with 0 and 1 labels


# In[ ]:


logist = LogisticRegression()
mnb = MultinomialNB()


# In[ ]:


gkf = list(GroupKFold(n_splits=3).split(X, y, train.check_id.values)) # spliter for train


# In[ ]:


get_ipython().run_cell_magic('time', '', 'score = cross_val_score(logist, X, y, scoring=\'neg_log_loss\', cv = gkf, n_jobs = -1)\nprint("%.3f +- %.4f" % (-np.mean(score), np.std(score)))')


# In[ ]:


print('Score for 1st split for LogisticRegression = %.3f' % (-score[0]))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'score = cross_val_score(mnb, X, y, scoring=\'neg_log_loss\', cv = gkf, n_jobs = -1)\nprint("%.3f +- %.4f" % (-np.mean(score), np.std(score)))')


# In[ ]:


print('Score for 1st split for MultinomialNB = %.3f' % (-score[0]))


# ## Split data for train and valid set

# 
# #### As example use 1st slit for NB - SVM  improvement

# In[ ]:


X_train, X_valid, y_train, y_valid = X[gkf[0][0]], X[gkf[0][1]], labels[gkf[0][0], :], labels[gkf[0][1],:]


# In[ ]:


X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# In[ ]:


preds = []
for idx, category in enumerate(labeler.classes_):
    #print (' Prediction for %s' % category)
    mnb.fit(X_train, y_train[:, idx])
    ratio = mnb.feature_log_prob_[1] - mnb.feature_log_prob_[0]
    logist.fit(X_train.multiply(ratio), y_train[:, idx])
    preds.append(logist.predict_proba(X_valid.multiply(ratio))[:,1])


# In[ ]:


print('Score after applying NB-SVM on data = %.3f' % (log_loss(y_valid, np.array(preds).T)))


# In[ ]:




