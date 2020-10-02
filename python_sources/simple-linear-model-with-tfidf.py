#!/usr/bin/env python
# coding: utf-8

# Simple Linear Model with term-frequency-inverse-document-frequency

# In[ ]:


import numpy  as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy  as sp

from sklearn.feature_extraction.text import TfidfVectorizer

import sklearn.linear_model as lm
import sklearn.model_selection as ms


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


test= pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


# create labels. drop useless columns

y = train.is_duplicate.values

train['question1'] = train['question1'].astype(str)
train['question2'] = train['question2'].astype(str)  
    
test['question1']  = test['question1'].astype(str)
test['question2']  = test['question2'].astype(str)


# In[ ]:


# term-frequency-inverse-document-frequency

tfv = TfidfVectorizer(min_df=2,  max_features=None, strip_accents='unicode',  
      analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), 
      use_idf=1,smooth_idf=1,sublinear_tf=1)
            
# Fit TFIDF
    
tfv.fit(pd.concat([train['question1'],train['question2']]))
   
tr1 = tfv.transform(train['question1']) 
tr2 = tfv.transform(train['question2'])
    
ts1 = tfv.transform(test['question1']) 
ts2 = tfv.transform(test['question2'])


# In[ ]:


X = sp.sparse.hstack([tr1,tr2])
Z = sp.sparse.hstack([ts1,ts2])

print (X.shape)
print (Z.shape)
print (y.shape)


# In[ ]:


model = lm.LogisticRegression(C=1, class_weight=None, dual=True, fit_intercept=True,
        intercept_scaling=1.0, max_iter=1000, multi_class='ovr', n_jobs=1,
        penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
        verbose=1, warm_start=False)


# In[ ]:


def rmse_cv(model):
    rmse= ms.cross_val_score(model, X, y, scoring="neg_log_loss", cv = 10)
    return(rmse)


# In[ ]:


rmse_cv(model).mean()


# In[ ]:


model.fit(X,y)


# In[ ]:


#Prediction

p_test = model.predict_proba(Z)[:,1]

sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('linear_submission.csv', index=False)

