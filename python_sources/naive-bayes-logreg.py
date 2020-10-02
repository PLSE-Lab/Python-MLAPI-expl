#!/usr/bin/env python
# coding: utf-8

# # This Notebook is not aimed at high f1 score.
# # It's just a demonstration of how the Deterministic Naive Bayes Algorithm can be combined with Logistic regression to get decent results.

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


# In[ ]:


data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')


# In[ ]:


len(data[data['target']==1]); len(data)


# In[ ]:


data=data.to_numpy()


# In[ ]:


x_trn,y_trn = data[:,1:-1][:4500],data[:,-1][:4500]
x_vld,y_vld = data[:,1:-1][4500:],data[:,-1][4500:]


# In[ ]:


len(x_trn), len(x_vld)


# In[ ]:


x_trn[0]


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
veczr = CountVectorizer(ngram_range=(1,3))


# In[ ]:


trn_term = veczr.fit_transform(x_trn[:,2])
vld_term = veczr.transform(x_vld[:,2])


# In[ ]:


vocab = veczr.get_feature_names(); vocab[:100], len(vocab)


# In[ ]:


trn_term.shape


# In[ ]:


y_trn=y_trn.astype('int64')
y_vld=y_vld.astype('int64')


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as f1
m=LogisticRegression(C=1e8,dual=True,solver='liblinear',max_iter=400, random_state=1)
m.fit(trn_term,y_trn)
preds = m.predict(vld_term)
(preds==y_vld).mean(), f1(y_vld,preds)


# dual = True, when the data is **wider than taller.**

# In[ ]:


m=LogisticRegression(C=10,dual=True,solver='liblinear',max_iter=400, random_state=1)
m.fit(trn_term,y_trn)
preds = m.predict(vld_term)
(preds==y_vld).mean(), f1(y_vld,preds)


# In[ ]:


p = trn_term[y_trn==1].sum(0)+1
q = trn_term[y_trn==0].sum(0)+1

r = np.log(((p/p.sum())/(q/q.sum())))


# In[ ]:


m=LogisticRegression(C=0.1,dual=True,solver='liblinear',max_iter=300, random_state=1)
m.fit(trn_term.multiply(r),y_trn)
preds = m.predict(vld_term.multiply(r))
(preds==y_vld).mean(), f1(y_vld,preds)


# In[ ]:


x = veczr.fit_transform(data[:,3])
y = data[:,-1]


# In[ ]:


y


# In[ ]:


p = x[y==1].sum(0)+1
q = x[y==0].sum(0)+1
p,q


# In[ ]:


r = np.log((p/p.sum())/(q/q.sum()))
test = veczr.transform(test_data["text"])
test_data["text"]


# Multiplying the training data sparse matrix X by the Naive Bayes Matrix r, regularizes the data more efficiently.
# Submission1 implements Naive Bayes whereas Submission2 is our good old Logistic Regression.

# In[ ]:


m=LogisticRegression(C=0.1,dual=True,solver='liblinear',max_iter=500, random_state=69)
m.fit(x.multiply(r),y.astype('int64'))
submission["target"] = m.predict(test.multiply(r))
submission.to_csv("submission1.csv", index=False)


# In[ ]:


m=LogisticRegression(C=1e8,dual=True,solver='liblinear',max_iter=600, random_state=69)
m.fit(x,y.astype('int64'))
submission["target"] = m.predict(test)
submission.to_csv("submission2.csv", index=False)


# In[ ]:


# kaggle competitions submit -c nlp-getting-started -f submission.csv -m "Message"


# In[ ]:




