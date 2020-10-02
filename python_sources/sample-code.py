#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


import os
print(os.listdir("../input"))

from sklearn.metrics          import auc, roc_curve
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import accuracy_score, confusion_matrix
from sklearn.multiclass       import OneVsRestClassifier 


# In[2]:


def  mcauc(T,Y):
    d   = T.shape[1]
    m   = np.zeros(d)
    for ii in range(d):
        fpr, tpr, thresholds = roc_curve(T[:,ii], Y[:,ii], pos_label=1)
        m[ii] = auc(fpr, tpr)
    return np.mean(m)


# In[5]:


Xtr = pd.read_csv('train_data.csv',index_col=0).as_matrix()
Ttr = pd.read_csv('train_label.csv',index_col=0).as_matrix()
Xts = pd.read_csv('test_data.csv',index_col=0).as_matrix()


# In[4]:


dr = np.isnan(Xtr)
ds = np.isnan(Xts)
print np.sum(dr), np.sum(ds)
Xtr[dr] = 1
Xts[ds] = 1


# In[33]:


model_lr = OneVsRestClassifier(LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.1, 
    fit_intercept=True, intercept_scaling=1, class_weight='balanced', 
    random_state=None, solver='liblinear', max_iter=10, 
    multi_class='ovr', verbose=0, warm_start=False, n_jobs=1))

model_lr.fit(Xtr,Ttr)
Y = model_lr.predict_proba(Xtr)
print "logistic reg : " + str(mcauc(Ttr,Y))
Y = model_lr.predict_proba(Xts)


# In[34]:



Id = np.arange(Y.shape[0]).reshape(Y.shape[0],1).astype('int')
Y = np.concatenate([Id,Y],axis=1)
sample_submition = pd.DataFrame(Y,columns=['Id','C0','C1','C2','C3','C4','C5'])
sample_submition[['Id']] = sample_submition[['Id']].astype(int)
sample_submition.to_csv('sample_submition.csv',index_col = False)

