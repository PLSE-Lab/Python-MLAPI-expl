#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import scipy
from sklearn.linear_model import LogisticRegression
import optuna

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Credits to @Ants / Notebook : https://www.kaggle.com/superant/oh-my-cat

# # Load DataSet

# In[ ]:


train=pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv', index_col='id')
test=pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv', index_col='id')
submission=pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv', index_col='id')
Ytrain=train['target']
train=train[list(test)]
all_data=pd.concat((train, test))
print(train.shape, test.shape, all_data.shape)


# In[ ]:


encoded=pd.get_dummies(all_data, columns=all_data.columns, sparse=True)
encoded=encoded.sparse.to_coo()
encoded=encoded.tocsr()


# In[ ]:


Xtrain=encoded[:len(train)]
Xtest=encoded[len(train):]


# In[ ]:


kf=StratifiedKFold(n_splits=10)


# In[ ]:


def objective(trial):
    C=trial.suggest_loguniform('C', 10e-10, 10)
    model=LogisticRegression(C=C, class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)
    score=-cross_val_score(model, Xtrain, Ytrain, cv=kf, scoring='roc_auc').mean()
    return score
study=optuna.create_study()


# In[ ]:


#study.optimize(objective, n_trials=50)


# In[ ]:


#print(study.best_params)
#print(-study.best_value)
#params=study.best_params


# Trial Resulted in Value : C = 0.09536298444122952

# In[ ]:


model=LogisticRegression(C=0.09536298444122952, class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)
model.fit(Xtrain, Ytrain)
predictions=model.predict_proba(Xtest)[:,1]
submission['target']=predictions
submission.to_csv('Result.csv')
submission.head()

