#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer


# ## Load datasets

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# ## Prepare data

# In[ ]:


Y_train = train['species']
id_train = train['id']
X_train = train.drop(['species', 'id'], axis=1)
classes = np.unique(Y_train)
Y_train_bin = LabelBinarizer().fit_transform(Y_train)

id_test = test['id']
X_test = test.drop(['id'], axis=1)


# In[ ]:


lb = LabelBinarizer()
lb.fit(Y_train)
print(lb.classes_)


# In[ ]:


"Number of unique classes: {0}".format(len(classes))


# ## Run XGBoost and save results

# In[ ]:


xb = xgb.XGBClassifier(n_estimators=500, objective='multi:softprob')
probas = xb.fit(X_train, Y_train).predict_proba(X_test)


# In[ ]:


result = pd.DataFrame(index=id_test.values, columns=xb.classes_, data=probas)
result.to_csv("./xgb_benchmark.csv", index_label='id')


# In[ ]:


get_ipython().system(' head -n 2 xgb_benchmark.csv')


# In[ ]:


cross_val_score(xgb.XGBClassifier(n_estimators=100), X_train, Y_train, cv=5)


# In[ ]:




