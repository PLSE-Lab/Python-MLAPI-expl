#!/usr/bin/env python
# coding: utf-8

# ## The kernel basic on the kernel [Logistic Regression](https://www.kaggle.com/martin1234567890/logistic-regression)

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 


# In[ ]:


submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")
train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

labels = train.pop('target')
labels = labels.values
train_id = train.pop("id")
test_id = test.pop("id")


# In[ ]:


train.head(5)


# In[ ]:


data = pd.concat([train, test])
data["ord_5a"] = data["ord_5"].str[0]
data["ord_5b"] = data["ord_5"].str[1]
data.drop(["bin_0", "ord_5"], axis=1, inplace=True)


# In[ ]:


columns = [i for i in data.columns]

dummies = pd.get_dummies(data,
                         columns=columns,
                         drop_first=True,
                         sparse=True)

del data


# In[ ]:


train = dummies.iloc[:train.shape[0], :]
test = dummies.iloc[train.shape[0]:, :]

del dummies


# In[ ]:


train = train.fillna(0)
train.head(5)


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train = train.sparse.to_coo().tocsr()
test = test.sparse.to_coo().tocsr()


# In[ ]:


lr_cv = LogisticRegressionCV(Cs=7,
                        solver="lbfgs",
                        tol=0.0002,
                        max_iter=10000,
                        cv=5)

lr_cv.fit(train, labels)

lr_cv_pred = lr_cv.predict_proba(train)[:, 1]
score = roc_auc_score(labels, lr_cv_pred)

print("score: ", score)


# In[ ]:


submission["id"] = test_id
submission["target"] = lr_cv.predict_proba(test)[:, 1]


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission_lr_cv.csv", index=False)

