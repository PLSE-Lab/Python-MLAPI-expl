#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
import os


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")
train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

labels = train.pop('target')
train_id = train.pop("id")
test_id = test.pop("id")


# In[ ]:


train.head()


# In[ ]:


train.columns


# In[ ]:


labels


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


sns.countplot(labels)
plt.title("labels counts")
plt.show()


# In[ ]:


labels = labels.values


# In[ ]:


data = pd.concat([train, test])


# In[ ]:


data["ord_5a"] = data["ord_5"].str[0]
data["ord_5b"] = data["ord_5"].str[1]


# In[ ]:


data["ord_5a"]


# In[ ]:


data["ord_5b"]


# In[ ]:


data['bin_0'].value_counts()


# In[ ]:


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


print(train.shape)
print(test.shape)


# In[ ]:


train = train.sparse.to_coo().tocsr()
test = test.sparse.to_coo().tocsr()

train = train.astype("float32")
test = test.astype("float32")


# In[ ]:


lr = LogisticRegression()

lr.fit(train, labels)

lr_pred = lr.predict_proba(train)[:, 1]
score = roc_auc_score(labels, lr_pred)

print("score: ", score)


# In[ ]:


submission["id"] = test_id
submission["target"] = lr.predict_proba(test)[:, 1]


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv", index=False)


# In[ ]:




