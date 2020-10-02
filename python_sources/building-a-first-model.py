#!/usr/bin/env python
# coding: utf-8

# # Read the data
# 
# Read both the transactions and the identity datasets:

# In[ ]:


import pandas as pd

train_transaction = pd.read_csv("../input/train_transaction.csv")
train_identity = pd.read_csv("../input/train_identity.csv")
train_transaction.head()


# The identity dataset looks like:

# In[ ]:


train_identity.head()


# Joining the two datasets on key $\mathrm{TransactionID}$:

# In[ ]:


train = train_transaction.join(train_identity, on="TransactionID", lsuffix="_leftid")
train.head()


# In[ ]:


temp = pd.DataFrame({
    "Columns": train.columns,
    "Types": train.dtypes
})
print(temp)


# Only columns without missing values are considered for a first model:

# In[ ]:


column_not_contains_missing = train.isna().sum() == 0
train.loc[:, train.columns[column_not_contains_missing]].head()


# Dropping the columns containing the IDs and getting the dummy encoded variables:

# In[ ]:


df = train.loc[:, train.columns[column_not_contains_missing]].drop(["TransactionID_leftid", "TransactionDT"], axis=1)

Y = df["isFraud"].copy()
X = pd.get_dummies(df.drop(["isFraud"], axis=1)).copy()
X.head()


# In[ ]:


print(X.shape)


# The fraction of frauds is the following:

# In[ ]:


import numpy as np

np.mean(Y)


# Deleting the unused datasets:

# In[ ]:


import gc

del train_transaction, train_identity, train
gc.collect()


# # Train models

# In[ ]:


from sklearn.model_selection import cross_val_score

cv=9


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(n_estimators=100)
scores_rf = cross_val_score(model, X, Y, cv=cv, n_jobs=3, scoring="roc_auc")
print(np.mean(scores_rf), "+/-", np.std(scores_rf))


# In[ ]:


from xgboost import XGBClassifier


model = XGBClassifier()
scores_xgb = cross_val_score(model, X, Y, cv=cv, n_jobs=3, scoring="roc_auc")
print(np.mean(scores_xgb), "+/-", np.std(scores_xgb))


# In[ ]:


from lightgbm import LGBMClassifier


model = LGBMClassifier()
scores_gbm = cross_val_score(model, X, Y, cv=cv, n_jobs=3, scoring="roc_auc")
print(np.mean(scores_gbm), "+/-", np.std(scores_gbm))


# In[ ]:


from catboost import CatBoostClassifier


X2 = (df.drop(["isFraud"], axis=1)).copy()

model = CatBoostClassifier(cat_features=["ProductCD"])
scores_cat = cross_val_score(model, X2, Y, cv=cv, n_jobs=3, scoring="roc_auc")
print(np.mean(scores_cat), "+/-", np.std(scores_cat))


# In[ ]:


classifier = ["Random forest"] * cv + ["Xgboost"] * cv + ["Lightgbm"] * cv + ["Catboost"] * cv
performance = pd.DataFrame({
    "classifier": classifier,
    "scores": list(scores_rf) + list(scores_xgb) + list(scores_gbm) + list(scores_cat)
})


# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.boxplot(y="classifier", x="scores", data=performance).set(xlabel='', ylabel='')


# Considering the model training time and the performances:
#     
# | **Classifier** | **Training time** | **Performance** |
# |  :- |    :-: | :-: |
# | Random Forest  | 6m 59s | $0.857 \pm 0.016$ |
# | Xgboost | 3m 12s | $0.848 \pm 0.011$ |
# | Lightgbm| 26.8s | $0.882 \pm 0.013$ |
# | Catboost| 39m 41s | $0.879 \pm 0.016$ |
# 
# The next iterations for designing features will be done only on the Lightgbm classifier.
