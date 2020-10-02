#!/usr/bin/env python
# coding: utf-8

# # Categorical Feature Encoding Challenge II

# In[ ]:


import numpy as np 
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler


# Load the training data.

# In[ ]:


train_df = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")
train_df.head()


# Load the test data.

# In[ ]:


test_df = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")
test_df.head()


# Concatenate training and test data.
# 
# Afterwards convert the categorical data into one hot encoded columns.
# 
# 
# Note: The columns "nom_5","nom_6","nom_7","nom_8","nom_9" have a very high cardinality and are unpractical for one hot encoding.
# That's why I left them out for now.

# In[ ]:


total_df = train_df.append(test_df,sort=False).drop(columns=["target","id"])
X_cat = pd.get_dummies(total_df.drop(columns=["nom_5","nom_6","nom_7","nom_8","nom_9"])).fillna(0).astype("uint8")


# In[ ]:


X_cat.head()


# Split the transformed train and test data again.

# In[ ]:


X_train = X_cat[:600000]
X_test = X_cat[600000:]

y_train = train_df["target"]


# Scale the data.

# In[ ]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Get the crossvalidation score to test the model.

# In[ ]:


clf = LogisticRegression(solver="saga",max_iter=200,random_state=0)
print("crossvalidation score:",cross_val_score(clf,X_train,y_train,cv=3,scoring='roc_auc').mean())


# Train the classifier, predict the labels and create a submission.

# In[ ]:


clf.fit(X_train,y_train)
submission = pd.DataFrame()
submission["id"] = test_df["id"]
submission["target"] = clf.predict_proba(X_test)[:,1]
pd.DataFrame(submission).to_csv("submission.csv",index=False)
print(pd.read_csv("submission.csv"))

