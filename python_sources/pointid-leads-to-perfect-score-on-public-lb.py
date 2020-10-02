#!/usr/bin/env python
# coding: utf-8

# Training a simple decision tree classifier on the "pointid" feature leads to 1 AUC on the public leaderboard. Explanation is in the last paragraphs of this notebook.

# # Load data

# In[ ]:


import cufflinks as cf
import numpy as np 
import pandas as pd 

RANDOM_STATE = 1234


# In[ ]:


local_path = "./data/"
kaggle_path = "/kaggle/input/killer-shrimp-invasion/"
example_submission_filename = "temperature_submission.csv"
train_filename = "train.csv"
test_filename = "test.csv"

base_path = kaggle_path

temperature_submission = pd.read_csv(base_path + example_submission_filename)
test = pd.read_csv(base_path + test_filename)
train = pd.read_csv(base_path + train_filename)


# In[ ]:


train.head()


# In[ ]:


test.head()


# # Clean data

# In[ ]:


train_fill_na = train.fillna(method='ffill')
test_fill_na = test.fillna(method='ffill')


# # Keep only pointid as feature

# In[ ]:


train_fill_na = train_fill_na[["pointid", "Presence"]]
test_fill_na = test_fill_na[["pointid"]]


# In[ ]:


train_fill_na.head()


# In[ ]:


test_fill_na.head()


# # Decision tree classifier CV --> ~1 AUC CV score

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


# In[ ]:


classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)

X_full = train_fill_na[["pointid"]]
y_full = train_fill_na["Presence"]

n_splits = 10
scores = cross_val_score(classifier, X_full, y_full,
                                  scoring='roc_auc',
                                  cv=n_splits)

scores.mean()


# # Train decision tree classifier on whole train data and submit --> 1 AUC leaderboard score

# In[ ]:


classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
classifier.fit(X_full, y_full)


# In[ ]:


predictions = classifier.predict(test_fill_na)


# In[ ]:


temperature_submission['Presence'] = predictions
temperature_submission.to_csv('with_pointid.csv', index=False)


# # Why?

# All the 50 records with positive presence in the training set have pointid >= 2917769 and all the records with negative presence have pointid < 2917769, thus the decision tree classifier learns to map positive presence to all the records with pointid >= 2917769. In the test set this rule applies, so we get the perfect score.

# In[ ]:


from sklearn import tree


# In[ ]:


tree.plot_tree(classifier)


# In[ ]:


train[train["Presence"] == 1].sort_values("pointid")["pointid"].values


# In[ ]:


temperature_submission[temperature_submission["Presence"] == 1].sort_values("pointid")["pointid"].values


# In[ ]:


temperature_submission[temperature_submission["pointid"] >= 2917768.5].sort_values("pointid")["pointid"].values

