#!/usr/bin/env python
# coding: utf-8

# # A Haunted (random) Forest is, of course, the obvious solution to this spooky contest

# In[ ]:


# Always the Same 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# ### The Data

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# The target variable: type of creature (of the night)

# In[ ]:


train_outcome = train_df["type"]
train_df.drop(["type"], axis=1, inplace=True)


# Encoding the categorical data: the color of the apparition (with the complete set (train and test)) 

# In[ ]:


conjunto = pd.concat([train_df[["id", "color"]], test_df[["id", "color"]]])
conjunto_encoded = pd.get_dummies(conjunto, columns=["color"])
train_df = train_df.merge(conjunto_encoded, on="id", how="left")
test_df = test_df.merge(conjunto_encoded, on="id", how="left")
train_df.drop(["color"], axis=1, inplace=True)
test_df.drop(["color"], axis=1, inplace=True)


# Separating the IDs (And we don't really need the IDs of the training set)

# In[ ]:


train_id = train_df[["id"]]
test_id = test_df[["id"]]
train_df.drop(["id"], axis=1, inplace=True)
test_df.drop(["id"], axis=1, inplace=True)


# ### The Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# First, let's train a small(ish) haunted forest to get some metrics 

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(train_df, train_outcome, test_size=0.2)
forest = RandomForestClassifier(n_estimators=200, n_jobs=4)
forest.fit(X_train, y_train)
y_pred_val = forest.predict(X_val)


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_val, y_pred_val))
print("Haunting accuracy: {:.1%}".format(accuracy_score(y_val, y_pred_val)))


# Now, we'll train the Haunted Random Forest with the complete training set...

# In[ ]:


forest = RandomForestClassifier(n_estimators=500, n_jobs=4)
forest.fit(train_df, train_outcome)
y_pred = forest.predict(test_df)


# ... and we'll predict of the type of creature haunting the forest

# In[ ]:


results = pd.read_csv("../input/sample_submission.csv")
results["type"] = y_pred


# Finally, we'll save the results to haunt the leaderboard

# In[ ]:


results.to_csv("submission.csv", index=False)


# ### *0.71267 score on the leaderboard! Booo!*
