#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


# No missing data, that's a relief. 
data = pd.read_csv("../input/train.csv")
data.info()


# In[ ]:


# Let's clean up some data
data = pd.get_dummies(data, columns=["color"])


# In[ ]:


class_encoder = LabelEncoder()
data["type"] = class_encoder.fit_transform(data["type"])


# In[ ]:


sns.pairplot(data[["bone_length", "type"]])


# In[ ]:


class_labels = data["type"]
del data["type"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data, class_labels)


# In[ ]:


lr = LogisticRegression(C=0.1, penalty='l1')
forest = RandomForestClassifier(criterion="entropy", n_estimators=10, random_state=1, n_jobs=2)


# In[ ]:


lr.fit(X_train, y_train)
forest.fit(X_train, y_train)


# In[ ]:


print("Logistic Regression: \nTraining accuracy: {}\nTesting Accuracy: {}".format(lr.score(X_train, y_train), lr.score(X_test, y_test)))
print("*" * 10)
print("Random Forests: \nTraining Accuracy: {}\nTesting Accuracy: {}".format(forest.score(X_train, y_train), forest.score(X_test, y_test)))


# In[ ]:


# Not the best results, lots of overfitting in Random Forests. 
submission = pd.read_csv("../input/test.csv")
submission = pd.get_dummies(submission, columns=["color"])


# In[ ]:


labels = data.columns[1:]
importances = forest.feature_importances_
indices = 


# In[ ]:


predictions = [int(forest.predict(row.reshape(1, -1))) for row in submission.values]


# In[ ]:



predictions = class_encoder.inverse_transform(predictions)


# In[ ]:


final_submission = pd.read_csv("../input/test.csv")
final_submission["type"] = predictions


# In[ ]:


final_submission.head()


# In[ ]:


x =final_submission[["id", "type"]]


# In[ ]:


x.to_csv("predictions.csv")

