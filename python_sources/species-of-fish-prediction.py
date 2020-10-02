#!/usr/bin/env python
# coding: utf-8

# Usual actions

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **Import** libraries

# In[ ]:


# import others useful libraires
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib


# **Read** dataset

# In[ ]:


# open and read dataset file with pandas
data = pd.read_csv("/kaggle/input/fish-market/Fish.csv")
# some information about it
data.info()
data.head()


# **Prepare** Data

# In[ ]:


# preparing dataset for work
# create train and test things
X = data.drop(columns = ["Species"])
y = data.Species
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# create list with usefull cols
useful_features = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]


# Start** modeling**

# In[ ]:


# start working with logistic regression
model = LogisticRegression()
# fit
model.fit(X_train, y_train)


# In[ ]:


# prediction
model.predict(X_test)


# **Score** of prediction

# In[ ]:


# score
# score of train
print(model.score(X_train, y_train))
# score of test
print(model.score(X_test, y_test))

