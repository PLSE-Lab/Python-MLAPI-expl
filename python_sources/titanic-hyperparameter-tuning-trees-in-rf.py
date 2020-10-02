#!/usr/bin/env python
# coding: utf-8

# ### In this notebook, we'll outline a process to figure out the optimal no of trees required in a Random Forest,
# ### without training the model again and again. I learnt this approach in the [Competitive Data Science](https://www.coursera.org/learn/competitive-data-science/home/welcome) course 
# ### on Coursera, and found it to be a pretty useful trick.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import fastai_structured as fs  ## For handling categorical variables

get_ipython().run_line_magic('matplotlib', 'inline')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


X = pd.read_csv("/kaggle/input/titanic/train.csv")
X.head().T


# #### Creating training and validation sets
# Handling categorical data using fastai v0.7 functions which are added as a utility script

# In[ ]:


fs.train_cats(X)  ## Converts strings to categorical variables
X, y, nas = fs.proc_df(X, 'Survived')
X_train, X_val, y_train, y_val = train_test_split(X, y)
print(X_train.shape)
print(X_val.shape)


# In[ ]:


rf = RandomForestClassifier(n_estimators=500, max_depth=3, n_jobs=-1)
rf.fit(X_train, y_train)


# #### Get predictions for each tree separately

# In[ ]:


predictions = []
for tree in rf.estimators_:
    predictions.append(tree.predict_proba(X_val)[None, :])


# In[ ]:


predictions = np.vstack(predictions)
cum_mean = np.cumsum(predictions, axis=0)/np.arange(1, predictions.shape[0] + 1)[:, None, None]


# #### Get accuracy scores for each of the n_estimators value

# In[ ]:


scores = []
for pred in cum_mean:
    scores.append(accuracy_score(y_val, np.argmax(pred, axis=1)))


# In[ ]:


plt.figure(figsize=(10, 6))
plt.plot(scores, linewidth=3)
plt.xlabel('num_trees')
plt.ylabel('accuracy');


# ### We see that around 150 trees are enough to get a good accuracy and 
# ### there is not really any payoff for adding new trees in the forest after that
