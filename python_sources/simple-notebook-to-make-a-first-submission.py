#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Import Data

# In[ ]:


train = pd.read_csv("../input/learn-together/train.csv")
test = pd.read_csv("../input/learn-together/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.describe()


# Let's delete the Id column in the training set but store it for the test set before deleting

# In[ ]:


train = train.drop(["Id"], axis = 1)

test_ids = test["Id"]
test = test.drop(["Id"], axis = 1)


# # Model Training

# Let's use 80% of the Data for training, and 20% for validation. We'll then train a simple Random Forest Classifier with 100 trees

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(train.drop(['Cover_Type'], axis=1), train['Cover_Type'], test_size=0.2, random_state = 42)


# In[ ]:


X_train.shape, X_val.shape, y_train.shape, y_val.shape


# In[ ]:


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score


# In[ ]:


model.score(X_train, y_train)


# In[ ]:


predictions = model.predict(X_val)
accuracy_score(y_val, predictions)


# Our Model has a 100% accuracy on the training set and ~ 87% on the test set. A clear example of overfitting. But we won't get into that cause this notebook is to get started.

# # Predictions

# In[ ]:


test.head()


# In[ ]:


test_pred = model.predict(test)


# In[ ]:


# Save test predictions to file
output = pd.DataFrame({'ID': test_ids,
                       'Cover_Type': test_pred})
output.to_csv('submission.csv', index=False)


# # Future work
# 
# * Explorative Data Analysis to extract the most relevant features
# * Feature engineering
# * Cross-validation so we can use the entire training data
# * Grid-Search to find the optimal parameters for our classifier so we can fight overfitting
# * Try a different classifer. XgBoost for example (I suspect the winning solution will use an xgboost. highly recommended
# * Deep-learning ? hummm probably not. Overkill

# In[ ]:




