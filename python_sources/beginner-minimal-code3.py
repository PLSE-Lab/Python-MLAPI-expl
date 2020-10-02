#!/usr/bin/env python
# coding: utf-8

# # Beginner Code
# 
# This is a minimal beginner code using random forest
# It uses a validation set to evaluate the model, and makes
# its final prediction using the entire training dataset
# 
# By the way, I am looking for team members for a beginner friendly team.
# 
# Feel free to contact me if you are interested.

# In[ ]:


import numpy as np
import os
import pandas as pd

from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split


# In[ ]:


# Load the data
DATA_FOLDER = '/kaggle/input/learn-together'
df_train = pd.read_csv(f'{DATA_FOLDER}/train.csv', index_col='Id')
df_test = pd.read_csv(f'{DATA_FOLDER}/test.csv', index_col='Id')

X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_test = df_test


# In[ ]:


# df_train.head().T


# In[ ]:


# df_train.describe().T


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Train the model on the test dataset
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(X_train, y_train)

# Evaluate the model with the validation dataset
y_pred = rf.predict(X_val)
mae = mean_absolute_error(y_pred, y_val)
acc = accuracy_score(y_pred, y_val)
print(f'Mean Absolute Error = {mae}')
print(f'Accuracy Score = {acc}')


# In[ ]:


# Now train the final model on the entire dataset
rf.fit(X, y)
y_pred = rf.predict(X_test)


# In[ ]:


# Create a submission file from the final predictions
submission = pd.read_csv('../input/learn-together/sample_submission.csv', index_col='Id')
assert np.all(submission.index == X_test.index)
submission.iloc[:, -1] = y_pred
submission.to_csv('submission.csv')


# In[ ]:


get_ipython().system('head -5 submission.csv')


# In[ ]:


get_ipython().system('head -5 ../input/learn-together/sample_submission.csv')


# In[ ]:




