#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# ## Load Data

# In[ ]:


train = pd.read_csv('../input/learn-together/train.csv')
test = pd.read_csv('../input/learn-together/test.csv')


# ## Analyze the Data

# In[ ]:


train.head()


# In[ ]:


# check for missing values
print(train.isna().sum())


# ## Prepare Train and Test Data

# In[ ]:


y_train_copy = train['Cover_Type']
x_train_copy = train.drop(['Id', 'Cover_Type'], axis=1)
x_test = test.drop(['Id'], axis=1)

print(x_train_copy.shape)


# In[ ]:


# create train and validation set
x_train, x_val, y_train, y_val = train_test_split(x_train_copy, y_train_copy, train_size=0.8)


# In[ ]:


print(x_train.shape)
print(x_val.shape)


# In[ ]:


np.random.seed(42)


# ## Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=500)


# In[ ]:


clf = clf.fit(x_train, y_train)


# In[ ]:


y_preds = clf.predict(x_val)
print(accuracy_score(y_val, y_preds))


# In[ ]:


predictions = clf.predict(x_test)


# ## Submission

# In[ ]:


submission_df = pd.DataFrame({'ID': test['Id'],
                       'Cover_Type': predictions})
submission_df.to_csv('submission.csv', index=False)


# In[ ]:




