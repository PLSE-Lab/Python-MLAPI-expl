#!/usr/bin/env python
# coding: utf-8

# ## Baseline Kernel for WebClub Recruitment Test 2018

# ### Importing required packages

# In[ ]:


import os
print((os.listdir('../input/')))


# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# ### Reading the Train and Test Set

# In[ ]:


df_train = pd.read_csv('../input/web-club-recruitment-2018/train.csv')
df_test = pd.read_csv('../input/web-club-recruitment-2018/test.csv')


# ### Visualizing the Training Set

# In[ ]:


df_train.head()


# ### Separating the features and the labels

# In[ ]:


train_X = df_train.loc[:, 'X1':'X23']
train_y = df_train.loc[:, 'Y']


# ### Initializing Classifier

# In[ ]:


rf = RandomForestClassifier(n_estimators=50, random_state=123)


# ### Training Classifier

# In[ ]:


rf.fit(train_X, train_y)


# ### Calculating predictions for the test set

# In[ ]:


df_test = df_test.loc[:, 'X1':'X23']
pred = rf.predict_proba(df_test)


# ### Writing the results to a file

# In[ ]:


result = pd.DataFrame(pred[:,1])
result.index.name = 'id'
result.columns = ['predicted_val']
result.to_csv('output.csv', index=True)


# In[ ]:




