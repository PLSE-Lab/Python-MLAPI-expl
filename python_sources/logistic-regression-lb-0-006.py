#!/usr/bin/env python
# coding: utf-8

# Hi all, this is my first Kaggle kernel. Today we're going to use logistic regression.

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


train_data = pd.read_csv('../input/train.csv')


# Let's see how big the data is:

# In[ ]:


len(train_data)


# Now we're ready to do some training!

# In[ ]:


import sklearn.linear_model
model = sklearn.linear_model.LogisticRegression()
X = train_data.drop('target', axis = 1).as_matrix()
Y = train_data.as_matrix(columns = ['target']).flatten()
model.fit(X, Y)


# Let's see how this performs.

# In[ ]:


test_data = pd.read_csv('../input/test.csv')
predictions = pd.DataFrame(columns = ['id', 'target'])
predictions['id'] = test_data['id']
XP = test_data.as_matrix()
predictions['target'] = model.predict(XP)
predictions.to_csv('submission.csv', index = False)


# This model predicts 0 for everything, and scores -0.006 on the leaderboard. Better luck next time :)
