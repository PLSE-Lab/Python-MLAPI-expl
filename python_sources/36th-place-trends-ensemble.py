#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tqdm import tqdm


# From https://www.kaggle.com/aerdem4/rapids-svm-on-trends-neuroimaging/output

# In[ ]:


svr = pd.read_csv('/kaggle/input/rapids-svm-on-trends-neuroimaging/submission.csv')

svr


# In[ ]:


nn = pd.read_csv('/kaggle/input/trends-augmented-tabular/submission_10fold.csv')

nn


# In[ ]:


submission = nn.append(svr).groupby('Id').mean().reset_index()

submission


# In[ ]:


submission = submission.set_index('Id')
submission


# In[ ]:


df_1590 = pd.read_csv("/kaggle/input/fork-of-fork-of-kernel19ee18bcfe/submission.csv").set_index('Id')


# In[ ]:


df_1587 = pd.read_csv("/kaggle/input/trends-ensemble/submission_svr_nn.csv").set_index('Id')

df_pri_1588 = pd.read_csv("../input/trends-priyanshu-meta/submission.csv").set_index('Id')


# In[ ]:


submission = .5*submission + .5*df_1590

submission


# In[ ]:


submission = .4*submission + .6*df_1587 

submission = .6*submission + .4*df_pri_1588 

submission


# In[ ]:


submission['temp'] = submission.index

age = [ val for val in submission.temp.values if 'age' in val]
counter = 0

for index, row in tqdm(submission.iterrows(), total = submission.shape[0]):
    counter +=1

    if index in age:
        submission.loc[submission.temp == row.temp, 'Predicted'] = row.Predicted * 1.01
    


# In[ ]:


submission = submission.drop(['temp'], axis=1)
submission.head()


# In[ ]:


submission.to_csv("submission_svr_nn.csv")


# In[ ]:


submission.head()

