#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sub_df = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "for index, row in train_df.iterrows():\n    train_df.at[index,'len'] = len(row['text'])\n    \nfor index, row in test_df.iterrows():\n    test_df.at[index,'len'] = len(row['ciphertext'])")


# In[ ]:


train_df.head()


# In[ ]:


scaler = MinMaxScaler()
train_len_scaled = scaler.fit_transform(train_df['len'].values.reshape(-1, 1))
train_df['normalized_len'] = train_len_scaled


# In[ ]:


train_df.head()


# In[ ]:


test_len_scaled = scaler.fit_transform(test_df['len'].values.reshape(-1, 1))
test_df['normalized_len'] = test_len_scaled


# In[ ]:


test_df.head()


# In[ ]:


test_df.plot(x='len', y='difficulty')


# In[ ]:


train_df = train_df.sort_values(by=['normalized_len']).reset_index(drop=True).reset_index(drop=True)
test_df = test_df.sort_values(by=['normalized_len']).reset_index(drop=True)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


test_df['index'] = train_df['index']


# In[ ]:


test_df.head()


# In[ ]:


test_df = test_df.sort_values(by=['difficulty']).reset_index(drop=True)


# In[ ]:


test_df.head()


# In[ ]:


sub_df.head()


# In[ ]:


sub_df.update(test_df)


# In[ ]:


sub_df.head()


# In[ ]:


sub_df.to_csv('submission.csv', index=False)

