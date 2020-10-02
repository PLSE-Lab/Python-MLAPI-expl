#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

test_data = pd.read_csv('../input/test.csv')


# In[ ]:


predictions = np.zeros((len(test_data)))

for i in range(0, len(predictions)):
    predictions[i] = np.random.uniform(0, 1)


# In[ ]:


ids = test_data['Id'].values


# In[ ]:


submission = pd.DataFrame(np.transpose(np.array([list(ids), list(predictions)])))


# In[ ]:


submission.columns = ['Id', 'winPlacePerc']


# In[ ]:


submission['Id'] = np.int32(submission['Id'])


# In[ ]:


submission.to_csv('PUBG_random_preds.csv', index=False)


# In[ ]:




