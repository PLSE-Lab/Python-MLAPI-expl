#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


result = pd.read_csv('../input/gruresult/kernel_submission.csv')


# In[ ]:


result.head()


# In[ ]:


result['price'] =result['price'].apply(abs)


# In[ ]:


result['test_id'] = result['test_id'].astype(int)


# In[ ]:


result.to_csv('submission_gru_1230.csv',index=False)


# In[ ]:


result['price'].describe()


# In[ ]:




