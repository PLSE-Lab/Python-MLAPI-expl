#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow-data-validation')


# In[ ]:


import tensorflow_data_validation as tfdv


# In[ ]:


get_ipython().system('pip show tensorflow_data_validation')


# In[ ]:


get_ipython().run_line_magic('ls', '../input')


# In[ ]:


train_stats = tfdv.generate_statistics_from_csv("../input/train.csv")
tfdv.visualize_statistics(train_stats)


# In[ ]:


test_stats = tfdv.generate_statistics_from_csv("../input/test.csv")
tfdv.visualize_statistics(test_stats)


# In[ ]:




