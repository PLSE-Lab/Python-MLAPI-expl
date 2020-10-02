#!/usr/bin/env python
# coding: utf-8

# *emphasized text*

# In[ ]:


import tflearn

data, labels = tflearn.data_utils.load_csv('../input/training.csv', target_column=0, categorical_labels=True, n_classes=10)

data.head()


# In[ ]:


import tflearn

data, labels = tflearn.data_utils.load_csv('../input/training.csv', target_column=0, categorical_labels=True, n_classes=10)

data.head()


# I'll try to explore the dataset of handwriten vectorized digits in this notebook. Any suggestions more than welcome!

# In[ ]:




