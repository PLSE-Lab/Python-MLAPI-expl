#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


sample = pd.read_csv('../input/sample_submission.csv', nrows=10)
sample.head()


# In[ ]:


train = pd.read_csv('../input/clicks_train.csv', nrows=10)
train.head()


# In[ ]:


test= pd.read_csv('../input/clicks_test.csv', nrows=10)
test.head()


# In[ ]:


events_head = pd.read_csv('../input/events.csv', nrows=10)
events_head.head()


# In[ ]:


promoted_content = pd.read_csv('../input/promoted_content.csv', nrows=10)
promoted_content.head()


# In[ ]:


page_views = pd.read_csv('../input/page_views_sample.csv', nrows=10)
page_views.head()


# <h1>documents</h1>

# In[ ]:


doc_category = pd.read_csv('../input/documents_categories.csv', nrows = 10)
doc_category.head()


# In[ ]:


doc_meta = pd.read_csv('../input/documents_meta.csv', nrows = 10)
doc_meta.head()


# In[ ]:


doc_entities = pd.read_csv('../input/documents_entities.csv',nrows = 10)
doc_entities.head()


# In[ ]:


doc_topics = pd.read_csv('../input/documents_topics.csv',nrows = 10)
doc_topics.head()


# In[ ]:




