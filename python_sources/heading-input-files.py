#!/usr/bin/env python
# coding: utf-8

# # Data Samples
# 
# Here, I show a quick example of loading in each of the input files and showing the first ten rows of each.

# In[ ]:


import pandas as pd

pd.read_csv("../input/app_events.csv").head(10)


# In[ ]:


pd.read_csv("../input/app_labels.csv").head(10)


# In[ ]:


pd.read_csv("../input/events.csv").head(10)


# In[ ]:


pd.read_csv("../input/gender_age_test.csv").head(10)


# In[ ]:


pd.read_csv("../input/gender_age_train.csv").head(10)


# In[ ]:


pd.read_csv("../input/label_categories.csv").head(10)


# In[ ]:


pd.read_csv("../input/phone_brand_device_model.csv").head(10)


# In[ ]:


pd.read_csv("../input/sample_submission.csv").head(10)

