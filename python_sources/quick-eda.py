#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# ## Load Dataset

# In[6]:


train = pd.read_csv("../input/train.csv")
resource = pd.read_csv('../input/resources.csv')
test = pd.read_csv("../input/test.csv")


# ### training set and test set overview

# In[12]:


print("# of example in train set %d" % len(train) )
print("# of example in test set %d" % len(test) )


# In[16]:


train.dtypes


# we have:
# >182,080 examples in train set 
# 
# >78,035  examples in test set 
# 
# > 6 categorical features:  **teacher_id,  features teacher_prefix, school_state,  project_grade_category, project_subject_categories, project_subject_subcategories**
# 
# > 5 text features:  **project_title, project_essay_1, project_essay_2, project_essay_3, project_essay_4, project_resource_summary**
# 
# > 1 numeric features: **teacher_number_of_previously_posted_projects**
# 
# > 1 timestamp features: **project_submitted_datetime**

# ## Take a look at training set

# In[17]:


train.sample(2)


# ## Take a look at resource features
# each project in training set and test set has zero or more resources

# In[5]:


resource.sample(5)


# In[ ]:




