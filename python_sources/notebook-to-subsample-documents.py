#!/usr/bin/env python
# coding: utf-8

# ## Starter Code ##
# This code is designed to provide a few much smaller basic files as well as a few files to get you started on the project.

# In[ ]:


The following loads a library (pandas) so that I can use it in other co


# In[ ]:


import pandas as pd

train_df = pd.DataFrame.from_csv('../input/clicks_train.csv', index_col = None)
display_group = train_df.groupby('display_id')
ad_group = train_df.groupby('ad_id')


# In[ ]:




