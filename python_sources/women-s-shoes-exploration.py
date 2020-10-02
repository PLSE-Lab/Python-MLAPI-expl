#!/usr/bin/env python
# coding: utf-8

# ## A Quick Look at Women's Shoes Data ##
# 
# ### Contents ###
# 
#  - Shape and Dtypes
#  - Data Cleaning

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv('../input/7003_1.csv', low_memory=False)


# In[ ]:


data.shape


# In[ ]:


data.dtypes.value_counts()


# In[ ]:


col_unique_data = [(col, len(data[col].unique())) for col in data.columns]
col_unique_data = pd.DataFrame(col_unique_data, columns=['column', 'unique_len'])
col_unique_data.sort_values('unique_len')


# In[ ]:


data['prices.warranty'].unique()


# As you can see some things simply don't belong which moves us to the next part.
# 
# ### Cleaning ###
# For now I'm going to attempt to remove unwanted rows rather than attempt to shift them to their proper place assuming that unwanted rows do not take up a majority of the data.
# 
# To be continued....
