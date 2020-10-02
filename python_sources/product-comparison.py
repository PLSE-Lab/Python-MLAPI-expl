#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.


# In[ ]:


complaints = pd.read_csv('../input/consumer_complaints.csv', low_memory=False)


# In[ ]:


complaints_by_prod = complaints.groupby(['product','sub_product'])['date_received'].count()


# In[ ]:


#This looks hideous, I'll revisit with a nicer plotting library
complaints_by_prod.unstack().plot.bar(stacked=True, legend=False)


# In[ ]:




