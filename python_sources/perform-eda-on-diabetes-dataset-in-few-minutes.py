#!/usr/bin/env python
# coding: utf-8

# ### Perform EDA In the Best Possible Way Using Pandas Profiling

# In[ ]:


import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport


# In[ ]:


from sklearn.datasets import load_diabetes


# In[ ]:


diab_data=load_diabetes()


# In[ ]:


df=pd.DataFrame(data=diab_data.data,columns=diab_data.feature_names)


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


### To Create the Simple report quickly
profile = ProfileReport(df, title='Pandas Profiling Report')


# In[ ]:


profile.to_widgets()


# In[ ]:


profile.to_file("output.html")

