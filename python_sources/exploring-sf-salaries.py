#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


# In[ ]:


salaries=pd.read_csv("../input/Salaries.csv")


# In[ ]:


salaries.columns


# In[ ]:


salaries.hist('TotalPay', by = 'Year', sharex = True, sharey = True)


# In[ ]:


salaries.Status.value_counts()


# In[ ]:


salaries.JobTitle.value_counts()[:10]

