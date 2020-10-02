#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/Salaries.csv",low_memory=False)


# In[ ]:


plt.hist(df['TotalPay'], bins=20, range=[0, 300000])


# In[ ]:


df['TotalPay'].mean()


# In[ ]:




