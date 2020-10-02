#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv('/kaggle/input/love-generator/Love  generator data.csv')
df.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.lmplot('X','Y',data=df,fit_reg=False,scatter_kws={"marker": "D","s": 500}) 


# In[ ]:




