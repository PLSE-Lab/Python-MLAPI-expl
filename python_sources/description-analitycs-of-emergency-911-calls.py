#!/usr/bin/env python
# coding: utf-8

# > The english isn't my natal language, I speak in Spanish and
# > Portuguese, but I hope that you can to understand me.

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
df_call = pd.read_csv('../input/911.csv')


# In[ ]:


df_call['twp'].value_counts()
name_pol = [x for x in list(set(df_call['twp'])) if str(x) != 'nan']
x_twp=np.arange(len(df_call['twp'].value_counts()))
pd.DataFrame.transpose(pd.DataFrame(name_pol,x_twp))


# In[ ]:


plt.bar(x_twp,df_call['twp'].value_counts(),width=0.5)

