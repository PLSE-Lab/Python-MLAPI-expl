#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
mpl.style.use('ggplot')


# In[ ]:


df = pd.read_csv('../input/NationalNames.csv')


# In[ ]:


def plotname(name,sex):
    df_named=df[(df.Name==name) & (df.Gender==sex)]
    plt.figure(figsize=(12,8))
    plt.plot(df_named.Year,df_named.Count,'g-')
    plt.title('%s name variation over time'%name)
    plt.ylabel('counts')
    plt.xticks(df_named.Year,rotation='vertical')
    plt.xlim([1990,2014])    
    plt.show()    


# In[ ]:


characters = [('Rachel','F'),('Monica','M'),('Phoebe','F'),('Ross','F'),('Chandler','F'),('Joey','F')]


# In[ ]:


for character in characters:
    plotname(character[0],character[1])


# In[ ]:




