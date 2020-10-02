#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv("../input/daily-inmates-in-custody.csv")


# In[ ]:


data.head()


# In[ ]:


data.columns


# # Look at the distribution of ages

# In[ ]:


plt.hist(
    data.loc[~data['AGE'].isna(),'AGE']
    , bins=list(range(16,96,5))
)
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# # Look at distribution of gang affiliation

# In[ ]:


gang_affiliated = data.SRG_FLG.map(dict(N=0,Y=1))
plt.hist(gang_affiliated)
plt.xlabel("Gang Affiliated")
plt.ylabel("Count")
plt.show()


# # Look at the ages of gangsters vs non-gangsters

# In[ ]:


plt.hist(
    [
        data.loc[data.SRG_FLG == 'Y', 'AGE']
        , data.loc[data.SRG_FLG == 'N', 'AGE']
    ]
    , bins=list(range(16,96,5))
)
plt.legend(['Ganster', 'Non-Gangster'])
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# In[ ]:




