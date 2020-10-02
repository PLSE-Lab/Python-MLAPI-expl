#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


database = pd.read_csv(os.path.join('../input', 'database.csv'))
database.head(1)


# In[ ]:


database.plot(kind="scatter", x="Longitude", y="Latitude", alpha=.2, s=database['Magnitude']*2, c=database['Depth'], figsize=(10,7), cmap=plt.get_cmap('jet'))


# In[ ]:


years = []
for d in database['Date']:
    x = d.split("/")
    if len(x) == 1:
        x = d.split("-")
    if len(x[0]) == 4:
        years += [x[0]]
    else:
        years += [x[2]]


# In[ ]:


database['Year'] = years
database['Year'].factorize()


# In[ ]:


database.plot(kind="scatter", x="Longitude", y="Latitude", alpha=.2, s=database['Magnitude'], c=database['Year'], figsize=(10,7), cmap=plt.get_cmap('jet'))


# In[ ]:




