#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting
import seaborn as sns # fancier plotting
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


price = pd.read_csv("../input/DJIA_table.csv")


# In[ ]:


y= price['Close']
x= price['Date']
plt.scatter(x,y)

