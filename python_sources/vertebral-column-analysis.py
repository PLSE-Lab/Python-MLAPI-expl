#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df2 = pd.read_csv('../input/vertebralcolumndataset/column_3C.csv')
df2.describe()


# In[ ]:


sns.pairplot(df2, hue="class", size=4, diag_kind="kde")


# In[ ]:





# # In progress 

# # Final
