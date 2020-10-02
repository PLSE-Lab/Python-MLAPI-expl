#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt # For plot configuration
import numpy as np              # For numerical operations
import pandas as pd             # For database management
import seaborn as sns           # For plotting data easily

import warnings
warnings.filterwarnings("ignore")

sns.set() 


# In[ ]:


dat = pd.read_csv('../input/data-on-depression/Ginzberg.csv')


# In[ ]:


n_rows, n_cols = dat.shape
print('The dataset has {} rows and {} columns.'.format(n_rows, n_cols))


# In[ ]:


dat.head()


# In[ ]:


columns = dat.columns
print('Columns names: {}.'.format(columns.tolist()))


# In[ ]:


sns.pairplot(data=dat, 
             kind='reg')


# In[ ]:


corr = dat.corr()
corr.style.background_gradient()
corr.style.background_gradient().set_precision(2)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sb
sb.set(style="darkgrid")

cols = ['simplicity', 'fatalism', 'depression']
sb.pairplot(dat[cols])


# In[ ]:


set1 = ['simplicity']
sb.lineplot(data=dat[set1], linewidth=2.5)


# In[ ]:


set2 = ['fatalism']
sb.lineplot(data=dat[set2], linewidth=2.5)


# In[ ]:


set3 = ['depression']
sb.lineplot(data=dat[set3], linewidth=2.5)

