#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os; print(os.listdir("../input"))
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_palette(sns.color_palette('tab20', 20))


# In[ ]:


plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['font.size'] = 12
C = ['#3D0553', '#4D798C', '#7DC170', '#F7E642']


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head(3)


# ### **Distribution of each feature of train dataset**

# In[ ]:


get_ipython().run_cell_magic('time', '', 'for i in range(20):\n    for j in range(10*i, 10*(i+1)):\n        col = \'var_{}\'.format(j)\n        sns.distplot(train[col], label=col)\n    plt.legend()\n    plt.xlabel("value")\n    plt.show()')


# The distributions of some features are so different from each other.

# ### **Comparison of the distributions of train and test dataset**

# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(200):\n    col = 'var_{}'.format(i)\n    sns.distplot(train[col], label=col, color=C[0])\n    sns.distplot(test[col], label=col, color=C[3])\n    plt.legend()\n    plt.xlabel(col)\n    plt.show()")


# ***Two distributions of each feature are almost same, which probably contributes to the closeness of cv and lb. ***
# 
# (However, some of them are different just a little.)

# In[ ]:




