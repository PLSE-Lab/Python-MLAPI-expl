#!/usr/bin/env python
# coding: utf-8

# # Molecular properties EDA
# 
# **Hi everyone. So, here I want to share my data exploration of the training dataset.
# Firstly, lets define what is scalar coupling constant:**
# > The coupling constant is defined as nJA,X, where n is the number of chemical bonds between the two coupling atoms A and X. The coupling constant is independent of the field strength, and has a plus or minus prefix and it is mutual to the coupled atoms (nJA,X=nJX,A).
# 
# [From: NMR Spectroscopy in Pharmaceutical Analysis, 2008](https://www.sciencedirect.com/topics/neuroscience/coupling-constant)  
# **Now let's define what kind of problems it helps us to solve:**
# * *Helps in assignment of molecular fragments;*
# * *Conformation and relative stereochemistry;*
# * *J coupling is heavily used in the liquid state NMR (Nuclear magnetic resonance).*  
# **Now we will start with importing the needed libraries:**

# In[ ]:


import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import ast, json

from datetime import datetime
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Great! Let's load out training dataset as DataFrame.**

# In[ ]:


train = pd.read_csv("../input/train.csv")
train.head()


# ### Define min, max, delta values
# Our first task in this EDA (Exploratory Data analysis) is to find the max- and min scalar coupling constant values grouped by type.  
# We have only 8 categorical variables in our Type field:  
# <font color=blue>*['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']* </font>  
#  Also we are going to calculate the delta using the equation below:  
# <center>*delta = max - min*
# 

# In[ ]:


max_scalar_coupling_constant = train.sort_values(by="scalar_coupling_constant", ascending=False).groupby('type').head(1)
min_scalar_coupling_constant = train.sort_values(by="scalar_coupling_constant", ascending=True).groupby('type').head(1)

min_sc = min_scalar_coupling_constant[['type', 'scalar_coupling_constant']]
max_sc = max_scalar_coupling_constant[['type', 'scalar_coupling_constant']]

sc_type = max_sc.join(min_sc.set_index('type'), on='type', lsuffix='_max', rsuffix='_min')
sc_type = sc_type.assign(delta=sc_type['scalar_coupling_constant_max']-sc_type['scalar_coupling_constant_min'])
sc_type.rename(columns = {"scalar_coupling_constant_min": "min", 
                     "scalar_coupling_constant_max":"max"}, 
                                 inplace = True) 
sc_type


# In[ ]:


sc_type.set_index('type')      .reindex(sc_type.set_index('type').sum().sort_values().index, axis=1)      .T.plot(kind='bar', stacked=False,
              colormap=ListedColormap(sns.diverging_palette(145, 280, s=85, l=25, n=7)), 
              figsize=(18,9))
plt.xticks(rotation='horizontal')
plt.tick_params(labelsize=20)
plt.show()


# **Now I am going to extract number of chemical bonds between the two coupling atoms A and X (J) in the 'type' field.
# Performing that will help to find out correlationships later in the dataset,**

# In[ ]:


sc_type['j_number'] = sc_type['type'].astype(str).str[0]
sc_type.groupby(['j_number']).mean()
sc_type.groupby(
    ['j_number']
).agg(
    {
        'delta': ['mean'],
    }
)


# **Inresting! We can make such conclusions:  
# The lower number of couplings we have, the higher is the scalar coupling constant.  
# Let's explore the training dataset**

# In[ ]:


train['j_number'] = train['type'].astype(str).str[0]
j_number_group = train.groupby(
    ['j_number']
).agg(
    {
        # find the min, max, and sum of the duration column
        'scalar_coupling_constant': ['mean', 'max', 'min'],
         # find the number of network type entries
        'type': ["count"],
        # min, first, and number of unique dates per group
        'atom_index_1': ['nunique'],
        'atom_index_0': ['nunique']
    }
)
j_number_group


# In[ ]:


j_number_group['scalar_coupling_constant'].plot(kind='bar',stacked=False,figsize=(8,8))
plt.xticks(rotation='horizontal')
plt.show()


# In[ ]:


train['atom_0'] = train['type'].astype(str).str[2]
train['atom_1'] = train['type'].astype(str).str[3]
train.head()


# In[ ]:


train.groupby(
    ['atom_0', 'atom_1']
).agg(
    {
        'atom_index_0': ['nunique'],
        'atom_index_1': ['nunique'],
        'j_number': ['nunique']
    }
)


# That helps us conclude that we have 'H' is always the first element, and that we have only 2 possible  
# combinations of 'HH': *2JHH, 3JHH*. 
# Now we will find how many atom types are in our dataset in percentage. That will also allow us to find the most frequent one in the training set.

# In[ ]:


n_two_hh = train['type'][train['type']=='2JHH'].count()
n_three_hh = train['type'][train['type']=='3JHH'].count()
two_j_hh_mean = train['scalar_coupling_constant'][train['type']=='3JHH'].mean()
three_j_hh_mean = train['scalar_coupling_constant'][train['type']=='2JHH'].mean()
all_data = train['type'].count()
results = {}
for i in train['type'].unique():
    count = train['type'][train['type']==i].count()
    perc = (count/all_data)*100
    results[i] = perc
plt.bar(range(len(results)), list(results.values()), align='center')
plt.xticks(range(len(results)), list(results.keys()))
plt.figsize=(18,9)
plt.show()

