#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load data

# In[8]:


d = pd.read_csv('../input/Data_Cortex_Nuclear.csv')


# ## Data pre-processing

# In[9]:


# take a peek
d.head()


# In[10]:


# turn MouseID into a true multi-index
multi = pd.MultiIndex.from_tuples( [ tuple(s.split('_')) for s in d['MouseID'] ], names=('mouse_id','replicate_id') )
d = d.set_index(multi)


# In[16]:


# pivot data frame to facilitate facet plotting
e = pd.DataFrame(d.stack())
e.index.names = ['mouse_id','replicate_id','protein']
e.reset_index(inplace=True)
e = e.rename( {0:'value'}, axis='columns' )
e['replicate_id'] = e['replicate_id'].astype(int)


# In[17]:


# only keep protein expression columns
e = e[ ['_N' in s for s in e['protein']] ]


# In[18]:


# summarize per mouse data (for reference)
d.drop([ c for c in d.columns if '_N' in c ],axis=1).groupby( level=0 ).head(1)


# In[19]:


# add log(expression).  This pulls out more information from the low expressors
import math
e['log_ex'] = [ math.log(a+0.1) for a in e['value'] ]


# ## Plot protein expression levels by replicate ID for the first 6 control mice (all same treatment)

# In[26]:


i = e['mouse_id'].isin(('309','311','320','321','322','3415'))
g = sns.FacetGrid(e[i], col="protein", hue='mouse_id', col_wrap=5, size=2.5)
g = g.map(plt.plot, "replicate_id", 'log_ex', marker=".").add_legend()


# Notes:
# - There's clearly a per-mouse effect.  Some mice have systematically high/low protein levels.  Suggests that normalizing by mouse id is warranted.
# - There's clearly a per-replicate effect.  There is a large subset of proteins where the expression level decreases or increases as a function of the replicate number.
# - There's clearly a periodicity within the per-replicate effect.  It looks like various protein curves follow a periodicity-3 pattern.  We could call these triads a "replicate batch".
# - Something went systematically wrong for the last three replicates of mouse 3415.  This kind of observation would suggest that there's also a mouse_id * replicate_batch_id effect.
# - Overall this brings serious questions to the utility of these replicates.  The SOM analysis paper and the blurb at kaggle say that the replicates form independent measurements of the same protein response, but all of these observations make it clear that this is far from true.  Without more information it's entirely unclear how to reduce the 15 replicates into a single measurement value.  Our best hope is that the replicate effects are reproduced across all mice, and if we analyze replicates as separate experiments, we can recover some statistical power.

# ## Plot representative protein levels for every mouse across replication points

# In[27]:


i = e['protein'].isin( ('ELK_N','ADARB1_N','pP70S6_N','ERK_N','pELK_N'))
g = sns.FacetGrid(e[i], col="mouse_id", hue='protein', col_wrap=5, size=2.5)
g = g.map(plt.plot, "replicate_id", 'value', marker=".").add_legend()


# Notes:
# - Every mouse repeats the same story that replicate_id is a dominant effect in the observed expression level.
# - The replicate effects are mostly consistent across mice; this would permit a treatment where we analyze each
# replicate as a separate experiment.  However, while the majority of mice exhibit the period-3 pattern, some mice
# don't show this and others have a phase offset in the period-3 behavior (e.g. 3497).
# - While not as powerful as a full mixed-effects model, we might be able to proceed by considering each replicate as
# a separate experiment and requiring our putative correlations to be significant across a
# majority (or more) of the replicates.

# In[ ]:




