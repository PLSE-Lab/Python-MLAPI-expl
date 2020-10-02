#!/usr/bin/env python
# coding: utf-8

# ## Introduction (This is a work in progress, expect updates)

# #### This is an interesting concept, showing the results of the private with-held dataset and the public leaderboard.
# ####  Personally, I like the idea of Kernel only competitions. 

# ### Without further ado, let's take a look at what we're dealing with here:

# In[ ]:


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


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape


# ### We've got 257 delightfully named (but obscure) columns and one target column to predict 

# In[ ]:


train.describe()


# #### Looks like the columns have already been normalized as the means are close to 0 and the std is not far off from 1. Next let's see how the variables correlate with each other.  

# In[ ]:


trainWihoutId  = train.drop(['id'],axis=1)


# In[ ]:


trainWihoutId.head()


# In[ ]:


##  courtesy:https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
values = trainWihoutId.corr().unstack().sort_values(ascending=False).drop_duplicates()


# #### Let's see what the top 10 correlations are:

# In[ ]:


print(values[1:11])


# #### We can see a bunch of weak-moderate correlations. The correlation between the target and 
# #### dorky-turquoise-maltese-important could be useful (I wonder if the name signifies anything). 

# ##### Next, Let's build a simple Linear model using sci-kit and see how it does (To be completed)
