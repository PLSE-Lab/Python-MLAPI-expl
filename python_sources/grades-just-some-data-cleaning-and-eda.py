#!/usr/bin/env python
# coding: utf-8

# Just found this 'Dataset' and decided to practice my data cleaning skills :)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/mygrades/GRADES.csv')
df.head(100)


# Renaming the points column:

# In[ ]:


df.rename(columns={'FINAL GRADE [Total Pts: 100 Score] |547859': 'Points'}, inplace = True)


# Assuming that Zero is actually a point, the mean points is:

# In[ ]:


df['Points'].mean()


# And assuming that Zero is actually an error:

# In[ ]:


df['Points'].replace(0,np.nan).mean()


# Just out of curiousity, will *skipna* argument make any difference in the value, as the above already skipped *na*

# In[ ]:


df['Points'].replace(0,np.nan).mean(skipna = True)


# No difference I guess..

# Finally, just to create a histogram

# In[ ]:


df['Points'].replace(0,np.nan).hist(bins = 28)


# How would removing the two points below 70 affect the mean?

# In[ ]:


df[df['Points']>70].mean()


# What about 75?

# In[ ]:


df[df['Points']>75].mean()


# Anyway, that's all for this very small dataset..
