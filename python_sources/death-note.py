#!/usr/bin/env python
# coding: utf-8

# #Work out the top10 causes of death.
# 
# Presented as an excecise to help me learn pandas more than anything else :)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


# In[ ]:


# Any results you write to the current directory are saved as output.
#Read in the death records to pandas df
deaths = pd.read_csv('../input/DeathRecords.csv')
codes = pd.read_csv('../input/Icd10Code.csv')
manners = pd.read_csv('../input/MannerOfDeath.csv')
icd10 = pd.read_csv('../input/Icd10Code.csv')


# Just as a distraction, let's do a simple histogram now.

# In[ ]:


deaths[deaths['MannerOfDeath']==0]['Age'].hist(bins=range(102))


# Ok do a groupby / sort / head to work out top 10.

# In[ ]:


top10 = deaths[['Icd10Code', 'Id']]    .groupby(['Icd10Code'])    .count()    .sort_values(['Id'], ascending=False)    .head(10)


# Now join the top10 with the table that has the descriptions.

# In[ ]:


top10 = pd.merge(top10, icd10, left_index=True, right_on=['Code'])
top10


# Finally do a simple plot.

# In[ ]:


top10.plot(kind='bar', x='Description')


# In[ ]:




