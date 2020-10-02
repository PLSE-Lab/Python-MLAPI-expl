#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/directory.csv")
licensed = df[df['Ownership Type']=='Licensed']


# In[ ]:


#Inspired by code https://www.kaggle.com/nageshp/d/starbucks/store-locations/starbucks-top-20-cities-top-20-states
lis = licensed.groupby(['Country', 'State/Province']).count().reset_index().iloc[:,range(3)]
lis["State Province"] = lis["State/Province"]
lis.rename(columns={'Brand': 'StoreCount'}, inplace=True)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#Sorting data
lisUS = lis[lis["Country"]=="US"].sort_values(['StoreCount'], ascending=[0])

#Ploting Figure
ax=plt.figure(1,figsize=(10,22))
ax = sns.barplot(lisUS["StoreCount"],y=lisUS["State Province"],ci=0)
ax.set_title('Total of Licensed Stores ')
ax.set(xlabel='Total number of Stores', ylabel='State Province')

