#!/usr/bin/env python
# coding: utf-8

# Most number of Attacks and Proportion of People Died

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

pk = pd.read_csv('../input/PakistanDroneAttacks.csv', encoding='latin1')
pk.head()


# In[ ]:


pk.groupby(["Longitude", "Latitude"]).size()


# We can clearly see that, the most number of attacks occurred in two locations.
# Now let see the proportion of NaNs in each column.

# In[ ]:


plt.figure(figsize=(5, 20))
pk.isnull().mean(axis=0).plot.barh()
plt.title("Proportion of NaNs in each column")


# In[ ]:


pk.groupby(["Date","Time"]).size()


# Replacing Nan with -1 and drop the columns which are not useful.

# In[ ]:


pk.drop(["Comments", "References", "Special Mention (Site)"], 1, inplace = True)
pk.columns = map(lambda x: x.replace(".", "").replace("_", ""), pk.columns)
pk.fillna(value = -1, inplace = True)


# In[ ]:


print(pk.shape)
print(pk.dtypes)
print(pk.head(10))


# In[ ]:


df_loc_count = pk.groupby('Location')[['City']].count()
df_loc_count.columns = ['loc_count']
df_bc = df_loc_count.ix[df_loc_count.loc_count>1,:]
df_bc.plot.barh(title='Location of Attack', legend=True, figsize=(6,8))
plt.show()


# Now, the proportion of people died in an attack.

# In[ ]:


df_dead = pk.groupby('Total Died Mix')[['Location']].count()
df_dead.columns = ['dead_count']
df_maxdead = df_dead.ix[df_dead.dead_count>2,:]
df_maxdead.plot.pie(y='dead_count', autopct='%2f', title='Total people died',legend=False ,figsize=(6,6)) 
plt.show()

