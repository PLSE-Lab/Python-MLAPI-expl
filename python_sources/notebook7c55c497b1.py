#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')

sc2 = pd.read_csv('../input/starcraft.csv')


# In[ ]:


sc2.head()


# In[ ]:


sc2.info()


# In[ ]:


corrmat = sc2.corr()
# Generate a mask for the upper triangle
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)

# Use matplotlib directly to emphasize known networks
networks = corrmat.columns.get_level_values("network")
for i, network in enumerate(networks):
    if i and network != networks[i - 1]:
        ax.axhline(len(networks) - i, c="w")
        ax.axvline(i, c="w")
f.tight_layout()


# In[ ]:


sns.boxplot(x="LeagueIndex", y="ActionLatency", data=sc2, palette="PRGn")


# In[ ]:


sns.boxplot(x="LeagueIndex", y="APM", data=sc2, palette="PRGn")


# In[ ]:


df = sc2.drop(sc2[sc2['TotalHours'] > 5000].index)
sns.boxplot(x="LeagueIndex", y="TotalHours", data=df, palette="PRGn")


# In[ ]:


#sns.boxplot(x="LeagueIndex", y="TotalHours", data=sc2, palette="PRGn")
sns.boxplot(x="LeagueIndex", y="HoursPerWeek", data=sc2, palette="PRGn")


# In[ ]:


sns.boxplot(x="LeagueIndex", y="Age", data=sc2, palette="PRGn")


# In[ ]:


sns.boxplot(x="LeagueIndex", y="NumberOfPACs", data=sc2, palette="PRGn")


# **How do the replay attributes differ by level of player expertise?**
# 
#    Professional leagues are LeagueIndex 8 while Bronze League is 1. Players in higher leagues generally play over 1000 hours in total and more than 20 hours per week. Their APM is higher, ActionLatency is lower, and use techniques that require practice like SelectByHotkeys and AssignToHotKeys. 
# 
# **What are significant predictors of a player's league?**
# 
# Almost all variables are good indicators. However the best are:
# 
# 1. APM 
# 2. SelectByHotkeys 
# 3. AssignToHotKeys 
# 4. NumberOfPacs 
# 5. ActionLatency

# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt
G=nx.Graph()


# In[ ]:


sc2['quintile'] = pd.qcut(sc2['Age'], 5, labels=False)


# In[ ]:


sc2


# In[ ]:


sc2.hist()


# In[ ]:


from itertools import combinations
cnxns = []
for k,g in sc2.groupby('Age'):
    [cnxns.extend((n1,n2,k,len(g)) for n1,n2 in combinations(g['LeagueIndex'], 2))]

sc3 = pd.DataFrame(cnxns)


# In[ ]:


sc3


# In[ ]:


G=nx.from_pandas_dataframe(sc3, 0, 1, 3)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
nx.draw(G, labels = True)
plt.show() # display


# In[ ]:


sc2.describe()


# In[ ]:




