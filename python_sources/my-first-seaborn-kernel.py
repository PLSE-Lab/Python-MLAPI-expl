#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/world-happiness/2017.csv')
kill = pd.read_csv('../input/police-shootings/database.csv',encoding="windows-1252")
kill.head()
data.head()


# In[ ]:


country_list = list(data["Country"].unique())
freedom_list = []
for i in country_list:
    x = data[data["Country"] == i]
    sum_freedom = sum(x.Freedom)/len(x)
    freedom_list.append(sum_freedom)
data1 = pd.DataFrame({'country_list' : country_list,'freedom_list':freedom_list})
new_index = (data1['freedom_list'].sort_values(ascending=False)).index.values
sorted_data = data1.reindex(new_index)
plt.figure(figsize = (15,10))
sns.barplot(x = sorted_data['country_list'],y = sorted_data['freedom_list'])
plt.xticks(rotation = 90)
plt.xlabel("Country")
plt.ylabel("Freedom")
plt.title("Country and Freedom")


# **Joint Plot**

# In[ ]:


g = sns.jointplot(data.Family,data.Freedom,kind = "kde",size = 7)
plt.savefig('graph.png')
plt.show()


# In[ ]:


g = sns.jointplot("Family","Freedom",data = data,size = 5,ratio = 3,color = "r")


# **Pie Chart**

# In[ ]:


data.Country.dropna(inplace = True)
labels = data.Country.value_counts().index
colors = ["grey","blue"]
explode = []
for i in data.Country:
    explode.append(0)
sizes = data.Country.value_counts().values

plt.figure(figsize = (7,7))
plt.pie(sizes,explode = explode,labels = labels,colors=  colors,autopct='%1.1f%%')
plt.title("Country",color = "blue",fontsize = 15)


# **Lm Plot**

# In[ ]:


sns.lmplot(x="Family",y="Freedom",data = data)
plt.show()


# **Kde Plot**

# In[ ]:


sns.kdeplot(data.Family,data.Freedom,shade = True,cut = 3)
plt.show()


# **Violin Plot**

# In[ ]:


pal = sns.cubehelix_palette(2,rot=-.5,dark=.3)
sns.violinplot(data = data,palette=pal,inner="points")
plt.show()


# **HeatMap**

# In[ ]:


data.corr()


# In[ ]:


f,ax = plt.subplots(figsize =(5,5))
sns.heatmap(data.corr(),annot = True,linewidths = 0.5,linecolor = "red",fmt = '.1f',ax = ax)
plt.show()


# **Pair Plot**

# In[ ]:


sns.pairplot(data[['Family']])
plt.show()


# **Count Plot**

# In[ ]:


sns.countplot(kill.gender)
plt.title("Gender",color = "red",fontsize = 16)


# In[ ]:


armed = kill.armed.value_counts()

plt.figure(figsize=(10,7))
sns.barplot(x=armed[:7].index,y=armed[:7].values)
plt.xlabel("Index")
plt.ylabel("Weapon")
plt.title("Kill Weapon",color="red",fontsize = 18)


# In[ ]:


above_below = ['above18' if i>=18 else 'below18' for i in kill.age]
df = pd.DataFrame({"age":above_below})
sns.countplot(df.age)
plt.ylabel("Number of Killed People")
plt.title('Age of killed people',color = 'blue',fontsize=15)


# In[ ]:


sns.countplot(data = kill,x='race')
plt.title('Race of killed people',color = 'blue',fontsize=15)


# In[ ]:


city= kill.city.value_counts()
plt.figure(figsize = (10,7))
sns.barplot(x = city[:7].index,y = city[:7].values)
plt.xticks(rotation = 90)
plt.title('Most dangerous cities',color = 'blue',fontsize=15)


# In[ ]:


state = kill.state.value_counts()
sns.barplot(x = state[:20].index,y = state[:20].values)
plt.title("Most Dangerous State",fontsize = 18,color ="Grey")


# In[ ]:


sta = kill.state.value_counts().index[:10]
sns.barplot(x=sta,y=kill.state.value_counts().values[:10])
plt.title('Kill Numbers from States',color = 'blue',fontsize=15)


# In[ ]:




