#!/usr/bin/env python
# coding: utf-8

# **Questions: <br>
# 1. What is the most popular platform for video games. <br>
# 2. Which genre does each region prefer. <br>
# 3. Does the preferred genre change over time. <br>**

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


df = pd.read_csv("../input/vgsales.csv")
df #preview data


# In[ ]:


gc = df.Platform.value_counts().head(10).sort_values(ascending=True)
gc.plot.barh() #bar graph for top 10 platforms


# **The results show that the most popular platform for video games are DS, followed by PS2 and PS3.**

# In[ ]:


gc.sort_values(ascending=False) #actual values


# In[ ]:


grp = df.groupby('Genre')
import seaborn as sb
sb.heatmap(grp['NA_Sales','EU_Sales','JP_Sales','Other_Sales', 'Global_Sales'].sum(), annot=True, fmt=".2f")


# **Above is a heat map showing which genre is the most favored within each region.**

# In[ ]:


NA = grp['NA_Sales'].sum()
NA.sort_values(ascending=False).head(3)


# **The North American region's top 3 genres are action, sports, and shooter.**

# In[ ]:


EU = grp['EU_Sales'].sum()
EU.sort_values(ascending=False).head(3)


# **Similar to the North American region, the European region's top 3 are action, sports, and shooter.**

# In[ ]:


JP = grp['JP_Sales'].sum()
JP.sort_values(ascending=False).head(3)


# **The Japanese region's top 3 genres are RPG, action, and sports.**

# In[ ]:


Others = grp['Other_Sales'].sum()
Others.sort_values(ascending=False).head(3)


# **Again, similar to the North American and European regions, the top 3 genres in the regions excluding North America, Europe and Japan are action, sports, and shooter.**

# In[ ]:


Global = grp['Global_Sales'].sum()
Global.sort_values(ascending=False).head(5)


# **With three out of four regions preferring action, sports and shooter games, it isn't surprising that these three had the most global sales. RPGs, being the top 1 for the Japanese region, is ranked fourth in the global rankings.**

# In[ ]:


got = df.groupby(['Year','Genre']).size()
got


# In[ ]:




