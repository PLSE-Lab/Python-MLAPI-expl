#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.getcwd())
print(os.listdir())


# # Data Preprocessing

# ### 1) Reading Data

# In[ ]:


df = pd.read_csv("../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv")
df.head(5)


# ### 2) Checking Null values

# In[ ]:


df.isnull().any().sum()


# In[ ]:


from datetime import datetime
df['Date'] = pd.to_datetime(df['Date'],format="%d/%m/%Y %H:%M:%S %p")


# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt


# # Data Visualization

# ### 1) Statewise Border Crossing

# In[ ]:


sdf = df[['Value','State']].groupby(['State'], as_index=False).sum()
sdf = sdf.sort_values('Value',ascending=False)
sdf.head()


# In[ ]:


g=sns.barplot(y='State',x='Value',data=sdf)
plt.title("Statewise Border Crossing")
plt.xlabel("Count of People Entering US")


# *OBSERVATION - Texas is the state with most people inbound into US*

# ### 2) Important Ports of Entry

# In[ ]:


sdf = df[['Value','Port Name']].groupby(['Port Name'], as_index=False).sum()
sdf = sdf.sort_values('Value',ascending=False)
sdf.head()


# In[ ]:


g=sns.barplot(y='Port Name',x='Value',data=sdf.iloc[:10,])
plt.title("Top 10 Ports of Entry")
plt.xlabel("Count of People Entering US")


# *OBSERVATION - El Paso is the most important Gateway into US*

# ### 2) Yearwise Border Crossings

# In[ ]:


df['Year'] = df['Date'].dt.year
sdf = df[['Year','Value','Border']].groupby(['Year'], as_index=False).sum()
sdf.head()


# In[ ]:


g=sns.lmplot(x='Year',y='Value',ci=False,robust=True,data=sdf)
plt.ticklabel_format(style='plain', axis='y')
plt.title("Number of People coming into US is decreasing with Time")


# *OBSERVATION -* 
# 
# 1) We can see there is a decreasing trend in the number of people inbound to US overtime
# 
# 2) There is an outlier from year 2019 as the year is still not complete (So, we have used robust = True to not consider this in our Linear Regression Line)

# ### 3) Yearwise Border Crossings For US-Mexico and US-Canada Borders

# In[ ]:


sdf = df[['Year','Value','Border']].groupby(['Year','Border'], as_index=False).sum()
sdf.head()


# In[ ]:


g=sns.lmplot(x='Year',y='Value',ci=False,robust=True,hue='Border',data=sdf)
plt.ticklabel_format(style='plain', axis='y')
plt.title("People coming into US are mostly from Mexico")


# *OBSERVATION -* 
# 
# 1) We can see the decreasing trend at both the borders 
# 
# 2) We also observe the decline is faster in people coming from Mexico

# In[ ]:




