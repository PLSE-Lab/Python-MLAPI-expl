#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Let's look at conflicts in Africa through the lens of numbers. 
# In this notebook, the following key questions are answered. 
# On trend,
#     Are conflicts increasing on continental levels, regional level, and on a country level?
#     At what time does the conflict usually take place?
#     How many people have died so far?
#     What is the major causes of conflict?

# 

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


# In[ ]:


print(os.listdir('../input'))


# In[ ]:


#let's initialize the data and see a few info about it
df=pd.read_csv('../input/Africa_1997-2018_Dec15.csv')
df.info()


# In[ ]:


df.head()


# **> Let's count on a continent basis. **
# 

# In[ ]:


sns.set()
sns.set_context("poster", font_scale=0.5, rc={"lines.linewidth": 15})
ax = sns.catplot(x="YEAR", data=df, aspect=1.5,kind="count" )
ax.set_xticklabels(rotation=60)


# **Result 1**: From this data set, we can conclude that conflicts have increased in Africa. 

# > **Let's look at the regional level. **

# In[ ]:


sns.set()
sns.set_context("poster", font_scale=0.5, rc={"lines.linewidth": 15})
ax = sns.catplot(x="REGION", data=df, aspect=1.5,kind="count")
ax.set_xticklabels(rotation=90)


# **Result 2**: East Africa tops with 36% of all conflicts in Africa. 

# > Let's look at countries with the most conflict. 

# In[ ]:


sns.set()
sns.set_context("poster", font_scale=0.5, rc={"lines.linewidth": 15})
ax = sns.catplot(x="COUNTRY",data=df,aspect=1.5,kind="count")
ax.set_xticklabels(rotation=90)


# In[ ]:


#let's confirm which one of DRC and Nigeria has the least conflicts. 
df_nigeria=df.loc[df['COUNTRY'] == 'Nigeria']
print('Nigeria conflict count:',df_nigeria.count())
df_congo=df.loc[df['COUNTRY'] == 'Democratic Republic of Congo']
print('Democratic Republic of Congo:',df_congo.count())


# **Result 3**: Countries with the most conflicts
#     1. Somalia
#     2. Nigeria
#     3. DRC
#     4. Sudan
#     5. South Africa

# To be continued...

# In[ ]:




