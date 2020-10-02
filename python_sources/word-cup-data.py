#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/World_Cups.csv")


# In[ ]:


df


# In[ ]:





# In[ ]:


sns.pairplot(df)


# ####  An interesting trend,  Attendence  each year seem have increased year on year

# In[ ]:


attend = df[['Year', 'Attendance']]


# In[ ]:


plt.figure(figsize = (15,5))
ax = sns.barplot(data=attend, x='Year', y='Attendance' )
# plot regplot with numbers 0,..,len(a) as x value
sns.regplot(x=np.arange(0,len(attend)), y = 'Attendance', data = attend)
#sns.despine(offset=10, trim=False)
ax.set_ylabel("Attendance")
plt.show()


# In[ ]:


winners = df[['Year', 'Winner', 'Runners-Up']]


# In[ ]:


plt.figure(figsize = (15,5))
ax = sns.countplot(x="Winner", data=winners)


# #### Brazil is in the top winning 5 times world cup

# In[ ]:


stats = df[['Country','Goals Scored', 'Matches Played']]


# In[ ]:


grouped = stats.groupby('Country', as_index = False).sum().sort_values('Goals Scored', ascending = False)
grouped['Average'] = grouped['Goals Scored']/grouped['Matches Played']
grouped


# #### Few observations from the above
# ####  1. Brazil scored highest goals - 259
# ####  2. Switzerland  scored highest average scores per match
# #### 3. Germany played in highet number of matches - 102

# In[ ]:




