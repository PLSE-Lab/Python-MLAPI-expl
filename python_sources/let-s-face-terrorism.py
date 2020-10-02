#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))


# In[22]:


df=pd.read_csv("../input/globalterrorismdb_0718dist.csv",encoding='ISO-8859-1',
              usecols=['iyear','extended','country','country_txt','region','region_txt',
                      'provstate','city','success','attacktype1','attacktype1_txt',
                      'targtype1','targtype1_txt','nkill','nwound','weapdetail'])
# extended=whether attack was extended for more than 24hour


# In[21]:


df.sample(2)


# ## year wise terrorist attack

# In[36]:


plt.figure(figsize=(10,10))
sns.barplot(df.iyear.value_counts().index,df.iyear.value_counts().values)


# The data shows very high terrorist attacks in last 10years.
# 
# This means either terrorist attacks have actually grown or due to better IT sector, more attacks are being recorded now

# ## how long attack takes

# In[39]:


sns.barplot(df.extended.value_counts().index,df.extended.value_counts().values)
# 0 means < 24hours
# 1 means > 24hours


# Most of the attacks are pinned down under 24 hours.
# 
# This means the motive of terrorist is not to actually own something or win some parts, but to kill and destroy as much as
# possible in as less time as possible to spread fear.

# ## which country has most attack

# In[84]:


#top 50 countries with high attacks
country_with_high_attacks=df.country_txt.value_counts().sort_values(ascending=False)[:50].index 
df_country_with_high_attacks=df[df.country_txt.isin(country_with_high_attacks)]


# In[85]:


plt.figure(figsize=(15,15))
sns.barplot(df_country_with_high_attacks.country_txt.value_counts().index,
            df_country_with_high_attacks.country_txt.value_counts().values)
plt.xticks(range(df_country_with_high_attacks.country_txt.nunique()),rotation='vertical')
plt.show()


# ## type of attack 

# In[89]:


plt.figure(figsize=(10,10))
sns.barplot(df.attacktype1_txt.value_counts().index,df.attacktype1_txt.value_counts().values)
plt.xticks(range(df.attacktype1_txt.nunique()),rotation='vertical')
plt.show()


# Confirmining my preposition of spreading fear, type of attack supports it completely.
# 
# Bombing are the easiest way to kill people and spread fear, and they have the highest frequency.

# ## how many attacks are successful

# In[91]:


df_count=df.success.value_counts()
sns.barplot(df_count.index,df_count.values)


# So this suggests most of terrorist attacks are well coordinated, that's why with so well intercomm we stopped only few attacks.

# ## what are the main tragets of terrorist

# In[95]:


plt.figure(figsize=(10,10))
df_count=df.targtype1_txt.value_counts()
sns.barplot(df_count.index,df_count.values)
plt.xticks(range(df.targtype1_txt.nunique()),rotation='vertical')
plt.show()


# Ok this one is little surprise. But let me put it this way - millitary and police are supposed to be our sole protectors.
# So if our defence service is breached, that means we're breached.
# 
# The data shows that most of the attacks are on defence services only. That also confirms my hopythesis, because if they can
# breach our defence services, we're in danger as well. This inference is enough for spreading fear.

# In[ ]:




