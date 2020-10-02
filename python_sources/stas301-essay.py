#!/usr/bin/env python
# coding: utf-8

# # **STAS301: London Homicides**

# **This kernel is linked to my STAS301 essay based on London homicides for 2008-2018**

# Firstly, I import the necessary libraries and load the dataset.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_excel("../input/london-homicide-data-20082018/mmap.xlsx")


# Secondly I explore the data values and see that certain fields have missing values.

# In[ ]:


print(df.info())


# In[ ]:


plt.figure(figsize=(15,10))
plt.title("Homicides in London for the period 2008-2018", fontsize=25, y=1.02)
df.groupby(df.date.dt.year).size().plot(color='red', marker='o')
plt.ylabel("Number of Homicides",fontsize=20, labelpad=25)
plt.xlabel("Year", fontsize=20, labelpad=25)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('Homicides.png')
plt.show()


# **Case Status:**
# 

# In[ ]:


df.Status.value_counts() * 100 / df.Status.value_counts().sum()


# In[ ]:


plt.figure(figsize=(10,4))
(df.Status.value_counts() * 100 / df.Status.value_counts().sum()).plot(kind='bar')
plt.title("% Solved")


# From this, it is evident that most of the homicide cases over the last 11 years have been solved. 

# **Number of Homicides:**
# 
# Firstly, I look at the number of homicides per year and see that they are on the rise again.

# In[ ]:


df.groupby(df.date.dt.year).size()


# Secondly I retrieve the data needed to do a runs test for randomness to see if there is a pattern to the number of murders per month.

# In[ ]:


s2= df.groupby([df.date.dt.year, df.date.dt.month]).size()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(s2)


# In[ ]:


plt.figure(figsize=(10,5))
y = df.groupby(df.date.dt.month).size()
y.plot(kind='bar', color = 'r')
plt.ylim(top=150)
plt.xlabel("Month", fontsize=14, labelpad=15)
plt.ylabel("Number of Homicides", fontsize=14, labelpad=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Total Number of Homocides by Month in London (2008-2018)", y=1.01, fontsize=15)  


xlocs=[i+1 for i in range(-1,11)]
for i, v in enumerate(y):
    plt.text(xlocs[i] - 0.2, v + 1.5, str(v))
    

plt.savefig("Homicide_Monthly")
plt.show()


# From the above, it is evident that most homicides have taken place in March and August and that July and October have the least homicides

# In[ ]:


pd.crosstab(df.Status, df.vicsex, margins=True)


# In[ ]:


month = df.date.dt.month
df.groupby([month]).size().plot(kind='bar')
plt.title("Number of homocides by month")
df.groupby(df.date.dt.month).size()/10


# In[ ]:


df.groupby([df.date.dt.year]).catdom.value_counts()


# In[ ]:


df[df.vicsex=='M'].weapon.value_counts() * 100 / df[df.vicsex=='M'].weapon.value_counts().sum()


# In[ ]:


df[df.vicsex=='F'].weapon.value_counts() * 100 / df[df.vicsex=='F'].weapon.value_counts().sum()


# In[ ]:


df.weapon.value_counts() * 100 / df.weapon.value_counts().sum()


# In[ ]:


f, axes = plt.subplots(1,2, figsize=(12,5))

ax1 = (df[df.vicsex=='M'].weapon.value_counts() * 100 / df[df.vicsex=='M'].weapon.value_counts().sum()).plot(kind='bar', ax=axes[0], color = 'r')
ax1.set_ylabel("% of Total", labelpad=15)
ax1.set_xlabel("Weapon Used in Homicide", labelpad=15)
ax1.set_title("Male Victims By Weapon")


ax2 = (df[df.vicsex=='F'].weapon.value_counts() * 100 / df[df.vicsex=='F'].weapon.value_counts().sum()).plot(kind='bar', ax=axes[1], color = 'r')
ax2.set_ylabel("% of Total", labelpad=15)
ax2.set_xlabel("Weapon Used in Homicide", labelpad=15)
ax2.set_title("Female Victims by Weapon")
plt.tight_layout()
plt.savefig("VictimWeapon")
plt.show()


# In[ ]:


pd.crosstab(df[df.vicsex=='M'].weapon, df.Status)


# In[ ]:


pd.crosstab(df[df.vicsex=='F'].weapon, df.Status)


# In[ ]:


pd.crosstab(df.weapon, df.Status, normalize ='index')


# In[ ]:


pd.crosstab(df.weapon, df.Status, normalize ='columns')


# # Additional Exploration of Data not directly mentioned in Essay

# **Domestic Violence:**
# 
# From the graph below, it is evident that those aged between 25 and 34 are the most vulnerable to domestic abuse followed by 35-44 and 65+.

# In[ ]:


dom = df['catdom']==1
df_dom = df[dom]

order = ['A. Child 0-6', 'B. Child 7-12', 'C. Teen 13-16','D. Teen 17-19','E. Adult 20-24','F. Adult 25-34','G. Adult 35-44', 'H. Adult 45-54','I. Adult 55-64','J. Adult 65 over']
df_dom.vicagegp.value_counts()

df_weekday = df.copy().groupby(df_dom.vicagegp).size().reindex(order)
df_weekday
df_weekday.plot(kind = 'bar')


# Victim Ethnicity: 
# 
# From the graph below, we can see that the majority of victims have been White, closely followed by Black victims and then Asians.

# In[ ]:


plt.figure(figsize=(10,4))
(df.vicethnic.value_counts() * 100 / df.vicethnic.value_counts().sum()).plot(kind='bar')
plt.title("Ethnicity")


# Victim Gender:
# 
# From the information below, more men are victims than women.

# In[ ]:


plt.figure(figsize=(10,4))
(df.vicsex.value_counts() * 100 / df.vicsex.value_counts().sum()).plot(kind='bar')
plt.title("Sex")


# In[ ]:


df.vicsex.value_counts() * 100 / df.vicsex.value_counts().sum()


# Thank you for reading through this kernal.
