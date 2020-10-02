#!/usr/bin/env python
# coding: utf-8

# **Let's load the TANK!!!**

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **Read the data into dataframe. **

# In[ ]:


df = []
df = pd.read_csv('../input/IT_motives_2013.csv')
df.head()


# **Remove unnecessary characters from the column headers.**

# In[ ]:


df.columns = df.columns.str.replace('\/','') 
df.head()


# **If we look at the data carefully, we can find that there are 3 rows which we don't need. So, let's drop them!**

# In[ ]:


df = df.drop(df[df.StateUTs == 'Total (UTs)'].index)
df = df.drop(df[df.StateUTs == 'Total (States)'].index)
df = df.drop(df[df.StateUTs == 'Total (All India)'].index)


# **Let's see which states report the highest cyber crimes...**

# In[ ]:


plt.figure(figsize=(70,30))
sns.set(font_scale=6)
sns.barplot(x='StateUTs', y='Total', data = df)
plt.title('Cyber Crimes - Total')
plt.ylabel('Total')
plt.xticks(rotation=90)


# **Let's see which states reported cyber crimes related to 'Revenge Settling Scores'...**

# In[ ]:


plt.figure(figsize=(70,30))
sns.set(font_scale=6)
yCol = 'Revenge Settling scores'
sns.barplot(x='StateUTs', y = yCol, data = df)
plt.title('Cyber Crimes - ' + yCol)
plt.ylabel('Total')
plt.xticks(rotation=90)


# **Let's see which states reported cyber crimes related to 'Greed of Money'...**

# In[ ]:


plt.figure(figsize=(70,30))
sns.set(font_scale=6)
yCol = 'Greed Money'
sns.barplot(x='StateUTs', y = yCol, data = df)
plt.title('Cyber Crimes - ' + yCol)
plt.ylabel('Total')
plt.xticks(rotation=90)


# **Let's see which states reported cyber crimes related to 'Extortion'...**

# In[ ]:


plt.figure(figsize=(70,30))
sns.set(font_scale=6)
yCol = 'Extortion'
sns.barplot(x='StateUTs', y = yCol, data = df)
plt.title('Cyber Crimes - ' + yCol)
plt.ylabel('Total')
plt.xticks(rotation=90)


# **Let's see which states reported cyber crimes related to 'Cause Direpute'...**

# In[ ]:


plt.figure(figsize=(70,30))
sns.set(font_scale=6)
yCol = 'Cause Disrepute'
sns.barplot(x='StateUTs', y = yCol, data = df)
plt.title('Cyber Crimes - ' + yCol)
plt.ylabel('Total')
plt.xticks(rotation=90)


# **Let's see which states reported cyber crimes related to 'Prank or Satisfaction of Gaining Control'...**

# In[ ]:


plt.figure(figsize=(70,30))
sns.set(font_scale=6)
yCol = 'Prank Satisfaction of Gaining Control '
sns.barplot(x='StateUTs', y = yCol, data = df)
plt.title('Cyber Crimes - ' + yCol)
plt.ylabel('Total')
plt.xticks(rotation=90)


# **Let's see which states reported cyber crimes related to 'Fraud or Illegal Gain'...**

# In[ ]:


plt.figure(figsize=(70,30))
sns.set(font_scale=6)
yCol = 'Fraud Illegal Gain'
sns.barplot(x='StateUTs', y = yCol, data = df)
plt.title('Cyber Crimes - ' + yCol)
plt.ylabel('Total')
plt.xticks(rotation=90)


# **Let's see which states reported cyber crimes related to 'Eve Teasing and harassemet'...**

# In[ ]:


plt.figure(figsize=(70,30))
sns.set(font_scale=6)
yCol = 'Eve teasing Harassment'
sns.barplot(x='StateUTs', y = yCol, data = df)
plt.title('Cyber Crimes - ' + yCol)
plt.ylabel('Total')
plt.xticks(rotation=90)


# **Let's use melt() funciton of Pandas to create a new dataframe.**

# In[ ]:


df_new = df
df_new = df_new.drop(['Crime Head','Total'], axis = 1)
df_new = pd.melt(df_new, id_vars=['StateUTs','Year'], var_name = 'CrimeType')
df_new.head()


# **Create a new dataframe which holds the total count of each cyber crime types. This would make our life easy to plot based on the total occurance of each cyber crime type.**

# In[ ]:


df_totalCrimeType = pd.DataFrame({'TotalCrimes' : df_new.groupby(['CrimeType']).value.sum()}).reset_index()
df_totalCrimeType


# **Let's see which type of cyber crimes are occuring more...**

# In[ ]:


plt.figure(figsize=(70,30))
sns.set(font_scale=6)
sns.barplot(x='CrimeType', y='TotalCrimes', data=df_totalCrimeType)
plt.title('Cyber Crimes - Overall')
plt.xlabel('Type of Crime')
plt.ylabel('Total Crimes')
plt.xticks(rotation=90)


# **From the above analysis, we can come to below conclusions...**
# *  'Frad & Illegal Gain', 'Eve Teasing & Harrassment' and 'Greed for Money' seem to be the biggest motivators for Cyber crime in India in the year 2013
# * Cyber crimes with motive 'Frad & Illegal Gain' happen most in Uttar Pradesh followed by Andhra Pradesh & Maharashtra
# * Cyber crimes with motive 'Eve Teasing & Harrassment' happen most in Maharastra followed by Andhra Pradesh & Uttar Pradesh
# * Cyber crimes with motive 'Greed for Money' happen most in Bihar followed by Mahrashtra, West bengal, Karnatak, Uttar Pradesh and Andhra Pradesh
