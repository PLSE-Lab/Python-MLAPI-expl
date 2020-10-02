#!/usr/bin/env python
# coding: utf-8

# **BLACK FRIDAY SALES**

# ***Extracting insights from Black Friday Sales due to gender, marital status and any other indicators available for purchase insights.***

# **DATA SETUP**
# 

# *** Import numpy and pandas ***

# In[49]:


import numpy as np
import pandas as pd


# In[50]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Read in the csv file as a dataframe called df **

# In[51]:


df = pd.read_csv('../input/black-friday/BlackFriday.csv')


# In[52]:


df.info()


# In[53]:


df.head(5)


# **Clean up Null Values**

# In[54]:


df.isnull().sum()


# **Fill Empty Values with 0**

# In[55]:


modifieddf=df.fillna(0)


# **Check Empty Values fixed**

# In[56]:


df.isnull().sum()


# In[57]:


df.head(5)


# **Check insights of data**

# ****Data based on Age****

# In[58]:


df['Age'].value_counts().head(5)


# In[59]:


df['Insights'] = df['Age'].apply(lambda title: title.split(':')[0])
sns.countplot(x='Insights',data=df,palette='viridis')


# In[60]:


ByAge = df.groupby(by=['Gender','Age']).count()['Insights'].unstack()


# In[61]:


plt.figure(figsize=(12,6))
sns.heatmap(ByAge,cmap='viridis')


# **Data based on Gender**

# In[62]:


df['Gender'].value_counts().head()


# In[63]:


sns.countplot(df['Gender'],data=df,palette='viridis')


# In[64]:


ByGender = df.groupby(by=['Age', 'Gender']).count()['Insights'].unstack()


# In[65]:


plt.figure(figsize=(12,6))
sns.heatmap(ByGender,cmap='viridis')


# **Data Based on Occupation**

# In[66]:


df['Occupation'].value_counts().head(10)


# In[67]:


sns.countplot(df['Occupation'],data=df,palette='viridis')


# In[68]:


ByOccupation = df.groupby(by=['Occupation','Age']).count()['Insights'].unstack()


# In[69]:


plt.figure(figsize=(12,6))
sns.heatmap(ByOccupation,cmap='viridis')


# In[70]:


ByOccupation2 = df.groupby(by=['Occupation','Gender']).count()['Insights'].unstack()


# In[71]:


plt.figure(figsize=(12,6))
sns.heatmap(ByOccupation2,cmap='viridis')


# **Data Based on Length of Occupancy**

# In[72]:


df['Stay_In_Current_City_Years'].value_counts().head(10)


# In[73]:


sns.countplot(df['Stay_In_Current_City_Years'],data=df,palette='viridis')


# In[74]:


ByCityStay = df.groupby(by=['Age','Stay_In_Current_City_Years']).count()['Insights'].unstack()


# In[75]:


plt.figure(figsize=(12,6))
sns.heatmap(ByCityStay,cmap='viridis')


# **Data Based on Marital Status**

# In[76]:


df['Marital_Status'].value_counts().head(10)


# In[77]:


sns.countplot(df['Marital_Status'],data=df,palette='viridis')


# <img src="https://s3-eu-west-1.amazonaws.com/website38/DataAnaylisBlackFriday.JPG" width="1250px">

# 

# [](https://public.tableau.com/views/BlackFridaySalesAnalysis_15575467642560/Story1?:embed=y&:display_count=yes&publish=yes)

# **RECOMMENDATION**
# 
# **The results extracted from this dataset reflects that the main purchase buyers were Male and in the age range of 26-35. Bonus insights from this data analysis reveals that the main purchase buyers were those that had lived in the current residency for 1 year and also unmarried.**
# 
# **By utilising this data, you would be advised to have a survey that has a free gift card for completion of the survey competition, prior to the next sale, to find out which of your potential new customers fit the above category.**
# 
# **You could then target the next Black Friday sale buyers by; 1. Rewarding loyalty with a special promotion pre-sale for the Male unmarried 26-35 customers whom had lived at their current residency for 1 year and, 2. Spend money on advertising directed at this group of customers which are more likely to pay off if using this data as a guideline. I would recommend though, that you have a separate database for insights of any changes to your marketing strategy from your next sale and keep this same database selection so that you can compare the results from this sale with the results from subsequent sales as to make sure this result is not unique**
