#!/usr/bin/env python
# coding: utf-8

# This is zomato dataset. So Lets do a EDA(Exploratory data Analysis) and understand Bangalorean food preferences.
# 
# 
# Lets first load the data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
data=pd.read_csv("../input/zomato.csv")


# Check with first few rows.

# In[ ]:


data.head()


# What are the cloumns we have:

# In[ ]:


data.columns


# In[ ]:


print('We have nearly data of',data.shape[0],'food orders.' )


# This is enough to do our analysis and find some generalisation about Bangalorean people food habits.
# 
# So Lets get started.

# **First Question:**
# 
# **Does Bangalorean people prefers to order online or offline?**

# In[ ]:


import seaborn as sns
sns.barplot(data.groupby('online_order').count().head()['url'].index,data.groupby('online_order').count().head()['url'])


# * > **This clearly tells that Bangalorean prefers to order online.**

# **Second Question:**
# 
# **What type of Restaurants does Bangalorean preferes?**

# In[ ]:


plt.figure(figsize=(12,5))
sns.barplot(data['rest_type'].value_counts().head(8).index,data['rest_type'].value_counts().head(8))


# In[ ]:


print('SO This chart tells that most prefered hotel types are:' ,data['rest_type'].value_counts().head(8).index.values)


# **Third Question:**
# 
# **What are the most favourate foods of Bangalorean people?**

# In[ ]:


data=data[data['dish_liked'].notnull()]
data.index=range(data.shape[0])
import re
likes=[]
for i in range(data.shape[0]):
    splited_array=re.split(',',data['dish_liked'][i])
    for item in splited_array:
        likes.append(item)

sns.barplot(pd.DataFrame(likes)[0].value_counts().head(10),pd.DataFrame(likes)[0].value_counts().head(10).index,orient='h')


# In[ ]:


print("The above chart tells us that the Bangalorean people mostly prefer This foods:",pd.DataFrame(likes)[0].value_counts().head(10).index.values)


# **Question 4:**
# 
# **Then the question pops up into the mind is that What is least favorite food they prefer.****
# 
# Lets find out!!!!

# In[ ]:


sns.barplot(pd.DataFrame(likes)[0].value_counts().tail(10),pd.DataFrame(likes)[0].value_counts().tail(10).index,orient='h')


# In[ ]:


print("The above chart tells us that the Bangalorean people hardly prefer these foods:",pd.DataFrame(likes)[0].value_counts().tail(10).index.values)


# **Question 5:**
# 
# **Do you know which is highest rated restaurant in banglore?**
# 
# Then Lets find out!!!

# In[ ]:


rating_data=data[np.logical_and(data['rate'].notnull(), data['rate']!='NEW')]
rating_data.index=range(rating_data.shape[0])
import re
rating=[]
for i in range(rating_data.shape[0]):
    rating.append(rating_data['rate'][i][:3])

rating_data['rate']=rating
rating_data.sort_values('rate',ascending=False)[['name','location','rate']].head(60).drop_duplicates()


# In[ ]:


print('This are the highest rated hotels in banglore:\n',rating_data.sort_values('rate',ascending=False)[['name']].head(60).drop_duplicates().values)


# **Last Question 6:**
# 
# **Do you know which is least rated restaurant in banglore?**
# 
# Then Lets find out!!!

# In[ ]:


rating_data.sort_values('rate',ascending=True)[['name','location','rate']].head(50).drop_duplicates()


# In[ ]:


print('This are the highest rated hotels in banglore:\n',rating_data.sort_values('rate',ascending=True)[['name','location','rate']].head(50).drop_duplicates().values)


# **Thank you for your time.
# **
# If you like, Please upvote.**
# 
# **And If you have any questions about Bangalorean food habits, Please mention it in comment.**
# 
# **We will definatily try to find out.****
