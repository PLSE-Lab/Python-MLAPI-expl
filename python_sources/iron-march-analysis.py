#!/usr/bin/env python
# coding: utf-8

# # Analyzing the Iron March Data Leak

# In this project, we are going to do some data exploration on the Neo-Nazi/Facist dataset that was leaked on November 7th, 2019.
#  
# The files are located on archive.org and we will be downloading the csv's directly from the website. 
# 
# I will first explore the member database and eventually dig into the actual messages to see what kind of information we can gather.
#  
# I will be exploring some initial questions such as:
# 
# 1. How many members are in this database?
# 2. Where are they generally located?
# 3. What location has the most members?
# 
# We'll also be plotting and visualization the data as we go along.

# In[ ]:


import pandas as pd
import io
import requests
import matplotlib.pyplot as plt 
import seaborn as sns

df =pd.read_csv("../input/iron-march-fascist-social-network-dataset/csv/core_members.csv")


# We can also check the number of members. One note is that the index key on the left hand side doesn't match the member_ID which means there were deleted records.

# In[ ]:


df


# In[ ]:


df.member_id.size


# There's several columns that are interesting. We should keep the member_id as a primary key to link other information. If we want to find out people's information for a deeper analysis, we should keep their e-mails, the frequency of posts, their birthday, and the timezone they selected.

# In[ ]:


df_filter=df.filter(['member_id', 'name', 'email', 'joined', 'member_posts', 'bday_year','timezone'], axis =1)


# In[ ]:


df_filter


# At a first glance, there is some information that pops out. The highest member post comes from a Russian e-mail along with some UK and FR e-mails. I would like to know who is the most active on the website , and I'm also curious to know where they live. Let's sort by descending.

# In[ ]:


df_filter.sort_values(by=['member_posts'], ascending = False)


# Looks like the most active members come from Europe with America following. This doesn't tell the whole story though. Let's take a look at this database and figure out where these members live.

# In[ ]:


df_filter['timezone'].value_counts(dropna=True)


# Let's visualize the top 10 timezones these members come from.

# In[ ]:


time = df_filter['timezone'].value_counts(dropna=True).head(10)


# In[ ]:


labels=time.index


# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(x=time, y=labels)


# It looks like while the first members came from Russia and Europe, the majority of the members reside in America. The next largest groups are from London and the Berlin timezones.
# 
# Next let's find out which timezones are most active by their posts.

# In[ ]:


df_filter.groupby(['timezone'])['member_posts'].sum().reset_index().sort_values(by=['member_posts'], ascending = False)


# In[ ]:


active = df_filter.groupby(['timezone'])['member_posts'].sum().reset_index().sort_values(by=['member_posts'], ascending = False).head(10)


# In[ ]:


labels =active['timezone']
data = active['member_posts']
plt.figure(figsize=(15,8))
sns.barplot(x=data, y=labels)


# Not a surprise when America leads in active posts, but there are some important insights to pull from this graph. 
# 
# 1. While the America New York timezone had the most members, the America Chicago timezone is the most active.
# 2. The London timezone has fewer members, but ranked third in active posts.
# 3. The Asia/Riyadh timezone which is linked to one Russian e-mail ranks 6th in active posts.

# Before I explore the individual posts, I want to know who moderates this website or who is control. It is likely the moderators of the site can manipulate and encourage certain behaviors. Instead of looking at the database broadly, we can see how the moderators suggest actions.

# In[ ]:


df_modlog =pd.read_csv("../input/iron-march-fascist-social-network-dataset/csv/core_moderator_logs.csv")


# There's several columns, but I am only interested in the member_id and the member_name. I'll use this list and merge it with the first dataframe.

# In[ ]:


df_modlog


# In[ ]:


df_mods = df_modlog.filter(['member_id', 'member_name']).dropna().drop_duplicates(subset=['member_id'], keep='last').sort_values(by=['member_id'], ascending = True)
df_mods


# Let's get the other dataframe ready to merge. We will be filtering by member_id, member_posts, and the timezone.

# In[ ]:


df1 = df_filter.filter(['member_id', 'member_posts', 'timezone'])
df1


# In[ ]:


#top10mods
df_mods = df_mods.merge(df1).sort_values(by=['member_posts'], ascending = False).head(10)
df_mods


# Now we know who the top 10 moderators( or as I like to call them "influencers") are. I will be focusing on these members for future analysis.[](http://)

# In[ ]:




