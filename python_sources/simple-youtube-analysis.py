#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df1=pd.read_csv("../input/CAvideos.csv")
df2=pd.read_csv("../input/GBvideos.csv")
df1=df1.copy()
df2=df2.copy()


# In[ ]:


df1.head()


# # Which channel has got most likes

# In[3]:




plt.style.use("ggplot")
disliked=pd.DataFrame(df1.groupby(["channel_title"])["likes"].agg("mean").sort_values(ascending=False))[:15]
plt.figure(figsize=(10,8))
sns.barplot(y=disliked.index,x=disliked.likes,data=disliked)
plt.gca().set_xlabel("Mean Number of likes")
#plt.gca().set_xticklabels(disliked.likes,rotation="45")
plt.gcf().subplots_adjust(left=.3)
plt.show()


# # Channels that has maximum number of videos

# In[4]:



plt.style.use("ggplot")
channels=pd.DataFrame(df1.groupby(["channel_title"])["video_id"].agg("count").sort_values(ascending=False))[:20]
plt.figure(figsize=(8,8))
sns.barplot(y=channels.video_id,x=channels.index,data=channels)
plt.gcf().subplots_adjust(bottom=.45)
plt.gca().set_xticklabels(channels.index,rotation="90")
plt.gca().set_ylabel("Number of videos")
plt.gca().set_xlabel("Channel Name")
plt.show()


# # Likes,dislikes,comments for The young Turks

# In[ ]:


young=df1[df1["channel_title"]=="The Young Turks"][["likes","dislikes","publish_time","comment_count"]].reset_index()
young["publish_time"]=young["publish_time"].apply(lambda x : pd.to_datetime(x[:10]))
plt.figure(figsize=(8,8))
plt.plot(young["publish_time"],young["likes"],color="red",label="likes")
plt.plot(young["publish_time"],young["dislikes"],color="y",alpha=.7,label="dislikes")
plt.plot(young["publish_time"],young["comment_count"],color='c',alpha=.8,label="comments")
plt.gca().set_xlabel("Publish Date")
plt.gca().set_ylabel("Count")
plt.gca().set_title("The Young Turks")
plt.legend()
plt.show()


# ## Which channel gets its videos trending quickly

# In[9]:


trending=df1[["publish_time","trending_date","channel_title"]]
trending=trending.copy()
trending["publish_time"]=trending["publish_time"].apply(lambda x : pd.to_datetime(x).date())
trending["trending_date"]=trending["trending_date"].apply(lambda x: pd.to_datetime(x,format="%y.%d.%m").date())
trending["days"]=(trending["trending_date"]-trending["publish_time"])
trending.sort_values(by="days")
trending["days"]=trending.days.apply(lambda x : x.days)
mean=pd.DataFrame(trending.groupby("channel_title")["days"].agg("mean").sort_values(ascending=False))
mean[mean["days"]==0]


# ## Which channels took more than an year to make their videos as trending

# In[16]:


year=mean[mean["days"]>365]
plt.figure(figsize=(12,8))
plt.style.use("ggplot")
sns.barplot(y=year.index,x=year["days"])
plt.gcf().subplots_adjust(left=.3)
plt.gca().set_title("Channels that took more than one year to make video as trending")
plt.gcf().savefig("more_than_an_year")
plt.show()


# In[ ]:




