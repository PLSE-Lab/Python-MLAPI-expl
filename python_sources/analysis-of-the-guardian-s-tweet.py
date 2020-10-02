#!/usr/bin/env python
# coding: utf-8

# **Overview**
# 
# In this analysis I took into consideration the tweets published by the official Guardian profile and related them to users who tweeted or retweeted the same article. I mainly focused on analyzing the distribution of delays and above all analyzing the most intense tweet session.
# 
# 
# **NOTE: Article_id = User_id**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[ ]:


tg= pd.read_csv("../input/guardian_tweets_exe (1).csv")
display(tg.head(3))


# In[ ]:


ute= pd.read_csv("../input/user_tweets_exe.csv")
display(ute.head(3))


# In[ ]:


tgute=pd.merge(tg, ute, left_on = ['id'], right_on=['article_id'])   #merging id from users and aticle_id in order to evaluate the tweet of guardian and tweet of user time


# In[ ]:


tgute["new_x"] = pd.to_datetime(tgute["timestamp_x"])
tgute["new_y"]= pd.to_datetime(tgute["timestamp_y"])

#x tweet guardian
#y tweet users



tgute["new_x"] = pd.to_datetime(tgute["timestamp_x"])
tgute["new_y"]= pd.to_datetime(tgute["timestamp_y"])

    
tgute['delay'] = tgute["new_y"]-tgute["new_x"]
print(type(tgute["new_y"]))
tgute.head(10)


# In[ ]:


tgute['delay'].describe()


# In[ ]:


data = tgute[['user_x','user_y','delay']]
data1 = data.tail(1000)
print(type(data1['delay']))
#.astype(int)
print(type(data1['delay']))

temporanea = []
for i in range(len(data)):
    temporanea.append(data['delay'][i].total_seconds()/3600)


# In[ ]:



fig = plt.figure(figsize=(16,7))
plt.title('Delay [Tweet Time user - Tweet Time The Guardian]')
sns.distplot(temporanea)
plt.xlabel('Hours')


# **Consideration**
# The average delay can be seen in the 7 hours so generally the user tends to tweet after seeing the tweet on The Guardian's twitter profile. In some cases you notice that the user has anticipated the tweet of the newspaper, probably by accessing the website of The Guardian and tweeting before the official profile of the newspaper

# In[ ]:


#tweet del the guardian istogramma per orario(ore, minuti, secondi) ma senza data 


temporanea1 = []
for i in range(len(tgute)):
    temporanea1.append(
((tgute["new_x"][i].hour*60+tgute["new_x"][i].minute)*60+tgute["new_x"][i].second)/3600
    )
    
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(16,7))
plt.title('Distribution of tweet by The Guardian')
sns.distplot(temporanea1)    
plt.xlabel('Hours')


# **Consideration**
# The Guardian's twitter profile tends to tweet without precise distribution and throughout the day.

# In[ ]:


#tweet degli utenti istogramma per orario(ore, minuti, secondi) ma senza data 

temporanea2 = []
for i in range(len(tgute)):
    temporanea2.append(
((tgute["new_y"][i].hour*60+tgute["new_y"][i].minute)*60+tgute["new_y"][i].second)/3600
    )
    
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(16,7))
sns.distplot(temporanea2) 
plt.title('Distribution of tweet by Users')
plt.xlabel('Hours')


# **Consideration**
# In general, users tend to tweet in the evening and on average during free hours. It is noted that during the morning hours the number of tweets concerning The Guardian articles tend to be smaller.

# 
