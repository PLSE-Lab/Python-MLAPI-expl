#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


fbdata= pd.read_csv("../input/pseudo_facebook.csv")
fbdata.head()


# In[ ]:


#Exploring data
fbdata.describe()


# Age: Minimum-13, Maximum-113, Average age-37 DOB Year: Minimum-1900, Maximum-2000, Average age-37 Tenure: Minimum-0, Maximum-3139, Average age-537 Likes done are lower than likes received.

# In[ ]:


fbdata.info()


# In[ ]:


fbdata['gender'].value_counts()
#nearly 40:60 ratio for female to male


# In[ ]:


#Counting all vales here
fbdata['gender'].value_counts(dropna=False)


# In[ ]:


#divided the age into a group of 10. see last column
labels=['10-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','101-110','111-120']
fbdata['age_group'] = pd.cut(fbdata.age,bins=np.arange(10,121,10),labels=labels,right=True)
fbdata.head()


# In[ ]:


#Counting value in age groups
fbdata.age_group.value_counts()


# In[ ]:


#Females have more friends than males
sns.barplot(x=fbdata['age_group'],y=fbdata['friend_count'],hue=fbdata.gender)


# In[ ]:


#No of people having some friends
np.count_nonzero(fbdata.friend_count)


# In[ ]:


#All the people having zero friends
fc=fbdata.friend_count==0
fc.value_counts()


# This infers tha Males have more zero friends than females

# In[ ]:


#plotting the gender vs zero friend count people
#fc=fbdata.friend_count==0
sns.barplot(y=fbdata.friend_count==0,x=fbdata.gender)


# In[ ]:


fcmale=(fbdata.friend_count==0) & (fbdata.gender=='male')
fcmale.value_counts(dropna=False)
#true:1459
fcfemale=(fbdata.friend_count==0) & (fbdata.gender=='female')
fcfemale.value_counts(dropna=False)
#true:503
fc=fbdata.friend_count==0
fc.value_counts()
#sns.barplot(x=fcmale,y=fcfemale)


# Dividing tenure to lables

# In[ ]:


fbdata.tenure.interpolate(inplace=True)


# In[ ]:


tenlabel=['0-1 year','1-2 years','2-3 years','3-4 years','4-5 years','5-6 years','6-7 years','7-8 years','8-9 years']
fbdata['year_group']=pd.cut(fbdata.tenure,bins=np.arange(0,3300,365),labels=tenlabel,right=True)


# In[ ]:


fbdata.head()


# In[ ]:


fbdata.year_group.fillna(value='0-1 year',inplace=True)


# In[ ]:


fbdata.year_group.value_counts(dropna=False)


# In[ ]:


#Most liked people
fbdata.sort_values(by='likes_received',ascending=False)[:10]


# In[ ]:


#Calculating likes per day
fbdata['likes_per_day']=fbdata.likes_received/fbdata.tenure.where(fbdata.tenure>0)
fbdata.head()


# In[ ]:


#Top 10 users getting highest likes received
fbdata.sort_values(by='likes_received',ascending=False)[:10]


# In[ ]:


#Highest likes received per day
fbdata.sort_values(by='likes_per_day',ascending=False)[:10]


# In[ ]:


#Extracting famous people
famous=fbdata.sort_values(by='likes_per_day',ascending=False)[:10]
famous.head()


# In[ ]:


#plt.subplots(figsize=(12,10)
#plt.plot(y='userid',x='likes_per_day',data=famous)
famous.plot(x='userid',y='likes_per_day',kind='bar')
plt.ylabel("Likes per day")
plt.xlabel("User ID")
plt.title("Maximum likes per day")
plt.show()


# In[ ]:


#pivot table
fbdata.pivot_table(values=['mobile_likes_received','mobile_likes','www_likes_received','www_likes'],index='age_group',columns='gender')


# In[ ]:


fbdata.pivot_table(values=['mobile_likes_received','mobile_likes','www_likes_received','www_likes'],index='gender').plot()


# In[ ]:


#Getting those people who are most interested in sending friend requests
fbdata.sort_values(by='friendships_initiated',ascending=False)[:10]


# In[ ]:


followers=fbdata.sort_values(by='friendships_initiated',ascending=False)[:10]


# In[ ]:


#plt.subplots(figsize=(12,10)
#plt.plot(y='userid',x='likes_per_day',data=famous)
followers.plot(x='userid',y='friendships_initiated',kind='bar')
plt.ylabel("Friendship_count")
plt.xlabel("User ID")
plt.title("Maximum friendships initiated")
plt.show()


# In[ ]:


followers['fc_per_day']=followers.friendships_initiated / followers.tenure
followers


# In[ ]:


#plt.subplots(figsize=(12,10)
#plt.plot(y='userid',x='likes_per_day',data=famous)
followers.plot(x='userid',y='fc_per_day',kind='bar')
plt.ylabel("Friendship_count")
plt.xlabel("User ID")
plt.title('Maximum friendships initiated per day')
plt.show()


# In[ ]:




