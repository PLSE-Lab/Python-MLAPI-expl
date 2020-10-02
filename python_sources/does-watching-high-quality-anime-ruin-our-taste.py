#!/usr/bin/env python
# coding: utf-8

# **My Question is does watching good anime Ruin our taste for other anime ? or is it just in my head ? 
# Well lets find out :) **
# 

# In[91]:


#importing everything that's needed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy
from scipy import stats


# In[92]:


anime = pd.read_csv('../input/anime.csv')
rating = pd.read_csv('../input/rating.csv')


# Taking id of top 20 anime , as the list is in descnding order of rating 

# In[93]:


#Extracting top 20 anime 
animeid_top20 = anime.head(20)['anime_id']
print(animeid_top20)


# In[94]:


#Extracting users who have watched top 20 anime 

UserRatingForTop20=rating.loc[rating['anime_id'].isin(animeid_top20)]
Userwithalltop20 = UserRatingForTop20['user_id']


# In[95]:


anime.head()


# In[96]:


Userwithalltop20


# In[97]:


UserRatingForTop20.head()


# **Getting all users who have watched atleast 15 out of 20 top 10 anime**

# In[98]:


x=Userwithalltop20.value_counts()
a=pd.DataFrame(x)
a.reset_index(level=0, inplace=True)
final_UserID=a[a['user_id'] > 15]['index']


# In[99]:


#Getting the ratings for those users 
RatingsBytop20Watchers=rating.loc[rating['user_id'].isin(final_UserID)]


# In[100]:


#Ignoring the -1s 
RatingsWithout = RatingsBytop20Watchers[RatingsBytop20Watchers['rating']>-1]


# In[107]:


print(plt.hist(RatingsWithout['rating']))


# In[109]:


#Evaluating Ratings by all users
ratingsforall=rating[rating['rating']>-1]
print(plt.hist(ratingsforall['rating']))


# Average Rating for All users 

# In[111]:


np.mean(ratingsforall['rating'])


# Average Rating for users who have watched atleast 15 out of top 20 anime 

# In[113]:


np.mean(RatingsWithout['rating'])


# In[114]:


stats.ttest_ind(ratingsforall['rating'], RatingsWithout['rating'],equal_var = False)


# Well I guess , watching Good anime does effect how we rate other anime .
# Or I might be wrong , Its my first Kernel anyway . 
# I'll be thankful if you can comment what can be improved and please tell if I went wrong anywhere . 
