#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('/kaggle/input/AppleStore.csv')


# Let's see the genre makeup of the App Store.

# In[ ]:


df.sort_values('prime_genre')
freq = df['prime_genre'].value_counts()[0:4]
labels = list(freq.index)
labels.append('Other')
freq = list(freq)
freq.append(len(df['prime_genre'])-sum(freq))

plt.style.use('seaborn-white')
plt.pie(freq,labels=labels)
_ = plt.title('App Genres')


# That's a lot of games!

# Now let's take a look at how apps are generally rated.

# In[ ]:


sns.violinplot(x=df['user_rating'])
plt.xlim(0,5)
plt.xlabel('Rating (0 to 5 stars)')
plt.title('Distribution of App Ratings on the App Store')
_ = plt.style.use('seaborn-white')


# It seems like users are remarkably satisfied with their apps. The average rating appears to lie around a 4, which is pretty impressive. It's also funny to see the density of 0 star ratings. Some people are clearly very unhappy with their experience.
# 
# **An interesting note:**
# 
# In this dataset, there are 0 recorded user ratings of 0.5 stars. If this isn't a mistake in the data or a misunderstanding on my part, it's further evidence of how users rate in absolutes: they either love it or they hate it!

# In[ ]:


print(len(df[df['user_rating']==0.5]))


# **How are apps rated accross different price ranges?**
# 
# First, let's look at free apps.

# In[ ]:


sns.set_style('white')
sns.violinplot(x=df[df['price']==0.00]['user_rating'],color='red')
plt.xlim(0,5)
plt.xlabel('Rating (0 to 5 stars)')
_ = plt.title('Distribution of App Ratings (Free)')


# Now let's take a look at more expensive apps: those that cost at least $15.

# In[ ]:


sns.set_style('white')
sns.violinplot(x=df[df['price']>=15]['user_rating'],color='green')
plt.xlim(0,5)
plt.xlabel('Rating (0 to 5 stars)')
_ = plt.title('Distribution of App Ratings ($15+)')


# The distributions are similar, but it's interesting to see that the ratings of expensive apps are less concentrated around 0. Perhaps expensive apps are of higher quality.

# **Is there a correlation between Byte Size and Price?**
# 
# 
# I would think that a higher byte size means more work for the developers, which would drive prices up. 
# 
# Let's find out!

# In[ ]:


plt.style.use('seaborn-white')
plt.scatter(df['size_bytes'],df['price'])
plt.title('Byte Size vs. Price')
plt.xlabel('Size (Bytes)')
plt.ylabel('Price')
plt.xlim(0)
_ = plt.ylim(0)


# Apparently not. If anything, it almost seems that apps get slightly cheaper as the size increases. Weird!
# 
# To me, this suggests that the value of an app to the user isn't necessarily related to its size.

# **How are apps rated with different content ratings?**
# 
# Let's compare the user ratings of 4+ apps and 17+ apps.

# In[ ]:


bins = (0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5)
plt.style.use('seaborn-white')
plt.hist(df[df['cont_rating']=='17+']['user_rating'],alpha=.8,bins=bins,color='orange')
plt.xticks((0,1,2,3,4,5))
plt.title('User Ratings (17+)')
plt.xlabel('Rating')
plt.ylabel('Frequency')
_ = plt.xlim(right=5.5)


# In[ ]:


bins = (0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5)
plt.style.use('seaborn-white')
plt.hist(df[df['cont_rating']=='4+']['user_rating'],alpha=.8,bins=bins,color='purple')
plt.xticks((0,1,2,3,4,5))
plt.title('User Ratings (4+)')
plt.xlabel('Rating')
plt.ylabel('Frequency')
_ = plt.xlim(right=5.5)


# It's important to note that the 4+ category encompasses a wide variety of apps, so there are many more datapoints in the 4+ histogram.
# 
# That being said, it appears that 17+ apps generally are rated much lower than 4+ apps. Notably, 17+ apps have a surprising number of 0-star ratings. These histograms suggest that apps for mature audiences are generally more controversial or provocative, as one might expect.
# 
# 

# **Thank you for reading!**
# If you liked this kernel, please consider upvoting!
