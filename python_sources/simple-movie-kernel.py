#!/usr/bin/env python
# coding: utf-8

# Hi There,
# This is my first Kernel. As you gonna see i m at start of the road. (But working hard :) This is  my first week at Phyton and Data Science
# So please forgivem my for my mistakes.
# 
# In my first kernel i choosed a Movie data to compare some variable of movies. Because of my first kernel, maybe you can not find so many informations. But even this way you can get something :)

# ![](https://i.kinja-img.com/gawker-media/image/upload/s--0dJesQ5_--/c_scale,f_auto,fl_progressive,q_80,w_800/mljkqghipl6v4v88dyyz.jpg)

# **What Do We Have In Our Data**

# Let's start import our librarys that we gonna use. 

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

data = pd.read_csv('../input/tmdb_5000_movies.csv')

del data['id']


# It s good to start to get some information about our data so we can check what columns do we have

# In[ ]:


data.columns


# And small sample to check our data

# In[ ]:


data.head()


# **Let's See What We Can Learn From Data About Movies**

# In this example i tried to get most liked movies from data. This data may not be 100% perfect correct but should be similar to correct. Our Vote_Count column is shown us how many vote is done for movies. That column is important because just one person can give 1 vote for a unvoted movie and if user's vote is 10, it s gonna be top on the list. But to find the correct answer we should pay attention the scoring by more people. In my example i gave a limit value which is 1000. So to enter that list minmum 1000 person should be voted that movie. 

# In[ ]:


filter_of_vote_count = data.vote_count > 1000
filtered_data_vote_count = data[filter_of_vote_count] 
filtered_data_vote_count[['title', 'popularity','vote_count', 'vote_average']].nlargest(15, 'vote_average')


# Vote avaerages by year. Except for 2017, score averages of movies are similiar

# In[ ]:


def to_date(series):
    return str(series)[:4]

data['release_date'] = data['release_date'].apply(to_date)
xx = data.groupby('release_date').vote_average.mean().reset_index()

xx.tail(30)


# 15 movies which get most money 

# In[ ]:


filtered_data =  data.nlargest(15, 'revenue')

plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
title = filtered_data.title
y_pos = np.arange(len(title))
revenue = filtered_data.revenue
error = np.random.rand(len(title))

ax.barh(y_pos, revenue, xerr=error, align='center',
        color='blue', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(title)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Revenue (Billion $)')
ax.set_title('Who earned more?')

plt.show()


# In this graphic i tried show which duration is more successfull in movies. In the result we see that movies which are around 120 minutes is more welcome for people 

# In[ ]:


# film surelerine gore puan ortalamasi
data_new = data.groupby('vote_average').runtime.mean().reset_index()
data_new = data_new.drop([69,70],axis=0)
data_new.tail(15).plot(kind='scatter', x='runtime', y='vote_average',color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('Duration(minute)')
plt.ylabel('Raking')
plt.title('Movies Duration By Ranking')
plt.show()


# 15 movice whichs are spent more money

# In[ ]:


most_budget_data = data.nlargest(15, 'budget')

fig, axs = plt.subplots()

budget = most_budget_data.title
y_pos = np.arange(len(budget))
budget = most_budget_data.budget
error = np.random.rand(len(budget))

axs.barh(y_pos, budget, xerr=error, align='center',
        color='red', ecolor='black')
axs.set_yticks(y_pos)
axs.set_yticklabels(most_budget_data.title)
axs.invert_yaxis()  # labels read top-to-bottom
axs.set_xlabel('Revenue (Billion $)')
axs.set_title('Who spent more?')

plt.show()



# Let's see most popular 15 movie. It is not the same list with our highest rank list. Just 2 movies are at both list

# In[ ]:


data[['title', 'popularity','vote_count', 'vote_average','budget','revenue']].nlargest(15, 'popularity')


# In the last part we will look at correlations. There is no negative correlations between variables but we can see one 0.7 correlation in our heatmap. That information show us when you spend more money for movies you can also get money 

# In[ ]:


f,ax = plt.subplots(figsize=(18, 12))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# Thank you for chekced my kernel
# 
# Please feel free to comment to improve myself
