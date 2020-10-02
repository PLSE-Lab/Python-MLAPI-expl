#!/usr/bin/env python
# coding: utf-8

# Hi, I hope you enjoy this kernel. 
# Your comments & votes are very much appreciated! Especially tips on how to improve are very much welcome!!
# Jesse

# In[ ]:


import pandas as pd
import pickle
import numpy as np
from fastai.collab import *
from pprint import pprint
import matplotlib.pyplot as plt
import umap
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# the original csv from https://raw.githubusercontent.com/beefsack/bgg-ranking-historicals/master/
# The column ID is used in API calls to retrieve the game reviews
games = pd.read_csv('../input/2019-05-02.csv')
games.describe()
games.sort_values('Users rated',ascending=False,inplace=True)
games.rename(index=str, columns={"Bayes average": "Geekscore",'Name':'name'}, inplace=True)
games[:10]


# In[ ]:


reviews = pd.read_csv('../input/bgg-13m-reviews.csv',index_col=0)
print(len(reviews))
reviews.head()


# In[ ]:


reviews['rating'].hist(bins=10)
plt.xlabel('rating of review')
plt.ylabel('number of reviews')
plt.show()


# As you can see,most of the reviews are between a 6 and a 10.

# In[ ]:


games_by_all_users = reviews.groupby('name')['rating'].agg(['mean','count']).sort_values('mean',ascending=False)
games_by_all_users['rank']=games_by_all_users.reset_index().index+1
print(len(games_by_all_users))

games_by_all_users = games_by_all_users.merge(games[['name','Geekscore']],how='left',left_on=['name'], right_on=['name'])
games_by_all_users.head()


# In[ ]:


x = games_by_all_users['rank']
y = games_by_all_users['mean']
plt.figure(num=None, figsize=(7, 3), facecolor='w', edgecolor='k')
plt.scatter(x, y,s=0.5)
plt.xlabel('sorted games by average rating')
plt.ylabel('average rating')
plt.show()  # or plt.savefig("name.png")


# In[ ]:


games_by_all_users[['mean','Geekscore']].hist(bins=10)
plt.xlabel('averge rating of game')
plt.ylabel('number of games')
plt.show()


# If we compute the average rating of the games, it's even more visible that the ratings are really centered around the mean which is 6.4. The Geekscore is a score which penalizes games when they have few reviews, since the uncertainty around their 'true' score is higher. This is reflected in a lower average score of 5.6

# In[ ]:


x = games_by_all_users['count']
y = games_by_all_users['Geekscore']
y2 = games_by_all_users['mean']

df = pd.DataFrame({'X' : x, 'Y' : y})  #we build a dataframe from the data
data_cut = pd.cut(df.X,bins=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,1000,10000,100000])           #we cut the data following the bins
grp = df.groupby(by = data_cut)        #we group the data by the cut
ret = grp.aggregate(np.median)         #we produce an aggregate representation (median) of each bin


df2 = pd.DataFrame({'X' : x, 'Y' : y2})  #we build a dataframe from the data
data_cut = pd.cut(df2.X,bins=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,1000,10000,100000])           #we cut the data following the bins
grp = df2.groupby(by = data_cut)        #we group the data by the cut
ret2 = grp.aggregate(np.median)         #we produce an aggregate representation (median) of each bin

#plotting
plt.figure(num=None, figsize=(10, 4), facecolor='w', edgecolor='k')
plt.xscale('log')
plt.scatter(df2.X,df2.Y,alpha=.5,s=0.5)
plt.plot(ret.X,ret.Y,'g--',lw=4,alpha=0.5)
plt.plot(ret2.X,ret2.Y,'r--',lw=4,alpha=0.5)
plt.xlabel('number of reviews')
plt.ylabel('average rating of game')
plt.show()


# Plotted on a logaritmic scale, obviously most of the games don't have that many reviews. There appears to be a slight positive correlation between the amount of reviews and the average score. The red score marks the average score, the green the 'Geekscore', which again penalizes for a lower amount of ratings. 

# In[ ]:


reviews_by_user_count = reviews.groupby('user')['rating'].agg(['mean','count']).sort_values('count',ascending=False).reset_index()
print(len(reviews_by_user_count))
reviews_by_user_count.head()


# In[ ]:


reviews_by_user_count['count'].hist(log=True)
plt.xlabel('number of users')
plt.ylabel('number of reviews written')


# Most of the users have between 0-500 reviews, however there are some enthusiasts with >4000 reviews! That's some real dedication right there! User 'leffe dubbel' has rated in total 5984 games, just wow!

# In[ ]:


# select users that reviewed more than cutoff games
cutoff = 500
active_users = reviews_by_user_count[reviews_by_user_count['count']>cutoff]
active_users = active_users['user']
reviews_by_active_users = reviews[reviews['user'].isin(active_users)]
print(len(reviews_by_active_users))
reviews_by_active_users.head()


# In[ ]:


count_user, count_review = reviews[['user','name']].nunique()
print('density',len(reviews)/(count_user*count_review))
count_user, count_review = reviews_by_active_users[['user','name']].nunique()
print('density', len(reviews_by_active_users)/(count_user*count_review))


# In[ ]:


games_rated_by_active_users = reviews_by_active_users.groupby('name')['rating'].agg(['mean','count']).sort_values('mean',ascending=False)
games_rated_by_active_users['rank']=games_rated_by_active_users.reset_index().index+1

print('{} users original, with {} reviews'.format(reviews['user'].nunique(),len(reviews)))
print('{} users left({}% of the userbase), with {} reviews (this is {} of all reviews)'.format(len(active_users),len(active_users)/reviews['user'].nunique(),len(reviews_by_active_users),len(reviews_by_active_users)/len(reviews)))


# In[ ]:


games_rated_by_active_users['mean'].hist()
games_by_all_users['mean'].hist()


# Games seem to get a slightly higher shore when rated by all users

# In[ ]:


x = reviews_by_user_count['count']
y = reviews_by_user_count['mean']
plt.xscale('log')

plt.hist2d(x, y, bins=[np.logspace(np.log10(30),np.log10(1000),40),np.linspace(5,10,num=40)], cmap=plt.cm.jet)
plt.colorbar()
plt.xlabel('Number of reviews written')
plt.ylabel('Average score of user')
plt.show()


# This graph shows that users with many reviews tend to give lower scores compared to users that are less active (r2=0.09). Two explainations arise:
# - Active users simply rate more games, so they also play more bad games. Vice versa, casual users just play good games
# - Active users are more critical of games, so tend to give lower scores

# In[ ]:


merge = games_rated_by_active_users[['mean','count']].merge(games_by_all_users[['name','mean','count', 'Geekscore']],how='outer',left_index=True, right_on=['name'],suffixes=('active','all'),indicator=True)

merge['delta_active_all']=merge['meanactive']-merge['meanall']
merge['proportion_active']=merge['countactive']/merge['countall']
merge.sort_values('countall',ascending=False)[:5]


# In[ ]:


merge['delta_active_all'].median()


# The active users with >500 reviews per user make up 1% of the userbase, but almost submit 15% of all the reviews. Their scores per game are about 0.33 points lower compared to the whole population. To me this suggests they are more critical, since we are comparing per game. E.g. Catan has a mean score of 7.21 (meanall), but a mean score of 6.79 by the active users, leading to a delta of 0.43 (delta_active_all). Furthermore, the reviews of active users make up 3% of the reviews for Catan (proportion_active)

# In[ ]:


corr = merge.corr()
corr.style.background_gradient(cmap='coolwarm')


# Aside from the obvious correlations between the set of reviews by active users and all reviews, a few things stand out:
# - The geekscore has a higher correlation with the number of reviews (count) than the average score given (mean). Is this is problem?
# - The more the reviews of games consist of active users, the lower the score (fairly low negative correlation)

# In[ ]:


y = merge['meanall']
x = merge['proportion_active']

df = pd.DataFrame({'X' : x, 'Y' : y})  #we build a dataframe from the data
data_cut = pd.cut(df.X,bins=np.linspace(0,1,num=10))   
grp = df.groupby(by = data_cut)        #we group the data by the cut
ret = grp.aggregate(np.median)         #we produce an aggregate representation (median) of each bin

#plotting
plt.figure(num=None, figsize=(10, 4), facecolor='w', edgecolor='k')
plt.scatter(df.X,df.Y,alpha=.5,s=0.5)
plt.plot(ret.X,ret.Y,'g--',lw=4,alpha=0.5)
plt.xlabel('proportion of reviews given by active users')
plt.ylabel('average rating of game')
plt.show()


# Here you see the small effect that active users have on a lower average score of a game

# In[ ]:


x = merge['countall']
y = merge['proportion_active']

df = pd.DataFrame({'X' : x, 'Y' : y})  #we build a dataframe from the data
data_cut = pd.cut(df.X,bins=np.logspace(0,5,num=30))           #we cut the data following the bins
grp = df.groupby(by = data_cut)        #we group the data by the cut
ret = grp.aggregate(np.median)         #we produce an aggregate representation (median) of each bin

#plotting
plt.figure(num=None, figsize=(10, 4), facecolor='w', edgecolor='k')
plt.xscale('log')
plt.scatter(df.X,df.Y,alpha=.5,s=0.5)
plt.plot(ret.X,ret.Y,'g--',lw=4,alpha=0.5)
plt.xlabel('number of reviews for a game')
plt.ylabel('proportion of reviews given by active users')
plt.show()


# This makes sense, since games like Catan that have a huge number of reviews (85268) by majority are rated by casual users, since there are only about 2700 active users with >500 reviews.
# 
# A more 'clean' way to investigate this is to investigate per game how active it userbase is. How active users are can be defined by the mean number of reviews of the userbase.

# In[ ]:


merged_reviews = reviews.merge(reviews_by_user_count,how='left',on='user',suffixes=('','user'),indicator=True)
games_test = merged_reviews.groupby('name')[['rating','count']].agg(['mean','median','count']).sort_values(('count', 'count'),ascending=False)
games_test[:5]


# Catan has a mean rating of 7.2, with median score of 7.0 and 84613 reviews. The users that reviewed it rated on average 105 games, with a median of 51.
# The median is the best metric to gauge how active the users are, since it filters away the effect of extremely active users on the mean.

# In[ ]:


corr = games_test.corr()
corr.style.background_gradient(cmap='coolwarm')


# Again, positive correlations for the number of reviews and scores, and negative correlations for the activity of the users that rated a game and the score

# In[ ]:


games_test['count','median'].hist(bins=50)


# Nice distribution, lets investigate.

# In[ ]:


games_test = games_test.sort_values(('count', 'median'),ascending=True)
games_test[games_test['rating', 'count']>0][:30]


# The wall of shame: these games are all reviewed by users that are suspiciously inactive. Probably rated by fake accounts. But also not so many reviews. Kill the unicorns managed to get 200 though, with an median user that reviewed only 4 games.

# In[ ]:


y = games_test[games_test['rating', 'count']>0]['rating', 'mean']
x = games_test[games_test['rating', 'count']>0]['count', 'median']

df = pd.DataFrame({'X' : x, 'Y' : y})  #we build a dataframe from the data
data_cut = pd.cut(df.X,bins=np.logspace(0,3,num=40))   
grp = df.groupby(by = data_cut)        #we group the data by the cut
ret = grp.aggregate(np.median)         #we produce an aggregate representation (median) of each bin

#plotting
plt.figure(num=None, figsize=(10, 4), facecolor='w', edgecolor='k')
plt.xscale('log')
plt.scatter(df.X,df.Y,alpha=.5,s=0.5)
plt.plot(ret.X,ret.Y,'g--',lw=4,alpha=0.5)
plt.xlabel("median number of reviews of a game's userbase (the higher the more active the userbase of a game)")
plt.ylabel('average score for a game')
plt.show()


# The more active the userbase, the lower the average score a game receives.

# In[ ]:


x = games_test['rating', 'count']
y = games_test['count', 'median']


df = pd.DataFrame({'X' : x, 'Y' : y})  #we build a dataframe from the data
data_cut = pd.cut(df.X,bins=np.logspace(0,5,num=30))   
grp = df.groupby(by = data_cut)        #we group the data by the cut
ret = grp.aggregate(np.median)         #we produce an aggregate representation (median) of each bin

#plotting
plt.figure(num=None, figsize=(10, 4), facecolor='w', edgecolor='k')
plt.xscale('log')
plt.scatter(df.X,df.Y,alpha=.5,s=0.5)
plt.plot(ret.X,ret.Y,'g--',lw=4,alpha=0.5)
plt.xlabel('number of reviews per game')
plt.ylabel("median number of reviews of a game's userbase")
plt.show()


# Here you see the effect again of games with a very high amount of ratings.
# 
# All in all, making causal inferences based on a historical data is like threading on thin ice. I do think experienced users give lower ratings, so are more critical. But concluding that games are ending up with a lower average rating because of these experienced reviewers is probably a step too far.
# 
# That's it. I enjoyed making my first dataset and kernel. Would appreciate any comments!!
