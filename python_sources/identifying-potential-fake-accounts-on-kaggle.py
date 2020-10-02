#!/usr/bin/env python
# coding: utf-8

# # Strange Activity with some kernels. Potential fake accounts behind downvoting?
# 
# After metnioning to a user that they were copying other's work without referencing them, I noticed my comment that once had 4 upvotes, suddenly had -5 votes. I found this strange that so many people at once downvoted my comment. After a little investigation I noticed something strange that leads me to believe fake accounts were created.
# 
# **Update**: This thread brought more attention to the matter and I decided to make this kernel public: https://www.kaggle.com/product-feedback/76394

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt


# ## User in question
# The first thing I noticed was that the user in question had a handful of followers with generic sounding names, that only followed this user.

# In[ ]:


user = pd.read_csv('../input/Users.csv')
KernelVotes = pd.read_csv('../input/KernelVotes.csv')
Kernels = pd.read_csv('../input/Kernels.csv')
# Our guy
user.loc[user['Id'] == 949803]


# In[ ]:


his_kernels = Kernels.loc[Kernels['AuthorUserId'] == 949803]


# ## How many kernels does this user have?
# Many of them are tutorials and consist of the same material. The kernel `awesome-data-science-with-nfl-punt-analytics` is the kernel that I commented on. And it has an impressive 30+ votes.

# In[ ]:


len(his_kernels)


# In[ ]:


his_kernels.set_index('CurrentUrlSlug')['TotalVotes'].sort_values().plot(kind='barh',
                                                                         figsize=(15, 18),
                                                                         title='Kernels and Votes',
                                                                         color='grey')
plt.show()


# ## Who voted on these kernels?
# A user with 34 kernels. Oddly there are a group of users who upvoted vote on 20+ of them. This isn't too strange. Maybe these are just his friends?
# 
# **Update** It appears there are only 23 kernels by this user now. Maybe some were taken down. Also the number of votes by these potential fake accounts has changed. Maybe the user caught on and changed their voting patters to only 10 votes on a single kernel.

# In[ ]:


his_kernels_list = his_kernels['CurrentKernelVersionId'].tolist()
his_votes = KernelVotes.loc[KernelVotes['KernelVersionId'].isin(his_kernels_list)].copy()
his_votes['count'] = 1
his_votes.groupby('UserId').count()[['count']].sort_values('count',
                                                           ascending=False) \
    .plot(kind='bar', figsize=(15, 5), title='Number of Votes on His Kernel by User')
plt.show()


# In[ ]:


his_vote_user_count = his_votes.groupby('UserId').count()[['count']].sort_values('count', ascending=False)
his_vote_user_count_more_than_5 = his_vote_user_count.loc[his_vote_user_count['count'] >= 5]


# ## What other kernels do these users vote on? Maybe they just like upvoting?
# Nope. While they have all voted enough to have a `contributor` status. They all seem to overlap voting for this one user.

# In[ ]:


kernel_votes_with_kernel_details = pd.merge(KernelVotes, Kernels, how='left')
kernel_votes_with_kernel_details = pd.merge(kernel_votes_with_kernel_details, user, left_on='AuthorUserId', right_on='Id', how='left')
kernels_voted_on_by_friends = kernel_votes_with_kernel_details.loc[kernel_votes_with_kernel_details['UserId']
                                                                   .isin(his_vote_user_count_more_than_5.index.tolist())]


# In[ ]:


ax = KernelVotes.loc[KernelVotes['UserId'].isin(his_vote_user_count_more_than_5.index.tolist())]     .groupby('KernelVersionId')     .count()     .sort_values('Id')['Id'].plot(kind='bar', figsize=(15, 5), title='Each bar is a kernel voted on my one of these friends')
x_axis = ax.axes.get_xaxis()
x_axis.set_visible(False)
plt.show()


# In[ ]:


# What kernels do the friends vote on?
KernelVotes['count'] = 1
kernels_friends_voted_on = KernelVotes.loc[KernelVotes['UserId'].isin(his_vote_user_count_more_than_5.index.tolist())]     .groupby('KernelVersionId')     .count()[['count']]


# In[ ]:


kernels_friends_voted_on_more_than_once = kernels_friends_voted_on.loc[kernels_friends_voted_on['count'] > 1]
kernels_with_author_info = pd.merge(Kernels, user, left_on='AuthorUserId', right_on='Id')

kernels_with_author_info.loc[kernels_with_author_info['CurrentKernelVersionId'].isin(kernels_friends_voted_on_more_than_once.index.tolist())]     .groupby('UserName')     .count()[['Id_x']]     .sort_values('Id_x').plot(kind='barh', title='Count of Kernels Friends voted on more than once by Author',
                              figsize=(15, 5),
                              legend=False)
plt.show()


# ## Who are these friends?
# 
# Interesting... Two of them are DeepBrainz related.
# - 3 users created on 12/4/2018
# - 4 more users created 12/22/2018
# - Many use the same format `firstname` `last initial`
# - `Bill G` `Jobs S` `Cook T` - those names seem familiar.

# In[ ]:


his_friends = his_vote_user_count_more_than_5.reset_index()['UserId'].tolist()
user.loc[user['Id'].isin(his_friends)]


# ## Who do they follow?

# In[ ]:


following = pd.read_csv('../input/UserFollowers.csv')
following['count'] = 1
following_counts = following.loc[following['UserId'].isin(his_friends)]     .groupby('FollowingUserId')     .count()     .sort_values('count')[['count']].reset_index()


# In[ ]:


# All The Users these 'Friends are following'
pd.merge(following_counts.loc[following_counts['count'] > 1],
         user, left_on='FollowingUserId', right_on='Id', how='left')


# In[ ]:


pd.merge(following_counts.loc[following_counts['count'] > 1],
         user, left_on='FollowingUserId', right_on='Id', how='left') \
    .plot(x='UserName', y='count', kind='barh', title='Users who These Profiles follow (min 1)',
          legend=False, figsize=(15, 5))
plt.show()


# ## A closer look at each profile reveals that they all have generic titles at major corporations.
# 
# - https://www.kaggle.com/jobss11  
# - https://www.kaggle.com/cookt11  
# - https://www.kaggle.com/satyams2018  
# - https://www.kaggle.com/sundarml18  
# - https://www.kaggle.com/bill2k18  
# - https://www.kaggle.com/pagel98  
# - https://www.kaggle.com/brins04 

# In[ ]:




