#!/usr/bin/env python
# coding: utf-8

# This is not going to be a nice notebook to read nor it wants to be a personal attack against anyone but rather a quick snapshot of how healthy is our community.
# 
# It has been a while that I notice more and more users getting high kaggle ranks in a short amount of time (people becoming expert within a day, or masters in a month) and, reading the forums very frequently, I have also noticed that the comments are becoming more and more in the lines of *thanks for sharing* and *please upvote my other work*.
# 
# Very often I notice nice notebooks being published with even some non-trivial approach in it that could spark interesting conversations and a little army of users that just post links to their own notebook about another topic, completely unrelated to what they are commenting.
# 
# Therefore, I decided to check if it is me getting grumpy the older I get or the quality of the forums is significantly dropping.
# 
# This notebook will focus on 4 things: 
# 
# * Average number of questions asked over time
# * Average number of requests of upvotes over time
# * Average number of self-promoting comments over time
# * Average number of thank you over time
# 
# To do so, Meta Kaggle is the perfect dataset.

# In[ ]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
users['RegisterDate'] = pd.to_datetime(users['RegisterDate'])
users['Year'] = users['RegisterDate'].dt.year

kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
kernels['CreationDate'] = pd.to_datetime(kernels['CreationDate'])
kernels['Year'] = kernels['CreationDate'].dt.year


# In[ ]:


messages = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')

messages['Thank'] = messages.Message.str.lower().str.contains('thank|great work|good work').fillna(0).astype(int)
messages['Upvote'] = messages.Message.str.lower().str.contains('upvote').fillna(0).astype(int)
messages['Promotion'] = messages.Message.str.lower().str.contains('check my notebook|check my kernel|my other notebook|my other kernel|check out my notebook|check out my kernel').fillna(0).astype(int)
messages['Question'] = messages.Message.str.lower().str.contains('why|how|what|"\?"').fillna(0).astype(int)
messages['Medal'] = messages.Medal.fillna(0)
messages['PostDate'] = pd.to_datetime(messages['PostDate'])
messages['Year'] = messages.PostDate.dt.year
messages['Week'] = messages.PostDate.dt.week

messages.head()


# In[ ]:


fig = plt.figure(figsize=(12, 24), facecolor='#f7f7f7') 
fig.subplots_adjust(top=0.95)
fig.suptitle('Evolution of Comments on Kaggle', fontsize=18)

gs = GridSpec(5, 2, figure=fig)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, :])
ax4 = fig.add_subplot(gs[3, :])
ax5 = fig.add_subplot(gs[4, :])

pd.Series(users.groupby('Year').size()).plot(style='.-', ax=ax0, color='black')
pd.Series(messages.groupby('Year').size()).plot(style='.-', ax=ax1, color='black')
messages.groupby(['Year'])[['Question']].mean().plot(style='.-', ax=ax2, color='green')
messages.groupby(['Year'])[['Thank']].mean().plot(style='.-', ax=ax3, color='firebrick')
messages.groupby(['Year'])[['Upvote']].mean().plot(style='.-', ax=ax4, color='darkorange')
messages.groupby(['Year'])[['Promotion']].mean().plot(style='.-', ax=ax5, color='darkblue')

ax0.set_title('Number of new users per year', fontsize=14)
ax1.set_title('Number of comments per year', fontsize=14)
ax2.set_title('Average number of comments with a question', fontsize=14)
ax3.set_title('Average number of comments with a thank you', fontsize=14)
ax4.set_title('Average number of comments with a request of upvote', fontsize=14)
ax5.set_title('Average number of comments promoting a notebook', fontsize=14)

for ax in [ax2, ax3, ax4, ax5]:
    ax.legend().set_visible(False)


plt.show()


# One hand, yes, it is just me being more and more grumpy the older I get, since the requests of upvotes are about 2.5% (excluding the one contained in the Notebooks) and the self promotion is around 0.2% of the comments. However, it is undeniable how the trend is to ask less and less questions and the thirst for upvotes is getting bigger and bigger.
# 
# If we focus on the comments on Notebooks, we notice that the requests for upvotes are twice as many than the average.

# In[ ]:


comm_kernels = kernels[kernels.ForumTopicId.notna()].copy()

comm_kernels = pd.merge(comm_kernels, messages[['ForumTopicId', 'Medal', 'Question', 'Thank', 'Upvote', 'Promotion']], on='ForumTopicId')

comm_kernels.head()


# In[ ]:


fig = plt.figure(figsize=(12, 24), facecolor='#f7f7f7') 
fig.subplots_adjust(top=0.95)
fig.suptitle('Evolution of Comments on Kaggle Notebooks', fontsize=18)

gs = GridSpec(5, 2, figure=fig)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, :])
ax4 = fig.add_subplot(gs[3, :])
ax5 = fig.add_subplot(gs[4, :])

pd.Series(comm_kernels.groupby('Year').size()).plot(style='.-', ax=ax0, color='black')
comm_kernels.groupby('Year').TotalComments.sum().plot(style='.-', ax=ax1, color='black')
comm_kernels.groupby(['Year'])[['Question']].mean().plot(style='.-', ax=ax2, color='green')
comm_kernels.groupby(['Year'])[['Thank']].mean().plot(style='.-', ax=ax3, color='firebrick')
comm_kernels.groupby(['Year'])[['Upvote']].mean().plot(style='.-', ax=ax4, color='darkorange')
comm_kernels.groupby(['Year'])[['Promotion']].mean().plot(style='.-', ax=ax5, color='darkblue')

ax0.set_title('Number of Notebooks created per year', fontsize=14)
ax1.set_title('Number of comments on Notebooks per year', fontsize=14)
ax2.set_title('Average number of comments with a question', fontsize=14)
ax3.set_title('Average number of comments with a thank you', fontsize=14)
ax4.set_title('Average number of comments with a request of upvote', fontsize=14)
ax5.set_title('Average number of comments promoting a notebook', fontsize=14)

for ax in [ax2, ax3, ax4, ax5]:
    ax.legend().set_visible(False)


plt.show()


# Which is sad because many users put a lot of effort in creating quality content (not me, not this notebook) and all they get back is glorified spam.
# 
# Now, I don't deny it is nice the feeling of getting upvoted and a community that says thank you more often is not a negative thing. However, I fear the quality of the conversation in a community of data scientists eager to learn everything about their field is going to deteriorate to a simple hunt for upvotes like any forgettable social media.
# 
# The Kaggle ranking system, as far as I know, decays the points, not the achievements. Everyone will eventually become a Kaggle expert if we simply continue being active. There is no point in getting there faster and it doesn't build a portfolio that a company will appreciate (unless you are applying for social media manager, in that case your ability of getting likes is a must).
# 
# I don't want to open the pandora box of users putting upvotes only to get upvoted, or texting you in private on LinkedIn to arrange some sort of upvotes trade. None of this is against the rules and it gets, at worst, in the category of the poor taste but the reason we are here is the quality of the community and if we undermine it we won't have a place to learn the next new thing.
# 
# Speaking of poor taste, there are so many ways to get upvotes, like slapping a fairly non informative wordcloud

# In[ ]:


text = messages.Message.values
wordcloud = WordCloud(
    width = 1000,
    height = 500,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:




