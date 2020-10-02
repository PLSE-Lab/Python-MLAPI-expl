#!/usr/bin/env python
# coding: utf-8

# # Topic modeling for tweets on election

# In[ ]:


get_ipython().system('pip install mglearn')
import re
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import mglearn
import matplotlib.pyplot as plt


# ### Read tweet data

# In[ ]:


tweet_df = pd.read_csv('../input/auspol2019.csv')


# In[ ]:


tweet_df.head(5)


# ### Execute topic modeling
# 
# Divide tweets into two topic groups by topic modeling.

# In[ ]:


tweet_df['full_text2'] = tweet_df['full_text'].map(lambda x: re.sub('[,\.!?]', '', x))                                       # remove ,.!?
tweet_df['full_text2'] = tweet_df['full_text2'].map(lambda x: re.sub('#[A-Za-z0-9]+', '', x))                       # remove hashtag
tweet_df['full_text2'] = tweet_df['full_text2'].map(lambda x: re.sub('https://t.co/[A-Za-z0-9]+', '', x))  # remove link


vect = CountVectorizer(max_features=100, max_df=.15, stop_words='english')
X = vect.fit_transform(tweet_df["full_text2"])

topics =2
lda = LatentDirichletAllocation(n_components = topics, learning_method="batch", max_iter=5, random_state=0)
document_topics = lda.fit_transform(X)
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=range(topics), feature_names=feature_names, sorting=sorting, topics_per_chunk=5, n_words=10)


# ### Summary of results
# 
# - The two major topics are about two leaders, Scott Morrison and Bill Shorten (as exptected).
# - One of the most interested topics related to Bill Shorten seems climate changes.
# 
# ### Calculate topic score for each tweet
# 
# - Score = Topic1(Bill Shorten) - Topic0(Scott Morrison)
# - Tweets related to topics of Bill Shorten have scores close to 1, while Scott Morrison topics have scores close to -1.
# - Neutral topics have scores close to zero.

# In[ ]:


topic_df = pd.DataFrame(document_topics, columns=['topic0', 'topic1'])
tweet_df = pd.concat([tweet_df.reset_index(drop=True), topic_df.reset_index(drop=True)],axis =1)
tweet_df["topic_diff"] = tweet_df['topic1']-tweet_df['topic0']
tweet_df.head()


# ### See the distribution
# 
# 

# In[ ]:


plt.hist(tweet_df['topic_diff'], 20)
plt.xlabel("<--- Scott Morrison    Bill Shorten -->        ")
plt.show()


# - It seems that the distribution has three peaks; Scott Morrison topics(-.75 to -.50), neutral topics (around 0.0), Bill Shoren topics(.50 to .75)
# 
# ### Check the percentiles

# In[ ]:


tweet_df.describe(percentiles=[.95, .99, .999])


# ### What did they say in their popular tweets related to Scott Morrison and Bill Shorten?
# 
# We are going to check tweets meeting the following conditions.
# - They have a lot of favorite counts (over 99.9 percentile (=1224)).
# - Their topics are about either Scott Morrison (topic score < .75) or Bill Shorten (topic score > .75).
# 
# #### Favored tweets related to Scott Morrison

# In[ ]:


th99 = 1224

tweet0 = tweet_df.query('topic_diff < -0.75 & favorite_count > ' +str(th99))

tw0= tweet0['full_text'].values

for tw in tw0:
    print('\n' + tw.replace('\n', ' '))


# #### Favored tweets related to Bill Shorten

# In[ ]:


tweet1 = tweet_df.query('topic_diff > 0.75 & favorite_count > ' + str(th99))
tw1= tweet1['full_text'].values

for tw in tw1:
    print('\n' + tw.replace('\n', ' '))


# ### Summary
# 
# We can see the characteristics of these opinions as follows.
# 
# - There are a lot of emotional messages, rather than news contents conveying objective information.
# - They can be divided into the following three categories.
#   1. Joy in the victory
#   1. Grudge or anger at the defeat
#   1. Encouragement for voting

# In[ ]:




