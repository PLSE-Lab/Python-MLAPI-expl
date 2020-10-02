#!/usr/bin/env python
# coding: utf-8

# To start working on it, the first step to process the data for the analysis, as the data is in JSON, I am using import json lib.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
json_data=open('../input/cleaned_twitter_elections_data.json').read()
json_text=json.loads(json_data)
# Any results you write to the current directory are saved as output.


# In the general election of Pakistan 2018, the author share the tweets dataset of 4 days, The Analysis start but before start, there still some normalization require to analyse the data, so let's extract some further nested details :)

# In[ ]:


hashtags=[]
for iterator in json_text:
    for a in iterator['entities']['hashtags']:
        hashtags.append(a['text'])
unique_hashtags=list(set(hashtags))
count_of_hashtags={'hashtags':unique_hashtags,'count':[0 for a in range(len(unique_hashtags))]}
for a in range(len(unique_hashtags)):
    count_of_hashtags['count'][a]=hashtags.count(unique_hashtags[a])    
for a in range(4):
    print(count_of_hashtags['hashtags'][a],count_of_hashtags['count'][a])


# We already have hashtags with count, let prepare DataFrame()

# In[ ]:


df=pd.DataFrame(count_of_hashtags)
df.head()


# **AND the top 20 hashtags ARE**

# In[ ]:


df.sort_values(by='count',ascending=False).head(20)


# Now we are only working on top 10 tweets with highest retweet and hashtags

# In[ ]:


map_plot = df.sort_values(by='count',ascending=False).head(10)


# Let's visualize what we have done so far with top 30 tweets that has highest retweet

# In[ ]:


objects= map_plot['hashtags']
y_pos = np.arange(len(map_plot['hashtags']))
plt.bar(y_pos, map_plot['count'], align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('Retweets')
plt.title('Top 10 Hashtag with Highest Retweets')

plt.show()


# As we have JSON format file, we require more data processing to get more details related to each tweet, below is code to prepare twitter JSON data in array

# In[ ]:


hashtag=[]
mention=[]
created_at=[]
lang=[]
description=[]
favourites_count=[]
friends_count=[]
name=[]
tweet=[]
retweet_count=[]
followers_count=[]
location=[]
name=[]
id_str=[]
retweeted=[]
for iterator in json_text:
    hashtag.append(' '.join([a['text'] for a in iterator['entities']['hashtags']]))
    mention.append(' '.join([a['name'] for a in iterator['entities']['user_mentions']]))
    created_at.append(iterator['created_at'])
    lang.append(iterator['lang'])
    if 'retweeted_status' in  iterator:
        retweet_count.append(iterator['retweeted_status']['retweet_count'])
    else:
        retweet_count.append(0)
    
    description.append(iterator['user']['description'])
    favourites_count.append(iterator['user']['favourites_count'])
    followers_count.append(iterator['user']['followers_count'])
    friends_count.append(iterator['user']['friends_count'])
    location.append(iterator['user']['location'])
    name.append(iterator['user']['name'])
    tweet.append(iterator['text'])
    retweeted.append(iterator['retweeted'])

df_dict={'hashtag':hashtag,
'mention':mention,
'created_at':created_at,
'lang':lang,
'description':description,
'favourites_count':favourites_count,
'friends_count':friends_count,
'name':name,
'tweet':tweet,
'retweet_count':retweet_count,
'followers_count':followers_count,
'location':location,
'retweeted':retweeted
}
df_all=pd.DataFrame(df_dict)
df_all.head(5)


# We have all the required data now let's visulize with respect to location (Country & City)

# In[ ]:


tweet_loc  = df_all.groupby('location')['location'].count().sort_values(ascending=False).head(20)
tl_df = pd.DataFrame({'location':tweet_loc.index, 'count': tweet_loc.values})
y_pos = np.arange(len(tl_df['location']))

plt.bar(y_pos, tl_df['count'], align='center', alpha=0.2)
plt.xticks(y_pos, tl_df['location'],rotation='vertical')
#plt.text(v + 9, i + .25, str(v), color='blue', fontweight='bold')
for index,data in enumerate(tl_df['count']):
    plt.text(x=index , y =data+5 , s=f"{data}" ,horizontalalignment='center', fontdict=dict(fontsize=7))
plt.ylabel('Count')
plt.title('Locations')
plt.show()


# Analysis Conclusion:
# 1. In the election 2018 the Karachi city post highest tweets
# 2. Overall the people support General election hashtag but PML(N) hashtag has higher tweet compare to hashtag PTI.
# 3. In the election campaign International community also retweeted the tweets.
