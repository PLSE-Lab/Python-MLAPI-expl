#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Importing Libraries
# 
# We are going to use basics data analysis libraries along with seaborn for plotting

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Reading the datasets 

# In[ ]:


episodes = pd.read_csv('../input/chai-time-data-science/Episodes.csv')


# In[ ]:


episodes.head()


# df.info() to get information about the dataframe

# In[ ]:


episodes.info()


# In[ ]:


yt_thumbnails = pd.read_csv('../input/chai-time-data-science/YouTube Thumbnail Types.csv')


# In[ ]:


yt_thumbnails


# In[ ]:


an_thumbnails = pd.read_csv('../input/chai-time-data-science/Anchor Thumbnail Types.csv')


# In[ ]:


an_thumbnails


# In[ ]:


desc = pd.read_csv('../input/chai-time-data-science/Description.csv')


# ## This data set contains the description of the episode along with their episode number

# In[ ]:


desc.head()


# ## Let's clean this data and make new columns so we can find the most commnly used words.

# In[ ]:


import re
import string

def preprocess_two(x):
    x = x.lower()
    x = re.sub(r'@[A-Za-z0-9]+','',x)#remove usernames
    x = re.sub(r'^rt[\s]+', '', x)
    x = re.sub('https?://[A-Za-z0-9./]+','',x) # remove url
    x = re.sub(r'#([^\s]+)', r'\1', x)# remove hastags
    x = re.sub('[^a-zA-Z]', ' ', x)
    #x= re.sub(r':','',x)
    #tok = WordPunctTokenizer()
    #words = tok.tokenize(x)
    #x = (' '.join(x)).strip()
    #return [word for word in x if word not in stopwords.words('english')]
    
    clean_list.append(x)
    return x
    
    
desc_list= desc['description'].tolist()
clean_list=[]

for i in desc_list:
    preprocess_two(i)

desc['cleaned desc'] = clean_list


# In[ ]:


desc.head()


# In[ ]:


from wordcloud import WordCloud, STOPWORDS

unique_string=(" ").join(clean_list)
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)


# ## Flavours of tea used during the shows

# In[ ]:


plt.figure(figsize=(15,4))
sns.countplot(x="flavour_of_tea",data=episodes,palette='BrBG')


# ## Let's explore which flavours are most used by genders

# In[ ]:


plt.figure(figsize=(15,4))
plt.xticks(rotation=90)
sns.countplot(x="flavour_of_tea",data=episodes,palette='BrBG',hue='heroes_gender')


# We can see that Ginger chai is the most liked tea among males and Masala/Ginger chai most liked by females

# ## Let's check flavours of chai preferred by the categories

# In[ ]:


plt.figure(figsize=(15,8))
plt.legend(loc=2)
sns.countplot(x='category',data=episodes,hue='flavour_of_tea',palette='BrBG')


# Most of the heroes in the kaggle category prefer ginger chai, while the second most liked flavour is kesar rose chai by 'other category'. The industry category likes herbal chai the most

# ## Let's check flavour of tea by time of the day

# In[ ]:


plt.figure(figsize=(15,8))
plt.xticks(rotation=90)
sns.countplot(x='flavour_of_tea',data=episodes,hue='recording_time',palette='BrBG')


# Most preffered flavour by day -->
# * Morning - Masala Chai
# * Night - Kesar chai 
# * Afternoon - Masala chai, Ginger chai, Sulemani chai, Herbal tea, Paan Rose Green Tea
# * Evening - Herbal Tea
# 

# ## Gender Distribution

# In[ ]:


sns.countplot(x='heroes_gender',data=episodes,palette='BrBG')


# As we can see most of the heroes are males

# ## Most Discusses category

# In[ ]:


sns.countplot(x='category',data=episodes,palette='BrBG')


# 1. Industry
# 2. Kaggle
# 3. Other
# 4. Research

# ## Time of the day where most of the discussions are held

# In[ ]:


sns.barplot(x='recording_time',y='episode_duration',data=episodes,palette="BrBG")


# Most discussions are during the morning!
# Surprisingly the second most preferred time is afternoon!

# ## Preferred time of the day based on genders

# In[ ]:


sns.barplot(x='recording_time',y='episode_duration',data=episodes,palette="BrBG",hue='heroes_gender')


# Both the genders prefer night time.
# The least prefer time for females and males is evening

# ## Let's check how the time of the day relates to the duration of the show

# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(x='recording_time',y='episode_duration',data=episodes,palette="BrBG",hue='category')


# Night time recordings have the nighest durations. Most of the Kaggle and Research categories are discussed in the night.
# 'Other' category topics are least discussed in the evening and most discussed in the morning. The two most discussed categories are Kaggle and Industry.
# 

# ## Top 10 heroes with the most youtube subs

# In[ ]:


subs=episodes.sort_values(by=['youtube_subscribers'],ascending=False)
plt.figure(figsize=(15,4))
plt.tight_layout()
plt.xticks(rotation=90)
sns.barplot(x='heroes',y='youtube_subscribers',data=subs.head(10),palette='BrBG')


# As we can see Jeremy Howard has the most subs on youtube and Rohan Rao has the least in the 10th place. Parul Pandey has the second most subs

# ## Categories with the most views

# In[ ]:


sns.barplot(x='category',y='youtube_views',data=episodes,palette='BrBG')


# The Industry category has the most views on youtube followed by kaggle. 'Other' category has the least views.
# 

# ### So far we can see that Industry Kaggle and Research categories are the most discussed ones

# ## Categories that have the most youtube likes

# In[ ]:


sns.barplot(x='category',y='youtube_likes',data=episodes,palette='BrBG')


# Industry based videos have the most likes followed by Kaggle. Other has the least

# ## Youtube dislikes based on categories

# In[ ]:


sns.barplot(x='category',y='youtube_dislikes',data=episodes,palette='BrBG')


# Again we can see that Industry has the most dislikes and Research has no dislikes!

# ## Episodes and their durations

# In[ ]:


plt.figure(figsize=(15,4))
plt.xticks(rotation=90)
plt.tight_layout()
sns.lineplot(x='episode_id',y='episode_duration',data=episodes)


# As we can see, all of the episodes have are not evenly timed.

# In[ ]:


plt.figure(figsize=(15,4))
plt.xticks(rotation=90)
plt.tight_layout()
sns.lineplot(x='episode_id',y='episode_duration',data=episodes,hue='category')


# Episode 23 is the episode with the highest recording duration and it belongs to Kaggle topic.
# The second highest is Episode 63 and it belongs to Industry followed by Episode 5 which belongs to research

# ## Let's see which episodes have the highest watch duration on youtube

# In[ ]:


plt.figure(figsize=(15,4))
plt.xticks(rotation=90)
plt.tight_layout()
sns.lineplot(x='episode_id',y='youtube_avg_watch_duration',data=episodes,hue='category')


# Industry has the highest watch times followed by Kaggle. Research has one episode with the most watch time

# ## Episodes which are listend to the most on itunes

# In[ ]:


plt.figure(figsize=(15,4))
plt.xticks(rotation=90)
plt.tight_layout()
sns.lineplot(x='episode_id',y='apple_avg_listen_duration',data=episodes,hue='category')


# Kaggle has the highest listen duration, followed by industry

# In[ ]:


plt.figure(figsize=(15,4))
plt.xticks(rotation=90)
plt.tight_layout()
sns.lineplot(x='episode_id',y='spotify_streams',data=episodes,hue='category')


# ## Let's see which of the heros are popular across platforms 

# In[ ]:


plt.figure(figsize=(8,10))
sns.barplot(y='heroes',x='anchor_plays',data = episodes.sort_values(by=['anchor_plays'],ascending=False).head(10),palette='BrBG')


# Jeremy Howard has the most anchor plays followed by Andrey Lukyanenko. Now let see who has the least

# In[ ]:


plt.figure(figsize=(5,7))
sns.barplot(y='heroes',x='anchor_plays',data = episodes.sort_values(by=['anchor_plays'],ascending=True).head(10),palette='BrBG')


# Rachel Thomas has the least with only 250 anchor plays

# ## Let's check which heros got the most views on youtube

# In[ ]:


plt.figure(figsize=(8,12))
sns.barplot(y='heroes',x='youtube_avg_watch_duration',
            data=episodes.sort_values(by=['youtube_avg_watch_duration'],ascending=False).head(10),palette='BrBG')


# Since Robert Bracco has uncertainity in data, we can say that Tim Dettmers has the most average watch time on youtube.

# ## Let's see who got the most views

# In[ ]:


plt.figure(figsize=(8,12))
sns.barplot(y='heroes',x='youtube_views',
            data=episodes.sort_values(by=['youtube_views'],ascending=False).head(10),palette='BrBG')


# Jeremy Howard has the most views, followed by Parul Pandey and Jean Francois Puget has among top 10.

# ## Most plays on spotify

# In[ ]:


plt.figure(figsize=(8,12))
sns.barplot(y='heroes',x='spotify_streams',
            data=episodes.sort_values(by=['spotify_streams'],ascending=False).head(10),palette='BrBG')


# Abhishek Thakur is the most streamed on spotify, followed by Jeremy Howard. 
# I guess this is the only play Jeremy Howard isin't at top xD

# In[ ]:


plt.figure(figsize=(8,12))
sns.barplot(y='heroes',x='apple_listeners',
            data=episodes.sort_values(by=['apple_listeners'],ascending=False).head(10),palette='BrBG')


# Jeremy Howard, again, has the highest apple listeners followed by Abhishek Thakur.

# ## Nationality of the heros that appear on CTDS

# In[ ]:


plt.figure(figsize=(15,4))
plt.tight_layout()
plt.xticks(rotation=90)
sns.countplot(x='heroes_nationality',data=episodes,palette='BrBG')


# Most of the heros are from USA, followed by France and then India

# ## Heroes belonging to a nationality but live in different locations

# In[ ]:


cor=episodes.groupby(['heroes_location','heroes_nationality']).count()['heroes'].unstack()


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(cor,cmap='YlGnBu',annot=True)


# This is a heatmap to visualize Nationality vs Location. At the time of recording some heroes belong to A but live in B.
# 
# The above heatmap gives a count of the heros based on the nationality and where they currently are during the recording.
# 
# As we can see there is 1 hero who's nationality is Africa but is living in USA. Similarly 3 heroes are from France but live in USA.

# In[ ]:




