#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from warnings import filterwarnings
filterwarnings('ignore')


#  # API Connection

# In[ ]:


get_ipython().system('pip install tweepy')


# In[ ]:


import tweepy, codecs



consumer_key = '5ecrgpF8KDhSZlvM8o0xSASVh'
consumer_secret = '5MdKdPRMpgGylXBZUlHXPwnxsgwjczusIeYw1JMTUUGQ0Dcdpi'
access_token = '330297491-56WABQRvTJnWTLG7bMnJT27Jlt9T5TYwUPiTR7hh'
access_token_secret = 'aCmW6YbiBjIycroI5zmoXGgpsnN46P5rXbCfx69KmRBMt'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)


# In[ ]:


# we can tweet with this code
# api.update_status('hello from python')


# In[ ]:


# we can see our friends on twitter with this code
#api.friends()


# # Pull data from Twitter 

# In[ ]:


fk = api.me()


# In[ ]:


# you can follow me :)
fk.screen_name


# In[ ]:


fk.followers_count


# In[ ]:


#fk.friends


# In[ ]:


for friend in fk.friends(count=10):
    print(friend.screen_name)


# In[ ]:


dir(fk)


# In[ ]:


# she is my sister
user = api.get_user(id = 'bsrakrsn')


# In[ ]:


user.screen_name


# In[ ]:


user.followers_count


# In[ ]:


user.profile_image_url


# ## home timeline

# In[ ]:


public_tweets = api.home_timeline(count=10)


# In[ ]:


for tweet in public_tweets:
    print(tweet.text)


# ## user time line 

# In[ ]:


name = 'AndrewYNg'
tweet_count = 10

user_timeline = api.user_timeline(id = name, count=tweet_count)

for i in user_timeline:
    print(i.text)


# ## retweets

# In[ ]:


retweets = api.retweets_of_me(count=10)
for retweet in retweets:
    print(retweet.text)


# In[ ]:


retweets


# ## hastag

# In[ ]:


results = api.search(q = '#datascience',
                    lang = 'tr',
                    result_type = 'recent',
                    count = 1000000 )


# ## convert to dataframe 

# In[ ]:


import pandas as pd


# In[ ]:


def tweets_df(results):
    id_list = [tweet.id for tweet in results]
    data_set = pd.DataFrame(id_list,columns=['id'])
    
    
    data_set['text'] = [tweet.text for tweet in results]
    data_set['created_at'] = [tweet.created_at for tweet in results]
    data_set['retweet_count'] = [tweet.retweet_count for tweet in results]
    data_set['name'] = [tweet.author.name for tweet in results]
    data_set['user_screen_name'] = [tweet.author.screen_name for tweet in results]
    data_set['user_followers_count'] = [tweet.author.followers_count for tweet in results]
    data_set['user_location'] = [tweet.author.location for tweet in results]
    data_set['Hashtags'] = [tweet.entities.get('hashtags') for tweet in results]
    
    return data_set


# In[ ]:


data = tweets_df(results)


# In[ ]:


data.head()


# # Profile Analysis

# In[ ]:


AndrewNg = api.get_user('AndrewYNg')


# In[ ]:


AndrewNg.name


# In[ ]:


AndrewNg.id


# In[ ]:


AndrewNg.url


# In[ ]:


AndrewNg.verified


# In[ ]:


AndrewNg.screen_name


# In[ ]:


AndrewNg.statuses_count


# In[ ]:


AndrewNg.favourites_count


# In[ ]:


AndrewNg.friends_count


# In[ ]:


tweets = api.user_timeline(id = 'AndrewYNg')


# In[ ]:


"""for i in tweets:
    print(i.text)"""


# In[ ]:


def timeline_df(tweets):
    id_list = [tweet.id for tweet in tweets]
    data_set = pd.DataFrame(id_list,columns=['id'])
    
    
    data_set['text'] = [tweet.text for tweet in tweets]
    data_set['created_at'] = [tweet.created_at for tweet in tweets]
    data_set['retweet_count'] = [tweet.retweet_count for tweet in tweets]
    data_set['favorite_count'] = [tweet.favorite_count for tweet in tweets]
    data_set['source'] = [tweet.source for tweet in tweets]

    
    return data_set


# In[ ]:


timeline_df(tweets)


# In[ ]:


def timeline_df(tweets):
    df = pd.DataFrame()
    
    df['id'] = list(map(lambda tweet:tweet.id, tweets))
    df['created_at'] = list(map(lambda tweet:tweet.created_at, tweets))
    df['text'] = list(map(lambda tweet:tweet.text, tweets)) 
    df['favorite_count'] = list(map(lambda tweet:tweet.favorite_count, tweets))
    df['retweeted_count'] = list(map(lambda tweet:tweet.retweet_count, tweets))
    df['source'] = list(map(lambda tweet:tweet.source, tweets))
    return df


# In[ ]:


tweets = api.user_timeline(id = 'AndrewYNg',count=10000)


# In[ ]:


df = timeline_df(tweets)


# In[ ]:


df.info()


# In[ ]:


df.sort_values('retweeted_count', ascending= False)


# In[ ]:


df.sort_values('favorite_count', ascending= False)[['text', 'favorite_count']].iloc[0:3]


# In[ ]:


df.sort_values('favorite_count', ascending= False)['text'].iloc[0]


# # Distribution of Retweet & Favorite Counts

# In[ ]:


df.head()


# In[ ]:


get_ipython().run_line_magic('config', "InlineBacend.figure_format = 'retina'")
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.distplot(df.favorite_count, kde=False ,color='blue')
plt.xlim(-100,15000)


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(df.retweeted_count, color='red')
plt.xlim(-100,5000)


# In[ ]:


df['favorite_count'].mean()


# In[ ]:


df['favorite_count'].std()


# ## Distribution of Tweet-Hour  

# In[ ]:


df.head()


# In[ ]:


df['tweet_hour'] = df['created_at'].apply(lambda x: x.strftime('%H'))


# In[ ]:


df.head()


# In[ ]:


df['tweet_hour'] = pd.to_numeric(df['tweet_hour'])


# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(df['tweet_hour'], kde=True, color='blue')


# In[ ]:


df['days'] = df['created_at'].dt.weekday_name


# In[ ]:


df.head()


# In[ ]:


gun_freq = df.groupby('days').count()['id']


# In[ ]:


gun_freq.plot.bar(x='days', y='id')


# ## Source of Tweets

# In[ ]:


source_freq = df.groupby('source').count()['id']


# In[ ]:


source_freq.plot.bar(x='source', y='id')


# In[ ]:


df.groupby('source').count()['id']


# In[ ]:


df.groupby(['source', 'tweet_hour','days'])[['tweet_hour']].count()


# ## Followers and Friends Analysis

# In[ ]:


user = api.get_user(id = 'AndrewYNg', count= 10000)


# In[ ]:


friends = user.friends()
followers = user.followers()


# In[ ]:


def followers_df(follower):
    idler = [i.id for i in follower]
    df = pd.DataFrame(idler, columns=['id'])
    
    
    df['created_at'] = [i.created_at for i in follower]
    df['screen_name'] = [i.screen_name for i in follower]
    df['location'] = [i.location for i in follower]
    df['followers_count'] = [i.followers_count for i in follower]
    df['statuses_count'] = [i.statuses_count for i in follower]
    df['friends_count'] = [i.friends_count for i in follower]
    df['favourites_count'] = [i.favourites_count for i in follower]
    
    return df


# In[ ]:


df = followers_df(followers)


# In[ ]:


df.head()


# In[ ]:


df.info()


# ## Followers Segmentation

# In[ ]:


df.index = df['screen_name']


# In[ ]:


s_data = df[['followers_count', 'statuses_count']]


# In[ ]:


s_data


# In[ ]:


s_data['followers_count'] = s_data['followers_count'] +0.01


# In[ ]:


s_data['statuses_count'] = s_data['statuses_count'] +0.01


# In[ ]:


s_data


# In[ ]:


s_data = s_data.apply(lambda x:(x-min(x)) / (max(x)- min(x))) #doing standardization


# In[ ]:


s_data['followers_count'] = s_data['followers_count'] +0.01
s_data['statuses_count'] = s_data['statuses_count'] +0.01


# In[ ]:


s_data.head()


# In[ ]:


score = s_data['followers_count'] * s_data['statuses_count']


# In[ ]:


score


# In[ ]:


score.sort_values(ascending = False)


# In[ ]:


score[score>score.median() + score.std()/2].sort_values(ascending=False)


# In[ ]:


score.median()


# In[ ]:


s_data['score'] =score


# In[ ]:


import numpy as np


# In[ ]:


s_data['segment'] = np.where(s_data['score'] >=score.median() + score.std()/len(score) , 'A', 'B')


# In[ ]:


s_data


# In[ ]:


a = api.user_timeline(id= 'AndrewYNg',count=5)


# In[ ]:


for i in a:
    print(i.text)


# In[ ]:


def country_codes():
    places = api.trends_available()
    all_woeids = {place['name'].lower(): place['woeid'] for place in places}
    return all_woeids


# In[ ]:


# country_codes()


# In[ ]:


def country_woeid(country_name):
    country_name = country_name.lower()
    trends = api.trends_available()
    all_woeids = country_codes()
    return all_woeids[country_name]


# In[ ]:


country_woeid('turkey')


# In[ ]:


trends = api.trends_place(id= 23424969 )


# In[ ]:


import json
#print(json.dumps(trends, indent=1))


# In[ ]:


turkey = api.trends_place(id= 23424969 )
trends = turkey[0]['trends']


# ## Pull Data from Hashtag

# In[ ]:


tweets = api.search(q= '#datascience', lang='en',
                     result_type='recent', counts = 1000)


# In[ ]:


def hashtag_df(results):
    id_list = [tweet.id for tweet in results]
    data_set = pd.DataFrame(id_list,columns=['id'])
    
    
    data_set['text'] = [tweet.text for tweet in results]
    data_set['created_at'] = [tweet.created_at for tweet in results]
    data_set['retweeted'] = [tweet.retweeted for tweet in results]
    data_set['retweet_count'] = [tweet.retweet_count for tweet in results]
    data_set['name'] = [tweet.author.name for tweet in results]
    data_set['user_screen_name'] = [tweet.author.screen_name for tweet in results]
    data_set['user_followers_count'] = [tweet.author.followers_count for tweet in results]
    data_set['user_location'] = [tweet.author.location for tweet in results]
    data_set['Hashtags'] = [tweet.entities.get('hashtags') for tweet in results]
    
    return data_set


# In[ ]:


df = hashtag_df(tweets)


# In[ ]:


df.shape


# In[ ]:


df


# In[ ]:


df['tweet_hour'] = df['created_at'].apply(lambda x: x.strftime('%H'))


# In[ ]:


df['tweet_hour'] = pd.to_numeric(df['tweet_hour'])


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(df['tweet_hour'], kde=True, color='blue')


# In[ ]:


df['days'] = df['created_at'].dt.weekday_name


# In[ ]:


gun_freq = df.groupby('days').count()['id']


# In[ ]:


gun_freq.plot.bar(x='days', y='id')


# # Twitter Text Mining

# In[ ]:



df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

df['text'] = df['text'].str.replace('[^\w\s]', '')

df['text'] = df['text'].str.replace('[\d]','')


import nltk
from nltk.corpus import stopwords
sw = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

#lemmi
from textblob import Word
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()] ))

df['text'] = df['text'].str.replace('rt', '')


# In[ ]:


df.text


# In[ ]:


freq_df = df['text'].apply(lambda x:pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()


# In[ ]:


freq_df.columns = ['words', 'freqs']


# In[ ]:


freq_df.sort_values('freqs',ascending=False)


# In[ ]:


freq_df.shape


# In[ ]:


a = freq_df[freq_df.freqs > freq_df.freqs.mean() + 
       freq_df.freqs.std()] # this code for the being more meaningful


# In[ ]:


a.plot.bar(x= 'words', y= 'freqs')


# ## Word Cloud

# In[ ]:


import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud , STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


# In[ ]:


text = " ".join(i for i in df.text)


# In[ ]:


text


# In[ ]:


wc = WordCloud(background_color='white').generate(text)
plt.figure(figsize=(10,6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# # Twitter Sentiment

# In[ ]:


df


# In[ ]:


from textblob import TextBlob


# In[ ]:


def sentiment_score(df):
    text = df['text']
    
    for i in range(0, len(text)):
        textB = TextBlob(text[i])
        sentiment_score = textB.sentiment.polarity
        df.set_value(i, 'sentiment_score', sentiment_score)
        
        
        if sentiment_score < 0.00:
            sentiment_class = 'Negative'
            df.set_value(i, 'sentiment_class', sentiment_class)
            
        elif sentiment_score > 0.00:
            sentiment_class ='Positive'
            df.set_value(i, 'sentiment_class', sentiment_class)
        else:
            sentiment_class = 'Notr'
            df.set_value(i, 'sentiment_class', sentiment_class)
    return df


# In[ ]:


sentiment_score(df)


# In[ ]:


df.groupby('sentiment_class').count()['id']


# In[ ]:


sentiment_freq = df.groupby('sentiment_class').count()['id']


# In[ ]:


sentiment_freq.plot.bar(x= 'sentiment_class', y='id')


# In[ ]:


tweets = api.search(q = '#apple', lang='en', count=5000)


# In[ ]:


def hashtag_df(results):
    id_list = [tweet.id for tweet in results]
    data_set = pd.DataFrame(id_list,columns=['id'])
    
    
    data_set['text'] = [tweet.text for tweet in results]
    data_set['created_at'] = [tweet.created_at for tweet in results]
    data_set['retweeted'] = [tweet.retweeted for tweet in results]
    data_set['retweet_count'] = [tweet.retweet_count for tweet in results]
    data_set['name'] = [tweet.author.name for tweet in results]
    data_set['user_screen_name'] = [tweet.author.screen_name for tweet in results]
    data_set['user_followers_count'] = [tweet.author.followers_count for tweet in results]
    data_set['user_location'] = [tweet.author.location for tweet in results]
    data_set['Hashtags'] = [tweet.entities.get('hashtags') for tweet in results]
    
    return data_set


# In[ ]:


df = hashtag_df(tweets)


# In[ ]:


df.shape


# In[ ]:



df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

df['text'] = df['text'].str.replace('[^\w\s]', '')

df['text'] = df['text'].str.replace('[\d]','')


import nltk
from nltk.corpus import stopwords
sw = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

#lemmi
from textblob import Word
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()] ))

df['text'] = df['text'].str.replace('rt', '')


# In[ ]:


def sentiment_score(df):
    text = df['text']
    
    for i in range(0, len(text)):
        textB = TextBlob(text[i])
        sentiment_score = textB.sentiment.polarity
        df.set_value(i, 'sentiment_score', sentiment_score)
        
        
        if sentiment_score < 0.00:
            sentiment_class = 'Negative'
            df.set_value(i, 'sentiment_class', sentiment_class)
            
        elif sentiment_score > 0.00:
            sentiment_class ='Positive'
            df.set_value(i, 'sentiment_class', sentiment_class)
        else:
            sentiment_class = 'Notr'
            df.set_value(i, 'sentiment_class', sentiment_class)
    return df


# In[ ]:


df = sentiment_score(df)


# In[ ]:


sentiment_freq = df.groupby('sentiment_class').count()['id']


# In[ ]:


sentiment_freq


# In[ ]:


sentiment_freq.plot.bar(x = 'sentiment_class', y= 'id')


# ### If you like it please vote 
