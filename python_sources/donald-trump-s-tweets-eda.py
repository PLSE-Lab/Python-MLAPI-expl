#!/usr/bin/env python
# coding: utf-8

# # Donald Trump's Tweets EDA

# In this notebook we present a brief EDA of Donald Trump's tweets from 4th May 2009 to 17th June 2020.
# 
# This notebook does not have any political motivation. I did this project to learn a bit of plotly!

# **Contents** 
# 
# * [1. Loading and Describing Our Data](#section-one)
# 
# * [2. Favorites and Retweets](#section-two)
# 
# * [3. Hashtags and Mentions](#section-three)
# 
# * [4. Tweets Content and Length](#section-four)
# 
# * [5. Trump's Tweeting Habits](#section-five)
# 
# 

# To learn more about plotly, see the excellent plotly tutorial notebook - https://www.kaggle.com/kanncaa1/plotly-tutorial-for-beginners. This helped me a lot combined with the plotly documentation.

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly 
import plotly.express as px


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # 1. Loading and Describing Our Data
# <a id="section-one"></a>

# The goal of this section is to understand the structure of our dataset.

# In[ ]:


data=pd.read_csv('../input/trump-tweets/realdonaldtrump.csv')


# In[ ]:


data.shape


# In[ ]:


data.head(5)


# In[ ]:


print(data.mentions.mode())
print(data.hashtags.mode())


# Looks like Donald Trump's favorite person to talk about is himself! We see that the mentions are twitter handles and hashtags are as expected and thus both categorical. It is clear that a NaN will imply noone was mentioned in the tweet/no hashtag. 
# 
# What about if we have multiple mentions/hashtags? 
# 

# In[ ]:


big_mention=pd.DataFrame([mentions for mentions in data.mentions if len(str(mentions))>20])
big_hashtag=pd.DataFrame([hashtag for hashtag in data.hashtags if len(str(hashtag))>20])
big_mention.head(), big_hashtag.head()


# We see that in the case of multiple, they are just seperated by a comma. 

# In[ ]:


#looking at numerical values
data.describe()


# In[ ]:


#Looking at length of tweets
data['tweet_length'] = data['content'].apply(str)
data['tweet_length']=data['tweet_length'].apply(len)


# In[ ]:


data['tweet_length'].describe()


# In[ ]:


data.iloc[data.loc[data.tweet_length==data.tweet_length.max()].index[0], 2]


# The max character length of a tweet is 280. However, this can be increased with the inclusion of links as seen here. We will assume this is the case for any tweets above the limit of 280. 
# 

# In[ ]:


#shortest tweet 
print(data.iloc[data.loc[data.tweet_length==data.tweet_length.min()].index[0], 2])


# # 2. Favorites and Retweets
# <a id="section-two"></a>

# **Favorites**

# In[ ]:


print('Total favorites: ', data['favorites'].sum())


# In[ ]:


data['date']=data['date'].apply(pd.to_datetime)


# In[ ]:


fig=px.line(data, x='date', y='favorites', title='Favorites Time Series')
fig['data'][0]['line']['color']='blue'
fig.update_xaxes(rangeslider_visible=True)
fig.show()


# Use the range slider underneath the plot to zoom into/select a particular range.

# **Retweets**

# In[ ]:


print('Total retweets: ', data['retweets'].sum())


# In[ ]:


fig=px.line(data, x='date', y='retweets', title='Retweets Time Series')
fig['data'][0]['line']['color']='red'
fig.update_xaxes(rangeslider_visible=True)
fig.show()


# **Most favorited and retweeted tweet**
# 

# In[ ]:


#Most favorited tweet: 

print(data.iloc[data.loc[data.favorites==data.favorites.max()].index[0], 2])


# In[ ]:


#Most retweeted tweet:
print(data.iloc[data.loc[data.retweets==data.retweets.max()].index[0], 2])


# **Favorites and Retweets by Year**

# In[ ]:


#Slice the dataframe by year and store in dict
yr_data={}
for year in range(2009,2021):
    yr_data[year]=data[(data['date'] >= str(year)+'-01-01') & (data['date']<=str(year)+'-12-31')]
    yr_data[year]=yr_data[year].reset_index() #resets the index to start from 0 from each slice
    yr_data[year]=yr_data[year].drop('index', axis=1)


# In[ ]:


#create a new dataframe with key callouts for each year
years=list(range(2009,2021))
data_year=pd.DataFrame(data={'Year':years})
favorites_max, favorites_max_content, favorites_mean, retweets_max,retweets_max_content, retweets_mean=[],[],[],[],[],[]
favorites_max_content=[]
for year in years:
    favorites_max.append(yr_data[year].favorites.max())
    favorites_max_content.append(yr_data[year].iloc[yr_data[year].loc[yr_data[year].favorites==yr_data[year].favorites.max()].index[0],2])
    favorites_mean.append(int(yr_data[year].favorites.mean()))
    retweets_max.append(yr_data[year].retweets.max())
    retweets_max_content.append(yr_data[year].iloc[yr_data[year].loc[yr_data[year].retweets==yr_data[year].retweets.max()].index[0],2])
    retweets_mean.append(int(yr_data[year].retweets.mean()))
data_year['favorites_max']=favorites_max
data_year['favorites_max_content']=favorites_max_content
data_year['favorites_mean']=favorites_mean
data_year['retweets_max']=retweets_max
data_year['retweets_max_content']=retweets_max_content
data_year['retweets_mean']=retweets_mean


# In[ ]:


data_year


# In[ ]:


fig=px.scatter(data_year, x='Year', y='favorites_max', size='favorites_max', hover_name='Year', hover_data=['favorites_max_content'])
fig.add_scatter(x=data_year['Year'],y=data_year['favorites_mean'], mode='lines' , name='mean_favorites')
fig.update_layout(showlegend=True)
fig.show()


# In[ ]:


#fig=px.line(data_year, x='Year', y='retweets_max', hover_name='Year', hover_data=['retweets_max_content'])
fig=px.scatter(data_year, x='Year', y='retweets_max', size='retweets_max', hover_name='Year', hover_data=['retweets_max_content'])
fig.add_scatter(x=data_year['Year'],y=data_year['retweets_mean'], mode='lines' , name='mean_retweets')
fig.update_layout(showlegend=True)
fig.show()


# # 3. Hashtags and Mentions
# <a id="section-three"></a>

# In this section we investigate who Donald Trump most frequently tags in his tweets and which are his most frequently used hashtags.

# In[ ]:


#we create a function that will split our hashtag and mentions data and return a dict of individual hashtags and mentions with frequency
def freq(column):
    freq_dict={}
    for point in column: 
        sep_points=point.split(',')
        for point in sep_points:
            if point not in freq_dict.keys():
                freq_dict[point]=1
            else:
                freq_dict[point]+=1
    return freq_dict
        


# In[ ]:


#for entry in mentions column
#split into list
#for each entry in list, append to a master list/df
data['mentions']=data['mentions'].fillna('None')
data.mentions.apply(str)
mention_dict=freq(data.mentions)
mention_dict.pop('None')
    


# In[ ]:


len(mention_dict)


# Given the number of mentions, we will focus our attention on the 20 most frequent. 

# In[ ]:


mentions=pd.Series(mention_dict, name='num_mentions')
mentions.index.name='User'
mentions.sort_values(ascending=False, inplace=True)

fig=px.bar(mentions, x=mentions[19::-1], y=mentions.index[19::-1], labels={'x':'num_mentions','y':''})
fig.show()


# We can now for certain confirm that Trump does indeed love talking about himself. We now look at his favorite hashtags. 

# In[ ]:


data['hashtags']=data['hashtags'].fillna('None')
hashtags_dict=freq(data.hashtags)
hashtags_dict.pop('None')


# In[ ]:


hashtags=pd.Series(hashtags_dict, name='num_uses')
hashtags.index.name='hashtag'
hashtags.sort_values(ascending=False, inplace=True)

fig=px.bar(hashtags, x=hashtags[19::-1], y=hashtags.index[19::-1], labels={'x':'num_uses','y':''})
fig.show()


# It is unsurprising that the presidential campaign hashtags are most frequent. 
# 
# We see that around half of Donald Trump's tweets contained mentions (None: 22966), whereas far fewer used hashtags (None: 37769). 

# # 4. Tweets Content and Length
# <a id="section-four"></a>

# In this section we will look at the words/phrases Donald Trump frequently uses by creating a wordcloud. We also look at the length of his tweets once cleaned.
# 
# We initially start by cleaning our data. We remove capitalisation, links, mentions and hashtags to get a clearer idea of what Donald Trump himself is saying.

# In[ ]:


import re
def clean(text):
    text=str(text).lower() #lowercase
    text = re.sub('(https?:\/\/)(www\.)?\S+', '', text) #removes links
    text=re.sub('(pic\.)\S+','',text) #removes links to twitter pics/gifs
    text=re.sub(r'\@(\s)?\S+','', text) #removes mentions
    text=re.sub(r'\#\S+','',text) #removes hashtags
    text=re.sub(r'[^\w\s]',' ',text)  #remove punctuation (adds a space)
    text=re.sub(r'\s+', ' ', text)   #removes doublespace
    return text


# In[ ]:


from wordcloud import WordCloud
data['clean_text']=data.content.apply(lambda x: clean(x))
text=" ".join(tweet for tweet in data['clean_text'])
wordcloud=WordCloud(max_font_size=2000, max_words=2000, background_color='white',random_state=42).generate(text)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most popular words')
plt.show()


# In[ ]:


data['len_clean_text']=data.clean_text.apply(len)
data['len_clean_text'].describe()


# This looks much better in terms of now fitting withn the bounds of tweet character length (max. 280). Note that a tweet of 0 length is the one we previously saw: "...."
# 
# We now have a look at how tweet length may influence favorites and retweets. We will restrict this to favorites above 200000 and retweets above 75000.

# In[ ]:


fig1=px.scatter(data[data['favorites']>200000],x='len_clean_text',y='favorites', title='Length vs Favorites')
fig1.show()
fig2=px.scatter(data[data['retweets']>75000],x='len_clean_text', y='retweets' , title='Length vs Retweets')
fig2.show()


# There is no obvious trend. However, we do see that the most retweeted tweets tend to be shorter.This is also the case for the 4 most favorited. Perhaps a short, straight to the point message is more popular.

# # 5. Trump's Tweeting Habits
# <a id="section-five"></a>

# In[ ]:


#add day of week column
data['day_of_week']=data['date'].dt.day_name()


# In[ ]:


data[data['day_of_week']=='Monday'].shape[0]
tweet_days={}
for day in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']:
    tweet_days[day]=data[data['day_of_week']==day].shape[0]
tweet_days=pd.Series(tweet_days, name='num_tweets')
tweet_days.index.name='day'


# In[ ]:


fig=px.bar(tweet_days, x=tweet_days.index, y=tweet_days, labels={'x':'Day','y':'Number of tweets'})
fig.show()


# All this really tells us is that Donald Trump tweets more on weekdays. We could break down further into 'time', but I am conscious that it is not clear which timezone the times are from, and so this wouldn't give much insight. 
# 
# (**Note to self:** Could go and cross reference this with Twitter and work out. It might be quite interesting to look at the times Trump tweets since he became president vs his presidential schedule - which is available publically)

# In[ ]:


data['per_month']=data.date.dt.to_period('M')

def group_sum(date):
    return data[data['per_month']==date].shape[0]

monthly_tweets=pd.DataFrame(data['per_month'].unique(), columns=['date_month'])
monthly_tweets['num_tweets']=monthly_tweets['date_month'].apply(group_sum)
monthly_tweets


# In[ ]:


fig=px.bar(x=monthly_tweets['date_month'].apply(str), y=monthly_tweets['num_tweets'], hover_name=monthly_tweets['date_month'].apply(str), hover_data={'Daily':round(monthly_tweets['num_tweets']/monthly_tweets['date_month'].dt.days_in_month,1)}, title='Monthly tweets over time')

fig.show()


# It's interesting to see the most prolific tweeting was from before Trump was president. In fact, after his election he did say that his use of social media would be restrained, if any at all. We clearly see that in the past year it has began to increase. 
# 
# January 2015 was Trump's highest tweeting month, with approx. 37 tweets a day. We investigate this with another wordcloud. 

# In[ ]:


text_jan15=" ".join(tweet for tweet in data[data['per_month']=='2015-01'].clean_text)
wordcloud=WordCloud(max_font_size=2000, max_words=2000, background_color='white',random_state=42).generate(text_jan15)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most popular words')
plt.show()


# Seems like it was a big month for Celebrity Apprentice. Indeed, as it turns out January 2015 was when season 14 premiered. It also looking as though we're starting to see some tweets around the presidential campaign (although this was not formally announced until June 2015. 

# # 6. Next steps
# <a id="section-six"></a>

# Following this basic analysis of the data, I would like to do some NLP e.g. sentiment analysis. Once I develop these skills I will come back to this notebook. 
