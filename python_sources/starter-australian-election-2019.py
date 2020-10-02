#!/usr/bin/env python
# coding: utf-8

# # Australian Election 2019

# #    TABLE OF CONTENT:
# 
#     1/ Import data
#     2/ Checking data
#     3/ Summarizing data
#     4/ Checking missing values
#     5/ Checking values in location_geocode.csv
#     6/ Number of tweets in each days of week
#     7/ Number of tweets in each region
#     8/ Wordcloud
#     9/ Count number of tweets of each party
#     10/ Average number of favorite and Average number of retweet
#     11/ Separating to dataframe of "Liberal" party and "Labor" party
#     12/ Calculate the sentiment in tweets of each party
#     13/ Wordcloud and N-Diagram for tweets of Liberal Party
#         a/ Wordcloud
#         b/ 1-Diagram
#         c/ Bi-Diagram
#     14/ Wordcloud and N-Diagram for tweets of Labor Party
#         a/ Wordcloud
#         b/ 1-Diagram
#         c/ Bi-Diagram
#     15/ Wordcloud and N-Diagram for tweets of Neutral people
#         a/ Wordcloud
#         b/ 1-Diagram
#         c/ Bi-Diagram

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from plotly.offline import iplot
import plotly.graph_objs as go
from plotly import tools
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import geopandas as gpd
from geopandas import GeoDataFrame

import matplotlib
from IPython.display import display

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.stem.porter import *
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict
import networkx
from textblob import TextBlob

import warnings
warnings.filterwarnings("ignore")


# # checking data

# In[ ]:


aus = pd.read_csv('auspol2019.csv', parse_dates=['created_at','user_created_at'])
loca = pd.read_csv('location_geocode.csv')

aus.head()


# #following to the table above, this file contains data about:
# 
# 1/ created_at: Date and time of tweet
# 2/ id: Unique ID of the tweet
# 3/ full_text: Full tweet text
# 4/ retweet_count: Number of retweets
# 5/ favorite_count: Number of likes
# 6/ user_id: User ID of tweet creator
# 7/ user_name: Username of tweet creator
# 8/ user_screen_name: Screen name of tweet creator
# 9/ user_description: Description on tweet creator's profile
# 10/ user_location: Location given on tweet creator's profile
# 11/ user_created_at: Date of create of account of user who posted the tweet
# 

# In[ ]:


aus.shape


# #data in file auspol2019.csv contains 183379 rows and 11 columns

# # summarizing data

# In[ ]:


aus.describe(include='all')


# # Checking missing values

# In[ ]:


aus.info()


# #There are a lot of missing values, it is easily to understand because this is a social network, there are some people do not like to give information about their profiles, some tweets are not insteresting so they do not have responses,...

# In[ ]:


aus.isnull().sum()


# #as we can see here, there are some retweet_count, favorite_count and user_id are null. In order to tidy data, we will delete them

# In[ ]:


aus_drop = aus.user_id.isnull()
aus = aus.drop(aus[aus_drop].index)
aus.isnull().sum()


# # checking values in location_geocode.csv

# In[ ]:


loca.head()


# #This file contains the information about the locations of the users with their latitude and longitude. It has 11153 rows and 3 columns as the result below

# In[ ]:


loca.shape


# # Number of tweets in each days of week

# In[ ]:


aus['created_at'] = pd.to_datetime(aus['created_at'])  # change from string to date format
count_tweet_date = aus['created_at'].dt.date.value_counts()  # count number of tweets in a date
count_tweet_date = count_tweet_date.sort_index() #sort index - date
plt.figure(figsize = (14,6))
sns.barplot(count_tweet_date.index, count_tweet_date.values, alpha = 0.8, color = 'green') # use seabone to create bar graph

plt.xticks(rotation=30)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of tweets', fontsize=12)
plt.title("Number of tweets in each date")
plt.show()


# #This data is collected in 10 days, from 10/05/2019 - 20/05/2019. In 18/05/2019 number of tweets is maximum, may be because this date is week-end or a day having an important event. In order to know it, we will see the relation between number of tweets and days of week

# In[ ]:


aus['days_of_week'] = aus['created_at'].dt.weekday_name
count_tweet_weekday = aus['days_of_week'].value_counts()
count_tweet_weekday = count_tweet_weekday.sort_index()
pre_grp = {
    "data": [
        {
            "values": count_tweet_weekday.values,
            "labels": count_tweet_weekday.index,
            "domain": {"x":[0, .5]},
            "name": "Number of tweets each days of week",
            "hoverinfo": "label+percent+name",
            "hole": .3,
            "type": "pie"
        },
    ],
    "layout": {
        "title":"Percentage of tweets each days of week",
        "annotations": [
            {"font": {"size":20},
            "showarrow": False,
            "text": "Percentage of tweets in each days of week",
            "x": 0.50,
            "y": 1
            },
        ]
    }
}
iplot(pre_grp)
count_tweet_weekday



# #as predicted, the number of tweets is maximum on the week-end

# # Number of tweets in each region

# In[ ]:


aus.replace({'user_location':{'Sydney':'Sydney, New South Wales', 
                              'Sydney, Australia':'Sydney, New South Wales',
                              'Melbourne, Australia':'Melbourne, Victoria', 
                              'Melbourne':'Melbourne, Victoria',
                              'Brisbane, Australia':'Brisbane, Queensland', 
                              'Brisbane':'Brisbane, Queensland',
                              'Sydney Australia':'Sydney, New South Wales'
                             }}, inplace = True)



count_tweet_location = aus['user_location'].value_counts()  # count number of tweets in a date
plt.figure(figsize = (14,6))
sns.barplot(count_tweet_location.index[0:15], count_tweet_location.values[0:15], alpha = 0.8, color = 'green') # use seabone to create bar graph

plt.xticks(rotation=45)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Number of tweets', fontsize=12)
plt.title("Number of tweets in each region")
plt.show()


# #number of tweet reach maximum in Sydney and Melbourne. May be the reason is that these are two regions where people are most concerned about this election or where there are most candidates or there are some people write a lot of tweets. In order to check if there are some cases of one user writing many tweets, we will see number of users in each region.

# In[ ]:


count_user_location = aus.loc[:,['user_screen_name','user_location']] 
count_user_location = count_user_location.drop_duplicates(subset='user_screen_name', keep='first')

count_user_location = count_user_location['user_location'].value_counts()  # count number of tweets in a date
plt.figure(figsize = (14,6))
sns.barplot(count_user_location.index[0:15], count_user_location.values[0:15], alpha = 0.8, color = 'green') # use seabone to create bar graph

plt.xticks(rotation=45)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Number of user', fontsize=12)
plt.title("Number of user in each region")
plt.show()


# #Comparing to the figure "Number of tweets in each region", we can conclude that there are a lot of people concern about the election in the two regions: "Sydney" and "Melbourn"

# # Wordcloud of tweets

# In[ ]:


text = " ".join(review for review in aus.full_text)

stopwords = set(STOPWORDS)
stopwords.update(["https","wyJzmAcyiD","will"])
# Create and generate a word cloud image:
wordcloud = WordCloud(stopwords=stopwords, max_font_size=40, 
                      max_words=150,
                      
                      random_state=1705, 
                      background_color="white").generate(text)

# Display the generated image:
plt.figure(1, figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



# #From the figure above, we can see that "auvotes", "auspol" appear the most. May be because they are hastags. In order to have exactly the word appearing the most, we need to remove the hastags. We will remove also "Australia" and "election"

# In[ ]:


#Clean data
aus['full_text_tidy'] = aus['full_text']
aus['full_text_tidy'] = aus['full_text_tidy'].apply(lambda x: re.sub(r'#[^\s]+', '', x))
aus['full_text_tidy'] = aus['full_text_tidy'].apply(lambda x: re.sub(r'@[^\s]+', '', x))
aus['full_text_tidy'] = aus['full_text_tidy'].apply(lambda x: re.sub(r'www.[^\s]+ | http[^\s]+ | https[^\s]+', '', x))
aus['full_text_tidy'] = aus['full_text_tidy'].apply(lambda x: re.sub(r'\n[^\s]+', '', x))
aus['full_text_tidy'] = aus['full_text_tidy'].apply(lambda x: re.sub('[!@#$:).;,?&]', '', x))

aus['full_text_tidy'] = aus['full_text_tidy'].apply(lambda x: re.sub('Australia', '', x))
aus['full_text_tidy'] = aus['full_text_tidy'].apply(lambda x: re.sub('election', '', x))
aus['full_text_tidy'] = aus['full_text_tidy'].str.lower()

text = " ".join(review for review in aus.full_text_tidy)


# Create and generate a word cloud image:
wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, 
                      max_words=150,         
                      background_color="white").generate(text)

# Display the generated image:
plt.figure(1, figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()






# In[ ]:


aus.shape


# #It can be seen that there are at least two parties in Australia: "Labor" and "Liberal". One of these two parties is led by Mr.Scott Morrison, another one may be is led by Mr.Bill Shorten. Mr.Bob Hawke may be is also a candidate or a person who support either Mr.Scott or Mr.Bill. From the size of word "Prime Minister", we can figure out that one of these three persons is Prime Minister now of Australia.
# 
# #An organization is demanded the most is "amp". This is a financial services company in Australia and New Zealand providing superannuation and investment products, insurance, financial advice and banking. The "climate change" problem is also of great interest.
# 
# #One of thing is very interesting is that observing the sentiment word of each party: "Labor" and "Liberal". First of all, we will see the sentiment word which appears the most.

# In[ ]:


aus['full_text_tidy'].head()


# # Count number of tweets of each party

# In[ ]:


aus['Party'] = ''
aus['Party'] = aus.apply(lambda row: 'Liberal' if 'liberal' in row['full_text_tidy']
                                   else ('Labor' if 'labor' in row['full_text_tidy'] 
                                         else 'Neutre'), axis=1) #Classify each Party


# In[ ]:


count_party = aus['Party'].value_counts()  # count number of tweets in a date
plt.figure(figsize = (14,6))
sns.barplot(count_party.index[1:3], count_party.values[1:3], alpha = 0.8, color = 'green') # use seabone to create bar graph

plt.xticks(rotation=0)
plt.xlabel('Party', fontsize=12)
plt.ylabel('Number of tweets', fontsize=12)
plt.title("Number of tweets in each Party")
plt.show()


# #As this figure, we can see that the Labor party is mentioned more than Liberal 2 times. 
# #In order to evaluate which party uses more efficiently Twitter, we will evaluate the average number of favorite and average number of retweet.

# In[ ]:


count_party.head()


# # Average number of favorite and Average number of retweet

# In[ ]:


count_like_party = aus.groupby('Party')['favorite_count','retweet_count'].mean()
count_like_party.head()


# In[ ]:


plt.figure(figsize = (14,6))
sns.barplot(count_like_party['favorite_count'].index[0:2], count_like_party['favorite_count'][0:2], alpha = 0.8, color = 'green') # use seabone to create bar graph

plt.xticks(rotation=0)
plt.xlabel('Party', fontsize=12)
plt.ylabel('Average number of favorite', fontsize=12)
plt.title("Average number of favorite")
plt.show()


# In[ ]:


plt.figure(figsize = (14,6))
sns.barplot(count_like_party['retweet_count'].index[0:2], count_like_party['retweet_count'][0:2], alpha = 0.8, color = 'green') # use seabone to create bar graph

plt.xticks(rotation=0)
plt.xlabel('Party', fontsize=12)
plt.ylabel('Average number of retweet', fontsize=12)
plt.title("Average number of retweet")
plt.show()


# #It seems the Liberal party had a slightly stronger social media effort while using more tweets than Labor. However, the efficiency of these two parties is almost equal while comparing their "Average number of favorite" and "Average number of retweet".

# #Now I will analyze the sentiment in tweets of each party to evaluate which party has more supporters than the other

# # Separating to dataframe of "Liberal" party and "Labor" party

# In[ ]:


aus_liberal = aus[aus['Party']=='Liberal']
aus_liberal = aus_liberal.reset_index(drop = True)

aus_labor = aus[aus['Party']=='Labor']
aus_labor = aus_labor.reset_index(drop = True)

aus_neutre = aus[aus['Party']=='Neutre']
aus_neutre = aus_neutre.reset_index(drop = True)


# #First of all, we need to classify which tweets belong to Liberal, Labor or Neutre

# # Calculate the sentiment in tweets of each party

# In[ ]:


#Calculate polartity for sentiment words in tweets of Liberal party
sentiment_objects_li = [TextBlob(x) for x in aus_liberal['full_text_tidy']]
sentiment_values_li = [[x.sentiment.polarity, str(x)] for x in sentiment_objects_li]
sentiment_liberal = pd.DataFrame(sentiment_values_li, columns=['polarity', 'full_text_tidy'])
aus_liberal['polarity'] = sentiment_liberal['polarity']


#Calculate polartity for sentiment words in tweets of Labor party
sentiment_objects_la = [TextBlob(x) for x in aus_labor['full_text_tidy']]
sentiment_values_la = [[x.sentiment.polarity, str(x)] for x in sentiment_objects_la]
sentiment_labor = pd.DataFrame(sentiment_values_la, columns=['polarity', 'full_text_tidy'])
aus_labor['polarity'] = sentiment_labor['polarity']


#Calculate polartity for sentiment words in tweets of Neutre
sentiment_objects_ne = [TextBlob(x) for x in aus_neutre['full_text_tidy']]
sentiment_values_ne = [[x.sentiment.polarity, str(x)] for x in sentiment_objects_ne]
sentiment_neutre = pd.DataFrame(sentiment_values_ne, columns=['polarity', 'full_text_tidy'])
aus_neutre['polarity'] = sentiment_neutre['polarity']


# In[ ]:


sentiment_objects_aus = [TextBlob(x) for x in aus['full_text_tidy']]
sentiment_values_aus = [[x.sentiment.polarity, str(x)] for x in sentiment_objects_aus]
sentiment = pd.DataFrame(sentiment_values_aus, columns=['polarity', 'full_text_tidy'])
aus['polarity'] = sentiment['polarity']


count_sentiment = aus.groupby('Party')['polarity'].mean()

plt.figure(figsize = (14,6))
sns.barplot(count_sentiment.index, count_sentiment.values, alpha = 0.8, color = 'green') # use seabone to create bar graph

plt.xticks(rotation=0)
plt.xlabel('Party', fontsize=12)
plt.ylabel('Polarity', fontsize=12)
plt.title('Average of the sentiment of each party according to tweets')
plt.show()


# #In this part, I use the polarity point to evaluate the sentiment in each tweets. The figure above shows us the average sentiment of each party according to tweets. 
# 
# #The polarity represents the sentiment level in each tweets. 
# 
# #The polarity is higher the positif sentiment is higher. It can be seen that the positif sentiment in Liberal party is the highest, but the difference beween Labor party and Liberal party is not considerable.
# 
# #It can be seen that the two parties are mentioned with a positif sentiment with the same polarity point.

# In[ ]:


#Classify "positif", "negatif" or "neutral" mark for each tweets of each party
#If polarity > 0: "positif"; polarity < 0: "negatif" polarity = 0: "neutral

aus_liberal['sentiment'] = aus_liberal.apply(lambda row: 'positif' if row['polarity']>0 
                                             else ('negatif' if row['polarity'] < 0 else 'neutral'), axis = 1)
aus_labor['sentiment'] = aus_labor.apply(lambda row: 'positif' if row['polarity']>0 
                                             else ('negatif' if row['polarity'] < 0 else 'neutral'), axis = 1)
aus_neutre['sentiment'] = aus_neutre.apply(lambda row: 'positif' if row['polarity']>0 
                                             else ('negatif' if row['polarity'] < 0 else 'neutral'), axis = 1)


# In[ ]:





# # Wordcloud and N-Diagram for tweets of Liberal Party

# ### Wordcloud

# In[ ]:


neg_tweets_lib = aus_liberal
neg_string_lib = []
for t in neg_tweets_lib.full_text_tidy:
    neg_string_lib.append(t)
neg_string_lib = pd.Series(neg_string_lib).str.cat(sep=' ')
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string_lib)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# #As the image above, we can figure out that Mr. Scott Morrison is the candidate of Liberal party and his viral is Mr. Bill Shorten
# 
# #As we can see that "Climate change" is one of the most concerned issues
# 
# #There are some word supporting to Labor party, may be these are also tweets with negatif sentiment
# 
# #In order to evaluate in detailed, we will analyze the N-Diagram to see which words are used the most

# ### 1-Diagram

# In[ ]:


def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace


freq_dict = defaultdict(int)
for sent in aus_liberal["full_text_tidy"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(25), 'blue')


fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,
                          subplot_titles=["Frequent words"
                                          ])
fig.append_trace(trace0, 1, 1)

fig['layout'].update(height=700, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
iplot(fig, filename='word-plots.html')


# #As we can see, "Climate" is one of the issue which may be will influences directly to the result of the election . This is logical because Autralia is one of the countries bearing the heavy consequences of climate change as: "along the Murray Darling river system in which up to 1 million fish have died. In Queensland, floods have wiped out half a million cattle and bushfires have burned close to pristine rainforests. In the usually cool southern state of Tasmania more bushfires have raged across 190,000 hectares of land and devastated old-growth forests" (quoted: the gardient: https://www.theguardian.com/australia-news/2019/may/07/climate-change-takes-centre-stage-in-australias-election"
# 
# #We can see that Labor is mentioned many times, may be these are the negatif sentences, we will visualize the Bi-Diagram to observe

# ### Bi-Diagram

# In[ ]:


freq_dict = defaultdict(int)
for sent in aus_liberal["full_text_tidy"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(20), 'green')

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,horizontal_spacing=0.15,
                          subplot_titles=["Frequent bigrams"
                                          ])
fig.append_trace(trace0, 1, 1)
fig['layout'].update(height=700, width=1000, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")
iplot(fig, filename='word-plots')


# #First of all, Mr.Scott and "Climate change" are mentioned the most and the number of words of "Labor party" decreases significantly. 
# 
# #One interesting thing we can see that may be Mr.Scott is now Prime Minister because the word "Scott Morrison" and "Prime Minister" appear almost equaly.

# #we can see that Scotte Morisson may be is candidature of Liberal party, and now he is Prime Minister

# # Wordcloud and N-Diagram for tweets of Labor Party

# ### Wordcloud

# In[ ]:


neg_tweets_lab = aus_labor
neg_string_lab = []
for t in neg_tweets_lab.full_text_tidy:
    neg_string_lab.append(t)
neg_string_lab = pd.Series(neg_string_lab).str.cat(sep=' ')
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string_lab)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# #It is clearly to see that "Labor", "Bill Shorten", "Green", "Climate change" are the the words mentioned the most. We can one more time realize that Climate change is an important problem in this election. 
# 
# #Now we will evaluate the N-Diagram

# ### 1-Diagram

# In[ ]:



freq_dict = defaultdict(int)
for sent in aus_labor["full_text_tidy"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(20), 'blue')


fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,
                          subplot_titles=["Frequent words"
                                          ])
fig.append_trace(trace0, 1, 1)

fig['layout'].update(height=700, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
iplot(fig, filename='word-plots.html')


# ### Bi-Diagram

# In[ ]:


freq_dict = defaultdict(int)
for sent in aus_labor["full_text_tidy"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(20), 'green')

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,horizontal_spacing=0.15,
                          subplot_titles=["Frequent bigrams"
                                          ])
fig.append_trace(trace0, 1, 1)
fig['layout'].update(height=700, width=1000, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")
iplot(fig, filename='word-plots')


# #It can be seen also that Bob Hauwke is the person who supports Bill in this campaign, we see that the word "Prime Minister" appears many time as "Bob Hauwke", therefore Mr.Bob Hauwke may be is the former Prime Minister. With this discovery, we can believe that Mr.Scott is now Prime Minister.

# # Wordcloud and N-Diagram for tweets of Neutre

# ### Worddcloud

# In[ ]:


neg_tweets_ne = aus_neutre
neg_string_ne = []
for t in neg_tweets_ne.full_text_tidy:
    neg_string_ne.append(t)
neg_string_ne = pd.Series(neg_string_ne).str.cat(sep=' ')
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string_ne)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# ### 1-Diagram 

# In[ ]:


freq_dict = defaultdict(int)
for sent in aus_neutre["full_text_tidy"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(20), 'blue')


fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,
                          subplot_titles=["Frequent words"
                                          ])
fig.append_trace(trace0, 1, 1)

fig['layout'].update(height=700, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
iplot(fig, filename='word-plots.html')


# ### Bi-Diagram

# In[ ]:


freq_dict = defaultdict(int)
for sent in aus_neutre["full_text_tidy"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(30), 'green')

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,horizontal_spacing=0.15,
                          subplot_titles=["Frequent bigrams"
                                          ])
fig.append_trace(trace0, 1, 1)
fig['layout'].update(height=700, width=1000, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")
iplot(fig, filename='word-plots')


# #In group of Neutre, we can conclude that:
# 
# 1/ Mr.Scott is mentioned the most, he uses more efficiently the social network Twitter than Mr.Bill. He is now also the Prime Minister of Australia
# 
# 2/ "Climate change" is the most important problem in this election, this is the concerned issue of the two parties and may be this election is the election of "Climate change"
