#!/usr/bin/env python
# coding: utf-8

# **Data Load**
# > 

# 1.APIs-
# News API, 
# Twitter API 

# 2. Sprinklr-  
# Event based News,  
# Event based Tweeter data 

# In[ ]:


# !pip install textblob
# !pip install advertools
import requests
import pandas as pd
import json
from pandas.io.json import json_normalize
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)  
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' # high resolution plotting")
import matplotlib.pyplot as plt


# In[ ]:


api_key="9ec526b31a214d80bfeb5aa83451239c"
type_of_news="everything" ## everything || top-headlines
date_from="2019-09-05"
querying_keyword="Iron ore"
pageSize="100"
source = "Bloomberg"
#source="google-news"
news_feed_url = ('https://newsapi.org/v2/'+type_of_news+'?q='+querying_keyword+'&from='+date_from+'&sortBy=popularity&sources='+source+'&pageSize='+pageSize+'&apiKey='+api_key)


# In[ ]:


def get_data(news_feed_url):
    response = requests.get(news_feed_url)
    return response.json()

a=get_data(news_feed_url)


# In[ ]:


#print(a)
news_data=json_normalize(a['articles'])
type(news_data)


# In[ ]:


date_stamp=news_data["publishedAt"].str.split("T",n=1,expand=True)
news_data["published_date"]=date_stamp[0]
news_data["published_time"]=date_stamp[1].str.split("Z",n=1,expand=True)[0]


# In[ ]:


news_data.head(1)


# In[ ]:


news_data.to_csv('data_bloom_iron0906.csv')


# In[ ]:


from IPython.display import FileLink
FileLink(r'data_bloom_iron0906.csv')


# API News and Tweet load

# In[ ]:


news_data.head(1)


# In[ ]:


import pandas as pd
pdf_oilprice=pd.read_csv("../input/oiltweet280819/oilprice28082019.csv",encoding = "ISO-8859-1")

pdf_oilandgas=pd.read_csv("../input/oiltweet280819/oilandgas28082019.csv",encoding = "ISO-8859-1")

pdf_oilwar=pd.read_csv("../input/oiltweet280819/oilwar28082019.csv",encoding = "ISO-8859-1")

pdf_pertroleum=pd.read_csv("../input/oiltweet280819/petroleum28082019.csv",encoding = "ISO-8859-1")

pdf_oilspill=pd.read_csv("../input/oiltweet280819/oilspill28082019.csv",encoding = "ISO-8859-1")


# In[ ]:


pdf_tweetapi=pd.concat([pdf_oilspill,pdf_pertroleum,pdf_oilwar,pdf_oilandgas,pdf_oilprice])


# In[ ]:


pdf_tweetapi=pdf_tweetapi.reset_index(drop=True)


# In[ ]:





# In[ ]:


import pandas as pd

pdf_reutersoiltanker2807to2808=pd.read_csv("../input/news2807to2808/data_reuters_oil1.csv",encoding = "ISO-8859-1")
pdf_bloomberg2807to2808=pd.read_csv("../input/news2807to2808/data_bloomberg_oil2807.csv",encoding = "ISO-8859-1")

pdf_reutersoiltanker2707to2608=pd.read_csv("../input/news2807to2808/data_reuters_oil.csv",encoding = "ISO-8859-1")

pdf_googleoil807to2808=pd.read_csv("../input/news2807to2808/data_google_oil2807.csv",encoding = "ISO-8859-1")

pdf_reutersoil2807to2808=pd.read_csv("../input/reutersoil2807to2808/data_reuters_oil2807.csv",encoding = "ISO-8859-1")


# In[ ]:





# In[ ]:


pdf_newsapi=pd.concat([pdf_bloomberg2807to2808,pdf_reutersoiltanker2807to2808,pdf_reutersoiltanker2707to2608,pdf_googleoil807to2808,pdf_reutersoil2807to2808])


# In[ ]:


pdf_newsapi=pdf_newsapi.reset_index(drop=True)


# In[ ]:


pdf_newsapi.head(1)


# 2.Sprinklr- Events bases News,Tweeter loaded ....................................

# In[ ]:


ls ../input/tweeterevents/*.csv


# In[ ]:


# import pandas as pd
import glob
filepath="../input/tweeterevents/"
files=glob.glob(filepath+"*.csv")
print("number of file:"+str(len(files)))
# # #pdf_419junoiltanker=pd.read_csv("../input/newevents/4jun-19jun-oiltanker.csv")


# In[ ]:


import pandas as pd
pdf_gulfofomantweet=pd.read_csv("../input/tweeterevents/gulfofoman-twitter.csv",encoding = "ISO-8859-1")


# In[ ]:





# In[ ]:


data = pd.read_csv(files[0],encoding = "ISO-8859-1")
for file in files[1:]:
    newdata=pd.read_csv(file,encoding = "ISO-8859-1")
    data = pd.concat([data,newdata],axis=0)
pdf_tweet=data.reset_index(drop=True)


# In[ ]:


pdf_gulfofoman915may=pd.read_csv("../input/newevents/9may-15may-gulfofomn.csv",encoding = "ISO-8859-1")

pdf_1831janironoredam=pd.read_csv("../input/newevents/18jan031jan-ironore-dam.csv",encoding = "ISO-8859-1")

pdf_1831juloilspill=pd.read_csv("../input/newevents/18jul-31jul-oilspill.csv",encoding = "ISO-8859-1")

pdf_116jannews=pd.read_csv("../input/newevents/1jan-16jan-news.csv",encoding = "ISO-8859-1")

pdf_1831juloilspillchinese=pd.read_csv("../input/newevents/1jan-16jan-chinese.csv",encoding = "ISO-8859-1")

pdf_419junoiltanker=pd.read_csv("../input/newevents/4jun-19jun-oiltanker.csv",encoding = "ISO-8859-1")


# In[ ]:


pdf_news=pd.concat([pdf_gulfofoman915may,pdf_1831janironoredam,pdf_1831juloilspill,pdf_116jannews,pdf_1831juloilspillchinese,pdf_419junoiltanker])


# In[ ]:


len(pdf_news)


# In[ ]:


pdf_news=pdf_news.reset_index(drop=True)


# In[ ]:


len(pdf_tweet)


# In[ ]:


pdf_tweet


# In[ ]:


pdf_news.to_csv('allnews-sprinklr.csv')


# In[ ]:


from IPython.display import FileLink
FileLink(r'allnews-sprinklr.csv')


# In[ ]:


#Function to drop null//NAN


# In[ ]:





# In[ ]:





# In[ ]:





# #LIST OF FUNCTIONS

# In[ ]:


import re
def removeSpecialChar(text):
    s = re.sub(r"[^a-z0-9]"," ",str(text).lower())
    return s


# In[ ]:


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


def removeStopWords(text):
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    return filtered_sentence


# In[ ]:


def checkretweet(list_of_words):
    if list_of_words[0]=="rt":
        return 1
    else :
        return 0

    


# In[ ]:


from textblob import Word
def lemmatize(list_of_words):
    return [Word(word).lemmatize() for word in list_of_words]


# In[ ]:


def countLetters(text):
    return len(text)


# In[ ]:


def countWords(text):
    return len(text.split(" "))


# In[ ]:


def mediaScore(text):
    if str(text).find('LINK'):
        return 25
    elif str(text).find('PHOTO'):
        return 50
    elif str(text).find('VIDEO'):
        return 75
    else :
        return 0

def confidenceInterval(df):
    df['confidence_interval']=np.log(df['SenderAge'])+np.log(df['Sender Followers Count'])+np.log(df['Retweets']) + df['engagement score']+ np.log(df['tweet_number_of_char'])+np.log(df['count_noun'])+np.log(df['count_verb'])+np.log(df['Favorites'])+np.log(df['tweet_number_of_char'])+np.log(df['tweet_number_of_words'])+np.log(df['media_score'])
    return df


        


# In[ ]:


df['clean_text']=df['Message'].apply(removeSpecialChar)
df["clean_stop_words"]=df["clean_text"].apply(removeStopWords)
#df["is_retweet"]=df["clean_stop_words"].apply(checkretweet)
df["lemmatize_stop_clean_text"]=df["clean_stop_words"].apply(lemmatize)
df["tweet_number_of_char"]=df["clean_text"].apply(countLetters)
df["tweet_number_of_words"]=df["clean_text"].apply(countWords)
df["count_pronouns"]=df["Message"].map(lambda z: countTaggedTokens(z,"pronoun"))
df["count_verb"]=df["Message"].map(lambda z: countTaggedTokens(z,"verb"))
df["count_noun"]=df["Message"].map(lambda z: countTaggedTokens(z,"noun"))
df["lemmatize_stop_clean_text_rt"]=df["lemmatize_stop_clean_text"].apply(cleanRTName)


# In[ ]:


new_df=create_engagement_metric(df)

new_df['media_score']=new_df['MediaTypeList'].apply(mediaScore)
new_df=confidenceInterval(new_df)
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler().fit(new_df['confidence_interval'].values.reshape(1,-1))
#rescaledX = scaler.transform(new_df['confidence_interval'].values.reshape(1,-1))
#new_df['scaled_confidenceInterval']=scaler.fit_transform()
#new_df['scaled_confidenceInterval']=rescaledX

#new_df[['scaled_confidenceInterval','confidence_interval']].head()


# In[ ]:


#!pip install advertoolsa=new_df.groupby('Event Name')
a.first()
new_df.aggregate({"Retweets":['sum','max'], 
              "Favorites":['max', 'sum'], 
              "confidence_interval":['max']}) 


# In[ ]:


news['clean_text']=news['Message'].apply(removeSpecialChar)
news['Title']=news['Title'].apply(removeSpecialChar)


# In[ ]:





# In[ ]:


import advertools as adv


# In[ ]:


[x for x in dir(adv) if x.startswith('extract')]  # currently available extract functions


# In[ ]:


hashtag_summary = adv.extract_hashtags(pdf_gulfofomantweet['Message'])
hashtag_summary.keys()


# In[ ]:


hashtag_summary['overview']


# In[ ]:


mention_summary = adv.extract_mentions(pdf_gulfofomantweet['Message'])
mention_summary.keys()


# In[ ]:


mention_summary['overview']


# In[ ]:


word_summary = adv.extract_words(pdf_gulfofomantweet['Message'], 
                                 words_to_extract=['oil', 'tanker', 'attack',],
                                 entire_words_only=False)


# In[ ]:


word_summary.keys()


# In[ ]:


word_summary['overview']


# In[ ]:


word_summary['top_words'][:20]


# In[ ]:


word_summary_oil = adv.extract_words(pdf_gulfofomantweet['Message'],
                                          ['oil', 'tanker', 'trump', 'donald','Explosion','Gulf','Iran','Fire','Attack',''])


# In[ ]:


word_summary_oil.keys()


# In[ ]:


word_summary_oil['top_words'][:20]


# In[ ]:


emoji_summary = adv.extract_emoji(pdf_gulfofomantweet['Message'])
emoji_summary.keys()


# In[ ]:


emoji_summary['overview']


# In[ ]:


emoji_summary['emoji_flat'][:10]


# In[ ]:


emoji_summary['emoji_flat_text'][:10]


# In[ ]:


list(zip(emoji_summary['emoji_flat'][:10], emoji_summary['emoji_flat_text'][:10]))


# In[ ]:





# In[ ]:


plt.figure(facecolor='#ebebeb', figsize=(8, 8))
plt.bar([x[0] for x in emoji_summary['emoji_freq'][:15]],
        [x[1] for x in emoji_summary['emoji_freq'][:15]])
plt.title('Emoji frequency', fontsize=18)
plt.xlabel('Emoji per tweet', fontsize=12)
plt.ylabel('Number of tweets', fontsize=12)
plt.grid(alpha=0.5)
plt.gca().set_frame_on(False)


# In[ ]:


emoji_summary['top_emoji'][:20]


# In[ ]:


emoji_summary['top_emoji_text'][:20]


# > Emoji summary vertically

# In[ ]:


# plt.figure(facecolor='#ebebeb', figsize=(8, 8))
# plt.barh([x[0] for x in emoji_summary['top_emoji_text'][:20]][::-1],
#          [x[1] for x in emoji_summary['top_emoji_text'][:20]][::-1])
# plt.title('Top Emoji')
# plt.grid(alpha=0.5)
# plt.gca().set_frame_on(False)


# In[ ]:


mention_summary = adv.extract_mentions(pdf_gulfofomantweet['Message'])
mention_summary.keys()


# In[ ]:


mention_summary['overview']


# In[ ]:


mention_summary['mentions_flat'][:10]


# In[ ]:


mention_summary['mention_counts'][:20]


# In[ ]:


plt.figure(facecolor='#ebebeb', figsize=(8, 8))
plt.bar([x[0] for x in mention_summary['mention_freq'][:15]],
        [x[1] for x in mention_summary['mention_freq'][:15]])
plt.title('Mention frequency', fontsize=18)
plt.xlabel('Mention per tweet', fontsize=12)
plt.ylabel('Number of tweets', fontsize=12)
plt.xticks(range(15))
plt.yticks(range(0, 2800, 200))
plt.grid(alpha=0.5)
plt.gca().set_frame_on(False)


# In[ ]:


mention_summary['top_mentions'][:10]


# In[ ]:


plt.figure(facecolor='#ebebeb', figsize=(8, 8))
plt.barh([x[0] for x in mention_summary['top_mentions'][:15]][::-1],
         [x[1] for x in mention_summary['top_mentions'][:15]][::-1])
plt.title('Top Mentions')
plt.grid(alpha=0.5)
plt.xticks(range(0, 1100, 100))
plt.gca().set_frame_on(False)


# In[ ]:


question_summary = adv.extract_questions(pdf_gulfofomantweet['Message'])


# In[ ]:


question_summary.keys()


# In[ ]:


question_summary['overview']


# In[ ]:


question_summary['top_question_marks']


# In[ ]:


[(i,x) for i, x in  enumerate(question_summary['question_text']) if x][:15]


# In[ ]:


intense_summary = adv.extract_intense_words(pdf_gulfofomantweet['Message'], min_reps=3)


# In[ ]:


intense_summary['overview']


# In[ ]:


intense_summary['top_intense_words'][:20]


# In[ ]:


extracted_tweets =  (pdf_gulfofomantweet[['Message', 'SenderScreenName', 'Sender Followers Count']]
 .assign(hashtags=hashtag_summary['hashtags'],
         hashcounts=hashtag_summary['hashtag_counts'],
         mentions=mention_summary['mentions'],
         mention_count=mention_summary['mention_counts'],
         emoji=emoji_summary['emoji'],
         emoji_text=emoji_summary['emoji_text'],
         emoji_count=emoji_summary['emoji_counts'],))
extracted_tweets.head()


# In[ ]:


word_freq_mention = adv.word_frequency(extracted_tweets['mentions'].str.join(' '), 
                                       extracted_tweets['Sender Followers Count'].fillna(0))
word_freq_mention.head(10)


# In[ ]:


pdf_gulfofomantweet['CreatedTime'] = pd.to_datetime(pdf_gulfofomantweet['CreatedTime'])


# We have 10+ days  of tweets  Times appear with these dates, so let's create a new column to hold only the date component of this!

# In[ ]:


pdf_gulfofomantweet['Createddate'] = pd.to_datetime(pdf_gulfofomantweet['CreatedTime'].dt.date)


# In[ ]:





# In[ ]:


# Count the number of times a date appears in the dataset and convert to dataframe
tweet_trend = pd.DataFrame(pdf_gulfofomantweet['Createddate'].value_counts())

# index is date, columns indicate tweet count on that day
tweet_trend.columns = ['tweet_count']

# sort the dataframe by the dates to have them in order
tweet_trend.sort_index(ascending = True, inplace = True)


# In[ ]:


# make a line plot of the tweet count data and give some pretty labels! ;)
# the 'rot' argument control x-axis ticks rotation
plt.style.use('seaborn-darkgrid')
tweet_trend['tweet_count'].plot(linestyle = "-", figsize = (12,8), rot = 45, color = 'B',
                               linewidth = 3)
plt.title('Tweet counts by date', fontsize = 20)
plt.xlabel('Date', fontsize = 15)

plt.ylabel('Tweet Count', fontsize = 13)


# In[ ]:


#dates_list = ['2019-05-12', '2019-05-13', '2019-05-14', '2019-05-20']

# create a series of these dates.
#important_dates = pd.Series(pd.to_datetime(dates_list))
important_dates= list(pdf_gulfofomantweet['Createddate'])
# add columns to identify important events, and mark a 0 or 1.
tweet_trend['Important Events'] = False
tweet_trend.loc[important_dates, 'Important Events'] = True
tweet_trend['values'] = 0
tweet_trend.loc[important_dates, 'values'] = 1


# In[ ]:





# In[ ]:


# Calculate the percentage change in tweet counts
tweet_trend['Pct_Chg_tweets'] = tweet_trend['tweet_count'].pct_change()*100

# Lets see values only for the important dates. This Pct_Chg_tweets shows us the percentage
# change in tweets for the day of the event versus the previous day!
tweet_trend.loc[tweet_trend['values'] == 1,['tweet_count', 'Pct_Chg_tweets']]


# In[ ]:


# take a look at what the 'text' column holds
pdf_gulfofomantweet['Message'].tail(10)


# We can see that ------------------******
# Retweets begin with the keyword 'RT'. 
# These are followed by @userkey.
# Hashtags begin with a # and are one continuous string with a space next to them!
# 
# Links begin with https:// or http:// and can be present anywhere in the string.
# There can be multiple links and hashtags in a tweet, but retweet identifier is just one.
# User mentions begin with '@' and are a continuous word!

# In[ ]:





# In[ ]:


# define a function that takes in a tweet and throws out the text without the RT.
import re
def remove_retweet(tweet):
    '''Given a tweet, remove the retweet element from it'''
    text_only = []
    if len(re.findall("^RT.*?:(.*)", tweet)) > 0:
        text_only.append(re.findall("^RT.*?:(.*)", tweet)[0])
    else:
        text_only.append(tweet)
    return text_only[0]

# extract texts and place in a list
text_only = pdf_gulfofomantweet['Message'].map(remove_retweet)


# In[ ]:


# this method checks for links and removes these from the tweet provided!
def remove_links(tweet):
    '''Provide a tweet and remove the links from it'''
    text_only = []
    if len(re.findall("(https://[^\s]+)", tweet)) > 0:
        tweet = re.sub("(https://[^\s]+)", "", tweet)
    if len(re.findall("(http://[^\s]+)", tweet)) > 0:
        tweet = re.sub("(http://[^\s]+)", "", tweet)    
    text_only.append(tweet)
    return text_only[0]

text_no_links = text_only.map(remove_links)


# In[ ]:


def remove_hashtags(tweet):
    '''Provide a tweet and remove hashtags from it'''
    hashtags_only = []
    if len(re.findall("(#[^#\s]+)", tweet)) > 0:
        tweet = re.sub("(#[^#\s]+)", "", tweet) 
    hashtags_only.append(tweet)
    return hashtags_only[0]

text_all_removed = text_no_links.map(remove_hashtags)


# In[ ]:


def remove_extraneous(tweet):
    '''Given a text, remove unnecessary characters from the beginning and the end'''
    tweet = tweet.rstrip()
    tweet = tweet.lstrip()
    tweet = tweet.rstrip(")")
    tweet = tweet.lstrip("(")
    tweet = re.sub("\.", "", tweet)
    return tweet

text_clean = text_all_removed.map(remove_extraneous)


# In[ ]:


# in case hashtags are not found, we will use "0" as the placeholder
def extract_hashtags(tweet):
    '''Provide a tweet and extract hashtags from it'''
    hashtags_only = []
    if len(re.findall("(#[^#\s]+)", tweet)) > 0:
        hashtags_only.append(re.findall("(#[^#\s]+)", tweet))
    else:
        hashtags_only.append(["0"])
    return hashtags_only[0]

# make a new column to store the extracted hashtags and view them!
pdf_gulfofomantweet['tweet_hashtags'] = pdf_gulfofomantweet['Message'].map(extract_hashtags)
pdf_gulfofomantweet['tweet_hashtags'].head(10)


# In[ ]:


# create a list of all hashtags
all_hashtags = pdf_gulfofomantweet['tweet_hashtags'].tolist()

# Next we observe that our all_hashtags is a list of lists...lets change that
cleaned_hashtags = []
for i in all_hashtags:
    for j in i:
            cleaned_hashtags.append(j)

# Convert cleaned_hashtags to a series and count the most frequent occuring
cleaned_hashtag_series = pd.Series(cleaned_hashtags)
hashtag_counts = cleaned_hashtag_series.value_counts()


# In[ ]:


# Get hashtag terms from the series and convert to list
hashes = cleaned_hashtag_series.values
hashes = hashes.tolist()

# convert list to one string with all the words
hashes_words = " ".join(hashes)

# generate the wordcloud. the max_words argument controls the number of words on the cloud
from wordcloud import WordCloud
wordcloud = WordCloud(width= 1600, height = 800, 
                      relative_scaling = 1.0, 
                      colormap = "Blues",
                     max_words = 100).generate(hashes_words)

plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[ ]:


hashtag_date_df = pdf_gulfofomantweet[['Createddate', 'tweet_hashtags']]
hashtag_date_df = hashtag_date_df.reset_index(drop = True)

# extract a list of hashtags from the dataframe
all_hashtags = hashtag_date_df['tweet_hashtags'].tolist()

hashtag_date_df.head()


# In[ ]:





# Which users had the most influence?

# In[ ]:


pdf_gulfofomantweet[['SenderListedName', 'Language', 'Sender Followers Count']].sort_values('Sender Followers Count', 
                                                               ascending = False)[:20]


# In[ ]:


# First we get a count of users from each time-zone and language combination!
user_loc_lang = pdf_gulfofomantweet.groupby(['LanguageCode', 'Language'])['SenderUserId'].agg('count').reset_index()
user_loc_lang.rename(columns = {'SenderUserId':'user_count'}, inplace = True)
user_loc_lang.head(5)


# In[ ]:


user_tweet_count = pdf_gulfofomantweet.groupby('SenderUserId')['Message'].agg('count').reset_index()
user_tweet_count.rename(columns = {'text':'Tweet_count'}, inplace = True)


# In[ ]:


pdf_gulfofomantweet['Favorites'].fillna(0, inplace=True)


# In[ ]:


pdf_gulfofomantweet['Retweets'].fillna(0, inplace=True)


# In[ ]:


def create_engagement_metric(df):
    working_df = df.copy()
    
    from sklearn.preprocessing import MinMaxScaler
    # Favorites
    fav_eng_array = df['Favorites'] / df['Sender Followers Count']
    scaler = MinMaxScaler().fit(fav_eng_array.values.reshape(-1, 1))
    scaled_favs = scaler.transform(fav_eng_array.values.reshape(-1, 1))
    
    # Retweets
    rt_eng_array = df['Retweets'] / df['Sender Followers Count']
    scaler = MinMaxScaler().fit(rt_eng_array.values.reshape(-1, 1))
    scaled_rts = scaler.transform(rt_eng_array.values.reshape(-1, 1))
    
    mean_eng = (scaled_favs + scaled_rts) / 2
    working_df['engagement score'] = mean_eng
    
    return working_df


# In[ ]:


eng=create_engagement_metric(pdf_gulfofomantweet)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#drop 
#SenderProfileImgUrl
#SenderScreenName
#SenderProfileLink,SenderInfluencerScore,SenderAge,SenderGender,Title
#ReceiverId	ReceiverScreenName	AssignedBy	AssignedTo	Spam	Status	Intel Location	Priority	Star Rating
#Geo Target	Post Id	Associated Cases	Location	Country	State
#Sender Email
#manipulate SenderAge 
#explore sarcasm code - 
#world map 
#mention map for engagement
#
#feature creation with graphs eg age,engage etc


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') # one time execution
import re
get_ipython().system('pip install rouge')
from rouge import Rouge
rouge = Rouge()
import string
import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[ ]:


# Function to tokenize sentences properly
def clean_tokenize(full_text):
    full_text = re.sub(r"\\'", "'", full_text)        # few modifications in text
    full_text = re.sub(r"U.S.", "US", full_text)
    #full_text = re.sub(r"[^a-zA-Z0-9]", " ", full_text)
    full_text = full_text.replace('\n','')
  
  
    tokenized = nltk.sent_tokenize(full_text)         # nltk tokenizer
  
    for sentence in tokenized:                        # identifying correct positions for tokenization
        x=re.findall(r'\w+[.?!][A-Z]+', sentence)
        all_delm = []
        punctuation = [".","!","?"]
        for punct in punctuation:
            for occurrence in x:
                try:
                    idx1 = occurrence.index(punct)
                    idx2 = sentence.index(occurrence)
                    punct_idx = idx1+idx2
                    all_delm .append(punct_idx)
                except:
                    continue
          
        all_delm.sort()
        good_tok = []
        lower_idx = 0
        higher_idx = 0
    
        for i in range(len(all_delm)):                  # creating list of properly tokenized text
            if i!=0:
                lower_idx = all_delm[i-1]+1
            higher_idx = all_delm[i]+1
            good_tok.append(sentence[lower_idx:higher_idx])
        good_tok.append(sentence[higher_idx:])
    
        sent_idx = tokenized.index(sentence)            
        for i in range(len(good_tok)):
            tokenized.insert(sent_idx+i+1, good_tok[i])
        tokenized.pop(sent_idx)
  
    #print (tokenized)
    return tokenized


# In[ ]:


# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


# In[ ]:


embeddings_index = {}
f = open('../input/glove-840b-300d/glove.840B.300d.txt')
for line in f:
    values = line.split(' ')
    word = values[0] ## The first entry is the word
    coefs = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word
    embeddings_index[word] = coefs
f.close()

print('GloVe data loaded')


# In[ ]:


# convert sentence to vector
def convert_to_vectors(clean_sentences):
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([embeddings_index.get(w, np.zeros((300,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((300,))
        sentence_vectors.append(v)
    return sentence_vectors


# In[ ]:


# calculate similarity of sentences
def calculate_similarities(clean_sentences,sentence_vectors):
    sim_mat = np.zeros([len(clean_sentences), len(clean_sentences)])           # create similarity matrix
    for i in range(len(clean_sentences)):
        for j in range(len(clean_sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]
    return sim_mat


# In[ ]:


# use textrank algorithm to calculate similarity
def TextRank(sim_mat):
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    return scores


# In[ ]:



import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def find_relations(tweets, feature_extraction, tdm):
    """
    Go through all the tweets and create a map for each tweet and
    there cosine similarities
    """
    # a list of dictionaries containing list of related tweets and the
    # cosine value
    cosine_value_map = []
    for tweet in tweets:
        temp = {tweet:[]}
        query = feature_extraction.transform([tweet])
        cosine_similarities = linear_kernel(query, tdm).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-5:-1]
        for index in related_docs_indices:
            temp[tweet].append((tweets[index], cosine_similarities[index]))
        cosine_value_map.append(temp)
    return cosine_value_map


# In[ ]:


feature_extraction = TfidfVectorizer(analyzer="word")
tdm = feature_extraction.fit_transform(df["text"])
relations = find_relations(df["text"], feature_extraction, tdm)
data = {'tweet_one':[], 'tweet_two':[], 'cosine_relation':[]}
lower_threshold=0.5
higher_threshold=0.8
for item in relations:
    for key in item.keys():
        for processed_data in item[key]:
            if key != processed_data[0] and processed_data[1]>=lower_threshold and  processed_data[1]<=higher_threshold:
                data['tweet_one'].append(key)
                data['tweet_two'].append(processed_data[0])
                data['cosine_relation'].append(processed_data[1])
a=pd.DataFrame.from_dict(data)
        
a.head(100)


# In[ ]:



from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix
def generate_summary(sentences,top_n):
    stop_words = stopwords.words('english')
    summarize_text = []
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    print("Sentence Similarity Matrix Generation Completed")
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ".".join(summarize_text))


# In[ ]:


articles = list(news_data['title'])
summaries = list(news_data['description'])


# In[ ]:


def generate_summary(text):
    sentences = clean_tokenize(text)                                                              # tokenize text
    sentences = list(filter(lambda a: a != "", sentences))                                        # remove empty string if any (Encode methods gives error if empty string is encountered)
    clean_sentences = pd.Series(sentences)                                                        # make a copy in pandas for processing
    clean_sentences = clean_sentences.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation)))
    clean_sentences = [s.lower() for s in clean_sentences]                                        # make alphabets lowercase
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]                      # remove stopwords from the sentences
    sentence_vectors = convert_to_vectors(clean_sentences)                                        # create vectors for sentences
    sim_mat = calculate_similarities(clean_sentences,sentence_vectors)                            # calculate similarities between sentences
    scores = TextRank(sim_mat)                                                                    # applying textrank algorithm
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)      # sort sentences on the basis of similarity
    
   
    return ranked_sentences


# In[ ]:


generate_summary(summaries[5])


# In[ ]:


avg_rouge = [0]*9


# In[ ]:


for i in range(5):  #range = len(articles)
    actual_summary = summaries[i]
    hypothesis = generate_summary(articles[i])
    
    print ("Actual Summary : ", end = "")
    print (actual_summary)
    print ("\n")
    print ("Predicted Summary : ", end = "")
    print (hypothesis)
    print ("\n")
    print ("Rouge Score : ", end = "")
    score = rouge.get_scores(hypothesis, actual_summary)
    print (score)
    
    dic = score[0]
    j = 0
    for k1,v1 in dic.items():
        for k2,v2 in v1.items():
            avg_rouge[j%9]+= v2
            j+=1
    
    print ("\n")
    print ("\n")


# In[ ]:


n = len(avg_rouge)
for i in range(n):    
    avg_rouge[i]/=500    # divide by len(articles)


# In[ ]:


print ("-------Rouge Scores-------")
print ("-----------with-----------")
print ("-----GLOVE Embeddings-----")
'''
print ("Rouge 1")
print ("f = ",avg_rouge[0])
print ("p = ",avg_rouge[1])
print ("r = ",avg_rouge[2])
print ("\n")
print ("Rouge 2")
print ("f = ",avg_rouge[3])
print ("p = ",avg_rouge[4])
print ("r = ",avg_rouge[5])
print ("\n")
'''
print ("\n")
print ("Rouge L")
print ("f = ",avg_rouge[6])
print ("p = ",avg_rouge[7])
print ("r = ",avg_rouge[8])


# In[ ]:


import pyLDAvis
import pyLDAvis.gensim


# In[ ]:





# In[ ]:


# import networkx as nx
# import community
# import matplotlib.pyplot as plt

# G = nx.karate_club_graph()  # load a default graph

# partition = community.best_partition(G)  # compute communities

# pos = nx.spring_layout(G)  # compute graph layout
# plt.figure(figsize=(8, 8))  # image is 8 x 8 inches
# plt.axis('off')
# nx.draw_networkx_nodes(G, pos, node_size=600, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
# nx.draw_networkx_edges(G, pos, alpha=0.3)
# plt.show(G)

