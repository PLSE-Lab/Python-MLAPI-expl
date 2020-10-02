#!/usr/bin/env python
# coding: utf-8

# <font size="+3" color=purple ><b> <center><u>Tweet Sentiment Extraction</u></center></b></font>

# # Table of content
# * [1. Objective](#1)
# * [2. Evaluation Metric](#2)
# * [3. Data](#3)
#     - [3.1 Libraries](#3.1)
#     - [3.2 Load Data](#3.2)
#     - [3.3 Shape](#3.3)
#     - [3.4 Proportion](#3.4)
#     - [3.5 NAN](#3.5)
# * [4. Features](#4)
#     - [4.1 Target-Selected Text](#4.1)
#         - [4.1.1 Find URLs](#4.1.1)
#         - [4.1.2 Punctuations - Selected Text](#4.1.2)
#         - [4.1.3 Length - Punctuation](#4.1.3)
#         - [4.1.4 Length - Selected Text](#4.1.4)
#         - [4.1.5 Average Length - Selected Text](#4.1.5)
#         - [4.1.6 Most Words - Selected Text](#4.1.6)
#     - [4.2 Sentiment](#4.2)
#         - [4.2.1 Sentiment - Train](#4.2.1)
#         - [4.2.2 Sentiment - Test](#4.2.2)
#     - [4.3 Text](#4.3)
#         - [4.3.1 Punctuation - Text](#4.3.1)
#         - [4.3.2 Length - Text](#4.3.2)
#         - [4.3.3 Average Length - Text](#4.3.3)
#         - [4.3.4 Most Words - Text](#4.3.4)
#         - [4.3.5 Stopwords](#4.3.5)
# * [5. Comparison](#5)
#     - [5.1 N-grams](#5.1)
#         - [5.1.1 Train - Selected text](#5.1.1)
#         - [5.1.2 Text - Train vs Test](#5.1.2)
#     - [5.2 Venn](#5.2)
#         - [5.2.1 Venn - Text vs Selected Text](#5.2.1)
#     - [5.3 Wordcloud](#5.3)
#         - [5.3.1 Word Cloud - Selected Text](#5.3.1)
#         - [5.3.2 Word Cloud Train vs Text](#5.3.2)
#     - [5.4 Jaccard](#5.4)
#         - [5.4.1 Text vs Selected Text](#5.4.1)
#         - [5.4.2 Jaccard - Sentiment](#5.4.2)
#         - [5.4.3 Violin - Jaccard Score](#5.4.3)
#         - [5.4.4 Difference - Text vs Selected Text](#5.4.4)
#         - [5.4.5 Difference vs Jaccard](#5.4.5)
# * [6. roBERTa](#6)
#     - [6.1 Basic Setup](#6.1)
#     - [6.2 Mask - Train](#6.2)
#     - [6.3 Mask - Test](#6.3)
#     - [6.4 Model](#6.4)
#     - [6.5 Run Model](#6.5)
#     - [6.6 Submission](#6.6)
#     
#     

# <font size="+3" color="blue"><b>1. Objective</b></font><a id="1"></a>

# * **Competiton**     : [Tweet Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction)
# * **Predict**    : Support phrases from given tweet text
# * **Evaluation** : [Jaccard Score](https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50). We will come to know more about this in below sections.
# * **Last Date to Join this competition in kaggle** :June 9, 2020 - Entry Deadline.So dont get late to join.
# * **Stages of this kernel** : Data >> Features(EDA) >> Comparison(EDA) >> Model

# <font size="+1">Please show patience to go through my kernel and appreciate me with an <font color="red"><b>UPVOTE</b></font> which will encourage me to make more kernels

# **Note:**<br>
# * I have hidden helper function code input and plot code input for providing more readabilty view.
# * I have another kernel that has basic level of helper functions required for text processing.Please read it as well. https://www.kaggle.com/raenish/cheatsheet-text-helper-functions

# <font size="+3" color="blue"><b>2. Evaluation Metric</b></font><a id="2"></a>

# **Jaccard Score** is more about how exactly the predicted words match against actual words in a sentence.

# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

# First set of sentences
Actual_1 = 'My life is totally awesome'
Predict_1 = 'awesome life'

# First set of sentences
Actual_2 = 'We are active kagglers'
Predict_2 = 'We are active kagglers'
    
print("Jaccard score for first set of scentences: {}".format(jaccard(Actual_1,Predict_1)))
print("Jaccard score for second set of scentences: {}".format(jaccard(Actual_2,Predict_2)))


# <font size="+3" color="blue"><b>3. Data</b></font><a id="3"></a>

# The data is collected from twitter.And we have four columns of data

# | Columns       |      Description          | 
# |---------------|:-------------------------:|
# | ID            |  Unique ID for each tweet |       
# | Text          |  Whole content of tweet   |   
# | Selected Text |  Selected Text of tweet   |    
# | Sentiment     |  Sentiment of tweet       |

# <font size="+2" color="indigo"><b>Key Things before start</b></font><br>
# 
# There are few factors that can play a major part in terms of getting good score in competition.We may not consider all of them for EDA.
# 
# * **Do consider raw data**.As we have selected text in our data which is **completely filled with raw text**,we cannot avoid them.It is our target variable(Selected text).(I may call it as target variable in most part of my kernel)
# * **Sentiments are very important.** Sentiments play a vital part because it will identify few words for target.So this variable can always play a cameo in EDA and modelling too.(I have completely utilized Sentiments for my EDA presentation.I could gain some insights from them)
# * **Do not correct spell** Our metric is so strict that even a punctuation can ruin your predicted word.
# 
# 
# *My primary focus would be to perform deep EDA on **text,selected_text,sentiment** to get pattern of train and test data.Understanding data with these features is very important*

# <font size="+2" color="indigo"><b>3.1 Libraries</b></font><br><a id="3.1"></a>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import string
import matplotlib.pyplot as plt
import matplotlib_venn as venn
import seaborn as sns


from tqdm import tqdm
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch
from collections import defaultdict
from collections import  Counter


# sklearn 
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

#nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
stop=set(stopwords.words('english'))
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

#Avoid warning messages
import warnings
warnings.filterwarnings("ignore")

#plotly libraries
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers

from datetime import datetime as dt


# <font size="+2" color="indigo"><b>3.2 Load Data</b></font><br><a id="3.2"></a>

# In[ ]:


train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
train.sample(6)


# <font size="+2" color="indigo"><b>3.3 Shape</b></font><br><a id="3.3"></a>

# In[ ]:


print("There are {} rows and {} columns in train file".format(train.shape[0],train.shape[1]))
print("There are {} rows and {} columns in test file".format(test.shape[0],test.shape[1]))


# <font size="+2" color="indigo"><b>3.4 Proportion</b></font><br><a id="3.4"></a>

# In[ ]:


print("There are {} percentage of test data proportion compared to train data".format(round(test.shape[0]/train.shape[0]*100,2)))


# <font size="+2" color="indigo"><b>3.5 NAN</b></font><br><a id="3.5"></a>

# In[ ]:


# Function for missing value
def miss_val(df):
    total=df.isnull().sum()
    return pd.concat([total],axis=1,keys=['Total'])
print("Missing values for train dataset \n")
print(miss_val(train))
print("---------------------------------------------------------------------")
print("Missing values for test dataset \n")
print(miss_val(test))


# <font size="+1" color="green"><b>Observations:</b></font><br>
# 
# * Data proportion between train and test is almost **88% - 12%**
# * **4** object type variables.
# * Only one row in train data has null value in text and selected_text columns
# 

# <font size="+3" color="blue"><b>4. Features</b></font><br><a id="4"></a>
# 
# #### What are we going to do in this section?
# 
# * Data cleaning
# * EDA
# * Feature generation

# Since we have 1 NULL row,we will remove it from train data.

# In[ ]:


train=train.dropna()
train.shape


# <font size="+2" color="indigo"><b>4.1 Target-Selected Text</b></font><a id="4.1"></a>

# <font size="+1" color="chocolate"><b>EDA on Selected text</b></font> <br>
# 
# We will undergo some basic text prepocessing and EDA on our target field- **Selected Text**.This is to understand how this feature is distributed in train data.
# 
# * Find URLs
# * Punctuations
# * Length of tweets
# * Average of tweets
# * Most words 

# <font size="+1" color="chocolate"><b>4.1.1 Find URLs - Target</b></font> <a id="4.1.1"></a>

# #### Why to consider URL?
# 
# URLs makes no sense for extreme sentiments.There are chances that they stay on neutral side.Lets check how they are spread in selected text
# 

# In[ ]:


# Convert to lower
train['target']=train['selected_text'].str.lower()


# In[ ]:


# Find URL
def find_link(string): 
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    return "".join(url) 
train['target_url']=train['target'].apply(lambda x: find_link(x))
df=pd.DataFrame(train.loc[train['target_url']!=""]['sentiment'].value_counts()).reset_index()
df.rename(columns={"index": "sentiment", "sentiment": "url_count"})


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * As expected,our target variables that has urls are **neutral** tweets.Only few urls are found in positive and negative.
# * Our model would easily judge that if the text has URL it will be along neutral side.

# <font size="+1" color="chocolate"><b> 4.1.2 Punctuations - Selected Text</b></font> <a id="4.1.2"></a>

# #### Can punctutations/symbols play a part in modelling?
# 
# Since we are analysing sentimental tweets,people describe their emotions in symbols.Say symbols like continuous stars **( * )** is considered to be extreme emotions(happy,angry,delight etc).Other symbols like **(# - tagging)** or **(@ - mention)** are also used very often in tweets.
# 
# Lets analyse all of them including other punctuations

# In[ ]:


# Function to find punctuation
def find_punct(text):
    line = re.findall(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', text)
    string="".join(line)
    return list(string)


# In[ ]:


# New Features with punctuation and punctuation length
train['target_punct']=train['target'].apply(lambda x:find_punct(x))
train['target_punct_len']=train['target'].apply(lambda x:len(find_punct(x)))


# In[ ]:


punc_df=pd.DataFrame(train,columns=['target_punct','sentiment'])
punc_df=punc_df[punc_df['target_punct'].map(lambda d: len(d)) > 0]
punc_df=punc_df.explode('target_punct')

positive_df=pd.DataFrame(punc_df.loc[punc_df['sentiment']=="positive"]['target_punct'].value_counts()).reset_index().rename(columns={'index': 'punct','target_punct':'pos_punct'})
negative_df=pd.DataFrame(punc_df.loc[punc_df['sentiment']=="negative"]['target_punct'].value_counts()).reset_index().rename(columns={'index': 'punct','target_punct':'neg_punct'})
neutral_df=pd.DataFrame(punc_df.loc[punc_df['sentiment']=="neutral"]['target_punct'].value_counts()).reset_index().rename(columns={'index': 'punct','target_punct':'neut_punct'})


# In[ ]:


fig = make_subplots(rows=1, cols=3)

fig.append_trace(go.Bar(x=positive_df.punct[:10],y=positive_df.pos_punct[:10],name='Positive',marker_color='green'), row=1, col=1)
fig.append_trace(go.Bar(x=negative_df.punct[:10],y=negative_df.neg_punct[:10],name='Negative',marker_color='red'), row=1, col=2)
fig.append_trace(go.Bar(x=neutral_df.punct[:10],y=neutral_df.neut_punct[:10],name='Neutral',marker_color='orange'), row=1, col=3)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text="Selected Text - Sentiment vs Punctuation",title_x=0.5)
fig.show()


# #### How much impact does ( * ) have?

# In[ ]:


def find_star(text):
   # if len(text.split())<1:
    line=re.findall(r'[*]{2,5}',text)
    return len(line)


# In[ ]:


train['star']=train['target'].apply(lambda x:find_star(x))
train.loc[train['star']!=0]['sentiment'].value_counts().to_frame()


# Eventhough negative shows high counts.Still it describes about neutral tweets dependency.Let us analyse the tweet with only ( * ) in tweet.

# In[ ]:


def find_only_star(text):
    if len(text.split())==1:
        line=re.findall(r'[*]{2,5}',text)
        return len(line)
    else:
        return 0


# In[ ]:


# grt column value that has only * in its tweet
train['only_star']=train['target'].apply(lambda x:find_only_star(x))
train.loc[train['only_star']==1]['sentiment'].value_counts()


# This is interesting.Selected text with only **star** symbol has negative sentiments.
# Replace those star with word "**abusive**"

# In[ ]:


train['target']= np.where(train['only_star']==1,"abusive",train['target'])


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * All sentiment tweets have full stop and quotes on top of list which is expected.
# * Tweets with ( * ) only has negative sentiments 
# * ( # ) and ( @ ) are no where in top lists.

# <font size="+1" color="chocolate"><b>4.1.3 Length - Punctuation</b></font> <a id="4.1.3"></a>

# ### How will punctuations play part in this competition?
# 
# Yes we know that the jaccard score matches string with punctuations too.
# 
# <u>For example:</u> <br>
# a=good.<br>
# b=good <br>
# Since there is a punctuation in a,length of a is not equal to length of b.<br>
# 
# So punctuations needs more attention as well.We cant remove it from our text without convincing reason.

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Histogram(
    x=train[train['sentiment']=='positive']['target_punct_len'],
    #histnorm='percent',
    name='Positive', # name used in legend and hover labels
    xbins=dict( # bins used for histogram
        start=1,
        end=20,
        size=1
    ),
    marker_color='green',
    opacity=0.75
))
fig.add_trace(go.Histogram(
    x=train[train['sentiment']=='negative']['target_punct_len'],
   # histnorm='percent',
    name='Negative',
    xbins=dict(
        start=1,
        end=20,
        size=1
    ),
    marker_color='red',
    opacity=0.75
))

fig.add_trace(go.Histogram(
    x=train[train['sentiment']=='neutral']['target_punct_len'],
   # histnorm='percent',
    name='Neutral',
    xbins=dict(
        start=1,
        end=20,
        size=1
    ),
    marker_color='orange',
    opacity=0.75
))
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(
    title_text='Distribution - Selected Text Punctuation length', # title of plot
    xaxis_title_text='Length', # xaxis label
    yaxis_title_text='Count', # yaxis label
    title_x=0.5,
    bargroupgap=0.1 # gap between bars of the same location coordinates
)
fig.show()


# <font size="+1" color="chocolate"><b>4.1.4 Length - Selected Text</b></font> <br><a id="4.1.4"></a>
# 
# In this section we will analyse more on words of *selected_text*. Before extracting length of selected_text words,let us remove urls and punctuation
# 
# **Note**: We need to keep in mind that while modelling we cant strip them.They play part in jaccard score.Now we will do this check the distribution of words only.

# #### Remove URLs & Punctuation

# In[ ]:


def remove_link(string): 
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'," ",string)
    return " ".join(text.split())


# In[ ]:


def remove_punct(text):
    line = re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]+'," ",text)
    return " ".join(line.split())


# In[ ]:


train['target']=train['target'].apply(lambda x:remove_link(x))
train['target']=train['target'].apply(lambda x:remove_punct(x))


# In[ ]:


train['target_tweet_length']=train['target'].str.split().map(lambda x: len(x))
train['target_tweet_length'].describe().to_frame()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Histogram(x=train[train['sentiment']=='positive']['target_tweet_length'],name="Positive",marker_color='green'))
fig.add_trace(go.Histogram(x=train[train['sentiment']=='negative']['target_tweet_length'],name="Negative",marker_color='red'))
fig.add_trace(go.Histogram(x=train[train['sentiment']=='neutral']['target_tweet_length'],name="Neutral",marker_color='orange'))

# Overlay both histograms

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(
    title_text='Length of each Selected Text tweet', # title of plot
    xaxis_title_text='Length', # xaxis label
    yaxis_title_text='Count', # yaxis label
    title_x=0.5,
    barmode='overlay'
)

# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Average length of word in selected text is around 7 which seems bit high.This says we need to** predict more words per tweet**
# * Most of positive and negative tweets have selected text length **less than 5**.This is good significance as we can add exclude conditons for predicting words.
# * Neutral sentiments are distributed across all length.

# <font size="+1" color="chocolate"><b>4.1.5 Average Length - Selected Text</b></font><br><a id="4.1.5"></a>

# Averaging selected text can determine us how long the train data has accepted strings from whole tweet.We will find the distribution of the selected text or target variable

# In[ ]:


train['target_average_word_len']=train['target'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x))
train['target_average_word_len'].describe().to_frame()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Histogram(x=train[train['sentiment']=='positive']['target_average_word_len'], xbins=dict(
        start=1,
        end=30,
        size=1
    ),name="Positive",marker_color='green'))
fig.add_trace(go.Histogram(x=train[train['sentiment']=='negative']['target_average_word_len'], xbins=dict(
        start=1,
        end=30,
        size=1
    ),name="Negative",marker_color='red'))
fig.add_trace(go.Histogram(x=train[train['sentiment']=='neutral']['target_average_word_len'], xbins=dict(
        start=1,
        end=30,
        size=1
    ),name="Neutral",marker_color='orange'))

# Overlay both histograms
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(
    title_text='Target - Average length of Selected Text', # title of plot
    xaxis_title_text='Length', # xaxis label
    yaxis_title_text='Count', # yaxis label
    title_x=0.5,
    barmode='overlay'
)

# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Positive and negative are averaging around 4-5 per sentence
# * Neutral keeps their high density.

# <font size="+1" color="chocolate"><b>4.1.6 Most Words - Selected Text</b></font> <br><a id="4.1.6"></a>

# This is essential to know before we perform modelling.Before we perform GLOVE or any word embedding,it is good to know which all are the words often used.

# In[ ]:


def create_corpus(data,feature,sentiment):
    corpus=[]
    for x in data[data['sentiment']==sentiment][feature].str.split():
        for i in x:
            corpus.append(i)
    return corpus


# In[ ]:


def corpus_sentiment(data,feature,sentiment):
    corpus=create_corpus(data,feature,sentiment)
    dic=defaultdict(int)
    for word in corpus:
        if word not in stop:
            dic[word]+=1
    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)
    x,y=zip(*top)
    return x,y


# In[ ]:


f, axes = plt.subplots(1,3,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=list(corpus_sentiment(train,'target','positive')[0])[:15], x= list(corpus_sentiment(train,'target','positive')[1])[:15],color="green",ax=axes[0]).set_title("Positive",color="green")
sns.barplot(y=list(corpus_sentiment(train,'target','negative')[0])[:15], x=list(corpus_sentiment(train,'target','negative')[1])[:15],color="red", ax=axes[1]).set_title("Negative",color="red")
sns.barplot(y=list(corpus_sentiment(train,'target','neutral')[0])[:15], x=list(corpus_sentiment(train,'target','neutral')[1])[:15],color="orange", ax=axes[2]).set_title("Neutral",color="orange")

plt.suptitle("Most Common Words in Selected Text" ,fontsize=25,color="blue")
plt.show() 


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Positive produces words like **good,happy,thanks** which is quite expected
# * Negative produces words like **miss,sad,sorry,bad** which indicates negative emotions
# * Neutral produces words like **get,going,work** which is sort of common words
# * We could see "day" comes in all three sentiments.(These tweets might be indicating a event day.Hopefully we may come to know after competition)

# <font size="+2" color="indigo"><b>4.2 Sentiment</b></font><a id="4.2"></a>

# Sentiment variable is the theme of our data.Let us know how it is distributed accross whole data.As of now most of participants in this competition have done modeling based on sentiments.

# In[ ]:


# count unique values present in each column
def count_values(df,feature):
    total=df.loc[:,feature].value_counts(dropna=False)
    percent=round(df.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percent],axis=1,keys=['Total','Percent'])


# <font size="+1" color="chocolate"><b>4.2.1 Sentiment - Train</b></font><a id="4.2.1"></a>

# In[ ]:


count_values(train,'sentiment')


# <font size="+1" color="chocolate"><b>4.2.2 Sentiment - Test</b></font><a id="4.2.2"></a>

# In[ ]:


count_values(train,'sentiment')


# In[ ]:


sent_train=count_values(train,'sentiment')
sent_test=count_values(test,'sentiment')

colors = ['orange','green','red']

fig = make_subplots(rows=1, cols=2,specs=[[{"type": "pie"}, {"type": "pie"}]])

fig.add_trace(go.Pie(labels=list(sent_train.index), values=list(sent_train.Total.values), hoverinfo='label+percent', 
               textinfo='value+percent',marker=dict(colors=colors)),row=1,col=1)
fig.add_trace(go.Pie(labels=list(sent_test.index), values=list(sent_test.Total.values), hoverinfo='label+percent', 
               textinfo='value+percent', marker=dict(colors=colors)),row=1,col=2)
fig.update_layout( title_text="Sentiment - Train vs Test",title_x=0.5)
iplot(fig)


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * We could see the distribution of sentiment is spread equally in both train and test data.  
# * This indicates that the data is **randomly shuffled**.
# * Both data has **high number of neutral reviews**.Positive and negative reviews have just **2.9% difference** between them in terms of spread.
# 

# <font size="+2" color="indigo"><b>4.3 Text</b></font><br><a id="4.3"></a>
# 
# This is the variable which we are going to train and predict
# 
# Before doing analysis,let us merge train and test data

# In[ ]:


full_data=pd.concat([train,test])
full_data['text']=full_data['text'].str.lower()
full_data.shape


# <font size="+1" color="chocolate"><b>4.3.1 Punctuation - Text</b></font><a id="4.3.1"></a>

# Already we analysed punctuations of selected text,now lets have a peek into whole text.

# In[ ]:


# New Features with punctuation and punctuation length
full_data['text_punct']=full_data['text'].apply(lambda x:find_punct(x))
full_data['text_punct_len']=full_data['text'].apply(lambda x:len(find_punct(x)))


# In[ ]:


punc_text_df=pd.DataFrame(full_data,columns=['text_punct','sentiment'])
punc_text_df=punc_text_df[punc_text_df['text_punct'].map(lambda d: len(d)) > 0]
punc_text_df=punc_text_df.explode('text_punct')

positive_text_df=pd.DataFrame(punc_text_df.loc[punc_text_df['sentiment']=="positive"]['text_punct'].value_counts()).reset_index().rename(columns={'index': 'punct','text_punct':'pos_punct'})
negative_text_df=pd.DataFrame(punc_text_df.loc[punc_text_df['sentiment']=="negative"]['text_punct'].value_counts()).reset_index().rename(columns={'index': 'punct','text_punct':'neg_punct'})
neutral_text_df=pd.DataFrame(punc_text_df.loc[punc_text_df['sentiment']=="neutral"]['text_punct'].value_counts()).reset_index().rename(columns={'index': 'punct','text_punct':'neut_punct'})


# In[ ]:


fig = make_subplots(rows=1, cols=3)
fig.append_trace(go.Bar(x=positive_text_df.punct[:15],y=positive_text_df.pos_punct[:10],name='Positive',marker_color='green'), row=1, col=1)
fig.append_trace(go.Bar(x=negative_text_df.punct[:15],y=negative_text_df.neg_punct[:15],name='Negative',marker_color='red'), row=1, col=2)
fig.append_trace(go.Bar(x=neutral_text_df.punct[:15],y=neutral_text_df.neut_punct[:15],name='Neutral',marker_color='orange'), row=1, col=3)
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text="Text - Sentiment vs Punctuation",title_x=0.5)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Interestingly,we have ( @ ) and ( # ) in top positive and neutral tweets.
# * As expected ( * ) are found more in negative than rest two sentiments.May consider them as abusive language.

# <font size="+1" color="chocolate"><b>4.3.2 Length - Text</b></font><a id="4.3.2"></a>

# We would know the proportion between text and selected text.Now lets know the length of whole text.

# In[ ]:


full_data['text']=full_data['text'].apply(lambda x:remove_link(x))
full_data['text']=full_data['text'].apply(lambda x:remove_punct(x))


# Two records were found to be null after removal of punctuations.They are filled with values

# In[ ]:


full_data.loc[full_data['text']=="",['text']]="nothing"


# In[ ]:


full_data['text_tweet_length']=full_data['text'].str.split().map(lambda x: len(x))
full_data['text_tweet_length'].describe().to_frame()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Box(x=full_data[full_data['sentiment']=='neutral']['text_tweet_length'],name="Neutral",marker_color='orange',boxmean='sd'))
fig.add_trace(go.Box(x=full_data[full_data['sentiment']=='negative']['text_tweet_length'],name="Negative",marker_color='red',boxmean='sd'))
fig.add_trace(go.Box(x=full_data[full_data['sentiment']=='positive']['text_tweet_length'],name="Positive",marker_color='green',boxmean='sd'))

# Overlay both histograms
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(
    title_text='Distribution of Text Length', # title of plot
    xaxis_title_text='Length', # xaxis label
    yaxis_title_text='Count', # yaxis label
    title_x=0.5
)

# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * The average length is around.Selected text had only 5.One thing to notice that we merged whole train and test data.So this is more or less expected.
# * We observe almost similar kind of distribution among all sentiments.This describes that words are diversified with length among all sentiments. 

# <font size="+1" color="chocolate"><b>4.3.3 Average Length - Text</b></font><a id="4.3.3"></a>

# In[ ]:


full_data['text_average_word_len']=full_data['text'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x))
full_data['text_average_word_len'].describe().to_frame()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Box(x=full_data[full_data['sentiment']=='positive']['text_average_word_len'],name="Positive",marker_color='green'))
fig.add_trace(go.Box(x=full_data[full_data['sentiment']=='negative']['text_average_word_len'],name="Negative",marker_color='red'))
fig.add_trace(go.Box(x=full_data[full_data['sentiment']=='neutral']['text_average_word_len'],name="Neutral",marker_color='orange'))

# Overlay both histograms
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(
    title_text='Average Length of Tweet', # title of plot
    xaxis_title_text='Length', # xaxis label
    yaxis_title_text='Count', # yaxis label
    title_x=0.5,
)

# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * As we saw earlier,some sort of uniform distribution with more outlier are found.

# <font size="+1" color="chocolate"><b>4.3.4 Most Words - Text</b></font><a id="4.3.4"></a>

# We need to what sort of words are place on top for each sentiments

# In[ ]:


f, axes = plt.subplots(1,3,figsize=(20,12))
sns.set(font_scale=2)
sns.barplot(y=list(corpus_sentiment(full_data,'text','positive')[0])[:15], x= list(corpus_sentiment(full_data,'text','positive')[1])[:15],color="green",ax=axes[0]).set_title("Positive",color="green")
sns.barplot(y=list(corpus_sentiment(full_data,'text','negative')[0])[:15], x=list(corpus_sentiment(full_data,'text','negative')[1])[:15],color="red", ax=axes[1]).set_title("Negative",color="red")
sns.barplot(y=list(corpus_sentiment(full_data,'text','neutral')[0])[:15], x=list(corpus_sentiment(full_data,'text','neutral')[1])[:15],color="orange", ax=axes[2]).set_title("Neutral",color="orange")

plt.suptitle("Most Common Words - Text" ,fontsize=25,color="blue")
plt.show() 


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Positive Words like **"good","love","happy"** are in top of the list which is expected.
# * Neutral Words like **"get","go","day","lol"** are also sort of neutral words.
# * But we could see some difference in negative,some non negative words like **"get","miss","like","work"** are edging to top.

# <font size="+1" color="chocolate"><b>4.3.5 Stopwords</b></font><a id="4.3.5"></a>

# We cannot remove stopwords from this variable.They are also connsidered as per our metric.Still it is good to know how is distributed to each sentiments

# In[ ]:


def corpus_sentiment_stop(data,feature,sentiment):
    corpus=create_corpus(data,feature,sentiment)
    dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word]+=1
    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)
    x,y=zip(*top)
    return x,y


# In[ ]:


f, axes = plt.subplots(1,3,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=list(corpus_sentiment_stop(full_data,'text','positive')[0])[:15], x= list(corpus_sentiment_stop(full_data,'text','positive')[1])[:15],color="green",ax=axes[0]).set_title("Positive",color="green")
sns.barplot(y=list(corpus_sentiment_stop(full_data,'text','negative')[0])[:15], x=list(corpus_sentiment_stop(full_data,'text','negative')[1])[:15],color="red", ax=axes[1]).set_title("Negative",color="red")
sns.barplot(y=list(corpus_sentiment_stop(full_data,'text','neutral')[0])[:15], x=list(corpus_sentiment_stop(full_data,'text','neutral')[1])[:15],color="orange", ax=axes[2]).set_title("Neutral",color="orange")

plt.suptitle("Most Common Stop Words - Text" ,fontsize=25,color="blue")
plt.show() 


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * **"I" ,"the" ,"too" ,"a" ** are leading on all sentiments.This is very much expected. 

# <font size="+3" color="Blue"><b>5. Comparison</b></font><a id="5"></a>

# #### Why do we need comparison?
# 
# We are facing information extraction from a sentence/tweet.And we have selected text from train data too.So We have to kind of matching these two variables in all expects.(Train data -Text will compete with Test data - Text)
# 
# But for now ,I think extracting information from these two variables will be a key insight to predict selected text for test data
# 
# 
# This section holds :
# 
# * **(N)grams     - Train Selected text** 
# * **(N)grams     - Train text vs Test text** 
# * **Venn Diagram - Train text vs Selected text**
# * **Word cloud   - Train selected Text**
# * **Word cloud   - Train text vs Test Text**
# 

# In[ ]:


train_word=full_data[:train.shape[0]]
test_word=full_data[train.shape[0]:]


# <font size="+2" color="indigo"><b>5.1 N-grams</b></font><br><a id="5.1"></a>

# <font size="+1" color="chocolate"><b>5.1.1 Train - Selected text</b></font><br><a id="5.1.1"></a>

# In[ ]:


def ngrams_plot(corpus,ngram_range,n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer(stop_words = 'english',ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    total_list=words_freq[:n]
    df=pd.DataFrame(total_list,columns=['text','count'])
    return df


# In[ ]:


f, axes = plt.subplots(1,3,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='positive']['target'],(1,1),10)['text'],
            x= ngrams_plot(train_word.loc[train_word['sentiment']=='positive']['target'],(1,1),10)['count'],
            color="green",ax=axes[0]).set_title("Positive",color="green")
axes[0].set(ylabel=" ",xlabel=" ")
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='negative']['target'],(1,1),10)['text'],
            x=ngrams_plot(train_word.loc[train_word['sentiment']=='negative']['target'],(1,1),10)['count'],
            color="red", ax=axes[1]).set_title("Negative",color="red")
axes[1].set(ylabel=" ",xlabel=" ")
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='neutral']['target'],(1,1),10)['text'], 
            x=ngrams_plot(train_word.loc[train_word['sentiment']=='neutral']['target'],(1,1),10)['count'],
            color="orange", ax=axes[2]).set_title("Neutral",color="orange")
axes[2].set(ylabel=" ",xlabel=" ")
f.suptitle("Top 10 Selected Text - Unigram" ,fontsize=25,color="blue")
f.show() 

f, axes = plt.subplots(1,3,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='positive']['target'],(2,2),10)['text'],
            x= ngrams_plot(train_word.loc[train_word['sentiment']=='positive']['target'],(2,2),10)['count'],
            color="green",ax=axes[0]).set_title("Positive",color="green")
axes[0].set(ylabel=" ",xlabel=" ")
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='negative']['target'],(2,2),10)['text'],
            x=ngrams_plot(train_word.loc[train_word['sentiment']=='negative']['target'],(2,2),10)['count'],
            color="red", ax=axes[1]).set_title("Negative",color="red")
axes[1].set(ylabel=" ",xlabel=" ")
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='neutral']['target'],(2,2),10)['text'], 
            x=ngrams_plot(train_word.loc[train_word['sentiment']=='neutral']['target'],(2,2),10)['count'],
            color="orange", ax=axes[2]).set_title("Neutral",color="orange")
axes[2].set(ylabel=" ",xlabel=" ")
f.suptitle("Top 10 Selected Text - Bigram" ,fontsize=25,color="blue")
f.show() 

f, axes = plt.subplots(1,3,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='positive']['target'],(3,3),10)['text'],
            x= ngrams_plot(train_word.loc[train_word['sentiment']=='positive']['target'],(3,3),10)['count'],
            color="green",ax=axes[0]).set_title("Positive",color="green")
axes[0].set(ylabel=" ",xlabel=" ")
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='negative']['target'],(3,3),10)['text'],
            x=ngrams_plot(train_word.loc[train_word['sentiment']=='negative']['target'],(3,3),10)['count'],
            color="red", ax=axes[1]).set_title("Negative",color="red")
axes[1].set(ylabel=" ",xlabel=" ")
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='neutral']['target'],(3,3),10)['text'], 
            x=ngrams_plot(train_word.loc[train_word['sentiment']=='neutral']['target'],(3,3),10)['count'],
            color="orange", ax=axes[2]).set_title("Neutral",color="orange")
axes[2].set(ylabel=" ",xlabel=" ")
f.suptitle("Top 10 Selected Text - Trigram" ,fontsize=25,color="blue")
f.show() 


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * "Day" was always a part among all sentiment tweets and thus confirmed the event.Well,<b><i>"Happy Mothers day"</i></b>.All these tweets were collected around **May 2nd Week**.
# * Positive bigrams and trigrams words are more biased towards mothers day.
# * Negative ngrams are displaying negative emotional words.
# * Neutral shows common words.Nothing much inference from this sentiment.

# <font size="+1" color="chocolate"><b>5.1.2 Text - Train vs Test</b></font><br> <a id="5.1.2"></a>

# In[ ]:


f, axes = plt.subplots(1,2,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='positive']['text'],(1,1),15)['text'],
            x= ngrams_plot(train_word.loc[train_word['sentiment']=='positive']['text'],(1,1),15)['count'],
            color="green",ax=axes[0]).set_title("Positive",color="green")
axes[0].set(ylabel=" ",xlabel=" ")
f.suptitle("Tweet Unigrams - Train vs Test" ,fontsize=30,color="blue")
sns.barplot(y=ngrams_plot(test_word.loc[test_word['sentiment']=='positive']['text'],(1,1),15)['text'],
            x=ngrams_plot(test_word.loc[test_word['sentiment']=='positive']['text'],(1,1),15)['count'],
            color="green", ax=axes[1]).set_title("Positive",color="green")
axes[1].set(ylabel=" ",xlabel=" ")
f.show() 

f, axes = plt.subplots(1,2,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='negative']['text'],(1,1),15)['text'],
            x= ngrams_plot(train_word.loc[train_word['sentiment']=='negative']['text'],(1,1),15)['count'],
            color="red",ax=axes[0]).set_title("Negative",color="red")
axes[0].set(ylabel=" ",xlabel=" ")
sns.barplot(y=ngrams_plot(test_word.loc[test_word['sentiment']=='negative']['text'],(1,1),15)['text'],
            x=ngrams_plot(test_word.loc[test_word['sentiment']=='negative']['text'],(1,1),15)['count'],
            color="red", ax=axes[1]).set_title("Negative",color="red")
axes[1].set(ylabel=" ",xlabel=" ")
f.show() 

f, axes = plt.subplots(1,2,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='neutral']['text'],(1,1),15)['text'],
            x= ngrams_plot(train_word.loc[train_word['sentiment']=='neutral']['text'],(1,1),15)['count'],
            color="orange",ax=axes[0]).set_title("Neutral",color="orange")
axes[0].set(ylabel=" ",xlabel=" ")
sns.barplot(y=ngrams_plot(test_word.loc[test_word['sentiment']=='neutral']['text'],(1,1),15)['text'],
            x=ngrams_plot(test_word.loc[test_word['sentiment']=='neutral']['text'],(1,1),15)['count'],
            color="orange", ax=axes[1]).set_title("Neutral",color="orange")
axes[1].set(ylabel=" ",xlabel=" ")
f.show() 


# In[ ]:


f, axes = plt.subplots(1,2,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='positive']['text'],(2,2),15)['text'],
            x= ngrams_plot(train_word.loc[train_word['sentiment']=='positive']['text'],(2,2),15)['count'],
            color="green",ax=axes[0]).set_title("Positive",color="green")
axes[0].set(ylabel=" ",xlabel=" ")
f.suptitle("Tweet Bigrams - Train vs Test" ,fontsize=30,color="blue")
sns.barplot(y=ngrams_plot(test_word.loc[test_word['sentiment']=='positive']['text'],(2,2),15)['text'],
            x=ngrams_plot(test_word.loc[test_word['sentiment']=='positive']['text'],(2,2),15)['count'],
            color="green", ax=axes[1]).set_title("Positive",color="green")
axes[1].set(ylabel=" ",xlabel=" ")
f.show() 

f, axes = plt.subplots(1,2,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='negative']['text'],(2,2),15)['text'],
            x= ngrams_plot(train_word.loc[train_word['sentiment']=='negative']['text'],(2,2),15)['count'],
            color="red",ax=axes[0]).set_title("Negative",color="red")
axes[0].set(ylabel=" ",xlabel=" ")
sns.barplot(y=ngrams_plot(test_word.loc[test_word['sentiment']=='negative']['text'],(2,2),15)['text'],
            x=ngrams_plot(test_word.loc[test_word['sentiment']=='negative']['text'],(2,2),15)['count'],
            color="red", ax=axes[1]).set_title("Negative",color="red")
axes[1].set(ylabel=" ",xlabel=" ")
f.show() 

f, axes = plt.subplots(1,2,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='neutral']['text'],(2,2),15)['text'],
            x= ngrams_plot(train_word.loc[train_word['sentiment']=='neutral']['text'],(2,2),15)['count'],
            color="orange",ax=axes[0]).set_title("Neutral",color="orange")
axes[0].set(ylabel=" ",xlabel=" ")
sns.barplot(y=ngrams_plot(test_word.loc[test_word['sentiment']=='neutral']['text'],(2,2),15)['text'],
            x=ngrams_plot(test_word.loc[test_word['sentiment']=='neutral']['text'],(2,2),15)['count'],
            color="orange", ax=axes[1]).set_title("Neutral",color="orange")
axes[1].set(ylabel=" ",xlabel=" ")
f.show() 


# In[ ]:


f, axes = plt.subplots(1,2,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='positive']['text'],(3,3),15)['text'],
            x= ngrams_plot(train_word.loc[train_word['sentiment']=='positive']['text'],(3,3),15)['count'],
            color="green",ax=axes[0]).set_title("Positive",color="green")
axes[0].set(ylabel=" ",xlabel=" ")
f.suptitle("Tweet Trigrams - Train vs Test" ,fontsize=30,color="blue")
sns.barplot(y=ngrams_plot(test_word.loc[test_word['sentiment']=='positive']['text'],(3,3),15)['text'],
            x=ngrams_plot(test_word.loc[test_word['sentiment']=='positive']['text'],(3,3),15)['count'],
            color="green", ax=axes[1]).set_title("Positive",color="green")
axes[1].set(ylabel=" ",xlabel=" ")
f.show() 

f, axes = plt.subplots(1,2,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='negative']['text'],(3,3),15)['text'],
            x= ngrams_plot(train_word.loc[train_word['sentiment']=='negative']['text'],(3,3),15)['count'],
            color="red",ax=axes[0]).set_title("Negative",color="red")
axes[0].set(ylabel=" ",xlabel=" ")
sns.barplot(y=ngrams_plot(test_word.loc[test_word['sentiment']=='negative']['text'],(3,3),15)['text'],
            x=ngrams_plot(test_word.loc[test_word['sentiment']=='negative']['text'],(3,3),15)['count'],
            color="red", ax=axes[1]).set_title("Negative",color="red")
axes[1].set(ylabel=" ",xlabel=" ")
f.show() 

f, axes = plt.subplots(1,2,figsize=(20,12))
sns.set(font_scale =2)
sns.barplot(y=ngrams_plot(train_word.loc[train_word['sentiment']=='neutral']['text'],(3,3),15)['text'],
            x= ngrams_plot(train_word.loc[train_word['sentiment']=='neutral']['text'],(3,3),15)['count'],
            color="orange",ax=axes[0]).set_title("Neutral",color="orange")
axes[0].set(ylabel=" ",xlabel=" ")
sns.barplot(y=ngrams_plot(test_word.loc[test_word['sentiment']=='neutral']['text'],(3,3),15)['text'],
            x=ngrams_plot(test_word.loc[test_word['sentiment']=='neutral']['text'],(3,3),15)['count'],
            color="orange", ax=axes[1]).set_title("Neutral",color="orange")
axes[1].set(ylabel=" ",xlabel=" ")
f.show() 


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Just like above,we see same sort of words.But there are few new words found from test data.
# * We saw mostly words are resembling between train and test data

# <font size="+2" color="indigo"><b>5.2 Venn</b></font><br><a id="5.2"></a>

# <font size="+1" color="chocolate"><b>5.2.1 Venn - Text vs Selected Text</b></font><br><a id="5.2.1"></a>

# In[ ]:


pos_text_list=list(corpus_sentiment(train_word,'text','positive')[0])
pos_target_list=list(corpus_sentiment(train_word,'target','positive')[0])
neg_text_list=list(corpus_sentiment(train_word,'text','negative')[0])
neg_target_list=list(corpus_sentiment(train_word,'target','negative')[0])
neutral_text_list=list(corpus_sentiment(train_word,'text','neutral')[0])
neutral_target_list=list(corpus_sentiment(train_word,'target','neutral')[0])

pos_common_words_list=list(set(pos_text_list).intersection(pos_target_list))
neg_common_words_list=list(set(neg_text_list).intersection(neg_target_list))
neutral_common_words_list=list(set(neutral_text_list).intersection(neutral_target_list))


# In[ ]:


def venn_plot(x,y,common_list,title):
    plt.title(title)
    venn.venn2(subsets=(x,y,common_list), alpha = 0.5,set_labels=("# of unique words in Text","# of unique words in Selected text"))
    return plt.show()

venn_plot(len(pos_text_list),len(pos_target_list),len(pos_common_words_list),"# of Common Positive Words")
venn_plot(len(neg_text_list),len(neg_target_list),len(neg_common_words_list),"# of Common Negative Words")
venn_plot(len(neutral_text_list),len(neutral_target_list),len(neutral_common_words_list),"# of Common Neutral Words")


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Neutral have most shared words between text and selected text.This is true because unseperable words are pushed in neutral section.
# * Positive and Negative have almost same sort of allocation according to their own length.

# <font size="+2" color="indigo"><b>5.3 Wordcloud</b></font><br><a id="5.3"></a>

# <font size="+1" color="chocolate"><b>5.3.1 Word Cloud - Selected Text</b></font><br><a id="5.3.1"></a>

# In[ ]:


# d = '../input/twitter/'
# bird = np.array(Image.open(d + 'twitter_mask.png'))
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[30, 15])
# wordcloud1 = WordCloud( background_color='white',mask=bird,colormap="Greens",
#                         width=600,
#                         height=400).generate(" ".join(train_word.loc[train_word['sentiment']=="positive"]['target']))
# ax1.imshow(wordcloud1)
# ax1.axis('off')
# ax1.set_title('Positive Selected Text',fontsize=35);

# wordcloud2 = WordCloud( background_color='white',mask=bird,colormap="Reds",
#                         width=600,
#                         height=400).generate(" ".join(train_word.loc[train_word['sentiment']=="negative"]['target']))
# ax2.imshow(wordcloud2)
# ax2.axis('off')
# ax2.set_title('Negative Selected Text',fontsize=35);

# wordcloud3 = WordCloud( background_color='white',mask=bird,colormap="Blues",
#                         width=600,
#                         height=400).generate(" ".join(train_word.loc[train_word['sentiment']=="neutral"]['target']))
# ax3.imshow(wordcloud3)
# ax3.axis('off')
# ax3.set_title('Neutral Selected Text',fontsize=35);


# <font size="+1" color="chocolate"><b>5.3.2 Word Cloud Train vs Text</b></font><br><a id="5.3.2"></a>

# In[ ]:


# fig, ((ax1,ax3,ax5),(ax2, ax4,ax6)) = plt.subplots(2, 3, figsize=[30, 15])

# wordcloud1 = WordCloud( background_color='white',mask=bird,colormap="Greens",
#                         width=600,
#                         height=400).generate(" ".join(train_word.loc[train_word['sentiment']=="positive"]['text']))
# ax1.imshow(wordcloud1)
# ax1.axis('off')
# ax1.set_title('Train - Positive text',fontsize=35);

# wordcloud2 = WordCloud( background_color='white',mask=bird,colormap="Greens",
#                         width=600,
#                         height=400).generate(" ".join(test_word.loc[test_word['sentiment']=="positive"]['text']))
# ax2.imshow(wordcloud2)
# ax2.axis('off')
# ax2.set_title('Test - Positive text',fontsize=35);


# wordcloud3 = WordCloud( background_color='white',mask=bird,colormap="Reds",
#                         width=600,
#                         height=400).generate(" ".join(train_word.loc[train_word['sentiment']=="negative"]['text']))
# ax3.imshow(wordcloud3)
# ax3.axis('off')
# ax3.set_title('Train - Negative text',fontsize=35);


# wordcloud4 = WordCloud( background_color='white',mask=bird,colormap="Reds",
#                         width=600,
#                         height=400).generate(" ".join(test_word.loc[test_word['sentiment']=="negative"]['text']))
# ax4.imshow(wordcloud4)
# ax4.axis('off')
# ax4.set_title('Test - Negative text',fontsize=35);



# wordcloud5 = WordCloud( background_color='white',mask=bird,colormap="Blues",
#                         width=600,
#                         height=400).generate(" ".join(train_word.loc[train_word['sentiment']=="neutral"]['text']))
# ax5.imshow(wordcloud5)
# ax5.axis('off')
# ax5.set_title('Train - Neutral text',fontsize=35);


# wordcloud6 = WordCloud( background_color='white',mask=bird,colormap="Blues",
#                         width=600,
#                         height=400).generate(" ".join(test_word.loc[test_word['sentiment']=="neutral"]['text']))
# ax6.imshow(wordcloud6)
# ax6.axis('off')
# ax6.set_title('Test - Neutral text',fontsize=35);


# <font size="+2" color="indigo"><b>5.4 Jaccard</b></font><br><a id="5.4"></a>

# <font size="+1" color="chocolate"><b>5.4.1 Text vs Selected Text</b></font><br><a id="5.4.1"></a>

# #### How can we utilize jaccard metric here?
# 
# The next upcoming plots will not display predicted vs actual values.In fact we need to understand that **how much selected words are pulled out from original text**.
# 
# * If values is high or near to 1 ,the whole text is similar to selected text
# * If values is low or near to 0 ,only few selected text are taken from whole text.
# 
# **Note:** I have removed URLs and punctuations already.Eventhough they play a small part,i have reduced the noise from them.Let it be only text vs text.

# In[ ]:


results_jaccard=[]

for ind,row in train_word.iterrows():
    sentence1 = row.text
    sentence2 = row.target

    jaccard_score = jaccard(sentence1,sentence2) # Jaccard function is defined at top of kernel
    results_jaccard.append([sentence1,sentence2,jaccard_score])


# In[ ]:


jaccard = pd.DataFrame(results_jaccard,columns=["text","target","jaccard_score"])
train_word = train_word.merge(jaccard,how='outer')


# In[ ]:


train_word['jaccard_score'].describe().to_frame()


# In[ ]:


x = train_word['text_tweet_length']
y = train_word['jaccard_score']
fig = go.Figure()
fig.add_trace(go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'gray',
        reversescale = True,
        xaxis = 'x',
        yaxis = 'y'
    ))
fig.add_trace(go.Scatter(
        x = x,
        y = y,
        xaxis = 'x',
        yaxis = 'y',
        mode = 'markers',
        marker = dict(
            color = 'blue',  #'rgba(0,0,0,0.3)',
            size = 3
        )
    ))
fig.add_trace(go.Histogram(
        y = y,
        xaxis = 'x2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ))
fig.add_trace(go.Histogram(
        x = x,
        yaxis = 'y2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ))

fig.update_layout(
    autosize = False,
    xaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    yaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    xaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    yaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    height = 600,
    width = 600,
    bargap = 0,
    hovermode = 'closest',
    showlegend = False,
    title_text="Jaccard - Text vs Selected Text ",title_x=0.5
)

fig.show()


# In[ ]:


print("Jaccard score equals 1 : {}%".format(round(train_word.loc[train_word['jaccard_score']==1].shape[0]/train_word.shape[0]*100,2)))
print("Jaccard score less than 0.3 : {}%".format(round(train_word.loc[train_word['jaccard_score']<0.3].shape[0]/train_word.shape[0]*100,2)))


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Almost 80% of train data has jaccard score of 1 which means words in text equals seleted text
# * From above plot,we could gain agreat insight that **text less than 8 are almost equal to selected text(top left dense area)**.
# * **Text Words between 10 & 20 have low jaccard score.(bottom middle dense area around jacard score 0.1)**

# <font size="+1" color="chocolate"><b>5.4.2 Jaccard - Sentiment</b></font><br><a id="5.4.2"></a>

# In[ ]:


x = train_word.loc[train_word['sentiment']=="positive"]['text_tweet_length']
y = train_word.loc[train_word['sentiment']=="positive"]['jaccard_score']
fig = go.Figure()
fig.add_trace(go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'gray',
        reversescale = True,
        xaxis = 'x',
        yaxis = 'y'
    ))
fig.add_trace(go.Scatter(
        x = x,
        y = y,
        xaxis = 'x',
        yaxis = 'y',
        mode = 'markers',
        marker = dict(
            color = 'green',  #'rgba(0,0,0,0.3)',
            size = 3
        )
    ))
fig.add_trace(go.Histogram(
        y = y,
        xaxis = 'x2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ))
fig.add_trace(go.Histogram(
        x = x,
        yaxis = 'y2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ))

fig.update_layout(
    autosize = False,
    xaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    yaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    xaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    yaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    height = 600,
    width = 600,
    bargap = 0,
    hovermode = 'closest',
    showlegend = False,
    title_text="Postive Jaccard - Text vs Selected Text ",title_x=0.5
)

fig.show()


# In[ ]:


x = train_word.loc[train_word['sentiment']=="negative"]['text_tweet_length']
y = train_word.loc[train_word['sentiment']=="negative"]['jaccard_score']
fig = go.Figure()
fig.add_trace(go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'gray',
        reversescale = True,
        xaxis = 'x',
        yaxis = 'y'
    ))
fig.add_trace(go.Scatter(
        x = x,
        y = y,
        xaxis = 'x',
        yaxis = 'y',
        mode = 'markers',
        marker = dict(
            color = 'red',  #'rgba(0,0,0,0.3)',
            size = 3
        )
    ))
fig.add_trace(go.Histogram(
        y = y,
        xaxis = 'x2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ))
fig.add_trace(go.Histogram(
        x = x,
        yaxis = 'y2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ))

fig.update_layout(
    autosize = False,
    xaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    yaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    xaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    yaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    height = 600,
    width = 600,
    bargap = 0,
    hovermode = 'closest',
    showlegend = False,
    title_text="Negative Jaccard - Text vs Selected Text ",title_x=0.5
)

fig.show()


# In[ ]:


x = train_word.loc[train_word['sentiment']=="neutral"]['text_tweet_length']
y = train_word.loc[train_word['sentiment']=="neutral"]['jaccard_score']
fig = go.Figure()
fig.add_trace(go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'gray',
        reversescale = True,
        xaxis = 'x',
        yaxis = 'y'
    ))
fig.add_trace(go.Scatter(
        x = x,
        y = y,
        xaxis = 'x',
        yaxis = 'y',
        mode = 'markers',
        marker = dict(
            color = 'orange',  #'rgba(0,0,0,0.3)',
            size = 3
        )
    ))
fig.add_trace(go.Histogram(
        y = y,
        xaxis = 'x2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ))
fig.add_trace(go.Histogram(
        x = x,
        yaxis = 'y2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ))

fig.update_layout(
    autosize = False,
    xaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    yaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    xaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    yaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    height = 600,
    width = 600,
    bargap = 0,
    hovermode = 'closest',
    showlegend = False,
    title_text="Neutral Jaccard - Text vs Selected Text ",title_x=0.5
)

fig.show()


# <font size="+1" color="chocolate"><b>5.4.3 Violin - Jaccard Score</b></font><br><a id="5.4.3"></a>

# These plots can explain the distribution of jaccard score.

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Violin(y=train_word.loc[train_word['sentiment']=="positive"]['jaccard_score'], box_visible=False, line_color='black',
                               meanline_visible=True, fillcolor='green', opacity=0.6,name="Positive",
                               x0='Positive')
             )

fig.add_trace(go.Violin(y=train_word.loc[train_word['sentiment']=="negative"]['jaccard_score'], box_visible=False, line_color='black',
                               meanline_visible=True, fillcolor='red', opacity=0.6,name="Negative",
                               x0='Negative')
             )

fig.add_trace(go.Violin(y=train_word.loc[train_word['sentiment']=="neutral"]['jaccard_score'], box_visible=False, line_color='black',
                               meanline_visible=True, fillcolor='orange', opacity=0.6,name="Neutral",
                               x0='Neutral')
             )


fig.update_traces(box_visible=False, meanline_visible=True)
fig.update_layout(title_text="Violin - Jaccard score",title_x=0.5)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Positive and Negative tweets have almost similar sort of distribution.
# * Positive and Negative -text with less than 8 words have jaccard score near 1
# * Positive and Negative -Text around 10 - 20 are biased to low jaccard score
# * Despite the length of text most of them are similar to selected text.(Thereby jaccard score is high).Train data text are extracted as a whole and passed into selected text for most of tweets.

# <font size="+1" color="chocolate"><b>5.4.4 Difference - Text vs Selected Text</b></font><br><a id="5.4.4"></a>

# Difference variable would be difference between length of selected text and length of whole text.

# In[ ]:


train_word['difference']=abs(train_word['text_tweet_length']-train_word['target_tweet_length'])


# In[ ]:


fig = go.Figure()

fig.add_trace(go.Violin(y=train_word.loc[train_word['sentiment']=="positive"]['difference'], box_visible=False, line_color='black',
                               meanline_visible=True, fillcolor='green', opacity=0.6,name="Positive",
                               x0='Positive')
             )

fig.add_trace(go.Violin(y=train_word.loc[train_word['sentiment']=="negative"]['difference'], box_visible=False, line_color='black',
                               meanline_visible=True, fillcolor='red', opacity=0.6,name="Negative",
                               x0='Negative')
             )

fig.add_trace(go.Violin(y=train_word.loc[train_word['sentiment']=="neutral"]['difference'], box_visible=False, line_color='black',
                               meanline_visible=True, fillcolor='orange', opacity=0.6,name="Neutral",
                               x0='Neutral')
             )


fig.update_traces(box_visible=False, meanline_visible=True)
fig.update_layout(title_text="Violin - Difference",title_x=0.5)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * As we saw from jaccard distribution,the same result can be observed on this plot as well

# <font size="+1" color="chocolate"><b>5.4.5 Difference vs Jaccard</b></font><br><a id="5.4.5"></a>

# The difference between length of words and jaccard in same plot would tell us how it is baised.

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=train_word['difference'],
    y=train_word['jaccard_score'],
    marker=dict(color="blue", size=12),
    mode="markers"
))

fig.update_layout(title="Difference vs Jaccard",
                  xaxis_title="Difference",
                  yaxis_title="Jaccard Score",title_x=0.5)

fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * As difference decreases,jaccard score must go high.Then plot will look linear.
# * We have robust data in positive and negative tweets and so we could see our plot noisy.

# <font size="+3" color="violet"><b>Findings:</b></font><br><a id="Findings"></a>
# 
# * Sentiments are **splitted uniformly** between train and test set.But neutral has covered more data.
# * Neutral tweets Selected Text(target) words are spread across different length till **30 words**.But positive and negative are **almost less than 5 words**.
# * Sentiments distribution for positive,negative,neutral - **approx(31%,28%,49%)**
# * Most of tweets were around **Mothers Day** Celebration
# * Neutral have **most shared words between text and selected text**.This is true because unseperable words are pushed in neutral section.
# * Positive and Negative - Text with **less than 8 words have jaccard score near 1** whereas tweet length **around 10 - 20 are biased to low jaccard score**.
# * Neutral - Despite the length of text most of them are similar to selected text.(Thereby jaccard score is high).

# <font size="+3" color="Blue"><b>6. roBERTa</b></font><a id="6"></a>

# I would like to thank Chris Deotte for his wonderful [kernel](https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705]) .This gave me inspirational to understand roberta.Also i referenced [Kiram Al Karba kernel](https://www.kaggle.com/al0kharba/tensorflow-roberta-0-712) where he reduced the training time of the model.
# 
# I will try to explain codes as simple as possible.I tried working on most of above EDA outcomes on this model.But i could not find progress in CV.So I will stick with basic stuff for now done by Chris and Kiram.I will be updating this model once any of the EDA or prepocessing gets succeed with the score/CV.
# 
# **Note:** You can fork my kernel to get pretrained data.Else you can fetch them from below links <br> 
# https://www.kaggle.com/cdeotte/tf-roberta<br>
# https://www.kaggle.com/al0kharba/model4

# <font size="+2" color="indigo"><b>6.1 Basic Setup</b></font><br><a id="6.1"></a>

# In[ ]:


#Since we dont have length larger than 96
MAX_LEN = 96

# Pretrained model of roberta
PATH = '../input/tf-roberta/'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=PATH+'vocab-roberta-base.json', 
    merges_file=PATH+'merges-roberta-base.txt', 
    lowercase=True,
    add_prefix_space=True
)

# Sentiment ID value is encoded from tokenizer
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}


# <font size="+2" color="indigo"><b>6.2 Mask - Train</b></font><br><a id="6.2"></a>

# In[ ]:


train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')
ct=train.shape[0] #27481

# Initialising training inputs
input_ids=np.ones((ct,MAX_LEN),dtype="int32")          # Array with value 1 of shape(27481,96)
attention_mask=np.zeros((ct,MAX_LEN),dtype="int32")    # Array with value 0 of shape(27481,96)
token_type_ids=np.zeros((ct,MAX_LEN),dtype="int32")    # Array with value 0 of shape(27481,96)
start_tokens=np.zeros((ct,MAX_LEN),dtype="int32")      # Array with value 0 of shape(27481,96)
end_tokens=np.zeros((ct,MAX_LEN),dtype="int32")        # Array with value 0 of shape(27481,96)


# In below code ,please go through comments which i have mentioned between codes to identify variables progress line by line.I have added a sample row from train data for explanation.
# 
# > text1 = my boss is bullying me <br>
# > text2 = bullying me

# In[ ]:


for k in range(train.shape[0]):
#1 FIND OVERLAP
    text1 = " "+" ".join(train.loc[k,'text'].split())
    text2 = " ".join(train.loc[k,'selected_text'].split())
    
    # idx - position where the selected text are placed. 
    idx = text1.find(text2)   # we get [12] position
    
    # all character position as 0 and then places 1 for selected text position  
    chars = np.zeros((len(text1))) 
    chars[idx:idx+len(text2)]=1    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] 
    
    #tokenize id of text 
    if text1[idx-1]==' ': 
        chars[idx-1] = 1    
        enc = tokenizer.encode(text1)  #  [127, 3504, 16, 11902, 162]
        
#2. ID_OFFSETS - start and end index of text
    offsets = []
    idx=0
    for t in enc.ids:
        w = tokenizer.decode([t])
        offsets.append((idx,idx+len(w)))     #  [(0, 3), (3, 8), (8, 11), (11, 20), (20, 23)]
        idx += len(w) 
    
#3  START-END TOKENS
    toks = []
    for i,(a,b) in enumerate(offsets):
        sm = np.sum(chars[a:b]) # number of characters in selected text - [0.0,0.0,0.0,9.0,3.0] - bullying me
        if sm>0: 
            toks.append(i)  # token position - selected text - [3, 4]
        
    s_tok = sentiment_id[train.loc[k,'sentiment']] # Encoded values by tokenizer
    
    #Formating input for roberta model
    input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]   #[ 0   127  3504    16 11902   162     2     2  2430     2]
    attention_mask[k,:len(enc.ids)+5] = 1                                  # [1 1 1 1 1 1 1 1 1 1]
    
    if len(toks)>0:
        # this will produce (27481, 96) & (27481, 96) arrays where tokens are placed
        start_tokens[k,toks[0]+1] = 1
        end_tokens[k,toks[-1]+1] = 1 


# <font size="+2" color="indigo"><b>6.3 Mask - Test</b></font><br><a id="6.3"></a>

# In[ ]:


test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')

ct_test = test.shape[0]

# Initialize inputs
input_ids_t = np.ones((ct_test,MAX_LEN),dtype='int32')        # array with value 1 for shape (3534, 96)
attention_mask_t = np.zeros((ct_test,MAX_LEN),dtype='int32')  # array with value 0 for shape (3534, 96)
token_type_ids_t = np.zeros((ct_test,MAX_LEN),dtype='int32')  # array with value 0 for shape (3534, 96)

# Set Inputs attention 
for k in range(test.shape[0]):
        
#1. INPUT_IDS
    text1 = " "+" ".join(test.loc[k,'text'].split())
    enc = tokenizer.encode(text1)                
     
    # Encoded value of tokenizer
    s_tok = sentiment_id[test.loc[k,'sentiment']]
    
    #setting up of input ids - same as we did for train
    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask_t[k,:len(enc.ids)+5] = 1


# <font size="+2" color="indigo"><b>6.4 Model</b></font><br><a id="6.4"></a>

# In[ ]:



def scheduler(epoch):
    return 3e-5 * 0.2**epoch


# In[ ]:



def build_model():
    
    # Initialize keras layers
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    # Fetching pretrained models 
    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
    x = bert_model(ids,attention_mask=att,token_type_ids=tok)
    
    # Setting up layers
    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)
    x2 = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('relu')(x2)
    
#     x3 = tf.keras.layers.Dropout(0.1)(x[0]) 
#     x3 = tf.keras.layers.Conv1D(128, 2, padding='same')(x3)
#     x3 = tf.keras.layers.LeakyReLU()(x3)
#     x3 = tf.keras.layers.Conv1D(64, 2, padding='same')(x3)
#     x3 = tf.keras.layers.Dense(1)(x3)
#     x3 = tf.keras.layers.Flatten()(x3)
#     x3 = tf.keras.layers.Activation('softmax')(x3)


    # Initializing input,output for model.THis will be trained in next code
    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    
    #Adam optimizer for stochastic gradient descent. if you are unware of it - https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model
    


# <font size="+2" color="indigo"><b>6.5 Run Model</b></font><br><a id="6.5"></a>

# In[ ]:


start_time=dt.now()

n_splits=5 # Number of splits

# INitialize start and end token
preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))

DISPLAY=1
for i in range(5):
    print('#'*40)
    print('### MODEL %i'%(i+1))
    print('#'*40)
    
    K.clear_session()
    model = build_model()
    # Pretrained model
    model.load_weights('../input/model4/v4-roberta-%i.h5'%i)

    print('Predicting Test...')
    preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
    preds_start += preds[0]/n_splits
    preds_end += preds[1]/n_splits
    
end_time=dt.now()
print("   ")
print("   ")
print("Time Taken to run above code :",(end_time-start_time).total_seconds()/60," minutes")


# <font size="+2" color="indigo"><b>6.6 Submission</b></font><br><a id="6.6"></a>

# In[ ]:


all = []
for k in range(input_ids_t.shape[0]):
    # Argmax - Returns the indices of the maximum values along axis
    a = np.argmax(preds_start[k,])
    b = np.argmax(preds_end[k,])
    if a>b: 
        st = test.loc[k,'text']
    else:
        text1 = " "+" ".join(test.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc.ids[a-1:b])
    all.append(st)


# In[ ]:


test['selected_text'] = all
submission=test[['textID','selected_text']]
submission.to_csv('submission.csv',index=False)
submission.head(5)


# <font size="+3" color="violet"><b>What more to come</b></font><br>
# 
# * **More EDA** - As selected text is raw data,we may need to get more insights here for preprocessing.
# * **Data Modeling based on sentiments**-We will have different model clubbed together.
# * **Prediction** - We will try add more layers and try ensembling.

# <font size="+1" color="black"><b>This is not the end.There are lot more to come in next versions.So please stay tuned.</b></font><br><br>
# 
# <font size="+1" color="black"><b>I hope you find this kernel useful.You can fork it and explore.And please dont forget to appreciate me with an </b></font><font size="+2" color="red"><b>Upvote </b></font><font size="+1" color="black"><b>which motivates me to do more kernels.</b></font>

# <font size="+2" color="chocolate"><b>My other kernels</b></font><br>
# 
# 
# * Cheatsheet text Helper Functions  :  https://www.kaggle.com/raenish/cheatsheet-text-helper-functions
# * Time series on Air Quality( R )   :  https://www.kaggle.com/raenish/time-series-on-air-quality
# 
# 
# If these kernels impress you,give them an <font size="+2" color="red"><b>Upvote</b></font><br>

# <font size="+2" color="chocolate"><b>Reference</b></font><br>
# 
# * https://www.kaggle.com/parulpandey/eda-and-preprocessing-for-bert
# * https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model
# * https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705
# * https://www.kaggle.com/al0kharba/tensorflow-roberta-0-712

# **Version 17** :
# <font size="+3" color="red"><b><i>Loading...</i></b></font><br>
