#!/usr/bin/env python
# coding: utf-8

# # Part II of my EDA series.
# 
# 
# This notebook dives a little deeper into the unexplored part of the employee reviews dataset.
# The following columns of the dataset will be analysed extensively in this notebook (Moslty Textual data) - 
# 1. summary - Provides a brief about the review by summarising it 
# 2. pros - Pros about working with the company
# 3. cons - Cons about working with the company
# 4. advice-to-mgmt - Advice to management that the reviewer felt important

# In[25]:


# imports
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
from textblob import TextBlob
import collections
from nltk import ngrams
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import os
reviews = pd.read_csv("../input/employee_reviews.csv")


# Lets see now if our dataset has null values-

# In[26]:


reviews = reviews.drop("Unnamed: 0",axis=1)

def print_null(df):
    if df.isnull().sum().any() != 0:
        print("Null Values are present!")
        print(reviews.isnull().sum())
    else:
        print("Null values not present")
print_null(reviews)


# Null values are present in summary which i think is better to remove but a better option would be to replace null with ''none'' since many people had no advices for the management and wrote ''none'' elsewhere.

# In[27]:


# Filling null values in advice_to_mgmt with value none
reviews[['advice-to-mgmt']] = reviews[['advice-to-mgmt']].fillna(value='none')
reviews = reviews.dropna()
print_null(reviews)


# In[28]:


reviews.columns = reviews.columns.str.replace('-','_')
reviews = reviews.drop(["location","dates","job_title","work_balance_stars","culture_values_stars",
                       "carrer_opportunities_stars","comp_benefit_stars","senior_mangemnet_stars",
                       "helpful_count","link"],axis=1)
print("Current shape of the dataset is :",reviews.shape)


# In[29]:


reviews[['summary']] = reviews[['summary']].astype(str)
reviews['polarity'] = reviews.apply(lambda x: TextBlob(x['summary']).sentiment.polarity, axis=1)

 
def sentiment(df):
    if df['polarity'] > 0:
        val = "Positive"
    elif df['polarity'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val

reviews['sentiment_type'] = reviews.apply(sentiment, axis=1)

cmp_lc = reviews[["company","sentiment_type"]] 
cmp_lc = cmp_lc.groupby(["company", "sentiment_type"]).size().reset_index()
cmp_lc = cmp_lc.rename(columns={0: 'total_sentiment_types'})

#For plotting it
years = list(cmp_lc.company.unique())
company_data = []
v = True

for i in years:
    if i!='amazon':
        v=False
    data_upd = [dict(type='bar',
                     visible = v,
                     x = cmp_lc[cmp_lc['company']==i]['sentiment_type'],
                     y = cmp_lc[cmp_lc['company']==i]['total_sentiment_types'],
                     textposition = 'auto',
                     marker=dict(
                     color='orange',
                     line=dict(
                         width=1.5),
                     ),
                 opacity=0.6)]
    
    company_data.extend(data_upd)

years = [x.capitalize() for x in years]

# set menus inside the plot
steps = []
yr = 0
for i in range(0,len(company_data)):
    step = dict(method = "restyle",
                args = ["visible", [False]*len(company_data)],
                label = years[yr]) 
    step['args'][1][i] = True
    steps.append(step)
    yr += 1
    

sliders = [dict(active = 6,
                currentvalue = {"prefix": "Company: "},
                pad = {"t": 50},
                steps = steps)]

# Set the layout
layout = dict(title = 'Sentiment analysis for summary text data for each company',
              sliders = sliders)
fig = dict(data=company_data, layout=layout)
iplot(fig)


# In[30]:


#Comparing it with overall rating where 3 = neutral, below 3 is negative and above 3 is positive for amazon
pd.options.mode.chained_assignment = None  
cmp_lc = reviews[["company","sentiment_type","overall_ratings"]] 
cmp_lc  = cmp_lc.loc[cmp_lc["company"]=="amazon"]

def sentiment_overall(df):
    if df['overall_ratings'] >= 4:
        val = "Positive"
    elif df['overall_ratings'] == 3:
        val = "Neutral"
    else:
        val = "Negative"
    return val

cmp_lc['sentiment_type_from_overall_rating'] = cmp_lc.apply(sentiment_overall, axis=1)
cmp_lc1 = cmp_lc.groupby(["company", "sentiment_type"]).size().reset_index()
cmp_lc1 = cmp_lc1.rename(columns={0: 'total_sentiment_types'})
cmp_lc2 = cmp_lc.groupby(["company", "sentiment_type_from_overall_rating"]).size().reset_index()
cmp_lc2 = cmp_lc2.rename(columns={0: 'total_sentiment_type_from_overall_rating'})

trace0 = go.Bar(
    x=cmp_lc1.sentiment_type,
    y=cmp_lc1.total_sentiment_types,
    name='Sentiment analysed from Summary text',
    marker=dict(
        color='rgb(49,130,189)'
    )
)
trace1 = go.Bar(
    x=cmp_lc2.sentiment_type_from_overall_rating,
    y=cmp_lc2.total_sentiment_type_from_overall_rating,
    name='Sentiment from Overall ratings',
    marker=dict(
        color='rgb(204,204,204)',
    )
)

data = [trace0, trace1]
layout = go.Layout(title = 'Comparasion of Sentiment from text and the rating given by user',
    xaxis=dict(tickangle=-45),
    barmode='group',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[31]:


# from wordcloud import WordCloud,STOPWORDS
# stopwords = set(STOPWORDS)
# extras = ["great","work","company","amazon","good","employee"]
# stopwords.update(extras)
# temp = reviews.loc[reviews["company"]=="amazon"]
# text = " ".join(str(review) for review in temp.pros)
# wordcloud = WordCloud(stopwords=stopwords,collocations = False,width=1600, height=800, max_font_size=200).generate(text)
# plt.figure(figsize=(12,10))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title("What pros are the people talking about when it comes to Amazon?")
# plt.show()


# Summary by itself is not enough and can be biased.
# Hence i will create a new column which will contain text from all the other columns combined and will identify sentiment_type on that.

# In[32]:


#Combining all the text into a single column called Final
reviews['final_summary'] = reviews['summary'] + ' ' + reviews['pros'] + ' ' + reviews['cons'] + ' ' + reviews['advice_to_mgmt']
reviews = reviews.drop("overall_ratings",axis=1) #As we don't need it
# reviews[['summary']] = reviews[['summary']].astype(str)
reviews['polarity_final'] = reviews.apply(lambda x: TextBlob(x['final_summary']).sentiment.polarity, axis=1)


# In[33]:


def sentiment(df):
    if df['polarity_final'] > 0:
        val = "Positive"
    elif df['polarity_final'] == 0: 
        val = "Neutral"
    else:
        val = "Negative"
    return val

reviews['final_sentiment_type'] = reviews.apply(sentiment, axis=1)
reviews.tail(5)


# In[34]:


cmp_lc = reviews[["company","sentiment_type","final_sentiment_type"]] 
cmp_lc  = cmp_lc.loc[cmp_lc["company"]=="amazon"]
cmp_lc1 = cmp_lc.groupby(["company", "sentiment_type"]).size().reset_index()
cmp_lc1 = cmp_lc1.rename(columns={0: 'total_sentiment_types'})
cmp_lc2 = cmp_lc.groupby(["company", "final_sentiment_type"]).size().reset_index()
cmp_lc2 = cmp_lc2.rename(columns={0: 'total_final_sentiment_types'})

trace0 = go.Bar(
    x=cmp_lc1.sentiment_type,
    y=cmp_lc1.total_sentiment_types,
    name='Sentiment types from Summary',
    marker=dict(
        color='rgb(49,130,189)'
    )
)
trace1 = go.Bar(
    x=cmp_lc2.final_sentiment_type,
    y=cmp_lc2.total_final_sentiment_types,
    name='Sentiment types from taking all the text',
    marker=dict(
        color='rgb(204,204,204)',
    )
)

data = [trace0, trace1]
layout = go.Layout(title = 'Comparasion of Sentiments from Summary and the after taking all the text entered together!',
    xaxis=dict(tickangle=-45),
    barmode='group',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='angled-text-bar')


# What the above graph means to communicate is that when you take into account just the summaries, people had more neutral opinion about Amazon, but when you take into consideration the Pros, Cons as well as advice to management, people had more positive sentiments about it. Interesting!
# Great going Amazon!

# Now next steps - 
# Preprocessing textual data

# Normalising the summary

# In[35]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
extras = ["google","amazon","netflix","microsoft","apple","facebook"]
stop_words.update(extras)
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 
    tokens = nltk.word_tokenize(only_letters)[2:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
reviews['normalized_summary'] = reviews.final_summary.apply(normalizer)
amazon_reviews  = reviews.loc[reviews["company"]=="amazon"]
# amazon_reviews = amazon_reviews.drop(["company","summary","pros","cons","advice_to_mgmt","polarity","sentiment_type"],axis=1)
amazon_reviews[['final_summary','normalized_summary']].head(2)


# Applying n-grams to the summary now

# In[36]:


def ngrams(input_list):
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams+trigrams


amazon_reviews['grams'] = amazon_reviews.normalized_summary.apply(ngrams)
amazon_reviews = amazon_reviews.reset_index()
amazon_reviews[['grams']].head(2)


# Now let's see what people are saying about Amazon when it comes to positive reviews by counting the most commontly occuring phrases

# In[37]:


def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt

amazon_reviews[(amazon_reviews.final_sentiment_type == 'Positive')][['grams']].apply(count_words)['grams'].most_common(20)

