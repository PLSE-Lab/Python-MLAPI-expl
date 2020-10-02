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


# In[ ]:


df = pd.read_csv('/kaggle/input/zomato-restaurants-hyderabad/Restaurant names and Metadata.csv')
df1 = pd.read_csv('/kaggle/input/zomato-restaurants-hyderabad/Restaurant reviews.csv')


# In[ ]:


df.shape, df1.shape


# In[ ]:


df.describe(include = 'all')


# In[ ]:


df1.isnull().sum()[df1.isnull().sum()>0]


# In[ ]:


df1 = df1.dropna()


# In[ ]:


df1['Rating'] =df1['Rating'].str.replace('Like', '5')


# In 'Rating' column there were some columns given as like. I have converted them as 5 ratings.

# In[ ]:


df1['Rating'] = df1['Rating'].astype(float)


# In[ ]:


df1['Time'] = pd.to_datetime(df1['Time'])
df1['Time'].min(),df1['Time'].max()


# The given Restaurant Reviews Data is for 3  Years.

# In[ ]:


df1['Metadata'] =df1['Metadata'].str.replace(' Review', ' Reviews')


# In 'Metadata' column we can see there are number of reviews and number of followers given. And for some rows there are just '1 Review' but no followers given. In order to seperate the column in two different columns for number of 'reviews' and 'followers' I have replaced the Review with Reviews.

# In[ ]:


df1['reviews'] = df1['Metadata'].str.replace('[^0-9,]','').str.split(',').str[0].astype(float)
df1['followers'] = df1['Metadata'].str.replace('[^0-9,]','').str.split(',').str[1].astype(float)


# In[ ]:


df1['reviews'] = df1['reviews'].astype(float)


# In[ ]:


df1['followers'].fillna('0', inplace = True)


# In[ ]:


df1['followers'] = df1['followers'].astype(float)


# In[ ]:


df1['Time'] = pd.to_datetime(df1['Time'])
df1['Day'] = df1['Time'].dt.day
df1['Month'] = df1['Time'].dt.month
df1['Year'] = df1['Time'].dt.year


# ### Top Restaurants having 60% and above with 5 ratings

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(15, 8))
res_rating_5 = df1.groupby(['Restaurant','Rating'])['Rating'].count()
top_res_having_5_ratings = res_rating_5.sort_values(ascending = False).head(11)
chart1 = top_res_having_5_ratings[::-1].plot.bar()
for p in chart1.patches:
    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Number_of_5_Ratings')
plt.xlabel('Restaurant_Name')


# ### Restaurants in which Reviewers have posted 20 and more pictures

# In[ ]:


plt.figure(figsize=(15, 4))
res_max_pics = df1.groupby('Restaurant')['Pictures'].max()
res_with_more_pics = res_max_pics.sort_values(ascending = False).head(21)
chart1 = res_with_more_pics[::-1].plot.bar()
for p in chart1.patches:
    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Pictures_Count')
plt.xlabel('Restaurant_Name')


# From above observations we can conclude that people have attached more photos in their reviews for the restaurants where they didn't liked the food.

# ### Reviewers_names who have written review for more than 10 Restaurants

# In[ ]:


plt.figure(figsize=(20, 8))
chart1 = sns.countplot(x = 'Reviewer', data=df1,
              order=df1.Reviewer.value_counts().iloc[:10].index)
for p in chart1.patches:
    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')


# There are 1341 People who have written reviews for more than 1 Restaurant. Out of which 150 People have written reviews for 5 or more Restaurants.

# ### Top 10 Restaurants based on average Ratings.

# In[ ]:


plt.figure(figsize=(15, 4))
res_avg_rating = df1.groupby('Restaurant')['Rating'].mean()
top10_res = res_avg_rating.sort_values(ascending = False).head(10)
chart1 = top10_res[::-1].plot.bar()
for p in chart1.patches:
    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.xlabel('Restaurant_Name')
plt.ylabel('Avg_Rating')


# ### Reviewers who have posted 30 and more Pictures

# In[ ]:


plt.figure(figsize=(25, 6))
reviewers_total_pics_posted = df1.groupby('Reviewer')['Pictures'].sum()
reviewers_with_30_or_more_pics = reviewers_total_pics_posted.sort_values(ascending = False).head(33)
chart1 = reviewers_with_30_or_more_pics[::-1].plot.bar()
for p in chart1.patches:
    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Total Number of Pictures Posted by Reviewer')
plt.xlabel('Reviewer_Name')


# ### Top 10 Reviewers whose Reviews are most seen by other people

# In[ ]:


plt.figure(figsize=(15, 6))
total_reviews_of_reviewers = df1.groupby('Reviewer')['reviews'].sum()
top10_reviewers = total_reviews_of_reviewers.sort_values(ascending = False).head(10)
chart1 = top10_reviewers[::-1].plot.bar()
for p in chart1.patches:
    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.xlabel('Reviewers_Name')
plt.ylabel('Total_Views_on-the_Reviews_by_Reviewers')


# ### Top 10 Reviewers having most followers

# In[ ]:


plt.figure(figsize=(15, 6))
total_followers_of_reviewers = df1.groupby('Reviewer')['followers'].sum()
top10_reviewers = total_followers_of_reviewers.sort_values(ascending = False).head(10)
chart1 = top10_reviewers[::-1].plot.bar()
for p in chart1.patches:
    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.xlabel('Reviewers_Name')
plt.ylabel('Total_Followers_of_the_Reviewers')


# In[ ]:


plt.figure(figsize=(15, 4))
res_avg_rating = df1.groupby(['Restaurant', 'Year'])['Rating'].mean()
top10_res = res_avg_rating.sort_values(ascending = False).head(10)
chart1 = top10_res[::-1].plot.bar()
for p in chart1.patches:
    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 
                                                   p.get_height()), ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Rating')
plt.xlabel('Restaurant_Name')


# ### Top 10 Restaurants in 2019 based on average ratings
# - AB's - Absolute Barbecues
# - B-Dubs
# - 3B's - Buddies, Bar & Barbecue
# - Paradise
# - Flechazo
# - Cascade - Radisson Hyderabad Hitec City
# - The Indi Grill
# - Karachi Bakery
# - Zega - Sheraton Hyderabad Hotel
# - Over The Moon Brew Company

# ### Top 10 Restaurants in 2018 based on average ratings
# - Feast - Sheraton Hyderabad Hotel
# - Zega - Sheraton Hyderabad Hotel
# - Mazzo - Marriott Executive Apartments
# - Hyderabadi Daawat
# - Cascade - Radisson Hyderabad Hitec City
# - NorFest - The Dhaba
# - Udipi's Upahar
# - American Wild Wings
# - Amul
# - Barbeque Nation

# ### Only 8 Restaurants have got ratings in the year 2017
# - Labonel
# - Chinese Pavilion
# - Cascade - Radisson Hyderabad Hitec City
# - Collage - Hyatt Hyderabad Gachibowli
# - Al Saba Restaurant
# - T Grill
# - Dunkin' Donuts
# - KS Bakers

# ### And, only 2 Restaurants have got rating in the year 2016
# - Labonel
# - Chinese Pavilion

# ### As, we can see for some Restaurant the ratings went too down in 2019 as compared to 2018.
# - Hitech Bawarchi Food Zone---3.33---1.84       
# - Royal Spicy Restaurant---3.90---2.69       
# - Owm Nom Nom---3.65---2.09       
# - Hyderabadi Daawat---4.30---3.28       
# - Kritunga Restaurant---3.66---2.40       
# - Domino's Pizza---3.32---1.54       
# - Triptify---3.83---2.28       
# - Shree Santosh Dhaba Family Restaurant---3.10---1.75       
# - Mathura Vilas---3.24---2.41       
# - Mohammedia Shawarma---3.21---1.73       
# - La La Land - Bar & Kitchen---3.56---2.85       
# - SKYHY---3.75---2.91       
# - Green Bawarchi Restaurant---3.63---2.75       
# - Hotel Zara Hi-Fi---2.77---1.93
# - Collage - Hyatt Hyderabad Gachibowli---3.64---2.62
# - Al Saba Restaurant---3.25---2.50
# - Dunkin' Donuts---3.14---2.29
# - KS Bakers---3.75---3.27

# For Restaurants KS Bakers, Dunkin' Donuts , T Grill, Al Saba Restaurant, Collage - Hyatt Hyderabad Gachibowli Ratiings have been gradually decreasing over the year.
# 
# For Restaurants Cascade - Radisson Hyderabad Hitec City, Barbeque Nation, The Foodie Monster Kitchen, Dine O China, Pista House, Delhi-39, Sardarji's Chaats & More, Karachi Bakery Ratings have been increasing over the year.

# In[ ]:


plt.figure(figsize=(15, 4))
df1.resample('1D',on='Time')['Restaurant'].size().plot.line() ## Instead of '1Y' we can use '1d' or '1m' or '1H'
plt.xlabel('Date')
plt.ylabel('No. of Reviews')
plt.show()


# In[ ]:


from wordcloud import WordCloud

plt.figure(figsize=(15, 4))
ip_string = ' '.join(df1['Review'].dropna().to_list())

wc = WordCloud(background_color='white').generate(ip_string.lower())
plt.imshow(wc)


# In[ ]:


# Word_Count
df1['Word_Count'] = df1['Review'].apply(lambda x: len(str(x).split()))


# In[ ]:


# Character_Count
df1['Char_Count'] = df1['Review'].apply(lambda x: len(x))


# In[ ]:


# Count hashtags(#) and @ mentions

df1['hashtags_count'] = df1['Review'].apply(lambda x: len([t for t in x.split() if t.startswith('#')]))
df1['mention_count'] = df1['Review'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))


# In[ ]:


# If numeric digits are present in tweets

df1['numerics_count'] = df1['Review'].apply(lambda x: len([t for t in x.split() if t.isdigit()]))


# In[ ]:


# UPPER_case_words_count#

df1['UPPER_CASE_COUNT'] = df1['Review'].apply(lambda x: len([t for t in  x.split()
                                                             if t.isupper() and len(x)>3]))


# In[ ]:


import re


# In[ ]:


# Count and Removing Emails
df1['Emails'] = df1['Review'].apply(lambda x: re.findall(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',x))
df1['Review'] = df1['Review'].apply(lambda x: re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', '',x))


# In[ ]:


# Count and Remove URL's
df1['URL_Flags'] = df1['Review'].apply(lambda x: len(re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))
df1['Review'] = df1['Review'].apply(lambda x: re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x))


# In[ ]:


# Removing RE_REVIEWS
df1['Review'] = df1['Review'].apply(lambda x: re.sub('RT', '', x))


# In[ ]:


# Punctuation_Count
df1['punct_count'] = df1['Review'].apply(lambda x: len(re.findall('[^a-z A-Z 0-9-]+', x)))


# In[ ]:


# Removal of special chars and punctuation
df1['Review'] = df1['Review'].apply(lambda x: re.sub('[^a-z A-Z 0-9-]+', '', x))


# In[ ]:


# Removing_Multiple_Spaces
df1['Review'] = df1['Review'].apply(lambda x: ' '.join(x.split()))


# In[ ]:


# Preprocessing and cleaning

contractions = {
"aight": "alright",
"ain't": "am not",
"amn't": "am not",
"aren't": "are not",
"can't": "can not",
"cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"daren't": "dare not",
"daresn't": "dare not",
"dasn't": "dare not",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"d'ye": "do you",
"e'er": "ever",
"everybody's": "everybody is",
"everyone's": "everyone is",
"finna": "fixing to",
"g'day": "good day",
"gimme": "give me",
"giv'n": "given",
"gonna": "going to",
"gon't": "go not",
"gotta": "got to",
"hadn't": "had not",
"had've": "had have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'dn't've'd": "he would not have had",
"he'll": "he will",
"he's": "he is",
"he've": "he have",
"how'd": "how did",
"howdy": "how do you do",
"how'll": "how will",
"how're": "how are",
"I'll": "I will",
"I'm": "I am",
"I'm'a": "I am about to",
"I'm'o": "I am going to",
"innit": "is it not",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"may've": "may have",
"methinks": "me thinks",
"mightn't": "might not",
"might've": "might have",
"mustn't": "must not",
"mustn't've": "must not have",
"must've": "must have",
"needn't": "need not",
"ne'er": "never",
"o'clock": "of the clock",
"o'er": "over",
"ol'": "old",
"oughtn't": "ought not",
"'s": "is, has, does, or us",
"shalln't": "shall not",
"shan't": "shall not",
"she'd": "she had",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"somebody's": "somebody is",
"someone's": "someone is",
"something's": "something is",
"so're": "so are",
"that'll": "that will",
"that're": "that are",
"that's": "that is",
"that'd": "that would",
"there'd": "there had",
"there'll": "there will",
"there're": "there are",
"there's": "there is",
"these're": "these are",
"these've": "these have",
"they'd": "they had",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"this's": "this is",
"those're": "those are",
"those've": "those have",
"'tis": "it is",
"to've": "to have",
"'twas": "it was",
"wanna": "want to",
"wasn't": "was not",
"we'd": "we had",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'd": "what did",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"where'd": "where did",
"where'll": "where will",
"where're": "where are",
"where's": "where is",
"where's": "where does",
"where've": "where have",
"which'd": "which would",
"which'll": "which will",
"which're": "which are",
"which's": "which is",
"which've": "which have",
"who'd": "who would",
"who'd've": "who would have",
"who'll": "who will",
"who're": "who are",
"who's": "who does",
"who've": "who have",
"why'd": "why did",
"why're": "why are",
"why's": "why does",
"won't": "will not",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd've": "you all would have",
"y'all'dn't've'd": "you all would not have had",
"y'all're": "you all are",
"you'd": "you would",
"you'll": "you will",
"you're": "you are",
"you've": "you have",
" u ": "you",
" ur ": "your",
" n ": "and"
}


# In[ ]:


def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key,value)
        return x
    else:
        return x


# In[ ]:


df1['Review'] = df1['Review'].apply(lambda x: cont_to_exp(x))


# ### Finding the Polarity and Subjectivity of Reviews

# In[ ]:


from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

df1['polarity'] = df1['Review'].apply(pol)
df1['subjectivity'] = df1['Review'].apply(sub)
df1.head()


# In[ ]:


df1['Sentiments'] = df1['polarity'].apply(lambda v: 'Positive' if v>0.000000 else ('Negative' if v<0.000000 else 'Neutral'))


# Out of all Reviews there are 7457 Positive reviews, 1994 Negative Reviews and 504 Neutral Reviews.

# In[ ]:


import spacy
nlp = spacy.load('en_core_web_sm')
import string


# In[ ]:


from spacy.lang.en.stop_words import STOP_WORDS


# In[ ]:


stopwords = list(STOP_WORDS)


# In[ ]:


punct = string.punctuation


# In[ ]:


def text_data_cleaning(sentence):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        if token.lemma_ != '-PRON-':
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)
    
    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in punct:
            cleaned_tokens.append(token)
    return cleaned_tokens


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


tfidf = TfidfVectorizer(tokenizer = text_data_cleaning)
classifier = LinearSVC()


# In[ ]:


X = df1['Review']
y = df1['Sentiments']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


X_train.shape, X_test.shape


# In[ ]:


clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


confusion_matrix(y_test,y_pred)

