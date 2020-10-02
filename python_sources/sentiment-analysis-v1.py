#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis v1
# Creating a Sentiment Analysis using TextBlob and Vader libraries
# 
# 

# Let us import libraries needed

# In[ ]:


# ! pip install textblob
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
import numpy as np 
import pandas as pd 


# # Using TextBlob Library

# In[ ]:


feedback = "So I've been going to chili's since it opened but today the food was so bad , it had zero taste , the mushroom sauce chicken didn't have any mushrooms or sauce , the baked potatoes were without any seasoning and the beans were soggy , the burger's bun was weird tasting so was the patty and the onion rings in it . One star is for the chips and salsa , thankfully it's still the same"
testimonial = TextBlob(feedback)
testimonial.sentiment


# Polarity negative means negative sentiment. It ranges from -1 to 1
# Subjectivity tells how much subjective is the text. 0 is objective, 1 subjective. Ranges from 0 to 1

# # Using Vader Library

# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
def get_vader_score(sent):
    # Polarity score returns dictionary
    ss = sid.polarity_scores(sent)
    return ss
get_vader_score(feedback)        


# # Data Load
# The data is from UCI website. https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences#
# 
# This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015
# 
# It contains sentences labelled with positive or negative sentiment.
# 
# 
# ** Format: ** 
# 
# 
# 
# |sentence |  score |
# |-- | --|
# |Feedback Text | 0 or 1 |
# 
# 
# ** Details: **
# 
# Score is either 1 (for positive) or 0 (for negative)
# The sentences come from three different websites/fields:
# 
# - imdb.com
# - amazon.com
# - yelp.com
# 
# For each website, there exist 500 positive and 500 negative sentences. Those were selected randomly for larger datasets of reviews.
# We attempted to select sentences that have a clearly positive or negative connotaton, the goal was for no neutral sentences to be selected.
# 
# 
# 
# 

# In[ ]:


# !ls ../input/sentimentanalysis/amazon_cells_labelled.txt
df = pd.read_csv('../input/sentimentanalysis/amazon_cells_labelled.txt',sep='\t', header=None, names=['Text', 'Sentiment'])
df.head()


# # Data processing
# Use TexBlob and Vader library to get the attributes like Polarity, Subjectivity, Positive, Negative etc to create dataframe which can then be used to model the sentiment engine

# In[ ]:


def get_tb_sentiment(text, param):
#     print(text,param)
    textblob = TextBlob(text)
    if param =='polarity':
        return textblob.sentiment.polarity
    elif param == 'subjectivity':
        return textblob.sentiment.subjectivity
    else:
        return None

def get_vader_sentiment(text, param):
    vader = get_vader_score(text)
    if param =='pos':
        return vader['pos']
    elif param == 'neg':
        return vader['neg']
    elif param == 'neu':
        return vader['neu']
    elif param == 'compound':
        return vader['compound']
    return None


df['tb_polarity'] = df['Text'].apply(get_tb_sentiment, args=('polarity',))
df['tb_subjectivity'] = df['Text'].apply(get_tb_sentiment, args=('subjectivity',))
df['vader_pos'] = df['Text'].apply(get_vader_sentiment, args=('pos',))
df['vader_neg'] = df['Text'].apply(get_vader_sentiment, args=('neg',))
df['vader_neu'] = df['Text'].apply(get_vader_sentiment, args=('neu',))
df['vader_compound'] = df['Text'].apply(get_vader_sentiment, args=('compound',))

df.head()


# # Visualization

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


fig = plt.figure(1, figsize=(20,20))

row = 2
col = 2

plt.subplot(row, col, 1)
sns.scatterplot(x='vader_compound', y='tb_polarity',data=df, hue='Sentiment')
plt.subplot(row, col, 2)
sns.scatterplot(x='vader_pos', y='tb_polarity',data=df, hue='Sentiment')
plt.subplot(row, col, 3)
sns.scatterplot(x='vader_neg', y='tb_polarity',data=df, hue='Sentiment')
plt.subplot(row, col, 4)
sns.scatterplot(x='vader_compound', y='tb_subjectivity',data=df, hue='Sentiment')


# # Merge TextBlob and Vader
# Build a classifer to use both TextBlob and Vader attributes to find the correct polarity

# In[ ]:


from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


X = df[['tb_polarity', 'tb_subjectivity', 'vader_pos', 'vader_neg', 'vader_neu', 'vader_compound']]
y = df['Sentiment']


# ## Decision Tree Classifier

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
clf = tree.DecisionTreeClassifier(min_samples_split=200)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
score = accuracy_score(y_test, predictions)
print('Model Classification Score:' , score)


# # SVC Classifier

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
score = accuracy_score(y_test, predictions)
print('Model Classification Score:' , score)


# # Predicting sentiment using above mode
# Let us try to predict the sentiment polarity with above mode

# In[ ]:


feedback_test = pd.read_csv("../input/feedback-test1csv/feedback_test.csv")
len(feedback_test)


# In[ ]:


feedback_test['tb_polarity'] = feedback_test['Feedback'].apply(get_tb_sentiment, args=('polarity',))
feedback_test['tb_subjectivity'] = feedback_test['Feedback'].apply(get_tb_sentiment, args=('subjectivity',))
feedback_test['vader_pos'] = feedback_test['Feedback'].apply(get_vader_sentiment, args=('pos',))
feedback_test['vader_neg'] = feedback_test['Feedback'].apply(get_vader_sentiment, args=('neg',))
feedback_test['vader_neu'] = feedback_test['Feedback'].apply(get_vader_sentiment, args=('neu',))
feedback_test['vader_compound'] = feedback_test['Feedback'].apply(get_vader_sentiment, args=('compound',))

feedback_test.tail()


# In[ ]:


feedback_test1 = feedback_test[['tb_polarity', 'tb_subjectivity', 'vader_pos', 'vader_neg', 'vader_neu', 'vader_compound']]
predictions = clf.predict(feedback_test1)
predictions


# In[ ]:




