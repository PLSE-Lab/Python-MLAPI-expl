#!/usr/bin/env python
# coding: utf-8

# # Climate Change Belief Analysis

# <img src="https://images.pexels.com/photos/2990612/pexels-photo-2990612.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260" width="1000" height="400">
# <span>Photo by <a href="https://unsplash.com/@markusspiske?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Markus Spiske</a> on <a href="/?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>
# 
# 
# # Contents
# 
# ### 1. [Introduction](#Introduction)
# ### 2. [Importing Libraries](#Importing-libraries)
# ### 3. [Importing Datasets](#Importing-datasets)
# ### 4. [Data Description](#Data-Description)
# ### 4. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
# ### 5. [Preprocessing](#Preprocessing)
# ### 6. [Models](#Models)
# - [Random Forest](#Random-Forest)
# - [Support Vector Classifier](#Support-Vector-Classifier)
# - [Logistic Regression](#Logistic-Regression)
# - [K-Nearest Neighbours](#K-Nearest-Neighbours)
# - [XGBoost](#XGBoost)
# 
# ### 7. [Model Selection](#Model-Selection)

# # Introduction

# Identifying someone's viewpoint on climate change may potentially reveal a lot about a person, such as their personal values, their political inclination and their web behaviour. This information may prove useful for various business contexts, whereby it may provide businesses with more context into their target market. 
# 
# This notebook aims to use classification machine learning algorithms to determine whether or not a person believes in climate change based on a single tweet.

# # Importing libraries

# In[ ]:


#!pip install comet_ml


# ## Setting up Comet

# Link to comet.ml page can be found [here](https://www.comet.ml/fossilgenera/classification-climate-change/).

# In[ ]:


#from comet_ml import Experiment


# In[ ]:


# experiment = Experiment(api_key="MKFU0VUoOX8wJf4c0erGJ1YBY",
#                         project_name="classification-climate-change", workspace="fossilgenera")


# External packages to download:
# 
# - nltk - `pip install nltk==3.2.5`
# - spacy - `pip install spacy==2.2.4`
# - imblearn - `pip install imblearn==0.4.3`
# - XGBoost - `pip install xgboost==0.90`

# In[ ]:


# Common Python data science packages
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# NLP packages
import re
import nltk
import spacy
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk import ngrams
import collections
from wordcloud import WordCloud, STOPWORDS

# Pipeline
from imblearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Resampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Training
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack

# Machine Learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier

import pickle

# Metrics
from sklearn.metrics import confusion_matrix, f1_score, classification_report

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Importing datasets

# In[ ]:


train = pd.read_csv('/kaggle/input/climate-change-belief-analysis/train.csv')
test = pd.read_csv('/kaggle/input/climate-change-belief-analysis/test.csv')


# In[ ]:


train.head()


# # Data Description

# The dataset contains tweets pertaining to climate change. It contains three columns:
# - *Sentiment* - Sentiment of the tweet, indicating the tweet's stance on climate change
# - *Message* - The tweet itself
# - *Tweetid* - Unique tweet ID
# 
# Each tweet is labelled as one of the following classes:
# - (2) The tweet links to a factual news website
# - (1) The tweet supports the belief of man-made climate change
# - (0) The tweet neither supports or refutes the belief of man-made climate change
# - (-1) The tweet does not believe in man-made climate change

# In[ ]:


train.info()


# # Exploratory Data Analysis

# [Back to contents](#Contents)

# ## Count of each sentiment

# In[ ]:


sentiment_count = plt.figure(figsize=(10, 6))
sentiment_count = sns.countplot(x='sentiment', data=train, palette='rainbow')
sentiment_count = plt.xlabel('Sentiment')
sentiment_count = plt.ylabel('Count')


# In[ ]:


train['sentiment'].value_counts()


# Most of the tweets in the training set have been written by people who believe in man-made climate change, followed by those who provide links to factual news about climate change. This shows that there are *imbalanced classes* in the training data. Imbalanced classes can lead to the machine learning model making good predictions on the larger class but poorer predictions on the smaller classes. This will be fixed through resampling techniques when building the models.

# ## Additional text features

# The may be value in observing additional features about about each tweet. We will look at word count, length of tweet, and average word length in order to extract additional meaning from the tweets.

# In[ ]:


# Word count function
def word_count(tweet):
    """
    Returns the number of words in a tweet.
  
    Parameters:
            A pandas series (str)
    Returns:
            An length of the tweet string (int).
    """
    return len(tweet.split())


# Length of tweet function
def length_of_tweet(tweet):
    """
    Returns the number of characters in each tweet.
    
    parameters: 
            A pandas series (str)
    Returns:
            The number of characters in each tweet (int).
    """
    return len(tweet)


# Average word length function
def average_word_length(tweet):
    """
    Returns the avarage length of words withing each tweet.
    
    parameters: 
            A pandas series(str)
    Returns:
            The average length of words within each tweet (float).
    """
    words = tweet.split()
    average = sum(len(word) for word in words) / len(words)
    
    return round(average, 2)


# In[ ]:


# Creating additional features
train['word_count'] = train['message'].apply(word_count)
train['tweet_length'] = train['message'].apply(length_of_tweet)
train['avg_word_length'] = train['message'].apply(average_word_length)


# In[ ]:


train.head()


# ### Word count for each category

# In[ ]:


word_count_dist = plt.figure(figsize=(10, 6))
word_count_dist = sns.boxplot(x=train['sentiment'], y=train['word_count'],
                              palette='rainbow')
word_count_dist = plt.xlabel('Sentiment')
word_count_dist = plt.ylabel('Word Count')


# In[ ]:


# Saving averages
denier_word_count_avg = round(train[train['sentiment'] == -1]['word_count'].mean(), 2)
neutral_word_count_avg = round(train[train['sentiment'] == 0]['word_count'].mean(), 2)
believer_word_count_avg = round(train[train['sentiment'] == 1]['word_count'].mean(), 2)
factual_word_count_avg = round(train[train['sentiment'] == 2]['word_count'].mean(), 2)


# ### Tweet length for each category

# In[ ]:


tweet_length_dist = plt.figure(figsize=(10, 6))
tweet_length_dist = sns.boxplot(x=train['sentiment'], y=train['tweet_length'],
                                palette='rainbow')
tweet_length_dist = plt.xlabel('Sentiment')
tweet_length_dist = plt.ylabel('Tweet Length')


# In[ ]:


# Saving averages
denier_tweet_length_avg = round(train[train['sentiment'] == -1]['tweet_length'].mean(), 2)
neutral_tweet_length_avg = round(train[train['sentiment'] == 0]['tweet_length'].mean(), 2)
believer_tweet_length_avg = round(train[train['sentiment'] == 1]['tweet_length'].mean(), 2)
factual_tweet_length_avg = round(train[train['sentiment'] == 2]['tweet_length'].mean(), 2)


# ### Average word length

# In[ ]:


word_length_dist = plt.figure(figsize=(10, 6))
word_length_dist = sns.boxplot(x=train['sentiment'], y=train['avg_word_length'],
                                palette='rainbow')
word_length_dist = plt.xlabel('Sentiment')
word_length_dist = plt.ylabel('Average Word Length')


# The Neutral group has an outlier which makes visual comparison using a boxplot difficult- we will use a distribution plot to compare the categories instead.

# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# Deniers
sns.distplot(train[train['sentiment'] == -1]['avg_word_length'], ax=axes[0, 0])
axes[0, 0].set_title('Deniers')
axes[0, 0].set_xlabel('Average word length')

# Neutrals
sns.distplot(train[train['sentiment'] == 0]['avg_word_length'], ax=axes[0, 1])
axes[0, 1].set_title('Neutrals')
axes[0, 1].set_xlabel('Average word length')

# Believers
sns.distplot(train[train['sentiment'] == 1]['avg_word_length'], ax=axes[1, 0])
axes[1, 0].set_title('Believers')
axes[1, 0].set_xlabel('Average word length')

# Factuals
sns.distplot(train[train['sentiment'] == 2]['avg_word_length'], ax=axes[1, 1])
axes[1, 1].set_title('Factuals')
axes[1, 1].set_xlabel('Average word length')

plt.tight_layout()
plt.show()


# In[ ]:


# Saving averages
denier_word_length_avg = round(train[train['sentiment'] == -1]['avg_word_length'].mean(), 2)
neutral_word_length_avg = round(train[train['sentiment'] == 0]['avg_word_length'].mean(), 2)
believer_word_length_avg = round(train[train['sentiment'] == 1]['avg_word_length'].mean(), 2)
factual_word_length_avg = round(train[train['sentiment'] == 2]['avg_word_length'].mean(), 2)


# ### Comparing metrics between categories

# In[ ]:


# Creating dictionary of data
tweet_metrics = {'Average word count': [denier_word_count_avg,
                                        neutral_word_count_avg,
                                        believer_word_count_avg,
                                        factual_word_count_avg],
                 'Average tweet length': [denier_tweet_length_avg,
                                          neutral_tweet_length_avg,
                                          believer_tweet_length_avg,
                                          factual_tweet_length_avg],
                 'Average word length': [denier_word_length_avg,
                                         neutral_word_length_avg,
                                         believer_word_length_avg,
                                         factual_word_length_avg]}

# Converting dictionary to dataframe
tweet_metrics = pd.DataFrame.from_dict(tweet_metrics, orient='index',
                                       columns=['Deniers', 'Neutrals',
                                                'Believers', 'Factuals'])
tweet_metrics


# In[ ]:


# Divide "Average tweet length" by 10 so that it visualises nicely
tweet_metrics.iloc[1,:] = tweet_metrics.iloc[1,:].apply(lambda x: x / 10)


# The "Average tweet length" is scaled down by dividing it by 10 so that it displays well in the following plot.

# In[ ]:


# Create new melted table for visualisation
tweet_metrics = tweet_metrics.reset_index()
tweet_metrics_melted = pd.melt(tweet_metrics, id_vars=['index'],
                               value_vars=['Deniers', 'Neutrals',
                               'Believers', 'Factuals'])


# In[ ]:


# Visualise length metrics
plt.figure(figsize=(10, 6))
sns.barplot(x='variable', y='value', hue='index', data=tweet_metrics_melted,
            palette='rainbow')
plt.xlabel("Category")
plt.ylabel("Value")
plt.show()


# Tweets from users who believe in climate change tend to write longer tweets than other groups. This may be because they feel emotionally invested in the topic.
# 
# Tweets from users who provide a link to a factual news website tend to use longer but fewer words - the links provided in the data may be skewing this information. They also tend to use fewer words than all other groups.

# ## Common words for each category

# There may be useful insights in seeing the most commonly used words (or groups of words) from each category. By taking inspiration from [this Kaggle kernel](https://https://www.kaggle.com/githubsearch/sentiment-classification-of-tweets), we will apply some preprocessing to simplify the tweets into their most basic form and then observe each tweet's bi-grams and tri-grams.
# 

# A duplicate dataframe is created just for the purpose of this analysis.

# In[ ]:


train_analysis = train.copy()


# In[ ]:


lem = WordNetLemmatizer()


# ### Create a tweet normalizer function

# In[ ]:


def normalizer(tweet):
    """
    Normalises a tweet string by removing URLs, punctuation, converting to
    lowercase, tokenisation and lemmatization.
    
    parameters:
            tweet: (string) A tweet that will be normalised
    Returns:
            lemmas: A list of the preprocessed strings

    """
    
    tweet_no_url = re.sub(r'http[^ ]+', '', tweet) # Remove URLs beginning with http
    tweet_no_url1 = re.sub(r'www.[^ ]+', '', tweet_no_url) # Remove URLs beginning with http
    only_letters = re.sub("[^a-zA-Z]", " ",tweet_no_url1)  # Remove punctuation
    tokens = nltk.word_tokenize(only_letters) # Tokenization
    lower_case = [l.lower() for l in tokens] # Lowercase
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [lem.lemmatize(t) for t in filtered_result] 
    
    return lemmas


# The tweets are cleaned by removing URLs and punctuation. They are then tokenized, converted into lowercase, and lemmatized. By reducing the words in each tweet down to its most basic form we will be able to get an understanding of the common concepts that come up in each category.

# In[ ]:


# Display normalised messages
pd.set_option('display.max_colwidth', 500)
train_analysis['normalized'] = train_analysis['message'].apply(normalizer)
train_analysis[['message', 'normalized']].head()


# ### Bigrams and trigrams 

# The tweets' bi-grams and tri-grams are observed because they may provide more context than the lemmatized words on their own.

# In[ ]:


def ngrams(input_list):
    """
    Creates a list of 2 and 3 consecutive words within the input list.
    
    Parameters:
            input_list: A list of strings that come from a normalized tweet
    Returns:
            bigrams+trigrams: A list of the bigrams and trigrams for the input list
    """
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    
    return bigrams+trigrams


# In[ ]:


# Display ngrams
train_analysis['grams'] = train_analysis.normalized.apply(ngrams)
train_analysis[['grams']].head()


# In[ ]:


def count_words(input_list):
    """
    Counts the number of occurences of strings within the input list.
    
    Parameters:
        input_list: A list of strings
        
    Return:
        A list of tuples containing n-grams and a count of occurences for the n-gram
    """
    
    cnt = collections.Counter()
    
    for row in input_list:
        for word in row:
            cnt[word] += 1
            
    return cnt


# ### Display ngrams count for different classes

# #### Climate change deniers

# In[ ]:


# Most common bigrams and trigrams for climate change deniers
train_analysis[(train_analysis.sentiment == -1)][['grams']].apply(count_words)['grams'].most_common(15)


# In[ ]:


# Most common individual words
train_analysis[(train_analysis.sentiment == -1)][['normalized']].apply(count_words)['normalized'].most_common(10)


# In this dataset, climate change deniers seem to tend to retweet Donald Trump and Twitter user @SteveSGoddard, who has since changed his username to [@Tony__Heller](https://twitter.com/Tony__Heller). US President Donald Trump is generally known to not believe in climate change, and has, in the past, suggested that is a hoax invented by China. This ties in with one of the other common bigrams, which is 'created chinese', which suggests that some people who deny climate change may also believe that it is "created by China." Tony Heller is a conservative anti climate change activist. This data may suggest that those who don't believe in climate change may be aligned towards right-wing politics.

# #### Neutral tweets

# In[ ]:


# Most common bigrams and trigrams for neutral tweets
train_analysis[(train_analysis.sentiment == 0)][['grams']].apply(count_words)['grams'].most_common(20)


# In[ ]:


# Most common individual words
train_analysis[(train_analysis.sentiment == 0)][['normalized']].apply(count_words)['normalized'].most_common(15)


# These tweets neither supports or refutes the belief of man-made climate change. An interesting common bigram that comes up is "club penguin." Neutral tweets also mention Leonardo Dicaprio who created the climate documentary "Before the Flood."

# In[ ]:


# Looking at club penguin tweets just for interest's sake
train[train['message'].str.contains("club penguin")]['message'].head(10)


# #### Climate change believers

# In[ ]:


# Most common bigrams and trigrams for climate change believers
train_analysis[(train_analysis.sentiment == 1)][['grams']].apply(count_words)['grams'].most_common(20)


# In[ ]:


# Most common individual words
train_analysis[(train_analysis.sentiment == 1)][['normalized']].apply(count_words)['normalized'].most_common(15)


# Tweets in this category frequently mention the idea of dying as a result of climate change. The tweets that are frequently mentioning Twitter user @StephenSchlegel are retweets that are responding to a tweet by Melania Trump. Melania posted a picture of a sea creature with the caption, "What is she thinking?" and many people responded with, "She's thinking about how she's going to die because your husband doesn't believe in climate change." This may indicate that those who believe in climate change tend to not follow Donald Trump.

# In[ ]:


train[train['message'].str.contains("StephenSchlegel")]['message'].head(5)


# #### Tweets that link a factual news article

# In[ ]:


# Most common bigrams and trigrams for tweets that link a factual news website
train_analysis[(train_analysis.sentiment == 2)][['grams']].apply(count_words)['grams'].most_common(20)


# In[ ]:


train_analysis[(train_analysis.sentiment == 2)][['normalized']].apply(count_words)['normalized'].most_common(15)


# **Tweets that link a factual news website**
# 
# These tweets seem to be centered around issues relating to policy and Donald Trump and Scott Pruitt's (former Administrator of the U.S. Environmental Protection Agency) views on climate change. 
# 
# There is also mention of the [Paris Agreement](https://unfccc.int/process-and-meetings/the-paris-agreement/the-paris-agreement), which is an agreement with the United Nations Framework Convention on Climate Change which deals with the reduction of the impact of climate change. In 2017, President Donald Trump chose to withdraw the U.S.'s participation from this agreement.

# In[ ]:


train[train['message'].str.contains("Paris")]['message'].head(5)


# __________
# 

# ### Observations from ngrams analysis
# 
# These bigrams and trigrams show some interesting characteristics about each category, especially the fact that each category tends to mention or retweet similar people. The two most commonly-used bigrams for each category are 'climate change' and 'global warming' which doesn't really mean much since it is already established that this is a dataset containing tweets related to climate change/global warming.
# 
# From this data, there seems to be a political divide between climate change supporters and deniers, where climate change deniers tend to be supporters of Republican politics, whereas climate change believers may either be anti-Donald Trump or may be aligned towards Democratic politics. 
# 
# Identifying one's political position may indicate factors such as personal values and beliefs, social habits, the news sources that they read and the websites that they visit. This information could be useful to marketers.

# ## Most common hashtags for each category

# Hashtags are commonly used on social media for categorizing content into a specific theme or idea. Organisations often use hashtags to encourage conversation and engagement around a certain topic. 
# 
# By identifying common hashtags used in each category, one can identify the common themes that occur in discussions relating to that particular category. Analysing hashtags may offer more insights than an n-grams analysis because an n-grams analysis does not offer the sentiment attached to the use of a phrase. To clarify, if one individual is a follower of Donald Trump, and another individual wants to fight with Donald Trump online, "donald trump" will show up in the n-grams analysis for a climate change denier and believer. However, with hashtags, someone who is not a supporter of Donald Trump may not use the #donaldtrump as often as a supporter might. 
# 
# The Hashtag Dectector function that Team 10 (CPT) wrote during the Analyse sprint was used.

# In[ ]:


def find_hashtags(tweet):
  """
  Create a list of all the hashtags in a string

  Parameters:
    tweet: String 
  Outputs:
    hashtags: List of strings containing hashtags in input string

  """
  hashtags = []         
  for word in tweet.lower().split(' '): 
    #Appending the hashtag into the list hashtags
    if word.startswith('#'):        
        hashtags.append(word)        
  return hashtags


# In[ ]:


# Create new column for hashtags
train_analysis['hashtags'] = train['message'].apply(find_hashtags)


# In[ ]:


stopwords = set(STOPWORDS)

def show_wordcloud(data):
  """
  Create a wordcloud of the input data

  Parameters:
  data: list of strings
  Outputs:
  plt figure of a wordcloud
  """
  wordcloud = WordCloud(
      background_color='white',
      stopwords=stopwords,
      max_font_size=30,
      scale=3,
      random_state=1)
  
  wordcloud=wordcloud.generate(str(data))

  fig = plt.figure(1, figsize=(12, 12))
  plt.axis('off')

  plt.imshow(wordcloud)
  plt.show()


# #### Climate change deniers

# In[ ]:


print("Top 10 hashtags for climate change deniers:")
train_analysis[(train_analysis.sentiment == -1)][['hashtags']].apply(count_words)['hashtags'].most_common(10)


# In[ ]:


show_wordcloud(train_analysis[(train_analysis.sentiment == -1)][['hashtags']].apply(count_words)['hashtags'].most_common(50))


# #### Neutral about climate change

# In[ ]:


print("Top 10 hashtags for climate change neutrals:")
train_analysis[(train_analysis.sentiment == 0)][['hashtags']].apply(count_words)['hashtags'].most_common(10)


# In[ ]:


show_wordcloud(train_analysis[(train_analysis.sentiment == 0)][['hashtags']].apply(count_words)['hashtags'].most_common(50))


# #### Climate change believers

# In[ ]:


print("Top 10 hashtags for climate change believers:")
train_analysis[(train_analysis.sentiment == 1)][['hashtags']].apply(count_words)['hashtags'].most_common(10)


# In[ ]:


show_wordcloud(train_analysis[(train_analysis.sentiment == 1)][['hashtags']].apply(count_words)['hashtags'].most_common(50))


# #### Provided a link to factual news website

# In[ ]:


print("Top 10 hashtags for those who provided a link to a factual news site:")
train_analysis[(train_analysis.sentiment == 2)][['hashtags']].apply(count_words)['hashtags'].most_common(10)


# In[ ]:


show_wordcloud(train_analysis[(train_analysis.sentiment == 2)][['hashtags']].apply(count_words)['hashtags'].most_common(50))


# ### Insights from hashtags analysis

# The most commonly-used hashtags imply the topic of conversation in which these Twitter users are participating. These hashtags may confirm the insights offered through the n-grams analysis, where it seems that some users' opinions are politically driven. 

# ## Named Entity Recognition

# Named Entity Recognition (NER) is a function that is able to identify important named entities in a text such as people, organisations and currencies. This will help identify important information such as who and what the Twitter users are talking about in each category. The functions used to generate the NER information were largely inspired by [this article](https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools) on NLP EDA.

# In[ ]:


nlp = spacy.load("en_core_web_sm")


# Identifying the most frequently occurring types (labels) of entities in the dataset will make it easier to identify which entity types should be analysed.

# In[ ]:


def ner_labels(tweet):
  """
  Get a list of types of entities in a string

  Parameters:
  tweet: A tweet in the form of a string
  Returns:
  A list of the named entity labels
  """
  doc=nlp(tweet)
  return [X.label_ for X in doc.ents]


# In[ ]:


# Create a column for the entity labels
train_analysis['entity labels'] = train['message'].apply(lambda x: ner_labels(x))


# In[ ]:


# Most frequently occurring types of entities in the dataset
common_entity_labels = train_analysis[['entity labels']].apply(count_words)['entity labels'].most_common(10)

x_vals = []
y_vals = []

for i in common_entity_labels:
    x_vals.append(i[0])
    y_vals.append(i[1])

plt.figure(figsize=(18,6))
plt.title('Most frequently occurring types of entities')
sns.barplot(y_vals,x_vals, palette='rainbow')


# We can see that organisations, people and GPEs (geopolitical entities) are the most frequently used named entity amongst the dataset. We will look deeper into the the most frequently mentioned people and organisations by each category.

# In[ ]:


def ner_tokens(tweet,ent):
  """
  Return a list of the named entity tokens that are 
  of the given label.

  Parameters:
      tweet: A tweet in the form of a string
      ent: Entity label (default is "ORG" - organisation)
  Outputs:
      A list of the named entity tokens in a tweet that are
      of the given label.
  """
  doc=nlp(tweet)
  return [X.text for X in doc.ents if X.label_ == ent]


# In[ ]:


# Create new column of "ORG" entities
train_analysis['entity_tokens'] = train_analysis['message'].apply(lambda x: ner_tokens(x, "ORG"))


# #### Most frequently mentioned organisations

# In[ ]:


# Climate change deniers
denier_orgs = train_analysis[(train_analysis.sentiment == -1)][['entity_tokens']].apply(count_words)['entity_tokens'].most_common(5)
x_vals = []
y_vals = []

for i in denier_orgs:
    x_vals.append(i[0])
    y_vals.append(i[1])

plt.figure(figsize=(14,4))
sns.barplot(y_vals,x_vals, palette='rainbow')
plt.title('Most commonly mentioned organisations by climate change deniers')
plt.show()


# In[ ]:


# Climate change neutrals
neutral_orgs = train_analysis[(train_analysis.sentiment == 0)][['entity_tokens']].apply(count_words)['entity_tokens'].most_common(5)
x_vals = []
y_vals = []

for i in neutral_orgs:
    x_vals.append(i[0])
    y_vals.append(i[1])

plt.figure(figsize=(14,4))
sns.barplot(y_vals,x_vals, palette='rainbow')
plt.title('Most commonly mentioned organisations by climate change neutrals')
plt.show()


# In[ ]:


# Climate change believers
believer_orgs = train_analysis[(train_analysis.sentiment == 1)][['entity_tokens']].apply(count_words)['entity_tokens'].most_common(5)
x_vals = []
y_vals = []

for i in believer_orgs:
    x_vals.append(i[0])
    y_vals.append(i[1])

plt.figure(figsize=(14,4))
sns.barplot(y_vals,x_vals, palette='rainbow')
plt.title('Most commonly mentioned organisations by climate change believers')
plt.show()


# In[ ]:


# Climate change factuals
factual_orgs = train_analysis[(train_analysis.sentiment == 2)][['entity_tokens']].apply(count_words)['entity_tokens'].most_common(5)
x_vals = []
y_vals = []

for i in factual_orgs:
    x_vals.append(i[0])
    y_vals.append(i[1])

plt.figure(figsize=(14,4))
sns.barplot(y_vals,x_vals, palette='rainbow')
plt.title('Most commonly mentioned organisations by climate change factuals')
plt.show()


# - The EPA is the United States Environmental Protection Agency. 
# - CO2 was recognised as an organisation but is likely referring to carbon dioxide.
# - Exxon is an oil and gas corporation
# - NYT refers to the New York Times

# In[ ]:


# Create new column of "PERSON" entities
train_analysis['entity_tokens_people'] = train_analysis['message'].apply(lambda x: ner_tokens(x, "PERSON"))


# In[ ]:


# Climate change deniers
denier_per = train_analysis[(train_analysis.sentiment == -1)][['entity_tokens_people']].apply(count_words)['entity_tokens_people'].most_common(5)
x_vals = []
y_vals = []

for i in denier_per:
    x_vals.append(i[0])
    y_vals.append(i[1])

plt.figure(figsize=(14,4))
sns.barplot(y_vals,x_vals, palette='rainbow')
plt.title('Most commonly mentioned people by climate change deniers')
plt.show()


# In[ ]:


# Climate change neutrals
neutral_per = train_analysis[(train_analysis.sentiment == 0)][['entity_tokens_people']].apply(count_words)['entity_tokens_people'].most_common(5)
x_vals = []
y_vals = []

for i in neutral_per:
    x_vals.append(i[0])
    y_vals.append(i[1])

plt.figure(figsize=(14,4))
sns.barplot(y_vals,x_vals, palette='rainbow')
plt.title('Most commonly mentioned people by climate change neutrals')
plt.show()


# In[ ]:


# Climate change believers
believer_per = train_analysis[(train_analysis.sentiment == 1)][['entity_tokens_people']].apply(count_words)['entity_tokens_people'].most_common(5)
x_vals = []
y_vals = []

for i in believer_per:
    x_vals.append(i[0])
    y_vals.append(i[1])

plt.figure(figsize=(14,4))
sns.barplot(y_vals,x_vals, palette='rainbow')
plt.title('Most commonly mentioned people by climate change believers')
plt.show()


# In[ ]:


# Climate change factuals
factual_per = train_analysis[(train_analysis.sentiment == 2)][['entity_tokens_people']].apply(count_words)['entity_tokens_people'].most_common(5)
x_vals = []
y_vals = []

for i in factual_per:
    x_vals.append(i[0])
    y_vals.append(i[1])

plt.figure(figsize=(14,4))
sns.barplot(y_vals,x_vals, palette='rainbow')
plt.title('Most commonly mentioned people by climate change factuals')
plt.show()


# The highly frequent mentions of people such as Donald Trump, Scott Pruitt, Hillary Clinton and Rex Tillerson also confirm that the discussion of climate change is largely a political issue.

# # Preprocessing

# [Back to contents](#Contents)

# The Twitter data needs to be preprocessed in such a way that it makes it easier for the machine learning algorithms to analyse.

# In[ ]:


# REGEX PATTERNS FOR CLEANING
url_re = r'http[^ ]+' # Remove URL starting with http
www_re = r'www.[^ ]+' # Remove URL starting with www
hashtag_re = r'#([^\s]+)' # Convert hashtag into just word without #


# In[ ]:


# Preprocessing function
def preprocessing(tweet):
    """
    Use regex to clean tweets to prepare for classification algorithms.
    Tweets are converted to lowercase, @ symbols before handles are removed,
    # symbols before hashtags are removed, and http and www links are 
    removed.
    
    Parameters:
        tweet: The string that is going to be cleaned
        
    Returns:
        tweet: String that has been cleaned
    """
    # Convert tweet to lower-case
    tweet = tweet.lower()
    
    # Remove twitter handles
    tweet = re.sub(r'@', r'', tweet)
    
    # Remove hashtags
    tweet = re.sub(r'#', '', tweet)
    
    # Remove links
    tweet = re.sub(url_re, '', tweet)
    tweet = re.sub(www_re, '', tweet)
    
    return tweet


# Initially, the tweet preprocessing function included functionalities such as removing @ handles, removing stop words, and tokenization and lemmatization. These functions were eventually removed because it was found that the models performed better with this information. 
# 
# The model performed significantly better once the Twitter handles were added back in. The exploratory analysis that was performed confirmed this because it seems that tweets in each category seemed to be retweeting and @ mentioning similar Twitter users.

# In[ ]:


# Apply preprocessing function to clean training and test data, assign to
# new column
train['clean'] = train['message'].apply(preprocessing)
test['clean'] = test['message'].apply(preprocessing)

# Set max width so we can visually compare original and clean training data
pd.set_option('display.max_colwidth', None)
train[['message','clean']].head()


# # Models

# [Back to contents](#Contents)

# In[ ]:


# Set X and y
X = train['clean']
y = train['sentiment']

# Save X of test.csv so that we can make Kaggle submission from predictions
X_sub = test['clean']


# In[ ]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# As mentioned previously, the dataset is imbalanced where there are far more datapoints in category 1 than in any other category. This will be handled through resampling. Category 1, the largest category, contains 8530 entries, followed by 3640 entries in category 2, 2353 entries in category 0, and 1296 entries in category -1. 
# 
# The largest category will be undersampled when training each model so that it only has 4265 entries. 4265 was chosen because it is half of 8530, and losing more than half of a category's data may lead to losing a lot of valuable information. 
# 
# The smaller categories will be upsampled using SMOTE to create synthetic data. Each category will be upsampled to 4265 entries each so that it matches category 1.

# ## Random Forest

# In text classification, a random forest selects a subspace of features at each node to grow branches of a decision trees, then to use bagging method to generate training data subsets for building individual trees, finally to combine all individual trees to form random forests model. For this task, a random forest is ideal because with enough trees in the forest, the classifier is likely to not overfit the data. In classification problems, Random Forests create trees based on the probability of belonging to a certain class, therefore, they tend to work well with multiclass problems. This model is expected to do a good job compared to some of the other models
# 
# The hyperparameters used to train this models were acquired from a grid search performed with the variables shown below. The best parameters were saved for future use. 
# 
# `params_rf = {
#     'n_estimators' : [10, 25],
#     'max_features' : [5, 10],
#     'max_depth' : [30, 50, None],
#     'min_samples_split' : [2, 3]}`
#  
# Source: [How Random Forest Algorithm Works in Machine Learning](https://medium.com/@Synced/how-random-forest-algorithm-works-in-machine-learning-3c0fe15b6674)

# In[ ]:


# Perform grid search
# params_rf = {
#     'n_estimators' : [10, 25],
#     'max_features' : [5, 10],
#     'max_depth' : [30, 50, None],
#     'min_samples_split' : [2, 3]
# }

rf_model = RandomForestClassifier(n_estimators = 25, max_depth = None,
                                  max_features = 5, min_samples_split = 3)
# rf_grid = GridSearchCV(rf_model, params_rf, cv=3)


# In[ ]:


# Create pipeline for vectorization, training and fitting
pipe_rf = make_pipeline(TfidfVectorizer(),
                        SMOTE(sampling_strategy={-1: 4265, 0: 4265, 2: 4265}, random_state=42),
                        RandomUnderSampler(sampling_strategy={1:4265}),
                        rf_model)
    
# Fit model using pipeline
pipe_rf.fit(X_train, y_train)

# Make predictions
predictions_rf = pipe_rf.predict(X_test)


# In[ ]:


#rf_grid.best_params_


# Best parameters saved from the grid search:
# 
# {'max_depth': None,
#  'max_features': 5,
#  'min_samples_split': 3,
#  'n_estimators': 25}

# In[ ]:


rf_f1_score = f1_score(y_test, predictions_rf, average='macro')
print(f"Random Forest F1 score: {rf_f1_score}")


# In[ ]:


# LOGGING FOR COMET
# params = {'random_state': 1,
#           'model_type': 'randomforest',
#           'param_grid': str(params_rf)
    
# }
# metrics = {'f1': rf_f1_score}

# experiment.log_parameters(params)
# experiment.log_metrics(metrics)


# In[ ]:


print('Classification Report')
print(classification_report(y_test, predictions_rf, target_names=['-1: Denier', '0: Neutral', '1: Believer', '2: News article']))


# The Random Forest model is better at predicting climate change believers better than any of the other classes. For believers, it has a higher precision than recall, meaning that it is more likely to identify false negatives (identified as not-believer when it is actually a believer) than it is to identify false positives. The same can be said for the denier class, where it may falsely identify someone as not being a climate change denier when in fact they actually are. For the "provides a link to a factual news article" class, there is a higher recall than precision meaning that the model is more likely to false positives, where a tweet may be identified as part of this category when in fact they are not.

# ## Support Vector Classifier

# Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. However,  it is mostly used in classification problems. In the SVC (Support Vector Classifier) algorithm, each item is plotted as a point in n-dimensional space (n = number of features) with the value of each feature being the value of a particular coordinate. Then, classification is performed by finding the hyper-plane that differentiates the two classes. This Support Vector Classifier algorithm performs well when the target classes do not overlap each other. The model is expected to do well because the target classes (-1, 0, 1, 2) do not overlap.
# 
# [[Source]](https://medium.com/@dhiraj8899/top-4-advantages-and-disadvantages-of-support-vector-machine-or-svm-a3c06a2b107)
# 
# The hyperparameters were acquired from a grid search with the following variables:
# 
# `params_svc_grid = {'C': [0.1,1, 10, 100],
#               'gamma': [1,0.1,0.01,0.001],
#               'kernel': ['rbf','linear']}`
# 

# In[ ]:


# Perform grid search
# params_svc_grid = {'C': [0.1,1, 10, 100],
#               'gamma': [1,0.1,0.01,0.001],
#               'kernel': ['rbf','linear']}
# svc_model = SVC()
# svc_grid = GridSearchCV(svc_model, params_svc_grid, cv=3)

# Parameters come from best grid search result
params_svc = {'C': 10, 'gamma': 1, 'kernel':'linear', 'probability':'True'}
svc_grid = SVC(C=10, gamma=1, kernel='linear', probability=True)


# In[ ]:


# Create pipeline for vectorization, training and fitting
pipe_svc = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                         SMOTE(sampling_strategy={-1: 4265, 0: 4265, 2: 4265}, random_state=42),
                         RandomUnderSampler(sampling_strategy={1:4265}),
                         svc_grid)

# Fit model using pipeline
pipe_svc.fit(X_train, y_train)

# Make predictions
predictions_svc = pipe_svc.predict(X_test)


# In[ ]:


confusion_matrix(y_test, predictions_svc)


# In[ ]:


#svc_grid.best_params_


# {'C': 10, 'gamma': 1, 'kernel': 'linear'}

# In[ ]:


print('Classification Report')
print(classification_report(y_test, predictions_svc, target_names=['-1: Denier', '0: Neutral', '1: Believer', '2: News article']))


# The Support Vector Classifier model is better at predicting climate change believers better than any of the other classes. For believers, it has a higher precision than recall, meaning that it is more likely to identify false negatives (identified as not-believer when it is actually a believer) than it is to identify false positives. For all other classes it has a higher recall than precision, meaning that the model may falsely identify a tweet as part of that category when it is in fact part of a different class.

# In[ ]:


svc_f1_score = f1_score(y_test, predictions_svc, average='macro')
print(f'Macro f1: {svc_f1_score}')


# In[ ]:


# LOGGING FOR COMET
# params = {'model_type': 'SVC',
#           'param_grid': str(params_svc),
# }
# metrics = {'f1': svc_f1_score}

# experiment.log_parameters(params)
# experiment.log_metrics(metrics)


# In[ ]:


#experiment.end()


# In[ ]:


#experiment.display()


# In[ ]:


# Use to make Kaggle submission using test.csv predictions
kaggle_pred_svc = pipe_svc.predict(X_sub)
submission_svc = pd.DataFrame({'TweetID':test['tweetid'],'Sentiment':kaggle_pred_svc})
submission_svc = submission_svc.set_index('TweetID')


# In[ ]:


submission_svc.to_csv('support_vector_classifier.csv')


# In[ ]:


# SAVING PKL FILE
# model_save_path = 'svc_model2.pkl'
# with open(model_save_path, 'wb') as file:
#   pickle.dump(svc_grid, file)


# ## Logistic Regression

# Logistic regression is often the appropriate regression analysis to conduct when the dependent variable is binary (two possible classes). However, it can be used with multi-class classification by changing the `multi_class` hyperparameter.  Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables. Logistic regression is not necessarily a classifier but rather a probability estimator, in that it calculates the probability of a data point belonging to a certain class. 
# 
# 
# Source: [Logistic Regression](https://web.stanford.edu/~jurafsky/slp3/5.pdf)

# In[ ]:


# Perform grid search
# params_log = {'C': np.logspace(-3,3,7),
#               'multi_class':['ovr'],
#               'penalty': ['l1', 'l2']}

log_model = LogisticRegression(multi_class='ovr', C = 1)

#log_grid = GridSearchCV(log_model, params_log, cv = 3,
                        #verbose = 1, n_jobs = -1)


# In[ ]:


# Create pipeline for vectorization, training and fitting
pipe_log = make_pipeline(TfidfVectorizer(),
                         SMOTE(sampling_strategy={-1: 4265, 0: 4265, 2: 4265}, random_state=42),
                         RandomUnderSampler(sampling_strategy={1:4265}),
                         log_model)
# Fit model using pipeline
pipe_log.fit(X_train, y_train)

# Make predictions
predictions_log = pipe_log.predict(X_test)


# In[ ]:


print('Classification Report')
print(classification_report(y_test, predictions_log, target_names=['-1: Denier', '0: Neutral', '1: Believer', '2: News article']))


# The logistic regression model is able to predict climate change believers and those who provide a link to a factual news article better than other groups. It is likely to make false negative predictions for climate change believers, neutrals, and those who provide a link to a factual news article. For climate change believers, it may falsely identify the tweet as being part of a different class when in fact it is a climate change believer.

# In[ ]:


log_f1_score = f1_score(y_test, predictions_log, average='macro')
print(f"Logistic Regression F1 score: {log_f1_score}")


# In[ ]:


# SAVING PKL FILE
# model_save_path = 'log_model.pkl'
# with open(model_save_path, 'wb') as file:
#   pickle.dump(pipe_log, file)


# ## k-nearest neighbours

# The k-nearest neighbours algorithm assumes that similar data points are within close proximity to each other, and will only work well if this is true. According to [Argerich](https://www.quora.com/What-is-better-k-nearest-neighbors-algorithm-k-NN-or-Support-Vector-Machine-SVM-classifier-Which-algorithm-is-mostly-used-practically-Which-algorithm-guarantees-reliable-detection-in-unpredictable-situations), k-nearest neighbours need to be carefully tuned and may not perform well in a scenario where there are a few points in a high-dimensional space, in which case, a Support Vector Machine may perform better. 
# 
# [[Source]](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)

# In[ ]:


# Perform grid search
# params_knn_grid = {'n_neighbors': [3,5,11],
#                    'weights': ['uniform', 'distance'],
#                    'metric':['euclidian', 'manhattan']}

knn_model = KNeighborsClassifier(n_neighbors = 3,
                                 weights = 'distance',
                                 metric = 'manhattan')
# knn_grid = GridSearchCV(knn_model, params_knn_grid, cv=3, verbose = 1, n_jobs = -1)


# In[ ]:


# Create pipeline for vectorization, training and fitting
pipe_knn = make_pipeline(TfidfVectorizer(),
                         SMOTE(sampling_strategy={-1: 4265, 0: 4265, 2: 4265}, random_state=42),
                         RandomUnderSampler(sampling_strategy={1:4265}),
                         knn_model)

# Fit model using pipeline
pipe_knn.fit(X_train, y_train)

# Make predictions
predictions_knn = pipe_knn.predict(X_test)


# In[ ]:


#knn_grid.best_params_


# Best parameters from grid search: 
# 
# {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}

# In[ ]:


confusion_matrix(y_test, predictions_knn)


# In[ ]:


print('Classification Report')
print(classification_report(y_test, predictions_knn, target_names=['-1: Denier', '0: Neutral', '1: Believer', '2: News article']))


# The k-nearest neighbours model has a moderate f1 score for those who provide a link to a factual news article. For climate change believers it has a very high precision but a very low recall which means that it is very likely to falsely identify false negatives, meaning that it would identify the tweet as part of being a part of a different class when in fact they are actually climate change believers. For the neutral class the opposite can be said with its high recall and low precision, meaning that the model will likely identify false positives, where it may say that a tweet is part of the neutral class when in fact is is actually part of another group.

# In[ ]:


knn_f1_score = f1_score(y_test, predictions_knn, average='macro')
print(f'k-nearest neighbours f1: {knn_f1_score}')


# In[ ]:


# Use to make Kaggle submission using test.csv predictions
kaggle_pred_knn = pipe_knn.predict(X_sub)
submission_knn = pd.DataFrame({'TweetID':test['tweetid'],'Sentiment':kaggle_pred_knn})


# In[ ]:


# SAVING PKL FILE
# model_save_path = 'knn_model.pkl'
# with open(model_save_path, 'wb') as file:
#   pickle.dump(pipe_knn, file)


# In[ ]:


#experiment.end()


# ## XGBoost

# XGBoost is an improvement of the Random Forest model. It is penalizes models through L1 and L2 regularization to prevent overfitting and comes with a built-in cross-validation method. This model is expected to perform well with the data. Neither a grid search nor resampling was performed on this model due to time constraints, however, we feel that if these were implemented with this model that this model would have been the best-performing model for this data.
# 
# Additional feature engineering is performed on this XGBoost model, where the length of the tweet is added as an additional feature to aid in the prediction.
# 
# 
# It may seem clear that this XGBoost model is created very differently to the previous models - this is because this model was built by a different member of the group.

# In[ ]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# In[ ]:


# vectorise & transform
word_vectorizer= TfidfVectorizer(
    sublinear_tf= True,
    strip_accents= 'unicode',
    analyzer= 'word',
    token_pattern= r'\w{1,}',
    stop_words= 'english',
    ngram_range= (1, 1),
    max_features= 5000)

word_vectorizer.fit(train.clean)
train_word_features= word_vectorizer.transform(X_train)
test_word_features= word_vectorizer.transform(X_test)
sub_word_features = word_vectorizer.transform(X_sub)
    
char_vectorizer= TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range= (2, 6),
    max_features=50000)

char_vectorizer.fit(train.clean)
train_char_features= char_vectorizer.transform(X_train)
test_char_features= char_vectorizer.transform(X_test)
sub_char_features = char_vectorizer.transform(X_sub)
    
train_features= hstack([train_char_features, train_word_features])
test_features= hstack([test_char_features, test_word_features])
sub_features = hstack([sub_char_features, sub_word_features])

# TRAIN FEATURE ENGINEERING
x_len= X_train.apply(len)
X_train_aug= add_feature(train_features, x_len)
x_digit= X_train.apply(lambda x: len(re.sub('\D','', x)))
X_train_aug2= add_feature(X_train_aug, x_digit)

# TEST FEATURE ENGINEERING
x_len2= X_test.apply(len)
X_test_aug= add_feature(test_features, x_len2)
x_digit2= X_test.apply(lambda x: len(re.sub('\D','', x)))
X_test_aug2= add_feature(X_test_aug, x_digit2)        


# In[ ]:


# Instantiate XGBClassifier object
XGB= XGBClassifier(n_estimators=100, max_depth= 10)

# Fit XGBoost model
xg_model= XGB.fit(X_train_aug2, y_train)

# Make prediction
y_pred_test= xg_model.predict(X_test_aug2)


# In[ ]:


print('Classification Report')
print(classification_report(y_test, y_pred_test, target_names=['-1: Denier', '0: Neutral', '1: Believer', '2: News article']))


# The XGBoost model performs decently on climate change believers and those who provided a link to a news article. It performs relatively poorly on the neutrals and deniers. This may be because the data was not resampled prior to training the model. For climate change deniers the model has a high precision but low recall, meaning that it may incorrectly place tweets in a different category when in fact it belongs to the denier class. The same can be said for the neutral class. The model has a high recall for the believer group, meaning that it is very good at correctly sidentifying believers and is not likely to produce false negatives.

# In[ ]:


xg_f1_score = f1_score(y_test, y_pred_test, average='macro')
print(f'XGBoost neighbours f1: {xg_f1_score}')


# # Model selection

# In[ ]:


scores = {'Random Forest': rf_f1_score,
          'Support Vector Classifier': svc_f1_score,
          'N-nearest Neighbours': knn_f1_score,
          'Logistic Regression': log_f1_score,
          'XGBoost': xg_f1_score}


# In[ ]:


plt.figure(figsize=(10, 6))
plt.bar(scores.keys(), scores.values())
plt.xticks(rotation=30)
plt.title('Comparing validation macro f1 scores for each model')
plt.ylabel('Macro f1 score')
plt.show()


# The Support Vector Classifier model has the greatest macro f1 score when testing it against the validation set (which in this case was X_test and y_test resulting from the train-test split). However, when testing against the test set on Kaggle, the model seemed to perform even better with a score of 0.75142.
# 
# Unfortunately, despite the resampling that was done, none of the models were particularly good at identifying classes other than the "Climate change believer" and in some cases the "Links to factual news source" class. Future improvements would include resampling and performing a grid search on the XGBoost model, because it may have been the highest performing model in the end.
# 
# While the Support Vector Classifier performs the best, the Random Forest model, the SVC model, and the Logistic Regression model will be included in the Streamlit application so that the user can select the algorithm that they'd like to use for the prediction. 
