#!/usr/bin/env python
# coding: utf-8

# # **Getting Started with Transfer Learning Using Tensorflow Hub**

# Hi Kagglers! I have been a *R* lover for long now (and I still am!), but recently I have started developing interest for **Python**. Here is my first public **Python** notebook. **Please do read it and hit upvote, if you like it**. If you have any ideas to improve the analysis, please post it in the comments.

# Welcome to the introductory notebook on using **TensorFlow Hub** for building transfer learning models. This notebook is focused on quickly get you started with building sophisticated models with very little knowledge of modeling with **TensorFlow**. Let's get started!

# ### What is Transfer Learning?
# **Transfer Learning** is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem (Source: [Wikipedia](https://en.wikipedia.org/wiki/Transfer_learning)). 
# 
# Transfer Learning overcomes the problem of isolated learning by applying the knowledge gained during learning one task to another different but similar task. For example, knowledge gained while learning to recognize cars could be applied when trying to recognize trucks! The intution actually came from the humans. We humans always practice *Transfer Learning*. A person who knows how to drive a bike, finds it easy to learn how to drive a car. Similarly *Transfer Learning* helps in learning difficult tasks with less effort.

# ### How Does TensorFlow Hub Help Us?
# **TensorFlow Hub** is a repository of pre-trained machine learning models. These models are categorised in three broad problem domains -
# + Image
# + Text
# + Video
# 
# There are several models, which you can quickly start using without much hassle. Visit [TensorFlow Hub](https://tfhub.dev/) for more details.

# ### The Problem
# For the demonstration of the transfer learning technique, I will be using the case of [](http://)[this](https://www.kaggle.com/c/nlp-getting-started) competition. The goal here is to build a machine learning model which is able to predict whether a tweet belongs to a real disaster or not!

# ### Dataset
# Dataset given here is tweets from different users split into training and test set. **target** is the dependent varible, which we are interested in predicting for the test set. Let's begin with our analysis.

# ### Importing Libraries

# In[ ]:


# General Purpose Libraries
import numpy as np
np.random.seed(1)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Text Processing Libraries
import spacy
import re
import string
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

# Scikit Learn
from sklearn.model_selection import train_test_split

# TensorFlow
import tensorflow_hub as hub
import tensorflow as tf
tf.random.set_seed(1)

# Setting Pandas Display Option
pd.set_option('display.max_colwidth', 500)

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


# ## Loading Data
# Let's begin by loading the dataset in our environment and then take a peek at the dataset.

# In[ ]:


# Reading Data Files
train = pd.read_csv("../input/nlp-getting-started/train.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
print("Train Shape :", train.shape)
print("Test Shape :", test.shape)


# In[ ]:


# Viewing top rows of training set
train.head()


# In[ ]:


# Viewing top rows of test set
test.head()


# + There are *7613* rows and *5* columns in the training set.
# + There are *3263* rows and *4* columns in the test set.
# + *target* is the dependent variable, with values *0* and *1* indicating a fake and real disaster tweet respectively.
# + *id* is just an unique identifier of the tweet. Training and Test set are subsets of a sample of tweets collected.
# + There are many missing values in *location* variable.

# ## Basic EDA
# Let's do some basic EDA to better understand the dataset provided.

# ### Missing Values

# In[ ]:


train.isnull().sum()


# *location* has many missing values. Are those missing values indicate non-legitimate tweets? There might be some information here. We will look at it at a later stage.

# ### The Dependent Variable - *target*

# In[ ]:


# Proportion of classes
train['target'].value_counts(normalize = True)


# In[ ]:


# Visualize the classes
sns.countplot('target', data = train)


# ### Independent Variables
# #### *keyword*

# In[ ]:


# Unique Words Count
train.keyword.value_counts()


# There are *221* unique words present in training set. Some text cleaning is required here to extract some information from this variable.

# In[ ]:


# Visualize top 20 keywords
sns.barplot(y=train['keyword'].value_counts()[:20].index,x=train['keyword'].value_counts()[:20])


# In[ ]:


# Visualize bottom 10 keywords
sns.barplot(y=train['keyword'].value_counts()[-20:].index,x=train['keyword'].value_counts()[-20:])


# #### *location*

# In[ ]:


# Unique Words Count
train.location.value_counts()


# In[ ]:


# Visualize top 20 locations
sns.barplot(y = train['location'].value_counts()[:20].index,x = train['location'].value_counts()[:20])


# In[ ]:


# Visualize bottom 20 locations
sns.barplot(y=train['location'].value_counts()[-20:].index,x=train['location'].value_counts()[-20:])


# There are many repetitions of location in the dataset with multiple names. For example - USA, United States & New York are separate entries. Also there are some numbers in this variable. We need to address these issues before making use to this variable in our model.

# #### *text*
# This is our main variable which contains actual tweets from different users. Let's take a look at some random tweets to get a sense of how the data is and what kind of pre-processing is required.

# In[ ]:


train['text'].sample(5)


# As one can expect from any textual data, there are all sorts of messyness going on here. There are numbers, special characters, links, punctuation marks etc. present in the tweets. We need to clean these before we proceed with modeling, in order to get good results.
# 
# Now, let's see how a typical "Real" & "Not Real" tweet looks like.

# In[ ]:


# Real tweets indicating disaster
real_tweets = train[train['target']==1]['text']
real_tweets.values[0:5]


# In[ ]:


# Not Real (Fake) tweets
fake_tweets = train[train['target']==0]['text']
fake_tweets.values[0:5]


# ## Text Preprocessing

# ### Text Cleaning
# Lets begin our text preprocessing by creating a custom function to remove numbers, links, punctuations etc. The following function has been taken from [Parul Pandey](https://www.kaggle.com/parulpandey/getting-started-with-nlp-a-general-intro)'s notebook. 

# In[ ]:


# Custom Function for Text Cleaning

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[ ]:


# Applying the cleaning function to both test and training datasets
train['cleaned_text'] = train['text'].apply(lambda x: clean_text(x))
test['cleaned_text'] = test['text'].apply(lambda x: clean_text(x))

# Take a look at the cleaned text
train['cleaned_text'].head()


# ### Removing Stopwords
# Simply removing all stopwords from the text might lead to loss of semantic meaning of the sentences. Since, we intend to use models which capture the semantic meaning its important that we take care of this problem. We will be using *nlppreprocess* library from PyPI to remove only unwanted stopwords. To know more about why removing stopwords blindly might not be a good idea here, read this nice article [here](https://towardsdatascience.com/why-you-should-avoid-removing-stopwords-aa7a353d2a52). 

# In[ ]:


get_ipython().system('pip install nlppreprocess')
from nlppreprocess import NLP

nlp = NLP()

train['stopwords_cleaned'] = train['cleaned_text'].apply(nlp.process)
test['stopwords_cleaned'] = test['cleaned_text'].apply(nlp.process)  


# ### Lemmatization
# *Lemmatization* is a technique where a word is reduced to its base or dictionary form. Like - *Go*, *Going* & *Gone* will all be replaced by *Go*. Let's create a custom function to lemmatize our tweets. 

# In[ ]:


# Import spaCy's language model
en_model = spacy.load('en', disable=['parser', 'ner'])

# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in en_model(i)]
        output.append(' '.join(s))
    return output


# In[ ]:


# Applying the lemmatization function to both test and training datasets
train['lemmatized_text'] = lemmatization(train['stopwords_cleaned'])
test['lemmatized_text'] = lemmatization(test['stopwords_cleaned'])


# In[ ]:


# Let's take a look at the cleaned & lemmatized text
train.head(5)


# I have added three new variables to our training and testing set so that we can test which strategy is producing best results. 

# ## Wordcloud
# Let's visualize wordcloud for train and test set to see the distribution of words!

# In[ ]:


# Wordcloud for train set
plt.figure(figsize = (12,9))
wordcloud = WordCloud(min_font_size = 6,  max_words = 200 , width = 1000 , height = 600).generate(" ".join(train['stopwords_cleaned']))
plt.imshow(wordcloud,interpolation = 'bilinear')


# In[ ]:


# Wordcloud for test set
plt.figure(figsize = (12,9))
wordcloud = WordCloud(min_font_size = 6,  max_words = 200 , width = 1000 , height = 600).generate(" ".join(test['stopwords_cleaned']))
plt.imshow(wordcloud,interpolation = 'bilinear')


# Some additional efforts required in cleaning the tweets to achieve better accuracy. Nonetheless, let's move to medeling and see how good our model is.

# ## Modeling Using Pretrained Model

# I will be using [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4) for modeling. It encodes text into high-dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.
# 
# The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. There are two variations of this model on *TensorFlow Hub* -
# + Deep Averaging Network (DAN) Encoder.
# + Transformer Encoder.
# 
# I will be using the one with DAN architecture, since it is computationally less intensive. We will download this model as a *KearsLayer* and use it as input layer for our Keras Sequential Model.

# ### Creating Training & Testing Set
# I will be using **stopwords_cleaned** variable for training our classification model, as it produced better results than the other text variables.
# 

# In[ ]:


# Splitting training & testing set
x_train, x_test, y_train, y_test = train_test_split(train['stopwords_cleaned'], train['target'],  
                                                    test_size = 0.2, random_state = 1)


# In[ ]:


# Build a Keras Sequential Model using Pre-trained Universal Sentence Encoder as the Input Layer
hub_layer = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4', 
                        input_shape = [],
                        output_shape = [512],
                        dtype = tf.string, 
                        trainable = True)

model = tf.keras.models.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
model.summary()


# In[ ]:


# Complile the Model
model.compile(optimizer = 'adam', 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])


# In[ ]:


# Train the Model 
model.fit(x_train, 
          y_train, 
          epochs = 1,
          validation_data = (x_test, y_test))


# ## Prediction

# Let's use this model to predict on our test set now and see how it performs.

# In[ ]:


# Predict on Test Set
pred = model.predict_classes(test['stopwords_cleaned'])


# ## Prepare for Submission

# In[ ]:


# Load Submission File and 
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
submission['target'] = pred
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv", index = False)


# **Thanks for taking out time to go through my notebook! Go ahead and experiment with other *TensorFlow Hub* models and try to improve the results.**
