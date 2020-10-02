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


import re
import string
import numpy as np
import pandas as pd
import random
import missingno
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, recall_score, plot_confusion_matrix

from wordcloud import WordCloud

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data = pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')


# In[ ]:


data.head()


# In[ ]:


# checking missing data in our dataframe.
missingno.matrix(data)


# * As we can see their are a lot of null values in our dataset, so we need to figure out something later about it.

# In[ ]:


print(data.columns)
data.describe()


# * From describing our data we get to know that their are 4 columns named as job_id, telecommuting, has_company_logo and has_questions features which have numerical data. So we can easily remove these columns as they are of no use in text classification problems.
# * We can also see one numerical feature 'fraudulent' is basically column on which our model will be trained and predicted.

# In[ ]:


# Now lets see how many jobs posted are fraud and real.
sns.countplot(data.fraudulent)
data.groupby('fraudulent').count()['title'].reset_index().sort_values(by='title',ascending=False)


# * From the plot we can see their are very few fraud jobs posted.
# * Our data is very much imbalanced so its a hard work to make a good classifier, we will try best :-)

# ### **Now let's fill the nan values and get rid of the columns which are of no use to make things simpler.**

# In[ ]:


columns=['job_id', 'telecommuting', 'has_company_logo', 'has_questions', 'salary_range', 'employment_type']
for col in columns:
    del data[col]

data.fillna(' ', inplace=True)


# In[ ]:


data.head()


# **Let's check which country posts most number of jobs.**

# In[ ]:


def split(location):
    l = location.split(',')
    return l[0]

data['country'] = data.location.apply(split)


# In[ ]:


country = dict(data.country.value_counts()[:11])
del country[' ']
plt.figure(figsize=(8,6))
plt.title('No. of job postings country wise', size=20)
plt.bar(country.keys(), country.values())
plt.ylabel('No. of jobs', size=10)
plt.xlabel('Countries', size=10)


# * Most number of jobs are posted by US.

# Let's check about which type of experience is required in most number of jobs.

# In[ ]:


experience = dict(data.required_experience.value_counts())
del experience[' ']
plt.bar(experience.keys(), experience.values())
plt.xlabel('Experience', size=10)
plt.ylabel('no. of jobs', size=10)
plt.xticks(rotation=35)
plt.show()


# In[ ]:


# title of jobs which are frequent.
print(data.title.value_counts()[:10])


# **Now we should combine our text in a single column to start cleaning our data.**

# In[ ]:


data['text']=data['title']+' '+data['location']+' '+data['company_profile']+' '+data['description']+' '+data['requirements']+' '+data['benefits']
del data['title']
del data['location']
del data['department']
del data['company_profile']
del data['description']
del data['requirements']
del data['benefits']
del data['required_experience']
del data['required_education']
del data['industry']
del data['function']
del data['country']


# In[ ]:


data.head()


# **Now lets see what type of words are frequent in fraud and actual jobs using wordclouds**

# In[ ]:


fraudjobs_text = data[data.fraudulent==1].text
actualjobs_text = data[data.fraudulent==0].text


# In[ ]:


STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize = (16,14))
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(str(" ".join(fraudjobs_text)))
plt.imshow(wc,interpolation = 'bilinear')


# In[ ]:


plt.figure(figsize = (16,14))
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(str(" ".join(actualjobs_text)))
plt.imshow(wc,interpolation = 'bilinear')


# # Cleaning Data

# * Creating a function that accepts a sentence as input and processes the sentence into tokens, performing lemmatization, lowercasing, and removing stop words.
# * The function that i have used to do these work is found here https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/, i know that i cant write so neat so i just taken those functions.

# In[ ]:


# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens


# In[ ]:


# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()


# In[ ]:


# creating our bag of words
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,3))


# * BoW converts text into the matrix of occurrence of words within a given document. It focuses on whether given words occurred or not in the document, and it generates a matrix that we might see referred to as a BoW matrix or a document term matrix.

# In[ ]:


# splitting our data in train and test
X_train, X_test, y_train, y_test = train_test_split(data.text, data.fraudulent, test_size=0.3)


# # Creating Model

# * We are creating a pipeline with three components: a cleaner, a vectorizer, and a classifier. The cleaner uses our predictors class object to clean and preprocess the text. The vectorizer uses countvector objects to create the bag of words matrix for our text. The classifier is an object that performs the logistic regression to classify the sentiments.

# 1. Logistic Regression

# In[ ]:


clf = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', clf)])

# fitting our model.
pipe.fit(X_train,y_train)


# In[ ]:


# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, predicted))
print("Logistic Regression Recall:", recall_score(y_test, predicted))


# In[ ]:


plot_confusion_matrix(pipe, X_test, y_test, cmap='Blues', values_format=' ')


# 2. Random Forest Classifier

# In[ ]:


clf = RandomForestClassifier()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', clf)])

# fitting our model.
pipe.fit(X_train,y_train)


# In[ ]:


# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Model Accuracy
print("Random Forest Accuracy:", accuracy_score(y_test, predicted))
print("Random Forest Recall:", recall_score(y_test, predicted))


# In[ ]:


plot_confusion_matrix(pipe, X_test, y_test, cmap='Blues', values_format=' ')


# **3. Support Vector Machine Classifier**

# In[ ]:


clf = SVC()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', clf)])

# fitting our model.
pipe.fit(X_train,y_train)


# In[ ]:


# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Model Accuracy
print("SVC Accuracy:", accuracy_score(y_test, predicted))
print("SVC Recall:", recall_score(y_test, predicted))


# In[ ]:


plot_confusion_matrix(pipe, X_test, y_test, cmap='Blues', values_format=' ')


# 4. XGBoost Classifier

# In[ ]:


clf = XGBClassifier()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', clf)])

# fitting our model.
pipe.fit(X_train,y_train)


# In[ ]:


# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Model Accuracy
print("XGBoost Accuracy:", accuracy_score(y_test, predicted))
print("XGBoost Recall:", recall_score(y_test, predicted))


# In[ ]:


plot_confusion_matrix(pipe, X_test, y_test, cmap='Blues', values_format=' ')


# ### * Sorry Guyz for creating Prediction section so long from next time i will do all classifiers in a loop and will try to implement tuning as i am still learning best way :-)
# ### * If you like the notebook please Upvote it.
# ### * Any kind of suggestion are appreciated.

# In[ ]:




