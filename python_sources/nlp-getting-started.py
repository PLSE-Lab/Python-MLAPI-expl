#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


# scientific calculation and data analysis
import numpy as np
import pandas as pd
 
# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# missingno
import missingno

# SKLearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
## Model
from sklearn.naive_bayes import GaussianNB

# NLP
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# WordCloud
from wordcloud import WordCloud

# basic imports
import string


# ## Load Data

# In[ ]:


train = pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv('../input/nlp-getting-started/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Data Cleaning

# ### Missing Value

# In[ ]:


# train missing matrix
missingno.matrix(train)
plt.show()


# In[ ]:


# test missing matrix
missingno.matrix(test)
plt.show()


# In[ ]:


# missing bar plot train
missingno.bar(train)
plt.show()


# In[ ]:


# missing bar plot test
missingno.bar(test)
plt.show()


# In[ ]:


# null count in train and test
null_vals = pd.DataFrame(columns=['train', 'test'])
null_vals['train'] = train.isnull().sum()
null_vals['test'] = test.isnull().sum()
null_vals


#  The output shows <b>location</b> has many null value and <b>keyword</b> are few.

# In[ ]:


# drop location features
train.drop('location', axis=1, inplace=True)


# In[ ]:


# drop nan rows
train.dropna(axis=0, inplace=True)


# In[ ]:


# check missing value in train dataset
train.isnull().sum()


# ## Exploratory data analysis

# Real/Fake tweets: Let's plot count plot and Pie plot

# In[ ]:


fig, ax = plt.subplots(1,2)
sns.countplot(x='target', data=train, ax=ax[0])
ax[1].pie(train.target.value_counts(), labels=['Not Real', 'Real'], autopct='%1.1f%%')
plt.show()


# ## Data Pre-processing

# 1. Remove punctuations
# 2. Lowercase and alphanumeric
# 3. Remove stopword

# In[ ]:


# create remove_punctuation functionabs
def remove_punctuations(text):
    return "".join([c for c in text if c not in string.punctuation])


# In[ ]:


# Apply to text feature
train['text'] = train['text'].apply(lambda x: remove_punctuations(x))


# In[ ]:


# create lower_apha_num: convert to lower case and remove numerucal value
def lower_alpha_num(text):
    return [word for word in word_tokenize(text.lower()) if word.isalpha()]


# In[ ]:


# Apply lower_apha_num
train['text'] = train['text'].apply(lambda x: lower_alpha_num(x))


# In[ ]:


def remove_stopword(text):
    return [w for w in text if w not in stopwords.words('english')]


# In[ ]:


train['text'] = train['text'].apply(lambda x: remove_stopword(x))


# In[ ]:


train.head()


# ### Lemmatizing

# In[ ]:


# Initiate Lamitizer
lemmatizer = WordNetLemmatizer()

def word_lemmatizer(text):
    lem_text = " ".join([lemmatizer.lemmatize(i) for i in text])
    return lem_text


# In[ ]:


train['text'] = train['text'].apply(lambda x: word_lemmatizer(x))


# In[ ]:


train.head()


# In[ ]:


X = train['text']
y = train['target']


# ### TFIDF

# In[ ]:


vectorizer_tfidf = TfidfVectorizer()


# In[ ]:


X = vectorizer_tfidf.fit_transform(X)


# In[ ]:


pd.DataFrame(X.A, columns=vectorizer_tfidf.get_feature_names()).head()


# ### Train Test Split

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X.A, y, test_size=0.3)


# ## Model

# In[ ]:


classifier = GaussianNB()


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


classifier.score(X_train, y_train)


# In[ ]:


classifier.score(X_val, y_val)


# In[ ]:




