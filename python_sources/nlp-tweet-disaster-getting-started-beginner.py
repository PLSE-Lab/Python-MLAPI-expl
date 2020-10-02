#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Importing data sets

# In[ ]:


train=pd.read_csv(r"/kaggle/input/nlp-getting-started/train.csv")
test=pd.read_csv(r"/kaggle/input/nlp-getting-started/test.csv")
sample_submission=pd.read_csv(r"/kaggle/input/nlp-getting-started/sample_submission.csv")


# # Data Analysis

# In[ ]:


train.head()


# In[ ]:


train.shape


# *Lets check whether the data have missing values or not.*

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# We can see that keyword and location features have missing values in both test and train data

# ***Exploring the target column***

# In[ ]:


train['target'].value_counts()


# The train data is classified into 0 and 1 and we can also say that the target column is balanced properly with almost equal no of 0's and 1's therefore there is no chance of overfitting. 

# In[ ]:


import seaborn as sns


# In[ ]:


sns.barplot(train['target'].value_counts().index,train['target'].value_counts(),palette='rocket')


# Now lets look at text and target feature in the train data

# In[ ]:


train[['text','target']].head(20)


# In[ ]:


#lets look separately how non disaster tweet(0) look like

for i in range(10):
    res=train[train['target']==0]['text'].values[i] 
    print(res)


# Non disaster tweets looks like so casual tweets

# In[ ]:


#lets look separately how  disaster tweet(1) look like

for i in range(10):
    res=train[train['target']==1]['text'].values[i] 
    print(res)


# Disaster tweets look like some serious tweets

# In[ ]:





# ### Lets look at keyword feature in train dataset

# In[ ]:


pd.DataFrame(train['keyword'].value_counts()[:20])


# In[ ]:


sns.barplot(y=train['keyword'].value_counts()[:20].index,x=train['keyword'].value_counts()[:20],orient='h')


# **Even though the location feature  has a number of missing values, let's see the top 20 locations present in the dataset. Since some of the locations are repeated, this will require some bit of cleaning.**

# In[ ]:


pd.DataFrame(train['location'].value_counts()).head(20)


# Here we can observe that the location feature contains city names as well as country names

# In[ ]:


# Replacing the ambigious locations name with Standard names
train['location'].replace({'United States':'USA',
                           'New York':'USA',
                            "London":'UK',
                            "Los Angeles, CA":'USA',
                            "Washington, D.C.":'USA',
                            "California":'USA',
                             "Chicago, IL":'USA',
                             "Chicago":'USA',
                            "New York, NY":'USA',
                            "California, USA":'USA',
                            "FLorida":'USA',
                            "Nigeria":'Africa',
                            "Kenya":'Africa',
                            "Everywhere":'Worldwide',
                            "San Francisco":'USA',
                            "Florida":'USA',
                            "United Kingdom":'UK',
                            "Los Angeles":'USA',
                            "Toronto":'Canada',
                            "San Francisco, CA":'USA',
                            "NYC":'USA',
                            "Seattle":'USA',
                            "Earth":'Worldwide',
                            "Ireland":'UK',
                            "London, England":'UK',
                            "New York City":'USA',
                            "Texas":'USA',
                            "London, UK":'UK',
                            "Atlanta, GA":'USA',
                            "Mumbai":"India"},inplace=True)

sns.barplot(y=train['location'].value_counts()[:5].index,x=train['location'].value_counts()[:5],
            orient='h')


# # Data cleaning

# In[ ]:


import re


# In[ ]:


# Applying a first round of text cleaning techniques since we have seen the text is having noise

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    #text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Applying the cleaning function to both test and training datasets
train['text'] = train['text'].apply(lambda x: clean_text(x))
test['text'] = test['text'].apply(lambda x: clean_text(x))

# Let's take a look at the updated text
train['text'].head()


# # Tokenization

# Tokenization is a process that splits an input sequence into so-called tokens where the tokens can be a word, sentence, paragraph etc. Base upon the type of tokens we want, tokenization can be of various types, for instance

# In[ ]:


import nltk


# In[ ]:


text = "Are you coming , aren't you"
tokenizer1 = nltk.tokenize.WhitespaceTokenizer()
tokenizer2 = nltk.tokenize.TreebankWordTokenizer()
tokenizer3 = nltk.tokenize.WordPunctTokenizer()
tokenizer4 = nltk.tokenize.RegexpTokenizer(r'\w+')

print("Example Text: ",text)
print("------------------------------------------------------------------------------------------------")
print("Tokenization by whitespace:- ",tokenizer1.tokenize(text))
print("Tokenization by words using Treebank Word Tokenizer:- ",tokenizer2.tokenize(text))
print("Tokenization by punctuation:- ",tokenizer3.tokenize(text))
print("Tokenization by regular expression:- ",tokenizer4.tokenize(text))


# In[ ]:


# Tokenizing the training and the test set
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))
test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))
train['text'].head()


# > **StopWord Removal**

# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


def remove_stopwords(text):
    """
    Removing stopwords belonging to english language
    
    """
    words = [w for w in text if w not in stopwords.words('english')]
    return words


train['text'] = train['text'].apply(lambda x : remove_stopwords(x))
test['text'] = test['text'].apply(lambda x : remove_stopwords(x))
train.head()


# **Stemming And Lemmatization**

# In[ ]:


from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# In[ ]:


# Stemming and Lemmatization examples
text = "feet cats wolves talked"

tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokens = tokenizer.tokenize(text)

# Stemmer
stemmer = nltk.stem.PorterStemmer()
print("Stemming the sentence: ", " ".join(stemmer.stem(token) for token in tokens))

# Lemmatizer
lemmatizer=nltk.stem.WordNetLemmatizer()
print("Lemmatizing the sentence: ", " ".join(lemmatizer.lemmatize(token) for token in tokens))


# It is important to note here that stemming and lemmatization sometimes donot necessarily improve results as at times we donot want to trim words but rather preserve their original form. Hence their usage actually differs from problem to problem. For this problem, I will not use these techniques.

# In[ ]:


# After preprocessing, the text format
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

train['text'] = train['text'].apply(lambda x : combine_text(x))
test['text'] = test['text'].apply(lambda x : combine_text(x))
train['text']
train.head()


# # Bag of Words

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


#  count vectorizer
count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train['text'])
test_vectors = count_vectorizer.transform(test["text"])


# In[ ]:





# # TF-IDF

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


#Tf-Idf
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(train['text'])
test_tfidf = tfidf.transform(test["text"])


# # **Lets Build the Model now**

# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[ ]:


#fitting the model to countvectorizer
clf=LogisticRegression(C=1.0)
scores=cross_val_score(clf,train_vectors,train['target'],cv=7,scoring="f1")
scores


# In[ ]:


scores.mean()


# In[ ]:


clf.fit(train_vectors,train['target'])


# In[ ]:


# Fitting a simple Logistic Regression on TFIDF
clf_tfidf = LogisticRegression(C=1.0)
scores = cross_val_score(clf_tfidf, train_tfidf, train["target"], cv=5, scoring="f1")
scores


# In[ ]:


scores.mean()


# It appears the countvectorizer gives a better performance than TFIDF in this case.

# In[ ]:





# **Naive Bayes classifier**

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
# Fitting a simple Naive Bayes on Counts
clf_NB=MultinomialNB()
scores=cross_val_score(clf_NB,train_vectors,train['target'],cv=5,scoring="f1")
scores


# In[ ]:


scores.mean()


# In[ ]:


clf_NB.fit(train_vectors,train["target"])


# In[ ]:


# Fitting a simple Naive Bayes on TFIDF
clf_NB_TFIDF = MultinomialNB()
scores =cross_val_score(clf_NB_TFIDF, train_tfidf, train["target"], cv=5, scoring="f1")
scores


# In[ ]:


scores.mean()


# In[ ]:


clf_NB_TFIDF.fit(train_tfidf, train["target"])


# In[ ]:





# **Make Submission**

# I found Naive Bayes classification of countvectorizer is giving good accuracy,I decided to predict test records using it.

# In[ ]:


sample_submission['target']=clf_NB.predict(test_vectors)


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.to_csv("submission.csv", index=False)


# In[ ]:




