#!/usr/bin/env python
# coding: utf-8

# # First things first
# 
# First of all, we have to convert categorical data into a numerical form before we can pass it on to a machine learning algorithm. We doing so using the **bag-of-words** model. The idea behind it is quite simple and can be summarized as follows:
# 1. We create a **vocabulary** of unique **tokens** - for example, words - from the entire set of documents.
# 2. We construct a feature vector from each document that contains the counts of how ofter each word occurs in the particular document.
# 
# ### Transforming words into feature vectors
# 
# To construct the bag-of-words model, we can use the `CountVectorizer` class implemented in `scikit-learn`, as follows.

# In[ ]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining and the weather is sweet'
])

bag = count.fit_transform(docs)


# ### Assessing word relevancy via term frequency-inverse document frequency
# 
# This technique can be used to downweight frequently occuring words that don't contain useful or discriminatory information in the feature vectors. `scikit-learn` implements it with the `TfidfTransformer`.

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()
np.set_printoptions(precision=2)

print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


# # Cleaning text data
# 
# The first important step - before we build our bag-of-words model - its to clean the text data by stripping it of all unwanted characters. To illustrate, lets display the last 50 characters from a random document of the dataset:

# In[ ]:


import pandas as pd

df = pd.read_csv('../input/movie_data.csv')

df.loc[49941, 'review'][-50:]


# We will remove all HTML markup as well as punctuation and other non-letter characters and keep only emoticon characters since those are certainly useful for sentiment analysis. To accomplish that, we will use Python's regular expressions library `re`.

# In[ ]:


import re

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text = text + " ".join(emoticons).replace('-', '')
    return text

preprocessor("</a>This :) is :( a test :-)!")


# In[ ]:


df['review'] = df['review'].apply(preprocessor)

print(df.tail())


# # Processing documents into tokens
# 
# Now that we successfully prepared the dataset, we need to think how to split the text into individual elements. We can do so:

# In[ ]:


def tokenizer(text):
    return text.split()

tokenizer('runners like running and thus they run')


# Another strategy is **word stemming**, which is the process of transforming a word into its root form that allow us to map related words to the same stem. We will use **Porter stemming algorithm** implemented by `nltk` package.

# In[ ]:


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tokenizer_porter('runners like running and thus they run')


# Before we jump into the training of a machine learning model using the bag-of-word, lets remove those extremely common words (called **stop-words**) that doesn't add useful information to the text.

# In[ ]:


import nltk
nltk.download('stopwords')


# In[ ]:


from nltk.corpus import stopwords

stop = stopwords.words('english')

[w for w in tokenizer_porter('a runner likes running and runs a lot') if w not in stop]


# # Training a logistic regression model for document classification
# 
# In this section, we will train a logictic regression model to classify the movie reviews into positive and negative reviews. First, lets divide the `DataFrame` of cleaned text document into 50:50 ratio for training and testing.

# In[ ]:


X = df.review
y = df.sentiment

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


# Next we will use a `GridSearchCV` to hypertune our logistic regression (besides his name, its a classification model) model using 5-fold cross-validation:

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

param_grid = [{'vect__ngram_range': [(1,1)],
              'vect__stop_words': [stop, None],
              'vect__tokenizer': [tokenizer, tokenizer_porter],
              'clf__penalty': ['l1', 'l2'],
              'clf__C': [1.0, 10.0, 100.0]},
             {'vect__ngram_range': [(1,1)],
              'vect__stop_words': [stop, None],
              'vect__tokenizer': [tokenizer, tokenizer_porter],
              'vect__use_idf': [False],
              'vect__norm': [None],
              'clf__penalty': ['l1', 'l2'],
              'clf__C': [1.0, 10.0, 100.0]}]

lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s' % gs_lr_tfidf.best_params_)

