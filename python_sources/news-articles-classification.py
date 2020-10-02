#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

import time
import re
import pickle
from string import punctuation
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier


# # Loading Dataset

# In[ ]:


df = pd.read_json('/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json', lines=True)
df.head()


# Let's check is their any Nan values which we should take care of.

# In[ ]:


df.isna().sum()


# Our dataset have no empty values.

# In[ ]:


# top 5 Categories of news in our datset
df.category.value_counts()[:5]


# In[ ]:


# our unique labels of text which are to be classified.
np.unique(df.category)


# In[ ]:


df['text'] = df['headline'] + df['short_description'] + df['authors']
df['label'] = df['category']
del df['headline']
del df['short_description']
del df['date']
del df['authors']
del df['link']
del df['category']


# In[ ]:


df.head(10)


# Let's check total number of words in our dataset.

# In[ ]:


df['text'].apply(lambda x: len(x.split(' '))).sum()


# In[ ]:


def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# # Cleaning Data

# In this section we just removed lemmatization and cleaned our words by removing stopwords.
# 
# #### Lemmatization : It is the process of grouping together the different inflected forms of a word so they can be analysed as a single item. It is similar to stemming but it brings context to the words. So it links words with similar meaning to one word.

# In[ ]:


REMOVE_SPECIAL_CHARACTER = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))
punctuation = list(punctuation)
STOPWORDS.update(punctuation)

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # part 1
    text = text.lower() # lowering text
    text = REMOVE_SPECIAL_CHARACTER.sub('', text) # replace REPLACE_BY_SPACE symbols by space in text
    text = BAD_SYMBOLS.sub('', text) # delete symbols which are in BAD_SYMBOLS from text
    
    # part 2
    clean_text = []
    for w in word_tokenize(text):
        if w.lower() not in STOPWORDS:
            pos = pos_tag([w])
            new_w = lemmatizer.lemmatize(w, pos=get_simple_pos(pos[0][1]))
            clean_text.append(new_w)
    text = " ".join(clean_text)
    
    return text


# In[ ]:


df['text'] = df['text'].apply(clean_text)


# I am saving the cleaned data so that i dont have to go through this process again as it takes a lot of time.

# In[ ]:


# df.to_csv('news_text_cleaned.csv', columns=['text', 'label'])


# In[ ]:


df.head(10)


# Let's check how many words are remaining after cleaning the data.

# In[ ]:


df['text'].apply(lambda x: len(x.split(' '))).sum()


# # Splitting Data

# In[ ]:


X=df.text
y=df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Comparing Models

# In[ ]:


# Creating Models
models = [('Logistic Regression', LogisticRegression(max_iter=500)),('Random Forest', RandomForestClassifier()),
          ('Linear SVC', LinearSVC()), ('Multinomial NaiveBayes', MultinomialNB()), ('SGD Classifier', SGDClassifier())]

names = []
results = []
model = []
for name, clf in models:
    pipe = Pipeline([('vect', CountVectorizer(max_features=30000, ngram_range=(1, 2))),
                    ('tfidf', TfidfTransformer()),
                    (name, clf),
                    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    
    names.append(name)
    results.append(accuracy)
    model.append(pipe)
    
    msg = "%s: %f" % (name, accuracy)
    print(msg)


# # Saving Model

# In[ ]:


# Logistic Regression
filename = 'model_lr.sav'
pickle.dump(model[0], open(filename, 'wb'))

# Linear SVC
filename = 'model_lin_svc.sav'
pickle.dump(model[2], open(filename, 'wb'))


# # Loading and Testing

# In[ ]:


lr_model = pickle.load(open('model_lin_svc.sav', 'rb'))


# In[ ]:


text1 = 'Could have been best all-rounder India ever produced in ODIs: Irfan Pathan'
text2 = "Ashwin Kkumar dances to Kamal Haasan's Annathe song on a treadmill. Actor is proud"
print(lr_model.predict([text1, text2]))


# We can see our model predicted correct for text2 but wrong for text1 which should be a Sports category.

# ## If you like this notebook please consider upvoting it. Any suggestion for improvement is appreciated :-). Thanks for giving your time.
