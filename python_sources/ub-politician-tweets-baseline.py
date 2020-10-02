#!/usr/bin/env python
# coding: utf-8

# # Politician Tweets - Classification Baseline

# In[ ]:


# Import essentials
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# You should see `train.xlsx` and `test.xlsx` here


# ## 1. Load Data

# In[ ]:


train_df = pd.read_excel('../input/train.xlsx') # Load the `train` file
train_df.sample(frac=0.1)[:10] # Show a sample of the dataset


# In[ ]:


test_df = pd.read_excel('../input/test.xlsx') # Load the `test` file
test_df.sample(frac=0.1)[:10] # Show a sample of the dataset


# ## 2. Are you able to manually classify these tweets?

# In[ ]:


for x in test_df.sample(frac=0.1)[:5]['text']:
    print(x)
    print("---------------")


# ## 3. Data Processing

# In[ ]:


import re
from stop_words import get_stop_words

def stop_words():
    """Retrieve the stop words for vectorization -Feel free to modify this function
    """
    return get_stop_words('es') + get_stop_words('ca') + get_stop_words('en')


def filter_mentions(text):
    """Utility function to remove the mentions of a tweet
    """
    return re.sub("@\S+", "", text)


def filter_hashtags(text):
    """Utility function to remove the hashtags of a tweet
    """
    return re.sub("#\S+", "", text)


# In[ ]:


# Preprocess data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words=stop_words())
X = vectorizer.fit_transform(train_df['text']).toarray()
y = train_df['party'].values


# ## 4. Classification - Bernoulli Naive Bayes

# In[ ]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# My first Naive Bayes classifier!
clf = BernoulliNB()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print(np.mean([prediction == y_test]))


# ## 5. Create submission file
# 
# To create submission file we need to predict the categories for the test tweets, to do so we need to:
# 1. Train a classifier with `train.xslx` data.
# 2.  Predict the labels for `test.xlsx` tweets
# 3. Generate an output file in csv format
# 4. Upload the results to Kaggle

# In[ ]:


# 1 Train the classifier
clf = BernoulliNB()
clf.fit(X_train, y_train)

# 2 Predict the data (We need to tokenize the data using the same vectorizer object)
X_test = vectorizer.transform(test_df['text']).toarray()
prediction = clf.predict(X_test)

# 3 Create a the results file
output = pd.DataFrame({'Party': prediction})
output.index.name = 'Id'
output.to_csv('sample_submission.csv')

# TIP - Copy and paste this function to generate the output file in your code
def save_submission(prediction):
    import datetime
    t = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    output = pd.DataFrame({'Party': prediction})
    output.index.name = 'Id'
    output.to_csv(f'sample_submission{t}.csv')


# In[ ]:


output[:10] # This is how your result file might look like

