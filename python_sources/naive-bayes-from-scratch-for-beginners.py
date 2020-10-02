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


# ### Goal

# The goal of this notebook is to learn what is the Naive Bayes algorithm by implementing from the scratch the sklearn MultinomialNB class.

# ### Definition

# Naive Bayes is a classification model based on Bayes Theorem.
# Bayes Theorem universal example is to classify email messages between spam and ham.
# 
# The equations of Naive Bayes for this 'Real or Not?' is:
# 
# P(real | text) = (P(text | real) * P(real)) / P(text)<br>
# P(fake | text) = (P(text | fake) * P(fake)) / P(text)

# ### Load the data

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_X_y, check_array


# In[ ]:


train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
submit_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


# Labels data
train_y = train_df['target']


# In[ ]:


train_y.unique()


# ### Create the vocabulary, the word count and prepare the train_X

# In[ ]:


vocabulary = []
_ = [vocabulary.extend(x.split()) for i,x in enumerate(train_df['text'])]


# In[ ]:


vocabulary = np.array(vocabulary)


# In[ ]:


vocab = np.unique(vocabulary)


# In[ ]:


len(vocab)


# In[ ]:


vectorizer = CountVectorizer(vocabulary=vocab)
word_counts = vectorizer.fit_transform(train_df.text.to_numpy()).toarray()


# In[ ]:


train_x = pd.DataFrame(word_counts, columns=vocab).to_numpy()


# ### Implementation of NaiveBayes

# |  Variable      | Math     |  Description                                                      |
# |----------------|----------|-------------------------------------------------------------------|
# | prior          | P(y)     |  Probability of any random selected message belonging to a class  |
# | ik_word        | P(Xi&#124;y)  |  Likelihood of each word, conditional on message class            |
# | lk_message     | P(x&#124;y)   |  Likelihood of an entire message belonging to a particular class  |
# | normalize_term | P(x)     |  Likelihood of an entire message across all possible classes      |

# In[ ]:


class NaiveBayes():
    def __init__(self, alpha=1.0):
        self.prior = None
        self.word_counts = None
        self.lk_word = None
        self.alpha = alpha
        
    def fit(self, x, y):
        '''
        Fit the features and the labels
        Calculate prior, word_counts and lk_word
        '''
        x, y = check_X_y(x, y)
        n = x.shape[0]
        
        # calculate the prior - number of text belonging to a particular class (real or fake)
        x_per_class = np.array([x[y == c] for c in np.unique(y)])
        self.prior = np.array([len(x_class) / n for x_class in x_per_class])
        
        # calculate the likelihood for each word 'lk_word'
        self.word_counts = np.array([sub_arr.sum(axis=0) for sub_arr in x_per_class]) + self.alpha
        self.lk_word = self.word_counts / self.word_counts.sum(axis=1).reshape(-1, 1)
        
        return self
    
    def _get_class_numerators(self, x):
        '''
        Calculate for each class, the likelihood that an entire message conditional
        on the message belonging to a particular class (real or fake)
        '''
        n, m = x.shape[0], self.prior.shape[0]
        
        class_numerators = np.zeros(shape=(n, m))
        for i, word in enumerate(x):
            word_exists = word.astype(bool)
            lk_words_present = self.lk_word[:, word_exists] ** word[word_exists]
            lk_message = (lk_words_present).prod(axis=1)
            class_numerators[i] = lk_message * self.prior
        
        return class_numerators
    
    def _normalized_conditional_probs(self, class_numerators):
        '''
        Conditional probabilities = class_numerators / normalize_term
        '''
        # normalize term is the likelihood of an entire message (addition of all words in a row)
        normalize_term = class_numerators.sum(axis=1).reshape(-1,1)
        conditional_probs = class_numerators / normalize_term
        assert(conditional_probs.sum(axis=1) - 1 < 0.001).all(), 'rows should sum to 1'
        
        return conditional_probs
    
    def predict_proba(self, x):
        '''
        Return the probabilities for each class (fake or real)
        '''
        class_numerators = self._get_class_numerators(x)
        conditional_probs = self._normalized_conditional_probs(class_numerators)
        
        return conditional_probs
    

    def predict(self, x):
        '''
        Return the answer with the highest probability (argmax)
        '''
        return self.predict_proba(x).argmax(axis=1)


# In[ ]:


NaiveBayes().fit(train_x,train_y).predict(train_x)


# ### Prepare the submission.csv

# In[ ]:


submit_df.head()


# In[ ]:


# word counts for submit using vectorizer from sklearn
word_counts_submit = vectorizer.fit_transform(submit_df.text.to_numpy()).toarray()


# In[ ]:


submit_x = pd.DataFrame(word_counts_submit, columns=vocab).to_numpy()


# In[ ]:


# use naive bayes to predict the submission
result = NaiveBayes().fit(train_x,train_y).predict(submit_x)


# In[ ]:


final_result = list(map(list, zip(submit_df.id.to_numpy(), result)))


# In[ ]:


final_result[:5]


# In[ ]:


final_df = pd.DataFrame(final_result, columns=['id', 'target'])


# In[ ]:


final_df.to_csv("submission.csv",index=False)


# Thank you and don't forget to up-vote in order to support the community ;)
