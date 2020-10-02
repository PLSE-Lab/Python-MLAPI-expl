#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import logit, expit


# # Read the data
# 
# There are six classes in the data and also some NA values

# In[2]:


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']


# # Combine the comments from both train and test class for creating vocabulary
# 
# There are many words in the test set which differs from the train data. So combining the data into one frame to get all the words and then creating the vocabulary to get feature vector helps to improve the accuracy.
# 
# the main code is [here](https://www.kaggle.com/thousandvoices/logistic-regression-with-words-and-char-n-grams)
# 

# In[3]:


all_text = pd.concat([train_text, test_text])


# # Apply count vectorizer on word level
# 
# Firstly as there are six classes then we dont know if only the word level features will be enough so for better accuracy we are going to take the char level features too.
# 

# In[4]:


word_vectorizer = CountVectorizer(stop_words = 'english',analyzer='word')
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)


# # and on char level
# 
# Taking the char level features.
# 

# In[5]:


char_vectorizer = CountVectorizer(stop_words = 'english',analyzer='char')
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)


# # Combine word and char features

# In[6]:


train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])


# # Train the classifier and get the CV-Score

# In[7]:


losses = []
predictions = {'id': test['id']}
for class_name in class_names:
    train_target = train[class_name]
    classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=100, max_features=1000, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

    cv_loss = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='f1_micro'))
    losses.append(cv_loss)
    print('CV score for class {} is {}'.format(class_name, cv_loss))

    classifier.fit(train_features, train_target)
    predictions[class_name] = expit(logit(classifier.predict_proba(test_features)[:, 1]))

print('Total CV score is {}'.format(np.mean(losses)))


# # Write predictions in submission.csv

# In[8]:


submission = pd.DataFrame.from_dict(predictions)
submission.to_csv('submission.csv', index=False)

