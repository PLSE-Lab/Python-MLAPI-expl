#!/usr/bin/env python
# coding: utf-8

# In this kernel I'll make a simple logistic regression model to predict questions as sincere or insincere.

# In[ ]:


import pandas as pd
import numpy as np
import re
from textblob.classifiers import NaiveBayesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


print(df['target'].value_counts(), end = '\n\n')
print(sum(df['target'] == 1) / sum(df['target'] == 0) * 100, 'percent of questions are insincere.')


# The vast majority of questions are considered sincere, and any model we train to predict the sincerity of reviews should do significantly better than simply predicting every question as sincere (based on the above, that would be correct nearly 94% of the time).

# ### Training a Model

# I'll make a train and test set, with 80% of the data being the train set and 20% being the test set. The model will be trained on the training data and evaluated on the test data.

# In[ ]:


msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]


# Before feeding reviews into a model, they should be cleaned to remove punctuation, as that willl be useless from the computers point of view in trying to work out the sentiment of a text - for this case we're only interested in the words in a review. The code below (from https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184) gets rid of punctuation so a review is turned into words only.

# In[ ]:


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

train_clean = preprocess_reviews(train['question_text'])
test_clean = preprocess_reviews(test['question_text'])


# The next step is one hot encoding, where we turn each review into a very large matrix of 0's and 1's. A 0 would represent a certain word isn't included, whereas a 1 means that word is included. A short review of a few words would have a matrix of almost entirely 0's (ie. the vast majority of unique words across all reviews aren't present in the review), with just a small number of 1's. This is necessary for the logistic regression algorithm used below.

# In[ ]:


cv = CountVectorizer(binary=True)
cv.fit(train_clean)
X_train = cv.transform(train_clean)
X_test = cv.transform(test_clean)


# Now we can finally train a model on the train data with model.fit(). After that, model.predict() can be called on the test data to evaluate how good the model does.

# In[ ]:


target_train = train['target']
target_test = test['target']

model = LogisticRegression()
model.fit(X_train, target_train)
print("Accuracy: %s" % accuracy_score(target_test, model.predict(X_test)))


# With over 95% accuracy, this model clearly beats the naive baseline, but could clearly be improved. Finally to load up the Kaggle test.csv file and make predictions on the questions there, then save them into a submissions.csv file that will be evaluated by kaggle.

# In[ ]:


kaggle_test = pd.read_csv('../input/test.csv')
kaggle_test_clean = preprocess_reviews(kaggle_test['question_text'])
X_kaggle_test = cv.transform(kaggle_test_clean)
results = model.predict(X_kaggle_test)
submission = pd.DataFrame({"qid" : kaggle_test['qid'], "prediction" : results})
submission.to_csv("submission.csv", index=False)

