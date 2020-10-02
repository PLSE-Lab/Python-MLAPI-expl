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


# # Objection
# In this project I will implement some simple text processing and machine learning skills in order to be practicing and familiar with how to deal with data of text format. I would like to build a model to predict whether the rating (target) is good or bad based on the review (feature) it provided. The goal is to let the model learn the meaning in the text. 

# In[ ]:


train_df = pd.read_csv('../input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv')
test_df = pd.read_csv('../input/kuc-hackathon-winter-2018/drugsComTest_raw.csv')

train_df.head()


# # Distribution of the ratings

# In[ ]:


import matplotlib.pyplot as plt

plt.hist(train_df['rating']);


# # Extract feature and target
# For the purpose of this project, I only extract review column and rating column from the dataset.

# In[ ]:


# extract only review as feature, rating as target
X_train = train_df['review']
y_train = train_df['rating']
X_test = test_df['review']
y_test = test_df['rating']


# # Prediction1

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ngram_range=(1,2) so that the model can deal with text such as "not good" as one word
vect = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2))
X_train = vect.fit_transform(X_train.tolist())
X_test = vect.transform(X_test.tolist())


# In[ ]:


nb = MultinomialNB().fit(X_train, y_train)

train_pred = nb.predict(X_train)
test_pred = nb.predict(X_test)

print('Training Accuracy:', accuracy_score(y_train, train_pred))
print('Testing Accuracy:', accuracy_score(y_test, test_pred))


# It looks like the model is overfitting
# If we just predict the exact rating of each review, it might be hard for model to have a good prediction.
# Therefore, I decided to modify the rating columns to improve my model.

# # More Preprocessing
# To make the claddifier work better, I will only use ratings better than 7 and ratings worse than 4. The former one will be grouped as positive rating (1), and the latter one will be begative rating (0).

# In[ ]:


# only use rather extreme rating as predicting standard
train_df['new_rating'] = train_df[(train_df['rating'] > 7) | (train_df['rating'] < 4)]['rating']
test_df['new_rating'] = test_df[(test_df['rating'] > 7) | (test_df['rating'] < 4)]['rating']

# 1 is good rating, 0 is bad rating
train_df['new_rating'] = train_df['new_rating'].apply(lambda x: 1 if x > 7 else 0)
test_df['new_rating'] = test_df['new_rating'].apply(lambda x: 1 if x > 7 else 0)

train_df.head()


# # Prediction2

# In[ ]:


X_train = train_df['review']
X_test = test_df['review']
y_train = train_df['new_rating']
y_test = test_df['new_rating']


# In[ ]:


vect = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2))
X_train = vect.fit_transform(X_train.tolist())
X_test = vect.transform(X_test.tolist())

nb = MultinomialNB().fit(X_train, y_train)
train_pred = nb.predict(X_train)
test_pred = nb.predict(X_test)

print('Training Accuracy:', accuracy_score(y_train, train_pred))
print('Testing Accuracy:', accuracy_score(y_test, test_pred))


# Now we got a much better prediction accuracy due to preprocessing of rating columns. 
# However, the model is still overfitting. So, I am going to work on the model's hyperparameter to make it better.

# # Hyperparameter (alpha)
# the primary hyperparameter of MultinomialNB is alpha. I attempt to use 7 different alpha to see which one performs the best.

# In[ ]:


alphas = np.array([0.001, 0.01, 0.1, 0, 1, 10, 100])

train_accu = []
test_accu = []

for alpha in alphas:
    nb = MultinomialNB(alpha=alpha).fit(X_train, y_train)
    train_pred = nb.predict(X_train)
    test_pred = nb.predict(X_test)
    
    train_accu.append(accuracy_score(y_train, train_pred))
    test_accu.append(accuracy_score(y_test, test_pred))

print('Training Accuracies')
print(train_accu)
print('Testing Accuracies')
print(test_accu)


# In[ ]:


plt.figure(figsize=(10, 6))
plt.plot(list(range(len(alphas))),train_accu, label='Training');
plt.plot(list(range(len(alphas))), test_accu, label='Testing');
plt.xticks(list(range(len(alphas))), alphas);
plt.xlabel('Alpha');
plt.ylabel('Accuracy')
plt.legend();


# It looks like alpha did not have lots of impact on accuracy in the range from 0.001 to 1.

# # What I have learned
# In this project I have learned how to use CountVectorizer to extract meanings behind the reviews (either positive or negative in this case) and how hyperparameter impacts the ML model, although it did not differ a lot in this project.
