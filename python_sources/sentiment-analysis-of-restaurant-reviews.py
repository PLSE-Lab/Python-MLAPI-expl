#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis of Restaurant Reviews

# The purpose of this analysis is to build a prediction model to predict whether a review on the restaurant is positive or negative. To do so, we will work on Restaurant Review dataset, we will load it into predicitve algorithms Multinomial Naive Bayes, Bernoulli Naive Bayes and Logistic Regression. In the end, we hope to find a "best" model for predicting the review's sentiment.
# 
# Dataset: [Restaurant_Reviews.tsv](https://www.kaggle.com/hj5992/restaurantreviews) is a dataset from Kaggle datasets which consists of 1000 reviews on a restaurant.
# 
# To build a model to predict if review is positive or negative, following steps are performed.
# 
# * Importing Dataset
# * Preprocessing Dataset
# * Vectorization
# * Training and Classification
# * Analysis Conclusion

# ### Importing Dataset

# Importing the Restaurant Review dataset using pandas library.

# In[ ]:


# Importing the libraries
import numpy as np
import pandas as pd


# In[ ]:


# Importing the dataset
dataset = pd.read_csv('../input/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# ### Preprocessing Dataset

# Each review undergoes through a preprocessing step, where all the vague information is removed.
# 
# * Removing the Stopwords, numeric and speacial charecters.
# * Normalizing each review using the approach of stemming.

# In[ ]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# ### Vectorization

# From the cleaned dataset, potential features are extracted and are converted to numerical format. The vectorization techniques are used to convert textual data to numerical format. Using vectorization, a matrix is created where each column represents a feature and each row represents an individual review.

# In[ ]:


# Creating the Bag of Words model using CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# ### Training and Classification

# Further the data is splitted into training and testing set using Cross Validation technique. This data is used as input to classification algorithm.
# 
# **Classification Algorithms:**
# 
# Algorithms like Decision tree, Support Vector Machine, Logistic Regression, Naive Bayes were implemented and on comparing the evaluation metrics two of the algorithms gave better predictions than others.
# 
# * Multinomial Naive Bayes
# * Bernoulli Naive Bayes
# * Logistic Regression

# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# **Multinomial NB**

# In[ ]:


# Multinomial NB

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
score1 = accuracy_score(y_test,y_pred)
score2 = precision_score(y_test,y_pred)
score3= recall_score(y_test,y_pred)
print("\n")
print("Accuracy is ",round(score1*100,2),"%")
print("Precision is ",round(score2,2))
print("Recall is ",round(score3,2))


# **Bernoulli NB**

# In[ ]:


# Bernoulli NB

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB(alpha=0.8)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
score1 = accuracy_score(y_test,y_pred)
score2 = precision_score(y_test,y_pred)
score3= recall_score(y_test,y_pred)
print("\n")
print("Accuracy is ",round(score1*100,2),"%")
print("Precision is ",round(score2,2))
print("Recall is ",round(score3,2))


# **Logistic Regression**

# In[ ]:


# Logistic Regression

# Fitting Logistic Regression to the Training set
from sklearn import linear_model
classifier = linear_model.LogisticRegression(C=1.5)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
score1 = accuracy_score(y_test,y_pred)
score2 = precision_score(y_test,y_pred)
score3= recall_score(y_test,y_pred)
print("\n")
print("Accuracy is ",round(score1*100,2),"%")
print("Precision is ",round(score2,2))
print("Recall is ",round(score3,2))


# ### Analysis and Conclusion

# In this study, an attempt has been made to classify sentiment analysis for restaurant reviews using machine learning techniques. Two algorithms namely Multinomial Naive Bayes and Bernoulli Naive Bayes are implemented.
# 
# Evaluation metrics used here are accuracy, precision and recall.
# 
# Using Multinomial Naive Bayes,
# 
# * Accuracy of prediction is 77.67%.
# * Precision of prediction is 0.78.
# * Recall of prediction is 0.77.
# 
# Using Bernoulli Naive Bayes,
# 
# * Accuracy of prediction is 77.0%.
# * Precision of prediction is 0.76.
# * Recall of prediction is 0.78.
# 
# Using Logistic Regression,
# 
# * Accuracy of prediction is 76.67%.
# * Precision of prediction is 0.8.
# * Recall of prediction is 0.71.
# 
# From the above results, Multinomial Naive Bayes is slightly better method compared to Bernoulli Naive Bayes and Logistic Regression, with 77.67% accuracy which means the model built for the prediction of sentiment of the restaurant review gives 77.67% right prediction.
