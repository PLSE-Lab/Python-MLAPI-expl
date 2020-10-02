#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Import the dataset
dataset = pd.read_json('../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json', lines= True)


# In[ ]:


# Lets see how our data looks
dataset.head()


# In[ ]:


# Drop article_link column
dataset = dataset.drop(axis= 1, columns= 'article_link')


# In[ ]:


# Lets verify if the column was dropped
dataset.head()


# **NOW Lets apply some NLP Techniques **

# In[ ]:


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[ ]:


# Lets first get the string in each headline and then remove stop words using PorterStemmer
corpus = []
for i in range(0,26709):
    headline = re.sub('[^a-zA-Z]', ' ', dataset['headline'][i]) 
    headline = headline.lower()
    headline = headline.split()
    porterStemmer = PorterStemmer()
    headline = [porterStemmer.stem(word) for word in headline if not word in set(stopwords.words('english'))]
    headline = ' '.join(headline)
    corpus.append(headline)


# In[ ]:


corpus[0]


# Hurray!!, we have actually removed stop words from the text

# In[ ]:


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[ ]:


# Fit Logistic Regression Model to our training dataset
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)


# In[ ]:


# Make predictions
y_pred = regressor.predict(X_test)


# In[ ]:


# Analyze the results
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)


# In[ ]:


print(f'Accuracy with Logistic Regression model is {accuracy*100} %')


# In[ ]:


# Lets analyze the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


cm


# In[ ]:


# Lets check the precision
precision = cm[0][0]/sum(cm[0])
print(f'Precision for the model is {precision*100}')


# In[ ]:


# Now lets check the recall
recall = cm[0][0]/(cm[0][0]+cm[1][0])
print(f'Recall for the model is {recall*100}')


# In[ ]:


# Finally!, lets see the F1-Score
f1_score = 2 * ((precision*recall)/(precision+recall))
print(f'F1-Score :- {f1_score}')


# 

# **Further Improvements** :- Getting optimum values of parameters for LogisticRegression Model to enhance the performance or try out different model with the dataset and analyze the results.!!
