# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 12:26:09 2018

@author: aswin.senthilnathan
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../input/amazon_alexa.tsv',delimiter='\t',quoting=3)
X=dataset.iloc[:,[3,4]]
# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0, 3150):
    review = re.sub('[^a-zA-Z]', ' ', X['verified_reviews'][i])
    review = review.lower()
    review = review.split()
    lem=    WordNetLemmatizer()
    #ps = PorterStemmer()
    review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

