# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Importing the dataset
dataset = pd.read_csv('..//input//dataset//data3.txt', delimiter = '\t', quoting = 2)
dataset = dataset.iloc[:,0:4]
# We have encoded discount 5% as 0, discount 12% as 1, discount 18% as 2 and discount 28% as 3

# Formatting the texts
import re
corpus = []
for i in range(0, 9648):
    review = dataset['Product ID'][i]
    review = review.split()
    review = ' '.join(review)
    corpus.append(review)
    
cor = []
for i in range(0,9648):
    review = dataset["Date"][i]
    review = review.split()
    review = ' '.join(review)
    cor.append(review)

corp = []
for i in range(0,9648):
    review = dataset["Customer"][i]
    review = review.split()
    review = ' '.join(review)
    corp.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 9648)
cv2 = CountVectorizer(max_features = 9648)
cv3 = CountVectorizer(max_features = 9648)
X1 = cv.fit_transform(cor).toarray()
X2 = cv2.fit_transform(corp).toarray()
X3 = cv3.fit_transform(corp).toarray()
y = dataset.iloc[:, 3].values
train = np.concatenate((X1,X2,X3),axis = 1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.10,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 150, criterion = "entropy",random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the test set
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

# kfold validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,X = X_train,y = y_train,cv = 10,n_jobs=-1)
accuracies

# Importing the sample
score_df = pd.read_csv("..//input//dataset//testz.txt", delimiter = "\t")
score_df = score_df.iloc[:,1:]
# we have imported the testz file extracting data from other files like product details in the order of the Bill ID 
train_d = pd.read_csv("..//input//dataset//data3.txt",delimiter = "\t")
train_df = train_d.iloc[:,0:3]
y2 = train_d.iloc[:,3]
train_df['label'] = 'train'
score_df['label'] = 'score'

# Concat
concat_df = pd.concat([train_df , score_df])

# Create your dummies
features_df = pd.get_dummies(concat_df, columns=['Date', 'Customer','Product ID'], dummy_na=True)

# Split your data
train_df = features_df[features_df['label'] == 'train']
score_df = features_df[features_df['label'] == 'score']

# Drop your labels
train_df = train_df.drop('label', axis=1)
score_df = score_df.drop('label', axis=1)
train1 = train_df
test = score_df


# Building the model to predict the sample 
from sklearn.ensemble import RandomForestClassifier
classifier_final = RandomForestClassifier(n_estimators = 150, criterion = "entropy",random_state = 0)
classifier_final.fit(train1,y2)

# Predicitng sample
test_predict = classifier_final.predict(test)
print(test_predict)
# This prediction is in order of bill ID. In excel we have converted 0,1,2,3 into 0 and 1 as in the original train file.