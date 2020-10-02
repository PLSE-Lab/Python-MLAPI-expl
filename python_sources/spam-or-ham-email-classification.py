# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Source code and Expalnation: https://github.com/Balakishan77/Spam-Email-Classifier

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

dataset = pd.read_csv(r'../input/emails.csv')
dataset .columns #Index(['text', 'spam'], dtype='object')
dataset.shape  #(5728, 2)

#Checking for duplicates and removing them
dataset.drop_duplicates(inplace = True)
dataset.shape  #(5695, 2)
#Checking for any null entries in the dataset
print (pd.DataFrame(dataset.isnull().sum()))
'''
text  0
spam  0
'''
#Using Natural Language Processing to cleaning the text to make one corpus
# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#Every mail starts with 'Subject :' will remove this from each text 
dataset['text']=dataset['text'].map(lambda text: text[1:])
dataset['text'] = dataset['text'].map(lambda text:re.sub('[^a-zA-Z0-9]+', ' ',text)).apply(lambda x: (x.lower()).split())
ps = PorterStemmer()
corpus=dataset['text'].apply(lambda text_list:' '.join(list(map(lambda word:ps.stem(word),(list(filter(lambda text:text not in set(stopwords.words('english')),text_list)))))))

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus.values).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Fitting Naive Bayes classifier to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
classifier.fit(X_train , y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
Confusion Matrix
array([[863,  11],
       [  1, 264]])
'''
#this function computes subset accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) #0.9894644424934153
accuracy_score(y_test, y_pred,normalize=False) #1129 out of 1139

# Applying k-Fold Cross Validation
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)#array([ 0.98903509,  0.98903509,  0.99122807,  0.98026316,  0.98245614,0.98903509,  0.98901099,  0.99340659,  0.99340659,  0.98681319])
accuracies.mean()#0.9888085218938609
accuracies.std()#0.004090356321646494


