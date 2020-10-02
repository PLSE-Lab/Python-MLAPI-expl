# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# -*- coding: utf-8 -*-
# Importing the dataset
dataset1 = pd.read_csv('../input/spam.csv' , encoding='cp437')
y=dataset1.iloc[:,0].values
import re
import nltk
#nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
orpus=[]
for i in range(0,5572):
    review = re.sub('[^a-zA-Z]',' ',dataset1['v2'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    orpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer    
cv=CountVectorizer(max_features=3000)
x=cv.fit_transform(orpus).toarray()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)   

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
#predition
pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print('Accuracy score: {}'.format(accuracy_score(y_test, pred)))
print('Precision score: {}'.format(precision_score(y_test, pred)))
print('Recall score: {}'.format(recall_score(y_test, pred)))
print('F1 score: {}'.format(f1_score(y_test, pred)))

from sklearn.ensemble import RandomForestClassifier
classifier1=RandomForestClassifier(n_estimators=15,criterion='entropy')
classifier1.fit(X_train,y_train)
predRF=classifier1.predict(X_test)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print('Accuracy score: {}'.format(accuracy_score(y_test, predRF)))
print('Precision score: {}'.format(precision_score(y_test, predRF)))
print('Recall score: {}'.format(recall_score(y_test, predRF)))
print('F1 score: {}'.format(f1_score(y_test, predRF)))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.