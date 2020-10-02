# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
'''
The idea of this classification task is very simple, the name of the app should somewhat have
close relation with the category of the app, so we are trying to predict the category of
the App given the name of the App thus creating an auto-categorizer
'''


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

def process_text(text):
    '''
    What will be covered:
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(clean_words)
    
    
data = pd.read_csv('../input/googleplaystore.csv')

name = data['App'].apply(process_text)
cate = data['Category']
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(name).todense()
y = LabelEncoder().fit_transform(cate)

##check out the shape of X and y
X.shape
y.shape

## check how many categories are there
print("the number of categories are: ")
print(len(np.unique(y)))

## split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = None)
print('The length of train size : ', len(X_train))

###use naive bayes
clf = GaussianNB()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(classification_report(pred, y_test))


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100, 100), random_state=1)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(classification_report(pred, y_test))



















