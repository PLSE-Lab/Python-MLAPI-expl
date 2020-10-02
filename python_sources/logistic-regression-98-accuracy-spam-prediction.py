# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 09:54:53 2019

@author: USER
"""
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("./../input/spam.csv", encoding = 'latin-1')

# Checking the presence of null values
dataset.isnull().sum()
# need to drop few columns
dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis= 1, inplace = True)

# Cleaning the data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 5572):
    review = re.sub('[^a-zA-Z]', ' ', dataset['v2'][i])
    review = review.lower()
    review = review.split()
    
    review = [word for word in review if not word in set(stopwords.words('english'))]
    ps= PorterStemmer()
    review = [ps.stem(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer 
cv= CountVectorizer(max_features = 2000)
X= cv.fit_transform(corpus).toarray()

# we have a categorical values
spam = pd.get_dummies(dataset['v1'], drop_first = True)
Y= spam.iloc[:, 0].values

# splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size= 0.2, random_state= None)

# Apply logistic rgression
from sklearn.linear_model import LogisticRegression
LR_regressor = LogisticRegression()
LR_regressor.fit(X_train, Y_train)

Y_pred_LR = LR_regressor.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_LR)
cm 
# (983 +118)/(983+118+14)-- 98%

# k cross validation
#
    
#