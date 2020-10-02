# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score  

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
total_dataset = pd.read_csv("../input/Iris.csv")
train,test = train_test_split(total_dataset, test_size=0.8)

train_features = train.iloc[:,:-1]
train_labels = train.iloc[:,-1]
test_features = test.iloc[:,:-1]
test_labels = test.iloc[:,-1]

clf = GaussianNB()
clf.fit(train_features, train_labels)
predict_labels = clf.predict(test_features)
print(balanced_accuracy_score(test_labels, predict_labels))
