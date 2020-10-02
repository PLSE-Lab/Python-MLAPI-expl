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
# import required lobraries
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm, metrics

# read dataset
df = pd.read_csv('/kaggle/input/social-network-ads/Social_Network_Ads.csv')

# drop less valuable columns
df.drop('User ID', axis=1, inplace=True)

# basic Exploratory Data Analysis
print(df.isnull().sum())
print(df['Purchased'].value_counts())
print('Purchased:', 143/400)
print('Not Purchased:', 257/400)

# handle categorical data
df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})

# get feqtures and labels
X = df.drop('Purchased', axis=1)
X = np.array(X)
X = preprocessing.scale(X) # scaled the featured between -1 to 1

y = df['Purchased']
y = np.array(y).astype(float)

# train and test data splits
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

# create support vector machine model
clf = svm.SVC(kernel='linear', random_state=0)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
pred = clf.predict(X_test)

# evaluation techniques
conf_mat = metrics.confusion_matrix(y_test, pred)
acc_score = metrics.accuracy_score(y_test, pred)
precision = metrics.precision_score(y_test, pred)
recall = metrics.recall_score(y_test, pred)
f1score = metrics.f1_score(y_test, pred)
report = metrics.classification_report(y_test, pred)
print(report)