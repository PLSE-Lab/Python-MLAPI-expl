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

df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Drop unnecessary last and first column
df.drop(df.columns[[-1, 0]], axis = 1, inplace = True)

# Assigning M = 1.0 and B = -1.0
mp = {'M' : 1.0, 'B' : -1.0}
df['diagnosis'] = df['diagnosis'].map(mp)

# dependent variable
y = df.iloc[:, 0].values.reshape(-1,1)
# independent feature matrix
X = df.iloc[:, 1:].values

# data normalization
X = MinMaxScaler().fit_transform(X)

# inserting a column of ones for 'b'
ones = np.ones((X.shape[0], 1))
X = np.concatenate((X, ones), axis = 1)

# splitting into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

########################################### ON SKLEARN SVM ##############################################################

classifier_SVM = SVC(kernel='rbf', degree=4)
classifier_SVM.fit(X_train, y_train)

y_pred_SVM = classifier_SVM.predict(X_test)

acc = accuracy_score(y_test, y_pred_SVM)
print('Accuracy of SVM is {} %'.format(acc*100))

######################################### ON SKLEARN LOGISTICS REGRESSION ################################################

classifier_LR = LogisticRegression()
classifier_LR.fit(X_train, y_train)

y_pred_LR = classifier_LR.predict(X_test)

acc = accuracy_score(y_test, y_pred_LR)
print('Accuracy of Logistic Regression is {} %'.format(acc*100))