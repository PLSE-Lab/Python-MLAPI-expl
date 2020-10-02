# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('../input/HR_comma_sep.csv')

X = dataset.drop(['left'],axis=1)
y = dataset['left']

salary = pd.get_dummies(X['salary'],drop_first=True)
X.drop(['salary'],axis=1,inplace=True)
X = pd.concat([X,salary],axis=1)


dept = pd.get_dummies(X['sales'],drop_first=True)
X.drop(['sales'],axis=1,inplace=True)
X = pd.concat([X,dept],axis=1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

# Using Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
log_reg_score = log_reg.score(X_train,y_train)

pred = log_reg.predict(X_test)

# Using SVM
from sklearn import svm
clf = svm.SVC(kernel='rbf',C=500,gamma=0.9)
clf.fit(X_train,y_train)
svm_score = clf.score(X_train,y_train)
print(svm_score)
svm_test_score = clf.score(X_test,y_test)
print(svm_test_score)