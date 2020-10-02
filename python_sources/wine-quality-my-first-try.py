
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,accuracy_score,classification_report,confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import shuffle
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/winequality-red.csv")
df = shuffle(df)
bins = (2,6.5,8)
labels = ['bad','good']
df['quality'] = pd.cut(df['quality'],bins=bins,labels=labels)
df = df.drop(['free sulfur dioxide'],axis=1)
df = df.drop(['chlorides'],axis=1)
df = df.drop(['residual sugar'],axis=1)
df = df.drop(['pH'],axis=1)

labelEncoder = preprocessing.LabelEncoder()
df['quality'] = labelEncoder.fit_transform(df['quality'])

X = df.iloc[:,0:10]
Y = df['quality']

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

model = SVC()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
