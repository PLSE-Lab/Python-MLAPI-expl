# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
os.chdir("../input")
os.getcwd()

# Any results you write to the current directory are saved as output.

#Importing Dataset
dataset = pd.read_csv("../input/weatherAUS.csv")

#Preprocessing
        #Understanding the dataset w.r.t counts of different columns
dataset.info()
dataset.count().sort_values()
        #Removing columns 
dataset = dataset.drop(['Sunshine','Evaporation','Cloud3pm','Cloud9am','RISK_MM'], axis = 1)
dataset = dataset.dropna(axis = 'rows')

#X & y
X = dataset.iloc[:, 2:18]
y = dataset.iloc[:, 18]
data = dataset.head(1000)


            ###LabelEncoding###
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

temp = X.head(100)

"""WindGustDir"""
le = LabelEncoder()
X['WindGustDir'] = le.fit_transform(X['WindGustDir'])

"""WindDir9am"""
le1 = LabelEncoder()
X['WindDir9am'] = le1.fit_transform(X['WindDir9am'])

"""WindDir3pm"""
le2 = LabelEncoder()
X['WindDir3pm'] = le2.fit_transform(X['WindDir3pm'])

"""RainToday"""
le3 = LabelEncoder()
X['RainToday'] = le3.fit_transform(X['RainToday'])

            ###OneHotEncoder###
"""WindGustDir"""
ohe = OneHotEncoder(categorical_features=[3])
X = ohe.fit_transform(X).toarray()
X = X[:,1:]

"""WindDir9am"""
ohe1 = OneHotEncoder(categorical_features=[19])
X = ohe1.fit_transform(X).toarray()
X = X[:,1:]

"""WindDir3pm"""
ohe2 = OneHotEncoder(categorical_features=[33])
X = ohe2.fit_transform(X).toarray()
X = X[:,1:]

"""RainToday"""
ohe3 = OneHotEncoder(categorical_features=[55])
X = ohe3.fit_transform(X).toarray()
X = X[:,1:]


#splitting the data into training & test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Standardizating Dependent variable
from sklearn.preprocessing import StandardScaler
ss  = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

        ###Logistic Regression###
#Fitting the classifier to the dataset
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
classifier = lr.fit(X_train, y_train)

#Predicting the target variable
y_pred = classifier.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred = (y_pred == 'Yes')
y_test = (y_test == 'Yes')

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

