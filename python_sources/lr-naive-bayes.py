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
Created on Wed Apr 10 18:54:09 2019

@author: Roshan Zameer Syed
"""

path = "../input/adult.csv"
data = pd.read_csv(path)

cols = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','target']

data.columns = cols

data.info()
data.replace(' ?', np.NaN,inplace = True)
data.isna().any().any()
data.isna().any()

# Missing values in workclass, occupation, native-country
data.fillna(method = 'ffill', inplace = True)

# removing unwanted columns fnlwgt as it is descrete large number exponential and education - its the same as education-num
data.drop(['fnlwgt','education'],axis = 1,inplace =True)

# identify numerical and categorical columns 
print(data.select_dtypes('int64').columns)
num = ['age', 'education-num', 'capital-gain', 'capital-loss',
       'hours-per-week']
print(data.select_dtypes('object').columns)
# not selecting the target variable
cat = ['workclass', 'marital-status', 'occupation', 'relationship', 'race',
       'sex', 'native-country']

print(data[num].describe())

import matplotlib.pyplot  as plt
plt.hist(data['age'])               # Min = 17 , Max = 90
plt.hist(data['education-num'])     # ordinal 1 to 16
plt.hist(data['capital-gain'])      # Min = 0 , Max = 99999, 0-75% lies at 0 
plt.hist(data['capital-loss'])      # Min = 0 , Max = 4356, 0-75% lies at 0 
plt.hist(data['hours-per-week'])    # Min = 1 , Max = 99

print(data[cat].describe())
# occupation = 14 and native-country = 41 categories
data[cat].nunique()

# checking occupation categories
data['occupation'].value_counts().plot('bar')
data['native-country'].value_counts().plot('bar')

# convert categorical columns to numerical
data_cat = pd.get_dummies(data[cat],drop_first = True)
data_num = data[num]

#combine data_cat and data[num]
df = pd.concat([data_cat,data_num],axis = 1)

# target variable has 4 values
data['target'].value_counts()
data['target'].replace({' <=50K.':' <=50K', ' >50K.': ' >50K'}, inplace = True)
# convert target variable to numeric- binary
data['target'].replace({' <=50K':0,' >50K':1},inplace = True)

# define X and y
X = df.iloc[:,:].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)

#Training data
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
y_train_pred = lr.predict(X_train)
print(classification_report(y_train,y_train_pred))
print(confusion_matrix(y_train,y_train_pred))
print(accuracy_score(y_train,y_train_pred))

# Test data
y_pred = lr.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
gb.fit(X_train,y_train)

#Training data
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
y_train_pred = gb.predict(X_train)
print(classification_report(y_train,y_train_pred))
print(confusion_matrix(y_train,y_train_pred))
print(accuracy_score(y_train,y_train_pred))

# Test data
y_pred = gb.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

