#!/usr/bin/env python
# coding: utf-8




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print("\n Import Library Successfull")

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/online-shoppers-intention/online_shoppers_intention.csv')
df.head()

print("\n Import Data Successfull")

# **Exploratory Data Analysis**
print("\n Exploratory Data Analysis")

plt.figure(figsize=(10,10));
sums = df.Revenue.groupby(df.Month).sum()
plt.pie(sums, labels=sums.index);
plt.show()

plt.figure(figsize=(6,6));
sums1 = df.Revenue.groupby(df.VisitorType).sum()
plt.pie(sums1, labels=sums1.index);
plt.show()

sns.countplot(df['Revenue'])
plt.show()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(x='ProductRelated_Duration',y='BounceRates', data=df, hue='Revenue',palette='prism')
plt.show()

sns.scatterplot(x='PageValues',y='BounceRates', data=df, hue='Revenue', palette='prism')
plt.show()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(x='Informational_Duration',y='BounceRates', data=df, hue='Revenue',palette='prism')
plt.show()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(x='ProductRelated',y='ExitRates', data=df, hue='Revenue',palette='prism')
plt.show()

# **Data Preprocessing**

print('\n Data Preprocessing')

df.isnull().sum()

df.dropna(inplace=True)

df2 = df.drop(['Revenue','Month'], axis=1)

X = pd.get_dummies(df2,drop_first=True)

X.Weekend = X.Weekend.astype(int)

X.head()

y = df['Revenue']

print('\n Done')

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X = scalar.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Function for calculating accuracy
def accuracy(y_test,y_pred):
    from sklearn.metrics import confusion_matrix
    result = confusion_matrix(y_pred,y_test)
    acc = ((result[0][0]+result[1][1])/(len(y_test)))*100
    return acc

print("\n Standardising and train test split done Successfully")
# **MODEL BUILDING**
# 
# Support Vector Machine
print("\n MODEL BUILDING")

print("\n Building Support Vector Machine")
from sklearn.svm import SVC
svc = SVC()
model = svc.fit(X_train,y_train)
y_pred_svc = model.predict(X_test)
print("\n Done")
print("\n Accuracy of SVM: ",accuracy(y_test,y_pred_svc))

# Logistic Regression
print("\n Building Logistic Regression")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model_lr = lr.fit(X_train,y_train)
y_pred_lr = model_lr.predict(X_test)
print("\n Done")
print("\n Accuracy of Logistic Regression: ",accuracy(y_test,y_pred_lr))

print("\n Building Random Forest Model")
# Random Forest
from sklearn.ensemble import RandomForestClassifier
model_rf_classi = RandomForestClassifier()
model_rf = model_rf_classi.fit(X_train,y_train)
y_pred_enrf = model_rf.predict(X_test)
print("\n Done")
print("\n Accuracy of Random Forest: ",accuracy(y_test,y_pred_enrf))

# Neural Network Approach
print("\n Building Neural Network Model")
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
print("\n Training the model")
classifier = Sequential()
classifier.add(Dense(units = 128, activation = 'relu', input_dim = 17))
classifier.add(Dropout(0.20))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.20))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.20))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 32, epochs = 100, verbose=2)
print("\n Done")

y_pred_nn = classifier.predict_classes(X_test)

print("\n Accuracy of Neural Network: ",accuracy(y_test,y_pred_nn))

data= [['SVC',88.96],['LogisticRegression',88.18],['RandomForest',88.87],['NeuralNetwork',88.18]]
accuracy_compare = pd.DataFrame(data, columns = ['Method', 'Accuracy'])

sns.barplot(x=accuracy_compare['Method'],y=accuracy_compare['Accuracy'])
plt.show()