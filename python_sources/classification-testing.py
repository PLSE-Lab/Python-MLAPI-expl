# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:47:20 2018

@author: kartik.sharma10
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# import datasets
train = pd.read_csv('../input/train.csv')

# check for missing values
nan_train=train.shape[0]-train.dropna().shape[0]

# check which column has missing values 
train.isnull().sum()

# fill these column's missing values with most frequent values or use an Imputer
# if column contains numeric values
train.workclass.value_counts(sort=True)
train.workclass.fillna('Private',inplace=True)

train.occupation.value_counts(sort=True)
train.occupation.fillna('Prof-specialty',inplace=True)

train['native.country'].value_counts(sort=True)
train['native.country'].fillna('United-States',inplace=True)

# again check for missing value and see if they are filled
train.isnull().sum()

# check proportion of target variable
train.target.value_counts()/train.shape[0]

# encode the column values in training set
for i in train.columns:
    le=LabelEncoder()
    train[i]=le.fit_transform(train[i].values)

# assign x and y
x=train.iloc[:,:-1].values
y=train.iloc[:,-1].values


# split data into training and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.3, random_state = 0)

#y_train=y_train.reshape(1,-1)

# scale the data if required
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



'''
the dependent variable in this probelm has 2 classes (0,1) so if any linear regressor
is used then predicted values will go out of bound or will result float values.

so it is not a regression problem, its a classification problem. 


------------------------ create a linear regressor -------------------------

# Multiple Linear Regressor
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)

#SVR Regressor (requires feature scaling)
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'rbf')
svr_regressor.fit(x_train, y_train)

#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state = 0)
dt_regressor.fit(x_train, y_train)

#Random Forest Regressor 
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_regressor.fit(x_train, y_train)

----------------------------------------------------------------------------
'''

# create classifiers (requires feature scaling for better result)

''' Logistic classifier '''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

''' KNN classifier '''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

''' SVM classifier '''
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)

''' Kernal SVM classifier '''
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

''' Naive bayes classifier '''
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

''' Decision Tree classifier '''
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

''' Random Forest classifier '''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# predict the output
y_pred=classifier.predict(x_test)

'''
for linear regressions:
if y_pred in float then: y_pred=np.array(list(map(int,map(round,y_pred))))

# use if feature scaling is done on y_test
 y_pred = sc_y.inverse_transform(y_pred)
'''

# create a confusion matrix
li=[]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
efficiency=str(round((cm[0][0]+cm[1][1])/len(y_test),2)*100)+"%"
li.append(efficiency)

# my method calculate efficiency
li1=[]
count=0
for i in range(len(y_test)):
    if list(y_test)[i]==list(y_pred)[i]:
        count+=1
li1.append(round(count/len(y_pred),2))

# another method for accuracy
from sklearn.metrics import accuracy_score
acc =  accuracy_score(np.array(y_test),y_pred)
print(acc)