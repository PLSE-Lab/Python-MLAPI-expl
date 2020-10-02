#!/usr/bin/env python
# coding: utf-8

#                                                                                                 Titanic Model

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA

KNNclassifier=KNeighborsClassifier(n_neighbors=5)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras
from keras.optimizers import SGD
import graphviz

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# 1. Training data Pre-preocessing

# In[ ]:


#Importing the training and testing data
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


#Checking the first five row in training data
train.head()


# In[ ]:


#stats of the training data
train.describe()


# In[ ]:


#Familiarising with the Column name 
train.columns


# In[ ]:


#Checking the data type for better understanding
train.dtypes


# In[ ]:


#countplot of survived by sex
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train)


# In[ ]:


#Heatmap plot of the missing data or null present in the training data
sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis',cbar=False)


# In[ ]:


#Boxplot of the Pclass,Age and Survived 
plt.figure(figsize=(19, 8))
sns.boxplot(x='Pclass',y='Age',data=train,hue='Survived',color="cyan")


# *  Feature Engineering

# In[ ]:


# total null presnets in the training data
train.isna().sum().max()


# In[ ]:


#Function fortotal null  present
def missing_total(data):
    missing_total= data.isna().sum().sort_values(ascending=False)
    return missing_total


# In[ ]:


#Function for null percentage present
def missing_percent(data):
    missing_percent = ((data.isna().sum()/data.isna().count())*100).sort_values(ascending=False)
    return missing_percent


# In[ ]:


#Table for null percentage and total null present
train_missing = pd.concat([missing_total(train), missing_percent(train)], axis=1, keys=['missing_total', 'missing_percent'])
train_missing.head(50)


# In[ ]:


#further anaylsis on missing train data
#Removing any columns that contains nan greater than 20% which is only cabin
train = train.drop((train_missing[train_missing['missing_percent'] > 20.0]).index,1)


# In[ ]:


#Fill the nan with median of the columns
#train.fillna(train.median(), inplace=True)


# In[ ]:


#complete embarked with mode
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)


# In[ ]:


#Function to create more feature as Age Category
def age_group(val):
    if val<2:
        return 'Infant'
    elif val>2 and val < 10:
        return 'Child'
    elif val>10 and val < 17:
        return 'Adolescence'
    elif val>17 and val < 24:
        return 'Teen'
    elif val>24 and val < 65:
        return 'Adult'
    else:
        return 'Elderly'
train['Age_category']=train['Age'].apply(age_group)


# In[ ]:


#Splitting the data into categorical data, float and Varaible
train_Var = train[train.dtypes[train.dtypes == "int64"].index]

train_Cat = train[train.dtypes[train.dtypes == "object"].index]

train_float = train[train.dtypes[train.dtypes == "float"].index]


# In[ ]:


#Creating more features from Name
train_Cat['Title'] = train_Cat['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
stat_min = 10
title_names = (train_Cat['Title'].value_counts() < stat_min)
train_Cat['Title'] = train_Cat['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(train_Cat['Title'].value_counts())
print("-"*10)


# In[ ]:


# dropping Name and Ticket columns name because they are not important for the analysis
drop_column = ['Name', 'Ticket']
train_Cat.drop(drop_column, axis=1, inplace = True)


# In[ ]:


#Converting the categorical data into dummy variable for easy analysis
train_Cat = pd.get_dummies(train_Cat)


# In[ ]:


train_Cat.head(20)


# In[ ]:


train_Var


# In[ ]:


#Fill the nan with median of the columns
train_float['Age'].fillna(train_float['Age'].median(), inplace = True)


# In[ ]:


#dropping the PassengerId becuase it is not important for the prediction
train_Var.drop('PassengerId', axis=1, inplace = True)


# In[ ]:


#concatinating the three data type together
inputData=train_Var.join([train_Cat, train_float])


# In[ ]:


#Checking for the last time the total null value available
inputData.isna().sum().max()


# In[ ]:


inputData.head()


# In[ ]:


inputData.shape


# 2.  Test Data data preprocessing

# The same analysis will be performed on the testing data

# In[ ]:


#Checking the first five row in testing data
test.head()


# In[ ]:


#stats of the testing data
test.describe()


# In[ ]:


#Familiarising with the Column name 
test.columns


# In[ ]:


#Checking the data type for better understanding
test.dtypes


# In[ ]:


# total null presnets in the testing data
test.isna().sum().max()


# In[ ]:


#Table for null percentage and total null present
test_missing = pd.concat([missing_total(test), missing_percent(test)], axis=1, keys=['missing_total', 'missing_percent'])
test_missing.head(50)


# In[ ]:


#Removing any columns that contains nan greater than 20% which is only cabin
test = test.drop((test_missing[test_missing['missing_percent'] > 50.0]).index,1)


# In[ ]:


test.head(5)


# In[ ]:


drop_column = ['PassengerId', 'Ticket']
test.drop(drop_column, axis=1, inplace = True)


# In[ ]:


#Fill the nan with median of the columns
test['Age'].fillna(test['Age'].median(), inplace = True)
test['Fare'].fillna(test['Fare'].median(), inplace = True)


# In[ ]:


def age_cat(val):
    if val<2:
        return 'Infant'
    elif val>2 and val < 10:
        return 'Child'
    elif val>10 and val < 17:
        return 'Adolescence'
    elif val>17 and val < 24:
        return 'Teen'
    elif val>24 and val < 65:
        return 'Adult'
    else:
        return 'Elderly'
test['Age_category']=test['Age'].apply(age_cat)


# In[ ]:


#Splitting the data into categorical data and Varaible
test_Var = test[test.dtypes[test.dtypes == "int64"].index]

test_Cat = test[test.dtypes[test.dtypes == "object"].index]

test_float = test[test.dtypes[test.dtypes == "float"].index]


# In[ ]:


#Feature Engineering
test_Cat['Title'] = test_Cat['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
stat_min = 10
title_names = (test_Cat['Title'].value_counts() < stat_min)
test_Cat['Title'] = test_Cat['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(test_Cat['Title'].value_counts())
print("-"*10)


# In[ ]:


test_Cat.drop('Name', axis=1, inplace = True)


# In[ ]:


#Converting the categorical data into dummy variable for easy analysis
test_Cat = pd.get_dummies(test_Cat)


# In[ ]:


test_Cat.head(5)


# In[ ]:


test_Cat.isna().sum().max()


# In[ ]:


outputData=test_Var.join([test_Cat, test_float])


# In[ ]:


outputData.head(5)


# In[ ]:


outputData.isna().sum().max()


# Data Splitting

# In[ ]:


X_train = inputData.drop(['Survived'],axis=1).values
Y_train = inputData['Survived'].values
X_test =outputData.values


# In[ ]:


#Data Processing
#random seed------ Meaning of random seed is explained in the documentation
seed = 0
np.random.seed(seed)
X_training, X_testing, Y_training, Y_testing= train_test_split(X_train, Y_train.reshape(-1,1), test_size=0.1, random_state=seed)
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()

X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)
X_scaled_test = X_scaler.transform(X_test)


# In[ ]:


features_selection = LassoCV(0.5, cv=5)
reg = SelectFromModel(features_selection)
reg.fit(X_scaled_training, Y_scaled_training)

X_scaled_training = reg.transform(X_scaled_training)

#Printing the features selected
for feature_list_index in reg.get_support(indices=True):
    print([feature_list_index])


# In[ ]:


#Transforming the testing data to the actual number of features selected for the training data
X_scaled_testing = reg.transform(X_scaled_testing)
X_scaled_test = reg.transform(X_scaled_test)


# In[ ]:


KNNclassifier.fit(X_scaled_training ,Y_scaled_training)
y_pred = KNNclassifier.predict(X_scaled_testing)
print("Accuracy :",accuracy_score(Y_scaled_testing,y_pred)*100)
cm = confusion_matrix(Y_scaled_testing, y_pred)
print("Confusion Matrix:\n", cm)


# In[ ]:


SVMlinear=SVC(kernel='linear')
SVMlinear.fit(X_scaled_training ,Y_scaled_training)
y_pred = SVMlinear.predict(X_scaled_testing)
print("Accuracy :",accuracy_score(Y_scaled_testing,y_pred)*100)
cm = confusion_matrix(Y_scaled_testing,y_pred)
print("Confusion Matrix:\n", cm)


# In[ ]:


SVMrbf=SVC(kernel='rbf')
SVMrbf.fit(X_scaled_training ,Y_scaled_training)
y_pred = SVMrbf.predict(X_scaled_testing)
print("Accuracy :",accuracy_score(Y_scaled_testing,y_pred)*100)
cm = confusion_matrix(Y_scaled_testing,y_pred)
print("Confusion Matrix:\n", cm)


# In[ ]:


NB=GaussianNB()
NB.fit(X_scaled_training ,Y_scaled_training)
y_pred = NB.predict(X_scaled_testing)
print("Accuracy :",accuracy_score(Y_scaled_testing,y_pred)*100)
cm = confusion_matrix(Y_scaled_testing,y_pred)
print("Confusion Matrix:\n", cm)


# In[ ]:


DecisionTree=DecisionTreeClassifier(criterion='entropy',random_state=23)
DecisionTree.fit(X_scaled_training ,Y_scaled_training)
y_pred = DecisionTree.predict(X_scaled_testing)
print("Accuracy :",accuracy_score(Y_scaled_testing,y_pred)*100)
cm = confusion_matrix(Y_scaled_testing,y_pred)
print("Confusion Matrix:\n", cm)


# In[ ]:


RFC=RandomForestClassifier(n_estimators=17,criterion='entropy',random_state=0)
RFC.fit(X_scaled_training ,Y_scaled_training)
y_pred = RFC.predict(X_scaled_testing)
print("Accuracy :",accuracy_score(Y_scaled_testing,y_pred)*100)
cm = confusion_matrix(Y_scaled_testing,y_pred)
print("Confusion Matrix:\n", cm)


# In[ ]:


model = Sequential()

# Inputing the first layer with input dimensions
model.add(Dense(100, 
                activation='relu',  
                input_dim=X_scaled_training.shape[1],
                kernel_initializer='uniform'))

model.add(Dense(20,
                kernel_initializer='uniform',
                activation='relu'))

#adding second hidden layer 
model.add(Dense(1,
                kernel_initializer='uniform',
                activation='sigmoid'))
#With such a scalar sigmoid output on a binary classification problem, the loss
#function you should use is binary_crossentropy

#Visualizing the model
model.summary()


# In[ ]:


#Creating an Stochastic Gradient Descent
sgd = SGD(lr = 0.01, momentum = 0.9)

# Compiling our model
model.compile(optimizer = sgd, 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])
# Fitting the ANN to the Training set
model.fit(X_scaled_training ,Y_scaled_training, 
               batch_size = 70, 
               epochs = 120, verbose=2)


# In[ ]:


#Using KNNclassifier for final prediction 
yhat =SVMrbf.predict(X_scaled_test).astype(int)


# In[ ]:


#Transforming the prediction back to normal data before scaling
prediction = Y_scaler.inverse_transform(yhat.reshape(-1,1)).astype(int)


# In[ ]:


Id = pd.read_csv('../input/titanic/test.csv')
Prediction = pd.concat([pd.DataFrame(Id['PassengerId'], columns=['PassengerId']),pd.DataFrame(yhat, columns=['Survived'])],axis=1)


# In[ ]:


#saving the prediction
Prediction.to_csv('Prediction.csv', index=None)

