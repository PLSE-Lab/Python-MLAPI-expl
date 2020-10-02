#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data=pd.read_csv('/kaggle/input/titanic/train.csv')
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


#Check the list of columns with missing values in the data
train_data.columns[train_data.isnull().any()]


# In[ ]:


#Check the datatype of eachcolumn
train_data.dtypes


# In[ ]:


#Getting count of null values for each column
train_data.isnull().sum(axis=0)


# In[ ]:


#Counting the number of times each value has occured in the column
train_data['Embarked'].value_counts()


# In[ ]:


#Since most frequent value is S i.e. Southampton, we replace all null values in column Embarked with Southhampton
train_data=train_data.fillna({'Embarked':'S'})


# In[ ]:


#Computing the sum of null values of each column
train_data.isnull().sum(axis = 0)


# In[ ]:


#Computing percentage of null entries in the Cabin column for training set
(687/len(train_data))*100


# In[ ]:


#Computing percentage of null entries in the Cabin column for the test set
(327/len(test_data))*100


# Since the Cabin column has more than 75% of null values, we will drop the column

# In[ ]:


train_data.drop(columns=['Cabin'],inplace=True)


# In[ ]:


test_data.drop(columns=['Cabin'],inplace=True)


# In[ ]:


#Computing the sum of null values of each column
train_data.isnull().sum(axis = 0)


# In[ ]:


#Computing the sum of null values of each column
test_data.isnull().sum(axis = 0)


# Imputing Missing values for the age column

# In[ ]:


def get_Title(name):
    if 'Mr.'in name:
        return 'Mr'
    elif 'Miss.' in name:
        return 'Miss'
    elif 'Ms.' in name:
        return 'Miss'
    elif 'Mme.' in name:
        return 'Mrs'
    elif 'Mlle.' in name:
        return 'Miss'
    elif 'Master.' in name:
        return 'Master'
    elif 'Mrs.' in name:
        return 'Mrs'


# In[ ]:


train_data['Title']=[get_Title(item) for item in train_data['Name']]
test_data['Title']=[get_Title(item) for item in test_data['Name']]


# In[ ]:


test_data.head(20)


# In[ ]:


#checking details of females with missing Titles for train data
for ind,val in enumerate(train_data['Title']):
    if val==None and train_data.loc[ind,'Sex']=='female':
        print(train_data.loc[ind,'Name'], train_data.loc[ind,'Age'])


# In[ ]:


#Checking Details of male with Missing Titles for train data
for ind,val in enumerate(train_data['Title']):
    if val==None and train_data.loc[ind,'Sex']=='male':
        print(train_data.loc[ind,'Name'], train_data.loc[ind,'Age'])


# In[ ]:


#Filling in the Missing Titles for train data
for index,value in enumerate(train_data['Title']):
    if value==None:
        if train_data.loc[index,'Sex']=='male' and train_data.loc[index,'Age']!='NaN':
            if train_data.loc[index,'Age']<14:
                train_data.loc[index,'Title']='Master'
            else:
                train_data.loc[index,'Title']='Mr'
        elif train_data.loc[index,'Sex']=='female' and train_data.loc[index,'Age']!='NaN':
            train_data.loc[index,'Title']='Miss'


# In[ ]:


list(train_data['Title'])


# In[ ]:


#checking details of females with missing Titles for test data
for ind,val in enumerate(test_data['Title']):
    if val==None and test_data.loc[ind,'Sex']=='female':
        print(test_data.loc[ind,'Name'], test_data.loc[ind,'Age'])


# In[ ]:


#Checking Details of male with Missing Titles for test data
for ind,val in enumerate(test_data['Title']):
    if val==None and test_data.loc[ind,'Sex']=='male':
        print(test_data.loc[ind,'Name'], test_data.loc[ind,'Age'])


# In[ ]:


#Filling in the Missing Titles for test data
for index,value in enumerate(test_data['Title']):
    if value==None:
        if test_data.loc[index,'Sex']=='male' and test_data.loc[index,'Age']!='NaN':
            if test_data.loc[index,'Age']<14:
                test_data.loc[index,'Title']='Master'
            else:
                test_data.loc[index,'Title']='Mr'
        elif test_data.loc[index,'Sex']=='female' and test_data.loc[index,'Age']!='NaN':
            test_data.loc[index,'Title']='Miss'


# In[ ]:


#calculating Median values of Age for Each group in train file
age_group=train_data.groupby('Title').Age.agg('median')
age_group


# In[ ]:


#Filling the missing values in the age column with the median value of age for each group in train data
for ind,age in enumerate(train_data['Age']):
    if str(age).upper()=='NAN' or age==None:
        train_data.loc[ind,'Age']=age_group[train_data.loc[ind,'Title']]


# In[ ]:


#Computing the sum of null values of each column
train_data.isnull().sum(axis = 0)


# In[ ]:


#This calculation is not required as we are filling up the missing values with the median values of the train data itself
#calculating Median values of Age for Each group in test data
#age_group_test=test_data.groupby('Title').Age.agg('median')
#age_group_test


# In[ ]:


#Filling the missing values in the age column with the median value of age for each group calculated from train data
for ind,age in enumerate(test_data['Age']):
    if str(age).upper()=='NAN' or age==None:
        test_data.loc[ind,'Age']=age_group[test_data.loc[ind,'Title']]  #Changed on 10-12-19 by shri


# In[ ]:


#Computing the sum of null values of each column
test_data.isnull().sum(axis = 0)


# In[ ]:


train_data['Age'].plot.hist()


# In[ ]:


test_data['Age'].plot.hist()


# In[ ]:


#Converting categorical variables into numeric by encoding
train_data['Sex']=train_data['Sex'].replace(to_replace=['male','female'],value=[1,0])


# In[ ]:


#Converting categorical variables into numeric by encoding
test_data['Sex']=test_data['Sex'].replace(to_replace=['male','female'],value=[1,0])


# In[ ]:


train_data.dtypes


# In[ ]:


#Categorizing data with label
encode=LabelEncoder()
train_data['Ticket_categorized']=encode.fit_transform(train_data['Ticket'])
test_data['Ticket_categorized']=encode.fit_transform(test_data['Ticket'])
train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data['Embarked']=train_data['Embarked'].astype('category')
train_data['Embarked_coded']=train_data['Embarked'].cat.codes
test_data['Embarked']=test_data['Embarked'].astype('category')
test_data['Embarked_coded']=test_data['Embarked'].cat.codes


# In[ ]:


test_data.isnull().sum(axis=0)


# In[ ]:


test_data=test_data.fillna({'Fare':0})


# In[ ]:


train_data.dtypes


# In[ ]:


#Assigning training data to X_train
X_train=train_data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Ticket_categorized','Fare','Embarked_coded']]
y_train=train_data['Survived']


# In[ ]:


test_data.columns


# In[ ]:


X_test=test_data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Ticket_categorized','Fare','Embarked_coded']]


# Standardizing using StandardScalar

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[ ]:


X_train


# Lets fit a Logistic Regression Model to predict whether a passenger survived or not

# In[ ]:


#GridSearchCV Implementing for Log Regression Model
parameters={"C":[0.01,0.1,0.5,1]}


# In[ ]:


#Initializing the Logistic Regrssion Model
logis=LogisticRegression()
logis_cv=GridSearchCV(estimator=logis,param_grid=parameters,
                     scoring='accuracy',cv=5,
                     return_train_score=True)
#Fitting our model on the training Data
#logis.fit(X_train,y_train)


# In[ ]:


#Fitting our model on the training Data
logis_cv.fit(X_train,y_train)


# In[ ]:


logis_cv.cv_results_


# In[ ]:


# results of grid search CV
cv_results = pd.DataFrame(logis_cv.cv_results_)
cv_results


# In[ ]:


help(logis_cv)


# In[ ]:


#Plotting Test vs Train score graph
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.plot(cv_results['param_C'],cv_results['mean_test_score'])
plt.plot(cv_results['param_C'],cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend(['test accuracy', 'train accuracy'])#loc='upper left')


# In[ ]:


print(logis_cv.best_score_)
print(logis_cv.best_params_['C'])


# We shall now try and fit an SVM to the data to see if it improves accuracy

# In[ ]:


from sklearn import svm
svm_algo=svm.SVC()


# In[ ]:


#Parameters for GridSearchCV for SVM
svm_params={'C':[0.1,0.5,1,10,100]}
svm_gs=GridSearchCV(estimator=svm_algo,param_grid=svm_params,
                   cv=5,verbose=1,return_train_score=True)


# In[ ]:


svm_gs.fit(X_train,y_train)


# In[ ]:


svm_gs.cv_results_


# In[ ]:


svm_results=pd.DataFrame(svm_gs.cv_results_)
svm_results


# In[ ]:


print(svm_gs.best_score_)
print(svm_gs.best_params_['C'])


# In[ ]:


#Plotting Test vs Train score graph
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.plot(svm_results['param_C'],svm_results['mean_test_score'])
plt.plot(svm_results['param_C'],svm_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend(['test accuracy', 'train accuracy'])#loc='upper left')


# In[ ]:


#fitting the final Logistic Regression model with the best hyperparameter value
#model=LogisticRegression(C=logis_cv.best_params_['C'])
#model.fit(X_train,y_train)


# In[ ]:


#Fitting the Final Model with the best Hyper parameter
model=svm.SVC(svm_gs.best_params_['C'])
model.fit(X_train,y_train)


# In[ ]:


#Making predictions with a Logistic regression model
labels=model.predict(X_test)
output_label=pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':labels})


# In[ ]:


output_label.to_csv('mysubmission.csv',index=False)
print('Your submissions file was successfully saved!')


# In[ ]:




