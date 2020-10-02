#!/usr/bin/env python
# coding: utf-8

# Hello, Today I am going to give the titanic problem a try ! hopefully i archive good score

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder 
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# First thing first is reading the data

# In[ ]:


data= pd.read_csv('/kaggle/input/titanic/train.csv')


# getting the basic information from it

# In[ ]:


print("data shape: " ,data.shape)
print("data columns: " ,list(data.columns))


# In[ ]:


data.info()


# In[ ]:


data.describe()


# Okay!
# # Now some visualization

# In[ ]:


data['Survived'].value_counts().plot.pie(figsize=(6,6))


# In[ ]:


def plot_by(data,feature):
    groupedData = data[feature].groupby(data["Survived"])
    groupedData = pd.DataFrame({
        'Survived' : groupedData.get_group(1),
        'Dead': groupedData.get_group(0)
    })
    histogram = groupedData.plot.hist(bins=40,alpha=0.4)
    histogram.set_xlabel(feature)
    histogram.plot()


# In[ ]:


data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})
data['Sex'].values.reshape(-1,1)
enc = OneHotEncoder()
ss = pd.DataFrame(enc.fit_transform(data['Sex'].values.reshape(-1,1)).toarray(),columns=['male','female'])


# In[ ]:


plot_by(data,'Parch')


# In[ ]:


plot_by(data,'SibSp')


# In[ ]:


data['Familly'] = data['SibSp'] + data['Parch']
plot_by(data,'Familly')


# In[ ]:


#new features male and female using one hot encoders 
data['Male'] = ss['male']
data['female'] = ss['female']
data = data.loc[:, data.columns != 'Sex']
data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
data.head()


# In[ ]:


#new feature (title)
title_names = (data['Title'].value_counts() < 10)
data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
l = LabelEncoder()
data['Title'] = l.fit_transform(data['Title'])


# In[ ]:


Embarked = set(data.loc[data["Embarked"].notnull()]["Embarked"])
Embarked_to_number = {ni: indi for indi, ni in enumerate(set(Embarked))}
print(Embarked)
print(Embarked_to_number)
data['Embarked'] = data['Embarked'].map(Embarked_to_number)


# In[ ]:


data['alone'] = (data['Familly'] == 0).astype('int')


# In[ ]:


data['Embarked'].hist()


# Let's check ths null values now !

# In[ ]:


data.isnull().sum()


# Cabin has to many null values, so i think the best option is to delete it
# and PassengerId will not give us any information 

# In[ ]:


data = data.loc[:, data.columns != 'PassengerId']
data = data.loc[:, data.columns != 'Cabin']


# In[ ]:


print(data.shape)
#complete missing age with median
data['Age'].fillna(data['Age'].median(), inplace = True)

#complete embarked with mode
data['Embarked'].fillna(data['Embarked'].median(), inplace = True)

#complete missing fare with median
data['Fare'].fillna(data['Fare'].median(), inplace = True)
data = data.dropna(how='any')

#####################
data.isnull().sum()


# In[ ]:


data = data.loc[:, data.columns != 'Ticket']
data = data.loc[:, data.columns != 'Name']
data.info()


# In[ ]:


import seaborn as sns #the librery we'll use for the job xD

corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cbar=True, annot=True, square=True, vmax=.8);


# In[ ]:


data.head()


# In[ ]:


data.columns


# # Now let's train!

# importing the needed libraries!

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score, make_scorer,f1_score, precision_score, recall_score, confusion_matrix


# tweaks on the data !

# In[ ]:


#preparing the data
X_cols = list(data.columns)
X_cols.remove('Survived')
Y=data['Survived']
rescaledX = StandardScaler().fit_transform(data[X_cols])
X=pd.DataFrame(data = rescaledX, columns= X_cols)
# X=data[X_cols]
X_train,Y_train = X,Y


# now let's try on diffrent models

# In[ ]:


get_ipython().run_cell_magic('time', '', 'svm1 = svm.SVC(kernel=\'linear\',random_state = 42)\nsvm2 = svm.SVC(kernel=\'rbf\',random_state = 42) \nlr = LogisticRegression(random_state = 42)\ngb = GaussianNB()\nrf = RandomForestClassifier(random_state = 42)\nknn = KNeighborsClassifier(n_neighbors=15)\ntree = tree.DecisionTreeClassifier()\nmodels = {"Logistic Regression": lr, \'DecisionTreeClassifier\' : tree, "Random Forest": rf, "svm linear": svm1 , "svm rbf": svm2,"KNeighborsClassifier": knn ,\'GaussianNB\': gb}\nl=[]\nfor model in models:\n    l.append(make_pipeline(Imputer(),  models[model]))\n\n        \ni=0\nfor Classifier in l:    \n    accuracy = cross_val_score(Classifier,X_train,Y_train,scoring=\'accuracy\',cv=5)\n    print("===", [*models][i] , "===")\n    print("accuracy = ",accuracy)\n    print("accuracy.mean = ", accuracy.mean())\n    print("accuracy.variance = ", accuracy.var())\n    i=i+1\n    print("")\n    ')


# Okay let's pick the top four models 
# * svm rbf
# * KNeighborsClassifier
# * GaussianNB 
# * Random Forest 
# 
# let's start with knn 

# In[ ]:


k_number = [i for i  in range(1,100,2)]
acc = []
ks = []
for i in range(len(k_number)):
    knn = KNeighborsClassifier(n_neighbors=k_number[i])
    accuracy = cross_val_score(knn,X_train,Y_train,scoring='accuracy',cv=5)
    acc.append(accuracy.mean())
    ks.append(k_number[i])
plt.plot(ks,acc)


# In[ ]:


print("best knn score: ", max(acc))
print("best k is:", ks[acc.index(max(acc))] )


# **Great**, now let's move to Random forest, I would like to run few **grid search** iteration to find the best paramters/estimator, let's see :

# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True,False],
    'max_depth': [130,140,150,160, None],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 2],
    'min_samples_split': [12, 14, 16],
    'n_estimators': [100, 110, 120, 130,140]
}
# Create a based model
rf = RandomForestClassifier(random_state = 42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, Y_train)


# let's see the best score we got

# In[ ]:


print(grid_search.best_score_)
print(grid_search.best_estimator_)
rf = grid_search.best_estimator_


# As we you saw random forest did a good job, but not as much as SVM with the rbf kernel, so it will be out official estimator.

# In[ ]:


svm2 = svm.SVC(kernel='rbf',random_state = 42) 
svm2.fit(X_train, Y_train)


# # moving on to the test set
# don't try to read too much haha, I will just apply what I have done to the train data on the test data so I can run through the estimator.

# In[ ]:


data= pd.read_csv('/kaggle/input/titanic/test.csv')
test = data.copy()
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})
data['Sex'].values.reshape(-1,1)
enc = OneHotEncoder()
ss = pd.DataFrame(enc.fit_transform(data['Sex'].values.reshape(-1,1)).toarray(),columns=['male','female'])
data['Familly'] = data['SibSp'] + data['Parch']
data['Male'] = ss['male']
data['female'] = ss['female']
data = data.loc[:, data.columns != 'Sex']
data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
title_names = (data['Title'].value_counts() < 10)
data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
l = LabelEncoder()
data['Title'] = l.fit_transform(data['Title'])
Embarked = set(data.loc[data["Embarked"].notnull()]["Embarked"])
Embarked_to_number = {ni: indi for indi, ni in enumerate(set(Embarked))}
data['Embarked'] = data['Embarked'].map(Embarked_to_number)
data['alone'] = (data['Familly'] == 0).astype('int')
data = data.loc[:, data.columns != 'PassengerId']
data = data.loc[:, data.columns != 'Cabin']
data['Age'].fillna(data['Age'].median(), inplace = True)
data['Embarked'].fillna(data['Embarked'].median(), inplace = True)
data['Fare'].fillna(data['Fare'].median(), inplace = True)
data = data.dropna(how='any')
data.isnull().sum()
data = data.loc[:, data.columns != 'Ticket']
data = data.loc[:, data.columns != 'Name']
X_cols = list(data.columns)
rescaledX = StandardScaler().fit_transform(data[X_cols])
X=pd.DataFrame(data = rescaledX, columns= X_cols)
X_test = X


# In[ ]:


predictions = svm2.predict(X_test)
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})


# In[ ]:


filename = 'Titanic Predictions 1.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)

