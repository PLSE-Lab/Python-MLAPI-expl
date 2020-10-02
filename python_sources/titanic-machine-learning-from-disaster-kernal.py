#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


from sklearn.compose import ColumnTransformer
import re
from statistics import mean
import math
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , StandardScaler , LabelEncoder , MinMaxScaler
from sklearn.metrics import mean_absolute_error , mean_squared_error
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score , train_test_split , StratifiedKFold , GridSearchCV , learning_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
sub = pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


train_data.shape


# In[ ]:


print(train_data.isnull().sum())
(train_data[train_data['Embarked'].isnull()])


# In[ ]:


train_data.head()
#print(train_data.shape)


# In[ ]:


pd.crosstab(train_data.Pclass , train_data.Survived)


# In[ ]:


plt.figure()
ax = sns.countplot('Pclass' , hue = 'Survived' , data = train_data)
height = sum([p.get_height() for p in ax.patches])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 2,
            '{:1.2f}'.format(100* height/len(train_data)),
            ha="center") 
plt.show()


# In[ ]:


plt.figure()
ax = sns.countplot('Sex' , hue = 'Survived' , data = train_data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2 , height+2 , '{:1.2f}'.format(100 * height / len(train_data)) , ha = 'center')


# In[ ]:


pd.crosstab(train_data.Sex , train_data.Survived)


# In[ ]:


train_data.head()


# In[ ]:


pd.crosstab([train_data.Sex , train_data.Pclass] , train_data.Survived , margins = True)


# In[ ]:


train_data.head()


# In[ ]:


train_data[train_data['Age'].isnull()]


# In[ ]:


my_imputer = LabelEncoder()
train_data['Sex']= my_imputer.fit_transform(train_data['Sex'])
test_data['Sex']= my_imputer.fit_transform(test_data['Sex'])


# In[ ]:


train_data.head()


# In[ ]:


ax = sns.countplot('Embarked' , data = train_data)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2 , height+4, '{:1.2f}'.format(height) , ha = 'center')


# In[ ]:


train_data['Embarked'].fillna('S',inplace = True)
test_data['Embarked'].fillna('S',inplace = True)


# In[ ]:



sns.factorplot('Embarked' , 'Survived', data = train_data)


# In[ ]:


print(train_data['Embarked'].isnull().sum())


# for i in range (0,len(train_data)):
#     if(train_data['Name'][i].find('Mr.')!=-1):
#         train_data['Name'][i] = 'Mr.'
#     elif(train_data['Name'][i].find('Mrs.')!=-1):
#          train_data['Name'][i] = 'Mrs.'
#     elif(train_data['Name'][i].find('Miss')!=-1):
#          train_data['Name'][i] = 'Miss'
#     else:
#         train_data['Name'][i] = 'other'

# x = ['Mr.' , 'Mrs.' , 'Miss' , 'Other']
# y = [train_data[train_data['Name'] == 'Mr.'].Age.mean() , 
#       train_data[train_data['Name'] == 'Mrs.'].Age.mean() , 
#        train_data[train_data['Name'] == 'Miss'].Age.mean() , 
#         train_data[train_data['Name'] == 'other'].Age.mean()]
# 
# ax = sns.barplot(x , y)
# plt.xlabel('Category')
# plt.ylabel('average age')
# for p in ax.patches:
#     height = p.get_height()
#     ax.text(p.get_x() + p.get_width()/2 , height+0.5
#             , '{:1.2f}'.format(height) ,ha= 'center')

# In[ ]:


train_data['Title'] = train_data.Name.apply(lambda x : re.search('([A-Z][a-z]+)\.' , x).group(1))
test_data['Title'] = test_data.Name.apply(lambda x : re.search('([A-Z][a-z]+)\.' , x).group(1))


# In[ ]:


pd.crosstab(train_data.Title , train_data.Sex).T


# In[ ]:


train_data.Title.replace(['Capt' , 'Col' , 'Countess' , 'Don' , 'Dr' , 'Jonkheer' , 'Lady' , 'Major' , 'Mlle' , 'Mme',
                         'Ms' , 'Rev' , 'Sir'] , ['Mr' , 'Mr' , 'Mrs' , 'Mr' , 'Mr' , 'Mr' , 'Mrs' , 'Mr',
                                                 'Miss' , 'Miss' , 'Mrs' , 'Mr' , 'Mr'] , inplace = True)

test_data.Title.replace(['Capt' , 'Col' , 'Countess' , 'Don' , 'Dr' , 'Jonkheer' , 'Lady' , 'Major' , 'Mlle' , 'Mme',
                         'Ms' , 'Rev' , 'Sir'] , ['Mr' , 'Mr' , 'Mrs' , 'Mr' , 'Mr' , 'Mr' , 'Mrs' , 'Mr',
                                                 'Miss' , 'Miss' , 'Mrs' , 'Mr' , 'Mr'] , inplace = True)


# In[ ]:


pd.crosstab(train_data.Title , train_data.Survived).T


# In[ ]:


pd.crosstab([train_data.Title,train_data.Pclass] , train_data.Survived).T


# In[ ]:


sns.factorplot('Title' , 'Survived' , col = 'Pclass' , data = train_data)


# In[ ]:


print('Average Age of Mr is {0}'.format(train_data[train_data.Title == 'Mr'].Age.mean()))
print('Average Age of Mrs is {0}'.format(train_data[train_data.Title == 'Mrs'].Age.mean()))
print('Average Age of Master is {0}'.format(train_data[train_data.Title == 'Master'].Age.mean()))
print('Average Age of Miss is {0}'.format(train_data[train_data.Title == 'Miss'].Age.mean()))


# In[ ]:


train_data.loc[(train_data.Age.isnull())&(train_data.Title=='Mr'),'Age']=33
train_data.loc[(train_data.Age.isnull())&(train_data.Title=='Mrs'),'Age']=36
train_data.loc[(train_data.Age.isnull())&(train_data.Title=='Master'),'Age']=4.57
train_data.loc[(train_data.Age.isnull())&(train_data.Title=='Miss'),'Age']=22




test_data.loc[(test_data.Age.isnull())&(test_data.Title=='Mr'),'Age']=33
test_data.loc[(test_data.Age.isnull())&(test_data.Title=='Mrs'),'Age']=36
test_data.loc[(test_data.Age.isnull())&(test_data.Title=='Master'),'Age']=4.57
test_data.loc[(test_data.Age.isnull())&(test_data.Title=='Miss'),'Age']=22


# In[ ]:


train_data.Age.isnull().sum()
test_data.Age.isnull().sum()


# In[ ]:


sns.countplot('SibSp' , hue = 'Survived' , data = train_data)
sns.factorplot('SibSp' , 'Survived' , data = train_data)


# In[ ]:


pd.crosstab(train_data.Parch , train_data.Survived).T


# In[ ]:


sns.factorplot('Parch' , 'Survived' , data = train_data)


# In[ ]:


train_data['Age_range'] = 0
train_data.loc[(train_data.Age<=20) , 'Age_range'] = 0
train_data.loc[((train_data.Age>20) & (train_data.Age<=40)) , 'Age_range'] = 1
train_data.loc[((train_data.Age>40) & (train_data.Age<=60)) , 'Age_range'] = 2
train_data.loc[(train_data.Age>60) , 'Age_range'] = 3


test_data['Age_range'] = 0
test_data.loc[(test_data.Age<=20) , 'Age_range'] = 0
test_data.loc[((test_data.Age>20) & (test_data.Age<=40)) , 'Age_range'] = 1
test_data.loc[((test_data.Age>40) & (test_data.Age<=60)) , 'Age_range'] = 2
test_data.loc[(test_data.Age>60) , 'Age_range'] = 3


# In[ ]:


pd.crosstab(train_data.Age_range , train_data.Survived)


# In[ ]:


sns.factorplot('Age_range' , 'Survived' , data = train_data)


# In[ ]:


print('Min of Fare Value is: {0}'.format(train_data.Fare.min()))
print('Max of Fare Value is: {0}'.format(train_data.Fare.max()))
print('Avg of Fare Value is: {0}'.format(train_data.Fare.mean()))


# In[ ]:


sns.distplot(train_data['Fare']  )


# In[ ]:


my_encoder = LabelEncoder()
train_data.Fare = train_data.Fare.fillna(-0.5)

range1 = [-1 , 0 , 8 , 15 , 31 , 600]



train_data['Fare_range'] = pd.cut(train_data['Fare'] , range1)
train_data['Fare_range'] = my_encoder.fit_transform(train_data['Fare_range'])



test_data.Fare = test_data.Fare.fillna(-0.5)

range1 = [-1 , 0 , 8 , 15 , 31 , 600]


test_data['Fare_range'] = pd.cut(test_data['Fare'] , range1 )
test_data['Fare_range'] = my_encoder.fit_transform(test_data['Fare_range'])


# In[ ]:


pd.crosstab(train_data.Fare_range , train_data.Survived).T


# In[ ]:


sns.factorplot('Fare_range' , 'Survived' , data = train_data)


# In[ ]:


train_data


# In[ ]:


train_data['Family_Size'] = train_data['SibSp'] + train_data['Parch'] + 1

test_data['Family_Size'] = test_data['SibSp'] + test_data['Parch'] + 1


# In[ ]:


sns.countplot('Family_Size'  ,hue = 'Survived' , data = train_data)
sns.factorplot('Family_Size' , 'Survived' , data = train_data)


# In[ ]:


train_data.drop(['PassengerId' , 'Name' , 'Age' , 'SibSp' , 'Parch' , 'Ticket' , 'Fare' ,'Cabin'] , 1 , inplace = True)

test_data.drop(['PassengerId','Name' , 'Age' , 'SibSp' , 'Parch' , 'Ticket' , 'Fare' ,'Cabin'] , 1 , inplace = True)


# In[ ]:


test_data


# In[ ]:


test_data


# In[ ]:


encoder = LabelEncoder()
train_data['Embarked'] = encoder.fit_transform(train_data['Embarked'])
train_data['Title'] = encoder.fit_transform(train_data['Title'])


test_data['Embarked'] = encoder.fit_transform(test_data['Embarked'])
test_data['Title'] = encoder.fit_transform(test_data['Title'])


# In[ ]:


X = (train_data.iloc[: , 1:]).values
y = (train_data.iloc[: ,:1]).values


# In[ ]:


X_train , X_test , y_train , y_test = train_test_split(X , y , train_size = 0.8 , test_size = 0.2 , random_state = 0
                                                    )


# In[ ]:


kfold = StratifiedKFold(n_splits=10)


# In[ ]:


random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))


# In[ ]:


cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X, y , scoring = "accuracy", cv = kfold, n_jobs=4))


# In[ ]:


cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())


# In[ ]:


cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting"]})


# In[ ]:


cv_res


# In[ ]:


RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X , y)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_


# In[ ]:


pred = gsRFC.predict(test_data)
print(mean_absolute_error(pred , sub['Survived']))


# In[ ]:


print(pred)


# In[ ]:


sub['Survived'] = pred


# In[ ]:


sub


# In[ ]:


sub.to_csv('submission.csv' , index = False)


# In[ ]:




