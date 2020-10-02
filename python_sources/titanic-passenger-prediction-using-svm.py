#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
gender_submission = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


train.head()


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


pd.options.display.max_colwidth = 100


# ## 1) Missing Value Handling :

# In[ ]:


train.isnull().mean()


# In[ ]:


test.isnull().mean()


# ### Firstly,we'll drop the columns with missing values of above 40% or more,and impute the columns with less than 2-5% of values missing :

# In[ ]:


train = train.drop('Cabin',axis=1)
test = test.drop('Cabin',axis=1)


# In[ ]:


train['Embarked'].head()


# In[ ]:


train['Embarked'].mode()


# In[ ]:


train['Embarked'] = train['Embarked'].fillna('S')


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# Fare in test data :
fare_test = test[test['Fare'].notnull()]
sns.distplot(fare_test['Fare'],rug=True)
plt.show()


# In[ ]:


fare_test['Fare'].quantile(0.50) # Checking Median


# In[ ]:


# Let's impute it with Median and see first:
test['Fare'] = test['Fare'].fillna(test['Fare'].median())


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# ### Now,comes 'Age' column :

# In[ ]:


names = train['Name'].str.split('.')
names_1 = []
for i in names:
    names_1.append(i[0].split(',')[1])


# In[ ]:


names


# In[ ]:


names_1 = []
for i in names:
    names_1.append(i[0].split(',')[1])


# In[ ]:


names_1


# In[ ]:


train['Name_Referred'] = names_1


# In[ ]:


train['Name_Referred'].value_counts()


# In[ ]:


train.groupby('Name_Referred')['Survived'].value_counts()


# In[ ]:


train['Name_Referred'] = train['Name_Referred'].replace({' Mr':'Mr',' Mrs':'Mrs',' Ms':'Ms',' Rev':'Rev',' Capt':'Military Man',' Col':'Military Man',' Don':'Noble',' Jonkheer': 'Noble',' Lady':'Noble',' Major':'Military Man',' Mlle':'Miss',' Mme':'Ms',' Master':'Master',' Miss':'Miss',' Sir':'Noble',' the Countess':'Noble',' Dr':'Dr'})


# In[ ]:


train.groupby('Name_Referred')['Survived'].value_counts()


# ### Let's also do the same for testing data now :

# In[ ]:


names = test['Name'].str.split('.')
names_1 = []
for i in names:
    names_1.append(i[0].split(',')[1])


# In[ ]:


test['Name_Referred'] = names_1
test['Name_Referred'].value_counts()


# In[ ]:


test['Name_Referred'] = test['Name_Referred'].replace({' Mr':'Mr',' Mrs':'Mrs',' Ms':'Ms',' Rev':'Rev',' Col':'Military Man',' Dona':'Noble',' Master':'Master',' Miss':'Miss',' Dr':'Dr'})


# In[ ]:


test['Name_Referred'].value_counts()


# ### My target initially for imputing the missing values in 'Age' column is by using it as a dependent variable,and using all other columns,including the newly created 'Referred Name' as the predictors.
# 
# ### So,I split the 'train' data into 2 parts.One with the values present which I'll use to train,and the other with the values missing,which I'll use as the data for which I'll predict the values :

# In[ ]:


train_Age_NotMissing = train[train['Age'].notnull()] # Training data
# I should also avoid using the column 'Survived' as it's not present in the testing data provided by Kaggle.
Survived_AgeNotMissing = train_Age_NotMissing.pop('Survived')


# In[ ]:


print(train_Age_NotMissing.shape)


# In[ ]:


train_Age_NotMissing = train_Age_NotMissing.drop(['PassengerId','Name','Ticket','Fare'],axis=1)


# In[ ]:


train_Age_NotMissing = pd.get_dummies(train_Age_NotMissing,columns=['Embarked','Sex','Name_Referred'],drop_first = True)


# In[ ]:


train_Age_NotMissing.head()


# In[ ]:


train_Age_NotMissing_y = train_Age_NotMissing.pop('Age')
train_Age_NotMissing_X = train_Age_NotMissing


# In[ ]:


from sklearn import svm
ml = svm.SVR(kernel='linear',gamma='scale')


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(ml,train_Age_NotMissing_X,train_Age_NotMissing_y,cv=4)


# In[ ]:


scores.mean()


# In[ ]:


from sklearn.metrics import SCORERS
SCORERS.keys()


# In[ ]:


ml.fit(train_Age_NotMissing_X,train_Age_NotMissing_y)


# In[ ]:


test = test.drop(['PassengerId','Name','Ticket','Fare'],axis=1)
test = pd.get_dummies(test,columns=['Embarked','Sex','Name_Referred'],drop_first=True)


# In[ ]:


test_age = test['Age']
test_y = test.pop('Age')
test_X = test


# In[ ]:


train_Age_NotMissing_X.head()


# In[ ]:


test_X.head()


# In[ ]:


test['New_Age'] = ml.predict(test_X)


# In[ ]:


test['Age'] = test_age


# In[ ]:


test['Age'] = test['Age'].fillna(0)


# In[ ]:


test.Age.isnull().sum()


# In[ ]:


for i in range(0,len(test['Age'])):
    if test['Age'][i] == 0:
        test['Age'] = test['Age'].replace(test['Age'][i],test['New_Age'][i])


# In[ ]:


test[test['Age']==0]


# In[ ]:


test = test.drop(['New_Age'],axis=1)


# #### Similarly for train data :

# In[ ]:


train.head()


# In[ ]:


train_New = train.drop(['PassengerId','Name','Ticket','Fare','Survived'],axis=1)
train_New = pd.get_dummies(train_New,columns=['Embarked','Sex','Name_Referred'],drop_first=True)


# In[ ]:


train_New.head()


# In[ ]:


train_New_Age=train_New.pop('Age')
train_New_X = train_New
train_New['New_Age'] = ml.predict(train_New_X)


# In[ ]:


train_New['Age'] = train_New_Age
train_New['Age'] = train_New['Age'].fillna(0)


# In[ ]:


for i in range(0,len(train_New['Age'])):
    if train_New['Age'][i] == 0:
        train_New['Age'] = train_New['Age'].replace(train_New['Age'][i],train_New['New_Age'][i])


# In[ ]:


train_New[train_New['Age']==0]


# In[ ]:


train_New = train_New.drop(['New_Age'],axis=1)


# In[ ]:


train_New.head()


# In[ ]:


test.head()


# ## 2) Outlier Treatment :

# ### Before doing outlier treatment,let's get the 'Survived' column here,as that'll allow me to carry out outlier operations on the dataset :

# In[ ]:


train_New['Survived'] = train['Survived']


# ### Let's first do the outlier treatment for 'Age',now :

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(train_New['Age'])
plt.show()


# In[ ]:


sns.boxplot(test['Age'])
plt.show()


# In[ ]:


print(train_New['Age'].quantile(0.99))
print(test['Age'].quantile(1.0))


# In[ ]:


print(train_New['Age'].quantile(0.001))
print(test['Age'].quantile(0.001))


# In[ ]:


train_New = train_New[train_New['Age'] <= 76]


# In[ ]:


train_New.shape


# In[ ]:


train_New.corr()


# In[ ]:


print(train_New.isnull().sum())
print(test.isnull().sum())


# In[ ]:


sns.pairplot(train_New)
plt.show()


# In[ ]:


plt.figure(figsize=(26,20),dpi=30)
sns.set(font_scale=2)
sns.heatmap(train_New.corr(),cmap='YlGnBu')


# In[ ]:


train_New['Family_members'] = train_New['SibSp'] + train_New['Parch']
test['Family_members'] = test['SibSp'] + test['Parch']


# In[ ]:


train_New = train_New.drop(['SibSp','Parch'],axis=1)
test = test.drop(['SibSp','Parch'],axis=1)


# In[ ]:


train_New['Family_members'].value_counts()


# In[ ]:


test['Family_members'].value_counts()


# ## 3) Scaling :

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()
train_New.loc[:,['Age','Family_members']] = scaler.fit_transform(train_New.loc[:,['Age','Family_members']])


# In[ ]:


test.loc[:,['Age','Family_members']] = scaler.transform(test.loc[:,['Age','Family_members']])


# ## 4) Train-Test Split :

# In[ ]:


y_train = train_New.pop('Survived')
X_train = train_New
X_test = test


# ## 5) Modelling :

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn import svm
param_grid = {'kernel' : ['rbf','poly','linear']}
ml_1 = svm.SVC(gamma = 'scale')
grid_search = GridSearchCV(estimator = ml_1,param_grid=param_grid,cv = 4,scoring = 'accuracy', n_jobs=-1,verbose=1)
grid_search.fit(X_train,y_train)


# In[ ]:


print(grid_search.best_score_)
print(grid_search.best_params_)


# In[ ]:


ml_1 = svm.SVC(kernel='rbf',gamma='scale')


# In[ ]:


ml_1.fit(X_train,y_train)


# In[ ]:


y_pred = ml_1.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(gender_submission['Survived'],y_pred)


# ## 7) Submission :

# ### Here is how I create my submission file,viz.'submission_8'.

# In[ ]:


test_X['Survived'] = y_pred
test_X[test_X['Age']<1]


# In[ ]:


submission_8 = pd.DataFrame()
submission_8['PassengerId'] = gender_submission['PassengerId']
submission_8['Survived'] = y_pred


# In[ ]:


gender_submission.head() # To check the required form of submission.


# In[ ]:


submission_8.head()


# In[ ]:


submission_8.to_csv('titanic_submission_8.csv',index = False, encoding='utf-8')

