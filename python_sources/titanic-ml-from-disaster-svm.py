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


tr=pd.read_csv('/kaggle/input/titanic/train.csv')
te=pd.read_csv('/kaggle/input/titanic/test.csv')
sub=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[ ]:


y=tr['Survived']


# In[ ]:


tr


# In[ ]:


te


# To apply FE&EDA on both train and test sets , its better to concat them before performing FE &EDA

# In[ ]:


df = pd.concat([tr , te], axis=0,sort=False)


# In[ ]:


df


# * **FEATURE ENGINEERING & EXPLORATORY DATA ANALYSIS**

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# **We can strip Title and LastName from Name which would be a good feature**

# In[ ]:


df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
df['LastName'] = df.Name.str.split(',').str[0]
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))


# In[ ]:





# In[ ]:


df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
df.loc[(df['Deck'] == 'T'), 'Deck'] = 'A'


# In[ ]:


df.head()


# **df.info()** gives the details of dataframe like dtype, #non-null for every column

# In[ ]:


df['LastName'].value_counts()


# we can reduce and make better set of featues by replacing LastNames with count =1 as Single.(#LastNames reduced by 636)

# In[ ]:


title_last=(df['LastName'].value_counts()==1)
df['LastName'] = df['LastName'].apply(lambda x: 'Single' if title_last.loc[x] == True else x)

print(df['LastName'].value_counts())


# fillng the mssing values 
# 
# Age is to be filled with random integers between (ave_age-std_age , ave_age+std_age)
# 
# MODE FOR Embarked 
# 
# MEDIAN for Fare
# 
# as there are more number of missing values in cabin it should be handled in another way

# In[ ]:


age_avg = df['Age'].mean()
age_std = df['Age'].std()
age_null_count = df['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
df['Age'][np.isnan(df['Age'])] = age_null_random_list
df['Age'] = df['Age'].astype(int)


# In[ ]:


df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)
df['Fare'].fillna(df['Fare'].median(), inplace = True)


# creating bins with grouping age and Fare  as new features

# In[ ]:


df['FareBin'] = pd.qcut(df['Fare'], 4)
df['AgeBin'] = pd.cut(df['Age'].astype(int),5)


# In[ ]:


df.info()


# In[ ]:


df['Title'].value_counts()


# In[ ]:


#df['Title'] = df['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 
#                                             'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

title_names = (df['Title'].value_counts() < 10)
df['Title'] = df['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x) #orcan use comment by manual picking
print(df['Title'].value_counts())


# In[ ]:


#print(df['Deck'].value_counts())


# In[ ]:


df.info()


# we can dropout  columns that dont help in aking hypothesis like **Survived , PassengerId , Name, Ticket , Cabin**

# In[ ]:


drop_column = ['PassengerId','Cabin', 'Ticket', 'Name','Survived']
df.drop(drop_column,axis=1,inplace=True)


# Now after feature engineering i.e building possible features , we have to handle caegorical variables by LabelEncoding  or by  get_dummies.

# In[ ]:


df= pd.get_dummies(df)
df.head()


# In[ ]:


X=df.iloc[:891,:].values
Xtest=df.iloc[891:,:].values


# In[ ]:


'''from xgboost import XGBClassifier
classifier=XGBClassifier(n_estimators=350,reg_lambda=0.15,max_depth=3)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=35, max_depth=5, random_state=1)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0,max_iter=40)'''
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0 , probability=True)


# In[ ]:


classifier.fit(X, y)
predictions = classifier.predict(Xtest)
print('done')


# use gridsearch to get best hyperparameters 

# In[ ]:


'''from sklearn.model_selection import GridSearchCV
#parameters = [{'n_estimators': [200,300,400] ,'max_depth':[2,3],'reg_lambda':[0.1,0.2,0.25]}]
#parameters = [{'n_estimators': [16,18,15,14] ,'max_depth':[4,5,6]}]
#parameters = [{'max_iter':[100,90,80,5,10]}]
parameters = [{'degree':[1,2,3,5,4]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X, y)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print('done')'''


# use cross_val_score to know about the fit by comparing acc.mean() and acc.std() and tune the hyperparameters

# In[ ]:


'''from sklearn.model_selection import cross_val_score
acc=cross_val_score(estimator=classifier, X=X , y=y , cv=5)
acc'''


# In[ ]:


#acc.mean()
#acc.std()


# In[ ]:


print('done')


# In[ ]:




