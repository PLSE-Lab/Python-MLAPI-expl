#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Read all the datasets required

# In[ ]:


dataframe = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# Start the data analysis part where one has to understand what we are dealing with 

# In[ ]:



dataframe.info()
print(dataframe.columns.values)


# 1. We have seen Float, int and object(string or character) data types our raw dataset. 
# 2. The dataframe has columns with integers and Float values at PassengerID, Survived, Age, Sibsp, Parch, Fare, Pclass
# rest with object datatypes

# # ** Data Analysis**
# > Survival on this ship can be based on age, class and or the place they entered the ship 
# 
# Survival based on Sex
# 

# In[ ]:


sns.countplot(dataframe['Survived'], hue = dataframe['Sex'])
Dead, lives = dataframe.Survived.value_counts()
male, female = dataframe.Sex.value_counts()
print("Percentage of Male on ship:", round(male/(male+female)*100) )
print("Percentage of Female on ship:", round(female/(male+female)*100 ))


# * 65% of people on this sunked ship are males and rest are females
# * Also by looking at the graph of survival against the sex, there are more females survived than the male and the death count is also high for the males.
# * This shows the women are given the first seat at the rescue boats ..

# In[ ]:


from matplotlib import pyplot as plt
plt.figure(figsize=(40,8))
sns.countplot(dataframe['Age'], hue = dataframe['Survived'])


# *  The above graph of survival based on age is so obvious that the young and middle ages people from 16 to 40 has the highest death count as shown in blue bars on above graph 
# * the people above 70 might have voluntarily chose to stay back on ship 
# * The infants and kids are also saved as shown in above graph 
# 

# **# survived based on Passenger Class**

# There are 3 passenger classes named as 1st, 2nd and 3rd class

# In[ ]:


#find out how many classes on ship
dataframe.Pclass.unique()
#so there are 3 classes on ship
dataframe.Pclass.value_counts()
#in which 1st, 2nd and 3rd class has 216, 184 and 491 respectively 
#The graph shows the most people died in 3rd class which is obvious from the
#number of people who bought 3rd class tickets are high


# In[ ]:


sns.countplot(dataframe['Pclass'], hue = dataframe['Survived'])
t_p = dataframe.groupby('Pclass')['Survived']
print(t_p.sum())


# * although the class 1 and 2 has almost equal survival and death counts(class 1 has more survivals better than 2 as class 1 are "rich" people )
# * on the other hand class 3 has most death count
# 
# ![A_NIGHT_TO_REMEMBER_DISC01-07.jpg](attachment:A_NIGHT_TO_REMEMBER_DISC01-07.jpg)
# 

# In[ ]:


#The Embarked class does not give much info other than S class embarkment has ppl from all different classes
sns.countplot(dataframe['Embarked'], hue = dataframe['Survived'])


# In[ ]:


sns.distplot(dataframe['Fare'])
dataframe['Fare'].describe()


# # To fill the missing values in different columns starting with age

# In[ ]:


#find the null values in different columns first 
#Find the null values if any in our DataFrame
dataframe.isnull().values.any()
dataframe.isnull().sum()


# We have to deal with missing values in Age, Cabin and Embarked
# The Cabin data is not that helpful because when ship sinks everyone has to get out of their bunkers. phewwwwww 
# The Age can be replaced by simple mean of ages of all people

# In[ ]:


#for test dataset
test.isnull().sum()


# In[ ]:


dataframe['Age'].fillna(round(dataframe['Age'].mean()), inplace = True)
test['Fare'].fillna(test['Fare'].mean(), inplace = True)
#doing the same for Test
test['Age'].fillna(round(test['Age'].mean()), inplace = True)


# Check again for missing data after filling the age column

# In[ ]:


dataframe.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


dataframe['Age'].head(10)


# * The columns Name and Cabin has unique values so dropping it might be a good idea. 
# * Same with Test data

# In[ ]:


import seaborn as sns
correlations = dataframe[dataframe.columns].corr(method='pearson')
sns.heatmap(correlations, cmap="YlGnBu", annot = True)


# In[ ]:


import heapq

print('Absolute overall correlations')
print('-' * 30)
correlations_abs_sum = correlations[correlations.columns].abs().sum()
print(correlations_abs_sum, '\n')

print('Weakest correlations')
print('-' * 30)
print(correlations_abs_sum.nsmallest(4))


# In[ ]:


train_set = dataframe.drop( ['Name','Cabin', 'Ticket','PassengerId', ], axis = 1)
test_set = test.drop( ['Name','Cabin', 'Ticket', 'PassengerId', ], axis = 1)


# Also we had 2 missing values in Training set Embarked column
# 

# In[ ]:


test_set.isnull().sum()


# In[ ]:


train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace = True)


# In[ ]:


#train_set = train_set.dropna()
#test_set = test_set.dropna()


# 1. Selecting the dependent and Independent variables 
# Survival being the dependent and rest of the data as independent variables

# In[ ]:


train_set.head()


# In[ ]:


test_set.head()


# In[ ]:


y = train_set.iloc[:, 0].values
X = train_set.iloc[:, train_set.columns != 'Survived'].values
print(X[0])


# * Now we deal with categrical data in Train and Test set using LabelEncoder and OneHOtEncoder
# * Columns with Sex and Embarkment

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X[2])


# In[ ]:


#for test
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
test_set = np.array(ct.fit_transform(test_set))
print(test_set[1])


# Applying LabelEncoder to Sex Column

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 4 ] = le.fit_transform(X[:,4])

print(X[1])


# In[ ]:


#for test set
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
test_set[:, 4] = le.fit_transform(test_set[:,4])
print(test_set[2])


# # Splitting the training set and test set as X_train and X_test

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20 )


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test =sc.transform(X_test)


# # Now implementing various classification models

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import Ridge

rfc=RandomForestClassifier()
parameters= {'n_estimators':[ 100,200,300,400, 600],
             'max_depth':[3,4,6,7],
             'criterion':['entropy','gini']
    }

rfc=GridSearchCV(rfc, param_grid=parameters, cv = 5)
rfc.fit(X_train,y_train)
print("The best value of leanring rate is: ",rfc.best_params_, )


# In[ ]:


#RandomForest

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(criterion= 'gini', n_estimators = 100 ,max_depth = 6, random_state = 0)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
Random_forest_acc= accuracy_score(y_test, y_pred)
print('acc = ', Random_forest_acc )


# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
LogisticReg_acc= accuracy_score(y_test, y_pred)
print('acc = ', LogisticReg_acc )


# In[ ]:


from sklearn.svm import SVC
model = SVC(kernel = 'rbf', random_state = 0)
model.fit(X, y)
y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix
SVC_acc = accuracy_score(y_test, y_pred)
print('acc = ', SVC_acc )


# In[ ]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix
Gaussian_acc = accuracy_score(y_test, y_pred)
print('acc = ', Gaussian_acc )


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix
DT_acc = accuracy_score(y_test, y_pred)
print('acc = ', DT_acc )


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import Ridge

gbc=GradientBoostingClassifier()
parameters= {'n_estimators':[ 50,100,200,300, ],
             'max_depth':[3,4,6,7]
    }

gbreg=GridSearchCV(gbc, param_grid=parameters, cv = 5 )
gbreg.fit(X_train,y_train)
print("The best value of leanring rate is: ",gbreg.best_params_, )


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
model_gb = GradientBoostingClassifier(n_estimators = 100, max_depth =4, random_state = 42)
model_gb.fit(X_train, y_train)
y_pred = model_gb.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix
GB_acc = accuracy_score(y_test, y_pred)
print('acc = ', GB_acc )


# In[ ]:


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
xgb_acc = accuracy_score(y_pred, y_test)
print('acc=',xgb_acc)


# In[ ]:


print('RF_acc=', Random_forest_acc)
print('Logistic_acc=', LogisticReg_acc)
print('SVC_acc=', SVC_acc)
print('Gaussian_acc=', Gaussian_acc)
print('DecisionTree_acc=', DT_acc)
print('GradBoost_acc=', GB_acc)
print('XGBoost_acc=', xgb_acc)


# 
# 
# * Now test the gradient boost on Test dataset

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(criterion= 'gini', n_estimators = 100 ,max_depth = 6, random_state = 0)
rf_model.fit(X, y)


final_pred = rf_model.predict(test_set)

final_pred


# # Final Submission

# In[ ]:


survivors = pd.DataFrame(final_pred, columns = ['Survived'])
len(survivors)
survivors.insert(0, 'PassengerId', test['PassengerId'], True)
survivors


# In[ ]:


survivors.to_csv('Submission.csv', index = False)


# **Please upvote if you like this notebook**
