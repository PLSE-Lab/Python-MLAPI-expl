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


# Load data and import libraries

# In[ ]:


import pandas as pd
pd.set_option("display.max_columns",30)
pd.set_option('display.max_rows', 1000)


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

train = pd.read_csv(r'../input/titanic/train.csv')
test = pd.read_csv(r'../input/titanic/test.csv')


# Now I put togheter train and test to work simultaneusly on both of them.

# In[ ]:


combined_data = [train,test]


# In[ ]:


# info on the data
print(train.info())

print(train.head())


# In[ ]:


# check for null in training and test data
print("Null data before cleaning:")
for data in combined_data:
    print(data.isnull().sum())


# As you can see there are some missing values on Age and Cabin and only one missing value for Fare

# In[ ]:


#imputing missing values

for data in combined_data:
    data['Age'].fillna(value = data['Age'].mean(), inplace=True)
    data['Embarked'].fillna(value = data['Embarked'].mode()[0],inplace=True)
    data['Fare'].fillna(value = data['Fare'].mean(),inplace = True)

    
    
# drop cabin column , too many values at null
    
for data in combined_data:
    data.drop(columns=['Cabin','Name'], inplace=True)

print("Null data after cleaning:")

for data in combined_data:
    print(data.isnull().sum())


# After cleaning data noone columns has missing values. The next step is feature engineering, that is trying to create new variables starting from the exsisting ones.

# In[ ]:


# features engineering
    
for data in combined_data:
    data['Family_size'] = data['SibSp']+data['Parch']
    
#Ticket
for data in combined_data:
    data['Ticket_len'] = data['Ticket'].apply(lambda x: len(x))
    data['First_digit_ticket'] = data['Ticket'].apply(lambda x : x[0])

#drop ticket    
for data in combined_data:
    data.drop(columns=['Ticket'], inplace=True)


# Now what I want to do is find the relationships with our variables and the target variable using some graphs.

# In[ ]:


#survival rate by ticket lenght

print("Survival_rate_by_ticket_len", train.groupby(['Ticket_len'])['Survived'].mean())
print("Survival rate by first digit ticket ", train.groupby(['First_digit_ticket'])['Survived'].mean())    


# In[ ]:


# suvivival for categorical data
for x in train:
    if train[x].dtype ==  'object' and x != 'Ticket':
        print(train[train['Survived']==1].groupby(x)['Survived'].count())


# In[ ]:


#distribution of survival per age   
sns.kdeplot(data =train['Age'][train['Survived']==1] , shade=True,label = "Distribution of age for survived")
sns.kdeplot(data =train['Age'][train['Survived']==0] , shade=True,label = "Distribution of age for non survived")
plt.title("Distribution of survival by age")
plt.show()


# In[ ]:


#distrbibution of survived per age - pclass
sns.jointplot(x=train['Pclass'][train['Survived']==1], y=train['Age'][train['Survived']==1], data=train, kind="kde")
plt.title("Distribution of survived by age and Pclass")
plt.show()


# In[ ]:


#pairplot

sns.boxplot(x="Embarked", y="Fare",hue ="Survived" , data=train)
plt.show()


# In[ ]:


sns.relplot(x="Age", y="Fare", size="Fare", sizes=(15, 200),hue="Survived",  data=train);
plt.show()


# From the jointplto what we see is that  first class and third class have the highest probability of survival. 
# From the pairplot we can see that people who embarked in port C have more probability of survival and from the last graph we can see that the highest is the fare the highest is the probability of survive.

# In[ ]:


#ONE HOT ENCODING , to use categorical values in our model we need to hod encode them
colonne = ['Sex','Embarked']
    
train = pd.get_dummies(train, columns=colonne)
test = pd.get_dummies(test, columns=colonne)



    
columns_to_drop2 = ['Ticket_len']

train.drop(columns=columns_to_drop2, inplace=True)
test.drop(columns=columns_to_drop2, inplace=True)
    
    

label = LabelEncoder()
#train['Age_10bins_code'] = label.fit_transform(train['Age_10bins'])
#train['Fare_10bins_code'] = label.fit_transform(train['Fare_10bins'])
train['First_digit_ticket_code'] = label.fit_transform(train['First_digit_ticket'])
test['First_digit_ticket_code'] = label.fit_transform(test['First_digit_ticket'])
            
columns_to_drop3 = ['First_digit_ticket']

train.drop(columns=columns_to_drop3, inplace=True)
test.drop(columns=columns_to_drop3, inplace=True)   

    
             
print(" training list", train.columns.tolist())
print("test  lists",test.columns.tolist())


# Now we can build our models. I will use Random forest classifier, Random forest regressor and Logistic regressor. I will use grid search to find out the best parameters for our model.

# In[ ]:


features = [ 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Family_size', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'First_digit_ticket_code']

target = 'Survived'
X = train[features]
y = train[target]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 0 )


# In[ ]:



#random forest classifier
print("RANDOM FOREST CLASSIFIER")
rfc = RandomForestClassifier(max_features='auto',  n_jobs=-1)
params_rfc = {"criterion":['gini','entropy'],"n_estimators": [50, 100, 400, 700, 1000],"bootstrap":[True, False]}
grid_search_rfc = GridSearchCV(rfc, param_grid= params_rfc, cv=5, n_jobs=-1)
grid_search_rfc.fit(X_train,y_train)
print("score ", grid_search_rfc.score(X_train, y_train))
print("best parameter for random forest classifier ", grid_search_rfc.best_params_)


# In[ ]:


#random forest regressor
print("RANDOM FOREST REGRESSOR")
rfr = RandomForestRegressor(max_features='auto',  n_jobs=-1)
params_rfr = {"n_estimators": [50, 100, 400, 700, 1000],"bootstrap":[True, False]}
grid_search_rfr = GridSearchCV(rfr, param_grid= params_rfr, cv=5, n_jobs=-1)
grid_search_rfr.fit(X_train,y_train)
print("score ", grid_search_rfr.score(X_train, y_train))
print("best parameter for random forest classifier ", grid_search_rfr.best_params_)


# In[ ]:


#Logistic regression
print("LOGISTIC REGRESSION")
lr = LogisticRegression()
params_lr = {"penalty": ['l1','l2'],"C":[0.001,0.01,0.1,1,10,100,1000]}
grid_search_lr = GridSearchCV(lr, param_grid= params_lr, cv=5, n_jobs=-1)
grid_search_lr.fit(X_train,y_train)
print("score ", grid_search_lr.score(X_train, y_train))
print("best parameter for random logistic regression ", grid_search_lr.best_params_)


# Random forest regressor is the best model among the ones selected with a score of 0.98.  We can now test our model on test data with parameters {'bootstrap': True, 'criterion': 'entropy', 'n_estimators': 100} and submit the results.

# In[ ]:


rfc2 = RandomForestClassifier(max_features='auto',   n_jobs=-1, n_estimators = 100, bootstrap = True, criterion ='entropy')

rfc2.fit(X, y)

x_test = test[features]

Y_prediction = rfc2.predict(x_test)

my_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': Y_prediction})

