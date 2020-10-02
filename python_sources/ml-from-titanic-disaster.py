#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Loading the data and checking for missing values

# In[ ]:


train_data = pd.read_csv("../input/titanic/train.csv",index_col='PassengerId')
test_data = pd.read_csv("../input/titanic/test.csv", index_col='PassengerId')


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# Encoding categorical values into numeric ones 

# In[ ]:


train_data['Sex'].replace({'male':1 , 'female':0},inplace=True)
train_data['Embarked'].replace({'C': 1, 'S': 2 ,'Q': 3},inplace=True)
train_data['Embarked'].fillna(1,inplace=True)
test_data['Sex'].replace({'male':1 , 'female':0},inplace=True)
test_data['Embarked'].replace({'C': 1, 'S': 2 ,'Q': 3},inplace=True)


# Filling missing values in Age and Fare with their median and mean respectively

# In[ ]:


train_data['Age'].fillna(train_data['Age'].median(), inplace = True)
test_data['Age'].fillna(test_data['Age'].median(),inplace = True )
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace = True)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# It can be seen that only the Cabin column now has missing values.
# Since it won't be used as a feature, it can be ignored

# # ***Exploratory Data Analysis***

# In[ ]:


sns.barplot(x = train_data['Pclass'] , y = 1 - train_data['Survived'])
plt.title("No.of deaths according to Pclass")
plt.ylabel('Deaths')

# It can be observed that as the people from the 3rd class suffered the highest deaths


# In[ ]:


sns.barplot(x = train_data['Sex'] , y = train_data['Survived'])
plt.title("No.of survivals according to Sex")

#It is observed that most of the survivors were female


# In[ ]:


sns.barplot(x = train_data['SibSp'] , y = train_data['Survived'])
plt.title("Effect of sibling(s) and spouse on survival")

# It can be observed that people with 0,1,2 spouse/sibling(s) had the highest survival rate


# In[ ]:


sns.barplot(x = train_data['Parch'] , y = train_data['Survived'])
plt.title("Effect of parent and children on survival")

# It can be observed that parents with 0,1,2,3 children had the highest survival rate


# In[ ]:


sns.barplot(x = train_data['Embarked'] , y = train_data['Survived'])
plt.title("Ports Embarked and Survival")

# It can be observed that people who Embarked from Cherbourg had the highest survival rate


# Creating Age Group and Fare Group

# In[ ]:


#Creating age groups and plotting it

age_labels = ['Kids','Teenager','Adult','Middle aged','Retired','Old']
age_bins = [0,10,18,35,60,75,100]
train_data['binned_age'] = pd.cut(train_data['Age'], bins=age_bins, labels=age_labels)
train_data['binned_age'].replace({ 'Kids' : 1,'Teenager' : 2,'Adult' : 3,'Middle aged' : 4,'Retired' : 5,'Old' : 6 }, inplace= True)

test_data['binned_age'] = pd.cut(test_data['Age'], bins=age_bins, labels=age_labels)
test_data['binned_age'].replace({ 'Kids' : 1,'Teenager' : 2,'Adult' : 3,'Middle aged' : 4,'Retired' : 5,'Old' : 6 }, inplace= True)

sns.barplot(x = train_data['binned_age'] , y = train_data['Survived'])
plt.title("Survival according to age groups")

#It can observed that old people / group 6 survived the most.


# In[ ]:


#Creating Fare groups and plotting it

train_data['Fare'].value_counts()

fare_labels = ['Base Class','Express Class','Gold Class','Platinum Class']
fare_bins = [0,15,30,100,1000]
train_data['binned_fare'] = pd.cut(train_data['Fare'], bins=fare_bins, labels=fare_labels)
train_data['binned_fare'].replace({ 'Base Class' : 1,'Express Class' : 2,'Gold Class' : 3,'Platinum Class' : 4}, inplace= True)

test_data['binned_fare'] = pd.cut(test_data['Fare'], bins=fare_bins, labels=fare_labels)
test_data['binned_fare'].replace({ 'Base Class' : 1,'Express Class' : 2,'Gold Class' : 3,'Platinum Class' : 4}, inplace= True)

sns.barplot(x = train_data['binned_fare'] , y = train_data['Survived'])
plt.title("Survival according to fare groups")

#The plot shows that people travelling via the platinum class / group 4 survived a lot more than the other class of people


# Note - creating bins has resulted in some missing values in the binned_fare column. Replcing the missing values with the most frequent value

# In[ ]:


train_data['binned_fare'].fillna(4,inplace = True)
test_data['binned_fare'].fillna(1,inplace=True)


# Defining features and creating the test and train data

# In[ ]:


features = ['Pclass','Sex','binned_age','SibSp','Parch','Embarked','binned_fare']
X = train_data[features]
y = train_data.Survived
X_test = test_data[features]


# Final check to see if any null values exist

# In[ ]:


X_test.isnull().sum()
X.isnull().sum()


# Choosing Model and determining accuracy via Cross Validation and f1 score

# ##### Random Forest Classification

# In[ ]:


from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#n_est = 400 , depth = 10 , obtained by tuning

Rfc_model = RandomForestClassifier(n_estimators = 400,max_depth=10,n_jobs= -1,random_state=33)
scores = cross_val_score(Rfc_model,X,y,cv=5,scoring='f1')
print(round(scores.mean(),5)*100)


# Using Random Forest Classification has an accuracy of 73.13%

# KNN

# In[ ]:


from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

#n_neighbours = 5 , leaf_size = 50 has been obtained after tuning 

knn_model = KNeighborsClassifier(n_neighbors=5,leaf_size = 50,n_jobs=-1)
scores = cross_val_score(knn_model,X,y,cv=5,scoring='f1')
print(round(scores.mean(),5)*100)


# Using K-Nearest Neighbours has an accuracy of 74.52%

# Logistic Regression

# In[ ]:


from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

LoReg_model = LogisticRegression(random_state = 3,n_jobs = -1)
scores = cross_val_score(LoReg_model,X,y,cv=5,scoring='f1')
print(round(scores.mean(),5)*100)


# Using Logistic Regression gives an accuracy of 70.58%

# ###### Choosing the right model - Since all the three models have almost similar accuracy(70,73,74 %) approx , we can choose by directly submitting the result(s) and see which gives better result

# In[ ]:


#Fitting the data

Rfc_model.fit(X,y)
pred_RFC = Rfc_model.predict(X_test)
output = pd.DataFrame({'PassengerId': X_test.index, 'Survived': pred_RFC})
output.to_csv('Sub_RFC.csv', index=False)

knn_model.fit(X,y)
pred_knn = knn_model.predict(X_test)
output = pd.DataFrame({'PassengerId': X_test.index, 'Survived': pred_knn})
output.to_csv('Sub_knn.csv', index=False)

LoReg_model.fit(X,y)
pred_LoReg = LoReg_model.predict(X_test)
output = pd.DataFrame({'PassengerId': X_test.index, 'Survived': pred_LoReg})
output.to_csv('Sub_LoReg.csv', index=False)


print("Saved!")


# ##### The submission results show that the model results were -  
# * RandomForestClassification with score of 0.77033 
# * Logistic Regression with score of 0.7655
# * KNN with score of 0.74162

# Futher improvements to the model can be made via Feature Engineering, but since I am still unknown to that field, I will eventually delve into it. 
# For any suggestions, pls comment. I appreciate any and all suggestions.
