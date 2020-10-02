#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # <center>TITANIC SURVIVAL PREDICTIVE ANALYSIS<center>
# ![Titanic](https://wallup.net/wp-content/uploads/2019/09/202817-titanic-disaster-drama-romance-ship-boat-poster-gt.jpg)
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# ### Today we will be predicting the survival of passengers on the basis of other factos like age,class,fare etc
# 
# ## About Data
# 1)  Passenger Id- index of the data \
# 2)  Survived- 0 represent NO and 1 represent YES \
# 3)  Pclass- It represent ticket class (1=1st, 2=2nd, 3=3rd class) \
# 4)  Name: Name of Passenger \
# 5)  Sex: Male or Female \
# 6)  Age: Age of passenger \
# 7)  Sibsp: Number of Siblings-Spouses on board \
# 8)  Parch: Number of Parents-Children on board \
# 9)  Ticket: Ticket Number \
# 10) Fare: Passenger Fare\
# 11) Cabin: Cabin Numberm\
# 12) Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) 

# ### 1) Importing Libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# ### 2) Reading DataSet

# In[ ]:


import pandas as pd 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print(train.shape)
print(test.shape)


# ### 3) Removing or Replacing Null Values

# In[ ]:


print(train.info())
print("\r")
print(test.info())


# #### Remove features that do not influence the analysis, like Cabin (Too many NaN values), Name and Ticket.

# In[ ]:


train = train.drop(["Cabin","Name", "Ticket"], axis=1)
test = test.drop(["Cabin","Name", "Ticket"], axis=1)
train.describe()


# #### Replacing Null Values with Mean

# In[ ]:


train["Fare"] = train["Fare"].replace(np.nan, 32)
test["Fare"] = test["Fare"].replace(np.nan, 32)
train["Age"] = train["Age"].replace(np.nan, 30)
test["Age"] = test["Age"].replace(np.nan, 30)
train["Embarked"] = train["Embarked"].replace(np.nan, "C")


# In[ ]:


print(train.info())
print("\r")
print(test.info())


# ####  ---All null values Removed---

# ### 4) Categorozing Data as discreate integer values

# Changing categorical features Sex and Embarked by numbers

# In[ ]:


train["Sex"].replace(["female","male"] , [0,1], inplace = True)
test["Sex"].replace(["female","male"] , [0,1], inplace = True)
train["Embarked"].replace(["Q","S","C"],[0,1,2],inplace=True)
test["Embarked"].replace(["Q","S","C"],[0,1,2],inplace=True)
train.head()


# Categorizing Age by Age groups

# In[ ]:


bins = [0,8,15,20,40,60,100]
names=(['Baby', 'Child', 'Teenager', 'Youngster', 'Adult', 'Senior Citizen'])

train["Age"] = pd.cut(train["Age"], bins, labels = names)
test["Age"] = pd.cut(test["Age"], bins, labels = names)
train.head()


# Categorizing Fare by numerical values depicting differnt range of fare price

# In[ ]:


train["Fare"] = pd.cut(train.Fare,[-1, 130, 260, 390, 520], labels=['1', '2', '3', '4'])
test["Fare"] = pd.cut(test.Fare,[-1, 130, 260, 390, 520], labels=['1', '2', '3', '4'])

train.head()


# ## 5) Visualizing rate of survival with respect to other factors

# In[ ]:


train.pivot_table(index = "Sex", values = "Survived")


# In[ ]:


sns.barplot(x="Sex", y="Survived", data=train)
plt.show()


# #### Conclusion 1 - Female Passengers were given priority on men

# In[ ]:


train.pivot_table(index = "Pclass", values = "Survived")


# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=train)
plt.show()


# #### Conclusion 2 - First Class Passengers were given priority over others

# In[ ]:


train.pivot_table(index = "Age", values = "Survived")


# In[ ]:



sns.barplot(x="Age", y="Survived", data=train)
plt.show()


# #### Conclusion 3 - Babies and Childeren were the most out of all who survived

# Categorizing Age by integers values for ease

# In[ ]:


train["Age"].replace(["Baby","Child","Teenager","Youngster","Adult","Senior Citizen"] , [1,2,3,4,5,6], inplace = True)
test["Age"].replace(["Baby","Child","Teenager","Youngster","Adult","Senior Citizen"] , [1,2,3,4,5,6], inplace = True)
train.head()


# In[ ]:


train.dtypes    #Checking if all data types are redable or not


# ## 6) Building Model

# In[ ]:


#Declaring Model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
columns = ['Pclass', 'Sex', 'SibSp','Embarked', 'Age', 'Fare']


# In[ ]:


from sklearn.model_selection import train_test_split
X = train[columns]
y = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=0)


# In[ ]:


# Checking Accuracy
from sklearn.metrics import accuracy_score
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)


# In[ ]:


print("Intercept",lr.intercept_)
print("\r")
print("Coefficient",lr.coef_)   


# In[ ]:


lr.fit(X,y)
test_predictions = lr.predict(test[columns])


# In[ ]:


#Submission dataframe
test_ids = test["PassengerId"]
submission_df = {"PassengerId": test_ids,
                 "Survived": test_predictions}
submission = pd.DataFrame(submission_df)
submission.head(10)


# In[ ]:


submission.to_csv("submission.csv",index=False)


print(lr.score(X_test, y_test))


# In[ ]:


fig=plt.figure(figsize=(4,5))
sns.countplot(submission['Survived'])
plt.show()


# In[ ]:


print(submission["Survived"].value_counts())
print(submission.shape)


# In[ ]:





# # Final Results
# 
# * Only 157/418 Passengers Survived from the Data we used for testing pupose
# * Accuracy Of Linear Regression Model= 79.88%
# * 1st Class Passengers were given priority over other class passengers
# * Women and children were the category of people that survived most, they were allowed to board the lifeboats first

# In[ ]:





# In[ ]:




