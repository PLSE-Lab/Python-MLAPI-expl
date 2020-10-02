#!/usr/bin/env python
# coding: utf-8

# # Machine Learning For Titanic Dataset

# Predicting whether a passenger will survive or not.

# 1. Data Cleaning
# 2. Feature Engineering
# 3. Building Machine Learning Classifiers/Models.

# In[1]:


#IMPORTING THE DEPENDENCIES

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))


# ### 1. Data Cleaning 

# In[2]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# In[3]:


df_train.shape


# In[4]:


df_train.head()


# In[5]:


df_train.info()


# In[6]:


df_test.head()


# In[7]:


df_test.shape


# In[8]:


df_test.info()


# * Checking for Missing Values in data

# In[9]:


import missingno as mn


# The missingno package allows to visualize and analyse the missing values through graphical representations.

# * MISSING VALUES IN TRAINING DATA

# In[10]:


print("\t\tMISSING VALUES IN TRAINING DATA")
for i in df_train.columns:
    print("Missing Values In {} : {}".format(i,df_train[i].isnull().sum()))


# In[11]:


mn.matrix(df_train)


# The Cabin columns contains the maximum missing values as compared to Age and Embarked columns.

# In[12]:


mn.bar(df_train)


# * Missing Values in Testing Data

# In[13]:


print("\t\tMISSING VALUES IN TESTING DATA")
for i in df_test.columns:
    print("Missing Values In {} : {}".format(i,df_test[i].isnull().sum()))


# In[14]:


mn.matrix(df_test)


# In[15]:


mn.bar(df_test)


# Columns with Missing Values<br>
# 1. Cabin
# 2. Embarked
# 3. Fare
# 4. Age
# 

# 1. Cabin Column<br>
# The cabin values can be treated but can add noise to our data as it is of high proportion in the data.

# In[16]:


print("\tTRAINING DATA")
print("The number of missing values in Cabin: {}".format(df_train['Cabin'].isnull().sum()))
print("The percentage of missing values in Cabin: {} %".format(df_train['Cabin'].isnull().sum()*100/891))
print("")
print("\tTESTING DATA")
print("The number of missing values in Cabin: {}".format(df_test['Cabin'].isnull().sum()))
print("The percentage of missing values in Cabin: {} %".format(df_test['Cabin'].isnull().sum()*100/418))


# In[18]:


#Filling the unknown cabin with 'U'
df_train['Cabin'].fillna(value='U',inplace=True)


# In[22]:


#Using only the letter of the Cabin without the number
df_train['CabinType'] = df_train['Cabin'].apply(lambda i: i[:1])


# In[25]:


#Similar for testing Data
df_test['Cabin'].fillna(value='U',inplace=True)
#Using only the letter of the Cabin without the number
df_test['CabinType'] = df_test['Cabin'].apply(lambda i: i[:1])


# 2. Embarked Column

# In[23]:


print("\tTRAINING DATA")
print("The number of missing values in Embarked: {}".format(df_train['Embarked'].isnull().sum()))
print("The percentage of missing values in Embarked: {} %".format(df_train['Embarked'].isnull().sum()*100/891))
print("")
print("\tTESTING DATA")
print("The number of missing values in Embarked: {}".format(df_test['Embarked'].isnull().sum()))
print("The percentage of missing values in Embarked: {} %".format(df_test['Embarked'].isnull().sum()*100/418))


# In[24]:


embarked_common = df_train['Embarked'].value_counts().index[0]
df_train['Embarked'].fillna(value=embarked_common,inplace=True)


# 3. Fare Column <br>
# 

# In[27]:


print("\tTRAINING DATA\t")
print("The number of missing values in Fare: {}".format(df_train['Fare'].isnull().sum()))
print("The percentage of missing values in Fare: {}".format(df_train['Fare'].isnull().sum()*100/889))

print("")
print("\tTESTING DATA\t")
print("The number of missing values in Fare: {}".format(df_test['Fare'].isnull().sum()))
print("The percentage of missing values in Fare: {}".format(df_test['Fare'].isnull().sum()*100/418))


# In[28]:


df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())


# 4. Age Column<br>

# The age's missing values are of decent proportion which can be treated for both the training and testing data.<br>
# 
# The approach to impute Age column is studied from the following blog.<br>
# https://medium.com/i-like-big-data-and-i-cannot-lie/how-i-scored-in-the-top-9-of-kaggles-titanic-machine-learning-challenge-243b5f45c8e9
# 

# In[30]:


df_train['Title'] = df_train['Name'].apply(lambda i: i.split(',')[1].split('.')[0].strip())
df_train.head()


# In[31]:


standardized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}


# In[32]:


df_train['Title'] = df_train['Title'].map(standardized_titles)


# In[33]:


df_train.head()


# In[34]:


#Grouping Sex,Pclass and Title Together
df_grouped = df_train.groupby(['Sex','Pclass', 'Title'])


# In[35]:


df_grouped['Age'].median()


# In[37]:


df_train['Age'] = df_grouped['Age'].apply(lambda i: i.fillna(i.median()))          


# In[38]:


#Same procedure for testing data's Age column

df_test['Title'] = df_test['Name'].apply(lambda i: i.split(',')[1].split('.')[0].strip())
df_test['Title'] = df_test['Title'].map(standardized_titles)
df_grouped_test = df_test.groupby(['Sex','Pclass', 'Title'])
df_test['Age'] = df_grouped_test['Age'].apply(lambda i: i.fillna(i.median()))          
df_test['Age'].isnull().sum()


# In[39]:


print("\t\tMISSING VALUES IN TRAINING DATA")
for i in df_train.columns:
    print("Missing Values In {} : {}".format(i,df_train[i].isnull().sum()))
    
print("")
print("\t\tMISSING VALUES IN TESTING DATA")
for i in df_test.columns:
    print("Missing Values In {} : {}".format(i,df_test[i].isnull().sum()))


# ### 3. Feature Engineering

# In[41]:


#Storing the passengerId for future submissions.
passengerId = df_test['PassengerId']


# Combining the data for feature engineering

# In[42]:


df_titanic = pd.DataFrame()
df_titanic = df_train.append(df_test)


# In[43]:


train_index = len(df_train)
test_index = len(df_titanic) - len(df_test)


# In[44]:


df_titanic.head()


# In[45]:


print("\t\tMISSING VALUES IN COMBINED DATA")
for i in df_titanic.columns:
    print("Missing Values In {} : {}".format(i,df_titanic[i].isnull().sum()))


# The Cabin value will not be considered.<br>
# The 418 values in Survived are to be predicted.

# 1. Family Size

# In[46]:


df_titanic['FamilySize'] = df_titanic['Parch'] + df_titanic['SibSp'] + 1


# * Converting Features to Categorical or Numerical for modeling.

# In[47]:


df_titanic['Sex'] = df_titanic['Sex'].map({"male": 0, "female":1})


# In[48]:


PClass_dummy = pd.get_dummies(df_titanic['Pclass'], prefix="Pclass")
Title_dummy = pd.get_dummies(df_titanic['Title'], prefix="Title")
CabinType_dummy = pd.get_dummies(df_titanic['CabinType'], prefix="CabinType")
Embarked_dummy = pd.get_dummies(df_titanic['Embarked'], prefix="Embarked")


# In[49]:


df_titanic_final = pd.DataFrame()
df_titanic_final = pd.concat([df_titanic, PClass_dummy, Title_dummy, Embarked_dummy,CabinType_dummy], axis=1)


# In[50]:


df_titanic_final.head()


# * Separating the data back into training and testing

# In[51]:


df_train_final = df_titanic_final[ :train_index]
df_test_final = df_titanic_final[test_index: ]


# In[52]:


#If not converted to 'int', can result in 0 score after submission as it doesn't get converted to boolean.
df_train_final.Survived = df_train_final.Survived.astype(int)


# In[53]:


df_train_final.head()


# ### 3. Building Machine Learning Models

# In[54]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[55]:


df_train_final.columns


# In[57]:


df_test_final.columns


# In[58]:



X = df_train_final.drop(['Cabin', 'CabinType', 'Embarked','Name',
       'PassengerId', 'Pclass','Survived', 'Ticket', 'Title'], axis=1).values 
Y = df_train_final['Survived'].values


# In[59]:


df_test_final.columns


# In[60]:


X_test = df_test_final.drop(['Cabin', 'CabinType', 'Embarked','Name',
       'PassengerId', 'Pclass','Survived', 'Ticket', 'Title'], axis=1).values


# In[61]:


parameters_dict = dict(     
    max_depth = [n for n in range(10, 21)],     
    min_samples_split = [n for n in range(5, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 70, 10)],
)


# In[62]:


rfc = RandomForestClassifier()


# In[63]:


forest_gridcv = GridSearchCV(estimator=rfc, param_grid=parameters_dict, cv=5) 
forest_gridcv.fit(X, Y)


# In[64]:


print("Best score: {}".format(forest_gridcv.best_score_))
print("Optimal params: {}".format(forest_gridcv.best_estimator_))


# In[65]:


rfc_predictions = forest_gridcv.predict(X_test)


# In[66]:


rfc_predictions


# In[70]:


kaggle_final = pd.DataFrame({'PassengerId': passengerId, 'Survived': rfc_predictions})


# In[71]:


#save to csv
kaggle_final.to_csv('mysubmission3.csv', index=False)


# In[ ]:




