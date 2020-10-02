#!/usr/bin/env python
# coding: utf-8

# # Case Study For The Titanic Competition (Data Analyse&Logistic Regression)
# 
# # Step 1 Importing Necessary Libraries

# In[ ]:


#import os
#print(os.listdir("../input"))


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# # Step 2 Loading Data

# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


train.head(4)


# # Step 3 Exploratory Data Analysis

# In[ ]:


train.info()


# In[ ]:


train.isnull().sample(25)


# In[ ]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="viridis")


# As seen above Age column and Cabin column have lots of missing information (NaN values).

# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


sns.countplot(x='Survived',hue='Sex', data=train, palette='RdBu_r')


# It looks like people that did not survive(0) were much more likely to be male and people that survive were mostly female.

# In[ ]:


sns.countplot(x='Survived',hue='Pclass', data=train)


# We can see above chart people who did not survive overwhelmingly part of the third class.

# In[ ]:


sns.distplot(train['Age'].dropna(), kde=False, bins=30 )


# It is quite skewed towards younger passengers.

# In[ ]:


sns.countplot(x='SibSp', data=train)


# The accounts of siblings versus spouses on board. (Most of them were probably single)

# In[ ]:


train['Fare'].hist(bins=50, figsize=(10,4))


# It looks like most of the prices are between 0 and 50$ at that time (1912) .

# # Step 4 Cleanin Data

# Instead of dropping all the missing values we can do some better touches. To be specifik, impulation or averege age can be used to fill in missing data for the 'Age' column. 

# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass', y='Age', data=train)


# It seems the more you older the more wealthy. We can use these average age values in order to impute the age based off the passenger class. 

# In[ ]:


def impute_average(columns):
    Age = columns[0]
    Pclass = columns[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_average, axis=1)


# In[ ]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# It looks like we are no longer having any missing information for the age column.
# On the other hand there is so much missing points for the Cabin column. Best action for this column is drop that cabin.

# In[ ]:


train.drop('Cabin', axis=1, inplace=True)


# In[ ]:


train.head(3)


# In[ ]:


#there are few missing values to get rid of them we drop them.
train.dropna(inplace=True)


# In[ ]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# # Step 5 Feature Engineering

# We need to preapare data to ML algorithm in order to that we should deal some of the data features. 
# In this context we need to convert some columns, categorical features, into dummy variables, and drop some of the columns which we won't use.

# In[ ]:


sex = pd.get_dummies(train['Sex'], drop_first=True)


# In[ ]:


embark = pd.get_dummies(train['Embarked'], drop_first=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head(3)


# In[ ]:


train.drop(['Sex', 'Embarked', 'Name','Ticket'],axis=1, inplace=True)


# In[ ]:


train.head(3)


# One last thing for this step is PassengerId column. It is smilar to index column and we won't use in our model for that reason we can drop it. 

# In[ ]:


train.drop('PassengerId', axis=1, inplace=True)


# In[ ]:


train.head(3)


# # Step 6 Training the Model

# I choose to use Logistic Regression. You can try any convenient ML model.

# ## a. Selecting Train Target Data Set and Spliting Them.

# In[ ]:


X = train.drop('Survived', axis=1)
y = train['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model = LogisticRegression()


# In[ ]:


model.fit(X_train, y_train)


# # Step 7 Predicting the Model

# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test, predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test, predictions)


# In[ ]:


sns.heatmap(confusion_matrix(y_test, predictions),annot=True,fmt="d",cmap='ocean_r' ,robust=True)


# In[ ]:


# Lastly we can strengthen the results with cross_val_score
from sklearn.model_selection import cross_val_score


# In[ ]:





# In[ ]:


print(cross_val_score(model, X, y, cv=19).mean())


# # Great Job!
