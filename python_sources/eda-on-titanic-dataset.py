#!/usr/bin/env python
# coding: utf-8

# Working on Titanic Dataset from Kaggle

# In[ ]:


# Importing libraries

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


# In[ ]:


# Reading the dataset using pandas

data=pd.read_csv("../input/titanicdataset-traincsv/train.csv")
data.head()


# In[ ]:


# Finding out the missing values
# I am going to use the isnull() to check if there exists any null value or not
# If it gives true that means that particular value is null

data.isnull()


# In[ ]:


# Scrolling through the entire dataset of 891 rows and checking null values become tough
# Hence , we use visualization library seaborn to use it's property of heatmap to show in a visualised manner that
# which all columns have null values

# yticklabels specifies to mention the column names. xticklabels is for showing record numbers.
# cbar is for whether to show colorbar or not. 
# cmap is for colored visualization
# All null values are displyed in grey color

sns.heatmap(data.isnull(),yticklabels=False,cbar=True,cmap='Accent')


# In[ ]:


# 'survived' is the dependent variable in this dataset
# Some more visualization 

sns.countplot(x='Survived', data=data)


# In[ ]:


sns.countplot(x='Sex',data=data)


# In[ ]:


# Now step is to find No. of males and females who survived

sns.countplot(x='Survived', hue='Sex', data=data)


# In[ ]:


# Here we will find out how many passenger class(Pclass) people survived and how many didnot

sns.countplot(x='Survived',hue='Pclass', data=data)


# In[ ]:


#Checking normal distribution of age group by dropping the NaN values
# kde is Kernel density estimation which gets shown by default
#bins parameter means distribute given dataset in a particular range and show in bars

sns.distplot(data['Age'].dropna(), kde=False)


# In[ ]:


# SibSp is Sibling or spouse
# Determining how many SibSp are present


sns.countplot(x='SibSp', data=data)


# In[ ]:


data['Fare'].hist(color='green')


# Data Cleaning i.e. Removing the null vaues

# In[ ]:


# Column Age and Cabin has null values

#Going to plot a Boxplot to plot a relation of PClass and Age, how much percentile does each passenger class has

sns.boxplot(x='Pclass', y='Age', data=data)


# Based upon Pclass and Age , I am going to replace the NaN values Age.

# In[ ]:


# Writing a function to put values in Age column where there is no value present

def input_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if(pd.isnull(Age)):
        
        if(Pclass==1):
            return 37
        
        elif(Pclass==2):
            return 29
        
        else:
            return 24
    
    else:
        return Age     
    


# In[ ]:


# Applying the above function tot he dataset to replace all Missing values in Age Column

data['Age']=data[['Age','Pclass']].apply(input_age, axis=1)


# In[ ]:


# Checking the heat map again to see if age column has any missing values or not. It shows that it doesnot
sns.heatmap(data.isnull(),yticklabels=False,cbar=True,cmap='Accent')


# In[ ]:


#After dropping Cabin column
#data.drop('Cabin',axis=1,inplace=True)
data.head()


# In[ ]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=True,cmap='Accent')


# Handling Categorical Data

# In[ ]:


data.info()


# In[ ]:


# Categorical features are Name,Sex, Ticket, Embarked
#using get_dummies () going to convert them

pd.get_dummies(data['Embarked'],drop_first=True).head()


# In[ ]:


sex=pd.get_dummies(data['Sex'],drop_first=True)
embark=pd.get_dummies(data['Embarked'],drop_first=True)


# In[ ]:


data.drop(['Sex','Name','Ticket','Embarked'],axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


# Now the converted categorical feature should be concatenated in the data

data=pd.concat([data,sex,embark],axis=1)


# In[ ]:


data.head()


# Logistic Regression Model

# In[ ]:


#Splitting the data into trained data and test data

data.drop('Survived',axis=1).head()


# In[ ]:


data['Survived'].head()


# In[ ]:


#30% will go to test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop('Survived',axis=1), 
                                                    data['Survived'], test_size=0.30, 
                                                    random_state=101)


# Training and Predicting

# In[ ]:


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(y_test,predictions)
accuracy


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions)
accuracy


# In[ ]:


predictions


# Evaluating the model

# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

