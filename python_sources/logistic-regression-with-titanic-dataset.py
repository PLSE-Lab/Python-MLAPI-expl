#!/usr/bin/env python
# coding: utf-8

# # **Importing all the important Libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# **Input Data**

# In[ ]:


data = pd.read_csv("../input/titanic-extended/full.csv")


# In[ ]:


data.head(10) #view the first few column of the dataset


# In[ ]:


data.shape #view the shape of the dataset (m,n)


# # **Perform data analysis**

# In[ ]:


sns.countplot(x="Survived", hue = "Sex", data=data) #analysis with visulisation


# In[ ]:


sns.countplot(x="Survived", hue = "Pclass", data=data)


# In[ ]:


data["Age"].plot.hist()


# In[ ]:


data.info()


# # Perfom data cleaning

# In[ ]:


data.isnull().sum() #lists the number of null values in each feature


# In[ ]:


data.head(5)

Drop column with high number of null values and duplicate columns
# In[ ]:


data.drop(["Hometown",
           "Age_wiki",
           "Name_wiki", 
           "Ticket",
           "Cabin",
           "Destination",
           "Body",
           "WikiId", 
           "Name","Boarded",
           "Lifeboat",
           "Name",
           "PassengerId"],axis=1, inplace= True)


# In[ ]:


data.head(5)


# In[ ]:


data.dropna(inplace=True) #drop all the data entries with missing values


# In[ ]:


data.isnull().sum()


# **handling categorial variables******

# In[ ]:


sex = pd.get_dummies(data["Sex"], drop_first=True)
embark = pd.get_dummies(data["Embarked"], drop_first=True)
clss = pd.get_dummies(data["Class"], drop_first=True)
data.drop('Pclass', axis=1,inplace=True)


# In[ ]:


data=pd.concat([data,sex,embark,clss], axis=1)
data.head(5)


# In[ ]:


data.drop(["Sex","Embarked",'Class'],axis=1,inplace=True) 


# In[ ]:


data.head(5)


# # Split the data for training

# In[ ]:


x=data.drop("Survived",axis=1) #features
y=data["Survived"] #target variable


# In[ ]:



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)


# In[ ]:


#build the model and train the model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)


# In[ ]:


#perform predictions
predic = model.predict(x_test)


# In[ ]:


#analyse the accuracy of the model
report = classification_report(y_test,predic)


# In[ ]:


print(report)


# In[ ]:



accuracy_score(y_test, predic)

