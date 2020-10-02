#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df=pd.read_csv("../input/titanic.csv")
df.head()


# In[ ]:


df.isnull().sum()


# #Age and cabin has Nan missing values
# #Goal: we will check if the person is dead or alive using classification

# In[ ]:


#Lets do some initial analysis on the data
sns.countplot(x=df.Sex) # male and Female ratio at Ship


# Treating the missing value of Age

# In[ ]:


#Lets see the Age group of people with missing data ,Total missing value is 177.
df[df.Age.isnull()]


# In[ ]:


#lets see the  age with  High frequency in data set and mean value as well
df.Age.mode(),df.Age.mean()


# In[ ]:


#Lets fill the missing value as mean value of the people as 29
df.Age.fillna(value=29,inplace=True)
df.Age.isnull().sum() # all values are filled now 


# In[ ]:


#lets encode Sex as its a categorical value and needs to be encoded for machine learnin.
from sklearn.preprocessing import LabelEncoder
dfe=df.copy()
le=LabelEncoder()
Sex1 =le.fit_transform(dfe.Sex)
Sex1


# In[ ]:


dfe["Sex1"]=Sex1
dfe.head()


# In[ ]:


dfe.columns


# In[ ]:


#Machine learning process started
Y=dfe.Survived
my_col=['PassengerId', 'Survived', 'Name', 'Ticket',"Sex","SibSp","Parch", 'Cabin', 'Embarked']
dfe.drop(columns=my_col,axis="columns",inplace=True)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


# Machine Learning model Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,Y_train)


# In[ ]:


len(X_test)


# In[ ]:


lr.predict(X_test)


# Model Score

# In[ ]:


lr.score(X_test,Y_test)

