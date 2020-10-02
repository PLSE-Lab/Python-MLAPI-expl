#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()


# In[ ]:



plt.figure(figsize=(12,10))
sns.heatmap(train.corr(), annot=True ,cmap='coolwarm')


# 

# **visualizaion**

# In[ ]:





# In[ ]:


sns.countplot(x="Survived" , hue="Sex" , data=train)


# In[ ]:


sns.catplot(x="Pclass",y= "Survived"  ,kind="box", data=train)


# In[ ]:


sns.boxplot(x="Survived" ,y= "Age", data=train)


# In[ ]:


sns.barplot(x="Survived" ,y= "SibSp", data=train)


# In[ ]:


train['Age'].hist(bins=30,alpha=0.7)


# Missing Value

# In[ ]:


train.isnull().sum()


# In[ ]:



train["Age"].fillna(train["Age"].mean() , inplace=True)
train["Embarked"]=train["Embarked"].fillna("S")


# In[ ]:


train=train.drop(columns=["Cabin" , "Name","PassengerId","Ticket"] )


# In[ ]:


train.isnull().sum()


# In[ ]:


train.info()


# In[ ]:


train.columns


# In[ ]:


train.head()


# In[ ]:


labelencoder=LabelEncoder()
train["Sex"]=labelencoder.fit_transform(train["Sex"])
train["Embarked"]=labelencoder.fit_transform(train["Embarked"])

train.head()


# In[ ]:


y=train["Survived"]

#print(y)
train.columns
train=train.drop(columns=["Survived"])
train.columns


# In[ ]:



scaler=StandardScaler()
X=train
X=scaler.fit_transform(X)
print(X)


# In[ ]:



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=41)


# In[ ]:


lr=LogisticRegression()
lr.fit(X_train,y_train)


# In[ ]:


y_pred =lr.predict(X_valid)


# In[ ]:


print(lr.intercept_)


# In[ ]:


print(lr.coef_)


# In[ ]:


mae= mean_squared_error(y_valid,y_pred)
mae


# In[ ]:


test.head()


# In[ ]:


test=test.drop(columns=["Cabin" , "Name","Ticket"] )


# In[ ]:


#test_df=pd.DataFrame(test_df)  


# In[ ]:


test.isnull().sum()


# In[ ]:



test["Age"].fillna(test["Age"].mean() , inplace=True)
test["Fare"].fillna(test["Fare"].mean() , inplace=True)


# In[ ]:


test.isnull().sum()


# In[ ]:


test["Sex"]=labelencoder.fit_transform(test["Sex"])
test["Embarked"]=labelencoder.fit_transform(test["Embarked"])


# In[ ]:


test.head()


# In[ ]:


test_df=test.drop(columns="PassengerId")


# In[ ]:


test_df=scaler.fit_transform(test_df)


# In[ ]:


prediction=lr.predict(test_df)


# In[ ]:


print(prediction)


# In[ ]:


output=pd.DataFrame({"PassengerId":test.PassengerId ,"Survived":prediction})


# In[ ]:


output.head()


# In[ ]:


output.to_csv("Submission_csv",index=False)

