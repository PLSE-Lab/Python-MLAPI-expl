#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries
# ##### Import a few libraries you think you'll need (Or just import them as you go along!)

# In[ ]:


import numpy as np


# In[ ]:


import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### Get the Data
# Read in the titanic_train.csv file and set it to a data frame called train.

# In[ ]:


train=pd.read_csv("../input/titanic_train.csv")


# #### Check the head of train

# In[ ]:


train.head()


# # Explore Data Analysis

# # missing data

# ** Use info and describe() on train**

# In[ ]:


train.info()
train.describe()


# #### Create heatmap of null values in dataframe

# In[ ]:


sns.heatmap(train.isnull(), cbar=False)


# #### Create countplot for Survived by 'Sex'

# In[ ]:


sns.countplot(x='Survived',data=train)


# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=train)


# ### create countplot for Survived by 'pclass' 

# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=train)


# ### Create distplot According to the Age 

# In[ ]:


dis=sns.distplot(train['Age'].dropna(),bins=30,kde=False)
dis
#kde removes the line 


# ### create countplot according to the 'SibSp' 

# In[ ]:


sns.countplot(x='SibSp',data=train)


# ### Create hist According to the Fair

# In[ ]:


bs=train.iloc[:,9].values


# In[ ]:


plt.hist(bs,bins=20,rwidth=1)
plt.show()


# #### Create boxplot According to Pclass and Age

# In[ ]:


sns.boxplot(x="Pclass",y="Age",data=train)


# ### Create function which fill all the null values in age column according to the Pclass

# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
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


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis= 1) # hint apply function in age column


# #### Create heatmap of null values in dataframe

# In[ ]:


sns.heatmap(train.isnull(), cbar=False)


# ### drop column 'Cabin' who is have more null values 

# In[ ]:


train=train.drop(['Cabin'],axis=1)             # drop cabin columns


# In[ ]:


train.head()


# ### then Draw heatmap

# In[ ]:


sns.heatmap(train.isnull())


# ### Drop All null values in Dataframe

# In[ ]:


train.dropna()


# ### show the heatmap of null values

# In[ ]:


sns.heatmap(train.isnull(), cbar=False)


# ### Create new variable name Sex and store dummies in sex column

# In[ ]:


a=pd.get_dummies(train.Sex, prefix='Sex').loc[:, 'Sex_male':]


# In[ ]:


a.head()


# ### Create new variable name embark and store dummies in Embarked column

# In[ ]:


e=pd.get_dummies(train.Embarked).iloc[:, 1:]


# In[ ]:


e.head()


# #### show  head of embark

# In[ ]:


e.head()


# #### Create variable name train pass values cancat all above variable (train,sex,embark)

# In[ ]:


train = pd.concat([train,a,e],axis=1)
train.head(2)


# ### show head of data frame

# In[ ]:


train.head()


# #### drop ('Sex','Name','Embarked','Ticket') columns in dataframe

# In[ ]:


train=train.drop(['Sex','Name','Embarked','Ticket'],axis=1)


# In[ ]:


train.head()


# #### drop(PassengerId) column in dataframe

# In[ ]:


train=train.drop(columns=('PassengerId'))


# In[ ]:


train.head()


# #### show head of dataframe

# In[ ]:


train.head()


# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train on!

# ** Split the data into training set and testing set using train_test_split**

# In[ ]:


x=train.drop('Survived',axis=1).values


# In[ ]:


y=train['Survived'].values


# In[ ]:


z=train.nunique()
z


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[ ]:


x_train


# In[ ]:


x_test


# In[ ]:


y_train


# In[ ]:


y_test


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model.fit(x_train,y_train)


# In[ ]:


y_pred=log_model.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


score=accuracy_score(y_test,y_pred)


# In[ ]:


score


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


cm=confusion_matrix(y_test,y_pred)


# In[ ]:


cm


# In[ ]:


sns.heatmap(cm,annot=True)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




