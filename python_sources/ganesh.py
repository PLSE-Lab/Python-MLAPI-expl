#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


train=pd.read_csv("../input/train.csv")


# In[ ]:


train.head()


# In[ ]:


train.info()
train.describe()


# In[ ]:


sns.heatmap(train.isnull(), cbar=False)


# In[ ]:


sns.countplot(x='Survived',data=train)


# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=train)


# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=train)


# In[ ]:


dis=sns.distplot(train['Age'].dropna(),bins=30,kde=False)
dis
#kde removes the line 


# In[ ]:


sns.countplot(x='SibSp',data=train)


# In[ ]:


bs=train.iloc[:,9].values


# In[ ]:


plt.hist(bs,bins=20,rwidth=1)
plt.show()


# In[ ]:


sns.boxplot(x="Pclass",y="Age",data=train)


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


# In[ ]:


sns.heatmap(train.isnull(), cbar=False)


# In[ ]:


train=train.drop(['Cabin'],axis=1)             # drop cabin columns


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.isnull())


# In[ ]:


train.dropna()


# In[ ]:


sns.heatmap(train.isnull(), cbar=False)


# In[ ]:


a=pd.get_dummies(train.Sex, prefix='Sex').loc[:, 'Sex_male':]


# In[ ]:


a.head()


# In[ ]:


e=pd.get_dummies(train.Embarked).iloc[:, 1:]


# In[ ]:


e.head()


# In[ ]:


train = pd.concat([train,a,e],axis=1)
train.head(2)


# In[ ]:


train.head()


# In[ ]:


train=train.drop(['Sex','Name','Embarked','Ticket'],axis=1)


# In[ ]:


train.head()


# In[ ]:


train=train.drop(columns=('PassengerId'))


# In[ ]:


train.head()


# Logistic Regression
# Now it's time to do a train test split, and train our model!
# You'll have the freedom here to choose columns that you want to train on!

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




