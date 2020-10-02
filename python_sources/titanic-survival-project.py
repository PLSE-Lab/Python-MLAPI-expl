#!/usr/bin/env python
# coding: utf-8

# ### Import the Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Import the dataset

# In[ ]:


train = pd.read_csv('../input/titanic-train-public-dataset/titanic_train.csv')


# ### Display the dataset

# In[ ]:


train.head()


# ### Check the Null Values

# In[ ]:


train.isnull()


# ### Check the Null Values using Heatmap

# In[ ]:


sns.heatmap(train.isnull(),cbar=False,yticklabels=False,cmap='viridis')


# ### Exploratory Data Analysis

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=train)


# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=train)


# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False,bins=50)


# In[ ]:


train['Age'].plot.hist(bins=50)


# In[ ]:


train.info()


# In[ ]:


sns.countplot(x='SibSp',data=train)


# In[ ]:


train['Fare'].hist(bins=50,figsize=(10,4))


# In[ ]:


import cufflinks as cf
cf.go_offline()


# In[ ]:


train['Fare'].iplot(kind='hist',bins=50)


# In[ ]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)


# ### Dealing with Missing Values

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


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Here, we can see that there are No more Missing values

# In[ ]:


pd.get_dummies(train['Sex'])


# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)


# In[ ]:


sex


# In[ ]:


pd.get_dummies(train['Embarked'])


# In[ ]:


embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


embark


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


train.drop('PassengerId',axis=1,inplace=True)


# In[ ]:


train.head()


# ### Splitting the data into Train and Test set

# In[ ]:


X = train.drop('Survived',axis=1)
y = train['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predictions)


# In[ ]:




