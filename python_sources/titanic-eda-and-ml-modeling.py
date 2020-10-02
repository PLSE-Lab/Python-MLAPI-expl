#!/usr/bin/env python
# coding: utf-8

# # Titanic EDA and ML Modeling
# 
# Let's import some libraries to get started!

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## The Data
# 
# Let's start by reading the train.csv and test.csv file into a pandas dataframe.

# In[ ]:


train=pd.read_csv('../input/titanic/train.csv')
train.head()


# In[ ]:


test=pd.read_csv('../input/titanic/test.csv')
test.head()


# In[ ]:


train.describe()


# # Exploratory Data Analysis
# 
# Let's begin some exploratory data analysis! We'll start by checking out missing data!
# 

# In[ ]:


sns.heatmap(train.isnull(),cbar=False,yticklabels=False,cmap='viridis')


# In[ ]:


sns.heatmap(test.isnull(),cbar=False,yticklabels=False,cmap='viridis')


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later

# In[ ]:


train.info()


# In[ ]:


test.info()


# Let's continue on by visualizing some more of the data!

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[ ]:


sns.countplot(x='Survived',data=train,hue='Sex')


# In[ ]:


sns.countplot(x='Survived',data=train,hue='Sex')


# In[ ]:


sns.countplot(x='Survived',data=train,hue='Pclass')


# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False,color='blue')


# In[ ]:


sns.countplot(x='SibSp',data=train)


# In[ ]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[ ]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train)


# ## Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers.
# 
# check the average age by passenger class, For example:
# 

# In[ ]:


class_group=train.groupby('Pclass')
class_group.mean()


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 38.2

        elif Pclass == 2:
            return 29.8

        else:
            return 25.1

    else:
        return Age


# Now apply that function!

# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# Now let's check that heat map again!

# In[ ]:


sns.heatmap(train.isnull(),cbar=False,yticklabels=False,cmap='viridis')


# Great! Let's go ahead and drop the Cabin column and the row in Embarked that is NaN.

# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


train.info()


# Repeat the same procedure for test data also

# In[ ]:


class_group=test.groupby('Pclass')
class_group.mean()


# In[ ]:


def fill_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 40.6

        elif Pclass == 2:
            return 28.8

        else:
            return 24.3

    else:
        return Age


# In[ ]:


test['Age'] = test[['Age','Pclass']].apply(fill_age,axis=1)


# In[ ]:


test.drop('Cabin',axis=1,inplace=True)


# In[ ]:


test[test['Fare'].isnull()]


# In[ ]:


test.fillna(value=12.4,inplace=True)


# In[ ]:


test.info()


# ## Converting Categorical Features 
# 
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[ ]:


train['Sex']=train['Sex'].map({'female':0, 'male':1}).astype(int)


# In[ ]:


train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


test['Sex']=test['Sex'].map({'female':0, 'male':1}).astype(int)


# In[ ]:


test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


train.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


test.drop(['Name','Ticket'],axis=1,inplace=True)


# Great! Our data is ready for our model!
# 
# # Building Machine Learning models
# 
# Let's start by splitting our data into a training set and test set

# In[ ]:


x_train=train.drop('Survived',axis=1)
y_train=train['Survived']
x_test=test.drop('PassengerId',axis=1)


# ## Logistic Regression model

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel=LogisticRegression(max_iter=1000)
logmodel.fit(x_train,y_train)


# In[ ]:


predictions=logmodel.predict(x_test)


# In[ ]:


logmodel.score(x_train,y_train)


# ##  DecisionTreeClassifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtree=DecisionTreeClassifier()


# In[ ]:


dtree.fit(x_train,y_train)


# In[ ]:


Tpredictions=dtree.predict(x_test)


# In[ ]:


dtree.score(x_train,y_train)


# In[ ]:


df=pd.DataFrame({'PassengerId': test['PassengerId'],
                  'Predictions': Tpredictions  })


# In[ ]:


df.to_csv('submission1.csv',index=False)

