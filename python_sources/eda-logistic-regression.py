#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import cufflinks as cf
cf.go_offline()


# # Reading in the test and train files 

# In[ ]:


test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


train.head(2)


# # Exploratory Data Analysis

# # Missing Data

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Roughly 20 percent of the Age data is missing.
# Looking at the Cabin column, we are just missing too much of that data to do something useful with it.

# In[ ]:


sns.countplot(x='Survived',data=train,hue='Sex')


# After seeing the above graph, we can clearly say that women had a much higher survival rate than men.

# In[ ]:


sns.countplot(x='Survived',data=train,hue='Pclass')


# Most of the deceased people belonged to the third class of the ship. Hence the first and second class people were the first one's to be rescued.

# In[ ]:


train['Age'].iplot(kind='hist',bins=20,color='red')


# Majority of the people aboard on the ship were inbetween the age group of 20-30 years old

# In[ ]:


sns.countplot(x='SibSp',data=train)


# In[ ]:





# Based on the above graph, most of the people on the ship were single which included mainly the working class.
# The couples had the second best population.

# In[ ]:


sns.swarmplot(x='Survived',y='Fare', data=train)


# Generally those who paid more fare had a higher chance of survival.

# In[ ]:


train['Fare'].iplot(kind='hist',bins=30,color='green')


# Most of the people onboard had paid between around $20 for their tickets.

# # Data Cleaning

# Using imputation to fill in missing age data instead of just dropping the missing age data rows 

# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# The wealthier passengers in the higher classes tend to be older. Using these average age values to impute based on Pclass for Age.

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


# Checking that heat map again!

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# Dropped the cabin column completely

# In[ ]:


train.head()


# In[ ]:


train.dropna(inplace=True)


# # Converting Categorical Features

# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# # Logistic Regression

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# # Evaluation 

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:





# In[ ]:




