#!/usr/bin/env python
# coding: utf-8

# 
# # Logistic Regression with Python
# 
# We'll be trying to predict a classification- survival or deceased.Let's begin our understanding of implementing Logistic Regression in Python for classification.
# 
# ## Import Libraries
# Let's import some libraries to get started!

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## The Data
# 

# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.tail()


# # Exploratory Data Analysis
# 
# Let's begin some exploratory data analysis! We'll start by checking out missing data!
# 
# ## Missing Data
# 
# We can use seaborn to create a simple heatmap to see where we are missing data!

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"
# 
# #### Let's continue on by visualizing some more of the data! Check out the video for full explanations over these plots, this code is just to serve as reference.

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[ ]:


train['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[ ]:


sns.countplot(x='SibSp',data=train)


# In[ ]:


sns.countplot(x='Parch',data=train)


# In[ ]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# ___
# ## Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
# However we can be smarter about this and check the average age by passenger class. For example:
# 

# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

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


# Now apply that function!

# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


train['Embarked'] = train['Embarked'].fillna('S')


# Now let's check that heat map again!

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Great! Let's go ahead and drop the Cabin column and the row in Embarked that is NaN.

# We will sum of family member 

# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.dropna(inplace=True)


# ## Converting Categorical Features 
#   We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[ ]:


train.info()


# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# ### Great! Our data is ready for our model!
# 
# ## Building a Logistic Regression model
#  Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training).
# 
# ## Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'],axis=1), 
                                                    train['Survived'], test_size=0.10, 
                                                    random_state=101)


# ## Training and Predicting

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)
X_test.head()


# In[ ]:


predictions


# ## Evaluation

# We can check precision,recall,f1-score using classification report!

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


print(classification_report(y_test,predictions))


# # Decision Tree Classifiction

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt_model=DecisionTreeClassifier()
dt_model.fit(X_train,y_train)


# In[ ]:


dt_pred = dt_model.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,dt_pred))


# In[ ]:


print(classification_report(y_test,dt_pred))


# # Random Forest Classification

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf= RandomForestClassifier(n_estimators=500)
rf.fit(X_train,y_train)


# In[ ]:


rf_pre=rf.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,rf_pre))


# In[ ]:


print(classification_report(y_test,rf_pre))


# Now we will use test dataset

# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


sns.heatmap(test.isnull())


# In[ ]:


test.drop('Cabin',axis=1,inplace=True)


# In[ ]:


test['Fare'].fillna(test['Fare'].median(), inplace=True)


# In[ ]:


test.info()


# In[ ]:


test.head()


# In[ ]:


test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sex_test = pd.get_dummies(test['Sex'],drop_first=True)
embark_test= pd.get_dummies(test['Embarked'],drop_first=True)


# In[ ]:


test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


test = pd.concat([test,sex_test,embark_test],axis=1)


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


rf = RandomForestClassifier(n_estimators=1000)


# In[ ]:


rf.fit(train.drop(['Survived'],axis=1),train['Survived'] )


# In[ ]:


test_prediction = rf.predict(test)


# In[ ]:


test_prediction.shape


# In[ ]:


test_pred = pd.DataFrame(test_prediction, columns= ['Survived'])


# In[ ]:


new_test = pd.concat([test, test_pred], axis=1, join='inner')


# In[ ]:


new_test.head()


# In[ ]:


df= new_test[['PassengerId' ,'Survived']]


# In[ ]:


df.head()


# In[ ]:


df.to_csv('predictions.csv' , index=False)


# ## If you like it, please vote.
# # Thank you :)
