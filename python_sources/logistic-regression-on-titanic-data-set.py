#!/usr/bin/env python
# coding: utf-8

# ## This kernel performs simple logistic regression on the titanic dataset and predicts survival of passengers from test dataset

# ### Importing all required Python libraries

# In[ ]:


import os
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


cwd = os.getcwd()
os.chdir(r'/kaggle/input/titanic')
os.listdir()


# ## Performing EDA and Feature Engineering on the training data set

# In[ ]:


df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')


# In[ ]:


# df_train.info()
print(f'The training dataset has {df_train.shape[1]} attributes for {df_train.shape[0]} rows' )
for i in df_train.columns:
    print("Attribute {:11s} has {:4.3} % records with nulls.".format(i,(df_train[i].isna().sum()/df_train.shape[0]) * 100))


# In[ ]:


df_train.describe()


# In[ ]:


print(df_train['Sex'].unique())
print(df_train['Embarked'].unique())


# In[ ]:


# There are two passangers with missing Embarked station code. Trying to fill the NaNs in column "Embarked"
df_train[df_train['Embarked'].isna()]


# In[ ]:


# There are only two passangers with cabin code = B28 and none of them have Embarked station code. Hence Cabin code is not useful to find missing 
# values
df_train[df_train['Cabin']=='B28']


# In[ ]:


sns.boxplot(df_train['Embarked'].dropna(), df_train['Fare'])


# In[ ]:


sns.distplot(df_train['Fare'], bins=50)


# In[ ]:


df_train['Embarked'].fillna(value='C', inplace=True)
df_train[df_train['Cabin']=='B28']
df_train['Sex'].replace({'male':0,'female':1}, inplace=True)
df_train['Embarked'].replace({'S':0,'C':1,'Q':2}, inplace=True)
df_train.head()


# In[ ]:


sns.set(style="darkgrid")
sns.countplot('Survived',hue='Sex', data=df_train)
# Observation:
# Most of the deceased passengers were males.


# In[ ]:


sns.catplot(x="Pclass", hue="Sex", col="Survived", data=df_train, kind="count")
# Observation
# Lower class passengers lost their lives the most
# Higher class female passenger survived the most


# In[ ]:


# df_train.head()
sns.distplot(df_train['Age'].dropna())
print("Mean age = ", np.mean(df_train['Age'].dropna()))
print("Median age = ", np.median(df_train['Age'].dropna()))


# In[ ]:


sns.countplot(x='SibSp', hue='Survived', data=df_train)
# Observation
# Most of the victims were Single men/women.


# In[ ]:


# Visual analysis of missing values
sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False, cmap="viridis")
# Observation
# All yellow lines indicate missing data point. Cabin attribute has many missing values. Age attibute also has some missing values.


# In[ ]:


# Which age group suffered the most?
sns.boxplot(x='Pclass', y='Age', data=df_train, hue='Survived')
# Observation
# Higher class passengers were older compared to lower class
# Older people from each class died more in numbers


# ### Imputing data in Age attribute
# - Using mean age of the price class to impute the age for missing passengers
# - This is to have values in age column without losing the mean of existing age data

# In[ ]:


def impute_age(columns):
    #print(columns)
    if not pd.isnull(columns[1]):
        return columns[1]
    else:
        return round(df_train[df_train['Pclass'] == columns[0]]['Age'].mean(),2)
df_train['New_Age'] = df_train[['Pclass','Age']].apply(impute_age, axis=1)
df_train[df_train['Age'].isnull()].head()


# ### Dropping all unnecessary attributes from the training dataset
# - Name   : Character values which might not have impact on survival output
# - Age    : Imputed age in new column "New_Age"
# - Ticket : Character values without a fixed pattern
# - Cabin  : Many records don't have data in this column. Imputation is feasible.

# In[ ]:


df_train.drop(['Name','Age','Ticket','Cabin'], axis=1, inplace=True)
df_train.head()


# In[ ]:


# Checking the heatmap of the train dataset again to confirm that there are no more missing values
sns.heatmap(df_train.isnull(), yticklabels=False, cmap='viridis')


# ## Applying same data cleanup and formatting steps to "Test" data set. 
# Technically we could have concatenated test dataset with training dataset and performed this activity just one. But in practical scenarios we can have a different test dataset in future on which similar analysis is needed. Hence handling test dataset separately.  
# 
# - Convert Sex column to a categorical/numeric column
# - Convert Embarked column to a categorical/numeric column
# - Impute age 
# - Drop columns : 'Name','Age','Ticket','Cabin'

# In[ ]:


def impute_age1(columns):
    #print(columns)
    if not pd.isnull(columns[1]):
        return columns[1]
    else:
        return round(df_test[df_test['Pclass'] == columns[0]]['Age'].mean(),2)

df_test  = pd.read_csv('test.csv')
df_test['Sex'].replace({'male':0, 'female':1}, inplace=True)
df_test['Embarked'].replace({'S':0,'C':1,'Q':2}, inplace=True)
df_test['New_Age'] = df_test[['Pclass','Age']].apply(impute_age1, axis=1)
df_test.drop(['Name','Age','Ticket','Cabin'], axis=1, inplace=True)
df_test.head()


# In[ ]:


# Confirming that there are no more nulls in the test dataset
sns.heatmap(df_test.isnull(), yticklabels=False)


# In[ ]:


print(df_test.info())
print("Record with nulls : ")
print(df_test[df_test.isin([np.nan, np.inf, -np.inf]).any(1)])
# Shows that there is one row with nulls in the Fare column
# Dropping this row from the test dataset
# df_test.dropna(inplace=True)
print(df_test.info())


# In[ ]:


df_test[df_test['Pclass']==3]['Fare'].median() #7.8958
df_test[df_test['Pclass']==3]['Fare'].mean() #12.459677880184334
pd.DataFrame(df_test[df_test['Pclass']==3]['Fare']).describe()


# In[ ]:


# df_test.set_value(1044,'Fare', 12.46)
df_test.at[152,'Fare']= 12.46


# In[ ]:


df_test.iloc[152]


# ## Model creation for logistic regression

# In[ ]:


Y = df_train['Survived']
X = df_train.drop('Survived', axis=1)


# In[ ]:


lg_model = LogisticRegression()
lg_model.fit(X, Y)

# Predicting the Survived class of test dataset
df_test['predicted_survived'] = lg_model.predict(df_test)
df_test.head()


# In[ ]:


df_submit = df_test[['PassengerId', 'predicted_survived']]


# In[ ]:


df_submit.columns=['PassengerId','Survived']


# In[ ]:


import os
os.getcwd()
os.chdir('/kaggle/working')
os.listdir()


# In[ ]:


df_submit.to_csv('titanic_submission_Logistic_regression.csv')


# In[ ]:


os.listdir()


# In[ ]:


lg_model.score(X, Y)
# This indicates that this model can explaing 80% of the variation in the data


# ### Validating the predictions with visualizations:
# - These visualizations are in accordance with the observations drawn from the training data set

# In[ ]:


sns.set(style="darkgrid")
sns.countplot('predicted_survived',hue='Sex', data=df_test)
# Observation:
# Most of the deceased passengers were males.

sns.catplot(x="Pclass", hue="Sex", col="predicted_survived", data=df_test, kind="count")
# Observation
# Lower class passengers lost their lives the most
# Higher class female passenger survived the most


# ### Further steps :
# - Checking regression metrics
#     - Confusion matrix
#     - Classification report
# - Feature engineering on Name, Ticket attributes
# - Testing different imputation methods
# - Testing pd.get_dummies() function instead of categorical columns

# In[ ]:


df_test.shape

