#!/usr/bin/env python
# coding: utf-8

# ### Titanic Challenge
# 
# Diego N. Vilela - April 2020

# #### Loading the armament
# 
# ___

# In[ ]:


# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from xgboost import XGBClassifier as XGB
from sklearn.model_selection import train_test_split as TTS
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Loading data

df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')


# #### Knowing the enemy - Sun Tzu mode on
# 
# ___

# In[ ]:


# Cheking train data

df_train.head()


# In[ ]:


# Cheking test data

df_test.head()


# In[ ]:


# Cheking the null or blank values

pd.DataFrame({'Train': df_train.drop('Survived', axis = 1).isna().sum(), 'Test': df_test.isna().sum()})


# **Note:** With the exception of the 'survived' field, the sets share the same characteristics, as well as lacking some information. In this way, I believe that joining the sets into one will facilitate the treatment of the data.

# In[ ]:


# First, it is necessary to identify each data set so that they can be segmented after treatment.

df_train['Set'] = 'Train'
df_test['Set'] = 'Test'

# Now, blend.

df = pd.concat([df_train.drop('Survived', axis = 1), df_test])
df.head()


# Nice. It's time to show some art on train data.

# In[ ]:


df_train.dtypes


# Here, we have 3 data types, int64, float64 and object. Let's start on float data.

# In[ ]:


plt.figure(figsize=(10,5))
sb.distplot(df_train[df_train['Survived'] == 0]['Age'].dropna(), color='red')
sb.distplot(df_train[df_train['Survived'] == 1]['Age'].dropna(), color='blue')
plt.ylabel('Passenger count')
plt.title('Age distribution by survived')
plt.legend(['Died', 'Survived'])
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sb.distplot(df_train[df_train['Survived'] == 0]['Fare'].dropna(), color='red')
sb.distplot(df_train[df_train['Survived'] == 1]['Fare'].dropna(), color='blue')
plt.ylabel('Passenger count')
plt.title('Fare distribution by survived')
plt.legend(['Died', 'Survived'])
plt.show()


# In[ ]:


sb.catplot(x = 'Sex', y = 'Survived', data = df_train, kind = 'bar', color='lightblue')
plt.title('Number of suvivors by sex')
plt.show()


# In[ ]:


sb.catplot(x = 'Parch', y = 'Survived', data = df_train, kind = 'bar', color='lightblue')
plt.title('Number of suvivors by parch')
plt.show()


# In[ ]:


sb.catplot(x = 'SibSp', y = 'Survived', data = df_train, kind = 'bar', color='lightblue')
plt.title('Number of suvivors by SibSp')
plt.show()


# In[ ]:


sb.catplot(x = 'Pclass', y = 'Survived', data = df_train, kind = 'bar', color='lightblue')
plt.title('Number of suvivors by Pclass')
plt.show()


# In[ ]:


sb.catplot(x = 'Embarked', y = 'Survived', data = df_train, kind = 'bar', color='lightblue')
plt.title('Number of suvivors by Embarked')
plt.show()


# We can come to some conclusions:
# 
# * the social class (fare and class) indicate that the poorest have a high chance of death;
# 
# * the female sex, as well as having an average number of people in the family, have a great chance of surviving;

# #### Sharpening the knife
# 
# ___

# Time to get your hands dirty, working on missing values and creating new features.

# In[ ]:


# Extract the title

df['Title'] = [s.split(', ')[1].split('.')[0] for s in df['Name']]


# In[ ]:


# Crate a dictionary of age means by Title

age_dict = df.groupby('Title')['Age'].mean().astype(int).to_dict()


# In[ ]:


# Replace nan values with the age mean

df['Age'] = [age_dict[t] if pd.isna(a) else a for a, t in zip(df['Age'], df['Title'])]


# In[ ]:


# Age category

df['Age_Class'] = pd.cut(df['Age'].astype(int), 8, labels=range(8))
df['Age_Class'] = df['Age_Class'].astype(int)


# In[ ]:


# Family size

df['Family'] = df['SibSp'] + df['Parch'] + 1


# In[ ]:


# Change de sex values

df['Sex'].replace({'male':1, 'female': 0}, inplace = True)


# In[ ]:


# Change de embarked values

df['Embarked'].replace({'C':0, 'Q': 1, 'S': 2}, inplace = True)
df['Embarked'].fillna(df['Embarked'].mean(), inplace = True)


# In[ ]:


# Fare values

df['Fare'].fillna(df['Fare'].mean(), inplace = True)


# In[ ]:


# Titles

title_dict = {k:i for i,k in enumerate(df.Title.unique())}

df['Title'].replace(title_dict, inplace = True)


# In[ ]:


# Passenger have a cabin?


df['Cabin'] = [c[0] if not(pd.isna(c)) else 'X' for c in df['Cabin']]

cabin_dict = {k:i for i,k in enumerate(df.Cabin.unique())}

df['Cabin'].replace(cabin_dict, inplace = True)


# In[ ]:


# Fare by person

df['Cost'] = df['Fare'] / df['Family']


# In[ ]:


# Is single?

df['Single'] = np.where(df['Family'] == 1, 1, 0)


# In[ ]:


# Drop the columns and few null values

df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


df.isna().sum()


# Ok! Data sounds good. Now , time to use some computer brain.

# In[ ]:


# Split the data in train and test

df_train_n = pd.concat([df[df['Set'] == 'Train'].drop('Set', axis = 1), df_train['Survived']], axis = 1)
df_test_n = df[df['Set'] == 'Test'].drop(['Set'], axis = 1)


# In[ ]:


# Getting the vectors

X_train, X_test, y_train, y_test = TTS(df_train_n.drop('Survived', axis = 1), df_train_n['Survived'], test_size=0.33, random_state=0)


# In[ ]:


# Fit the model on train data

model = XGB(n_estimators = 100).fit(X_train, y_train)
model.score(X_train, y_train)


# Wow. Nice score!

# In[ ]:


# Checking the fatures importance

pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)


# Let's predict the data from Kaggle :)

# In[ ]:


# Predicting 

y_pred_k = model.predict(df_test_n)


# In[ ]:


# Rounding the probabilities

y_pred_k = np.where(y_pred_k >= 0.5, 1, 0)


# In[ ]:


# Creating the dataframe to export

df_k = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_pred_k})


# In[ ]:


df_k.head()


# In[ ]:


# Exporting the data

df_k.to_csv('Titanic_prediction.csv', sep = ',', index = None)

