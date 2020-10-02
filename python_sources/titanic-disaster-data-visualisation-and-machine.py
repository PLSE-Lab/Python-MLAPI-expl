#!/usr/bin/env python
# coding: utf-8

# This notebook contains my very first public machine learning and data science, visualization approach for specific dataset. Titanic dataset is the most popular on Kaggle to deal with so that is why I decided to start my kernels from here. 

# 1. Importing Libraries and Packages

# In[ ]:


from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')


# 2. Loading and viewing Dataset

# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
train.name = 'Train Data'
test = pd.read_csv('../input/titanic/test.csv')
test.name = 'Test Data'


# In[ ]:


# to play with data I create copies of them

train_copy = train.copy(deep=True)
train_copy.name = 'Train Data'
test_copy = train.copy(deep=True)
test_copy.name = 'Test Data'


# In[ ]:


train.head(6)


# In[ ]:


# Printing datasets' info

for data in [train, test]:
    print('Info of %s \n' %data.name), 
    print(data.info())
    print('\n')
    
# In test data one column is missing - 'Survived'
# It will be the feature which we will want to predict


# In[ ]:


# Describing train data

train_copy.describe()


# In[ ]:


data_cleaner = [train, test]


# In[ ]:


# Seeing null values in datasets

for data in data_cleaner:
    print('Null values in %s'%data.name), 
    print('in every column: \n')
    print(data.isnull().sum())
    print('\n')


# 3. Data cleaning
# 
# We have to see how null values deal with the rest of dataset.
# 
# 'Cabin' column has to be removed from dataset, because it contains more null values than 
# normal. Also 'Ticket' column is going to be deleted because contains messy values, not connected with this task.
# 
# For 'Age' column we have to see how age in this dataset is distributed and decide what mean values assign. 
# 
# For 'Embarked' column I will replace null values with 'S'

# In[ ]:


# Heatmap to see null values - training data

plt.figure(figsize=(10, 6))

sns.heatmap(data=train.isnull(), cmap='plasma', yticklabels=False, cbar=False)
plt.show()


# In[ ]:


# Viewing Age column 

plt.figure(figsize=(10, 6))
plt.title('Age distribution in every class', fontsize=15)
sns.boxenplot(x='Pclass', y='Age', data=train, palette='GnBu_d')


# In[ ]:


# Removing Cabin, Ticket and PassengerId column

train.drop(columns=['Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)
test.drop(columns=['Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)


# In[ ]:


# For Embarked for now I decide to replace NaN values with 'S'

train['Embarked'].fillna(value='S', inplace=True)


# In[ ]:


# For Age column I decide to see age distribution and deicde which mean value assign
# I have to see how age is connected with Pclass

plt.figure(figsize=(10, 6))
plt.title('Age distribution', fontsize=15)
sns.distplot(train['Age'].dropna(), kde=True, bins=40)


# In[ ]:


# For mean age for every class i want to replace null values with mean age for specific class
# I am preparing a funcition 

class_mean_age = pd.DataFrame(train.groupby('Pclass')['Age'].mean())
class_mean_age


# In[ ]:


def mean_age(col):
    age = col[0]
    pclass = col[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return 38
        elif pclass == 2:
            return 30
        else:
            return 25
    else:
        return age

    
# Applying function to Age column to set mean values for missing ones    
train['Age'] = train[['Age', 'Pclass']].apply(mean_age, axis=1)
test['Age'] = test[['Age', 'Pclass']].apply(mean_age, axis=1)


# In[ ]:


# For test dataset one values is missing also in Fare column. I decided to replace this value with mean value of Fare which is 32

test.fillna(value=32, inplace=True)


# 4. Data Visualisation

# In[ ]:


train.head(6)


# In[ ]:


# Visualisation 1 - survived people in each class
# Result: overwhelmingly more people from third class died in disaster. 

plt.figure(figsize=(10, 6))
plt.title('Number of survived people versus classes', fontsize=15)
sns.countplot(data=train, x='Pclass', hue='Survived', palette='Blues')

# Number of dead people in every class - i want to sum and print percentage of people
class3 = train[(train_copy['Pclass'] == 3) & (train_copy['Survived'] == 0)].count()['Pclass']
class2 = train[(train_copy['Pclass'] == 2) & (train_copy['Survived'] == 0)].count()['Pclass']
class1 = train[(train_copy['Pclass'] == 1) & (train_copy['Survived'] == 0)].count()['Pclass']

sum_dead = class3+class2+class1
class1_dead = round((class1/sum_dead)*100, 2)

print('Percentage of people from First Class who died is: %s' %class1_dead),
print('%')


# In[ ]:


# Visualisation 2 - survived people in exact age
# Result: as we can see there is blue peak for survived babies and kids.
# Also there is more older people who survivde (30-60 years) but in age 70-80 year more people died.

plt.figure(figsize=(10, 6))
plt.title('Survived people vs Age', fontsize=15)
g = sns.kdeplot(train['Age'][train['Survived']==0], color='red', shade=True)
g = sns.kdeplot(train['Age'][train['Survived']==1], color='blue', shade=True)
g.set_xlabel('Age')
g.set_ylabel('Frequency')
g.legend(['Not Survived', 'Survived'])


# In[ ]:


# Visualisation 3 - survived people based on gender
# Result: in every class more females survived than males. 

sns.catplot(data=train, x='Pclass', y='Survived', hue='Sex', palette='GnBu_d', kind='bar')
plt.title('Distribution of Survival based on Gender', fontsize=15)
plt.ylabel('Survival Probability')


# In[ ]:


# Mean values of females and males who survived
# Result: much more females survived during the disaster

train[['Sex', 'Survived']].groupby('Sex').mean()


# In[ ]:


# Visualisation 4 - Embarked and survived categorical plot
# Result: people who embarked in Cherbourg had more chance to survive

plt.figure(figsize=(10, 6))
plt.title('Survived people with specific place of embarkation', fontsize=15)
plt.xlabel('Survival probability')
sns.barplot(data=train, x='Embarked', y='Survived', palette='GnBu_d')


# In[ ]:


# Table shows descending survival rate versus place of embark

train[['Embarked', 'Survived']].groupby('Embarked').mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# Visualisation 5 - siblings/spouses abord
# Result: one or two sibling/spuses had more chance to survive

plt.figure(figsize=(10, 6))
plt.title('Survived vs sibling/spouses aboard', fontsize=15)
sns.barplot(data=train, x='SibSp', y='Survived', palette='GnBu_d')


# 5. Machine learning - logistic regression
# 
# This dataset consists of typical categorical features which i want to use to build machine learning approach with logistic regression to predict survival. 

# In[ ]:


train_corr = train.corr()

plt.figure(figsize=(10, 6))
plt.title('Correlations between features', fontsize=15)
sns.heatmap(train_corr, cmap='Blues', annot=True, linewidths=.5)

# As we can see the biggest correlation is between Survived&Fare, Fare&SibSp and Parch&Fare


# In[ ]:


train.head()


# In[ ]:


train.drop('Name', axis=1, inplace=True)


# In[ ]:


test.drop('Name', axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


sex = pd.get_dummies(train['Sex'], drop_first=True)
embarked = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embarked], axis=1)


# In[ ]:


sex = pd.get_dummies(test['Sex'], drop_first=True)
embarked = pd.get_dummies(test['Embarked'], drop_first=True)
test = pd.concat([test, sex, embarked], axis=1)


# In[ ]:


train.drop(['Sex', 'Embarked'], axis=1, inplace=True)
test.drop(['Sex', 'Embarked'], axis=1, inplace=True)


# In[ ]:


train.rename(columns={'male': 'Sex'}, inplace=True)
test.rename(columns={'male': 'Sex'}, inplace=True)


# In[ ]:


train.head(6)


# In[ ]:


test.head()


# In[ ]:


# Preparing X features and y value to predict for training and test data

X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(X_train, y_train)


# In[ ]:


y_predictions = logmodel.predict(X_test)


# In[ ]:


# Accuracy = (TP+TN)/total

acc_matrix = round((133+74)/268, 2)
acc_matrix


# In[ ]:


# Error rate = (FP+FN)/total

error_matrix = round((40+21)/268,2)
error_matrix


# In[ ]:


# Printing accuracy

accuracy_log = round(logmodel.score(X_train, y_train) * 100, 2)
accuracy_log


# In[ ]:


# Printing correlations

coef_df = pd.DataFrame(train.columns[1:])
coef_df.columns = ['Feature']
coef_df
coef_df['Correlation'] = pd.Series(logmodel.coef_[0])
coef_df.sort_values(by='Correlation', ascending=False)


# 6. Machine Learning - k-Nearest Neighbors

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_predictions = knn.predict(X_test)

accuracy_knn = round(knn.score(X_train, y_train) * 100, 2)
accuracy_knn


# For now better accuracy is for KNN method - with my further learning I am going to make another approaches to better predict data. 
