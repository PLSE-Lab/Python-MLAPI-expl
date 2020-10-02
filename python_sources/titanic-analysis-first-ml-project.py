#!/usr/bin/env python
# coding: utf-8

# Import  necessary libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


import os
print(os.listdir("../input"))


# **Collecting the data**
# 
# Load the Train and Test datasets.

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# **Exploratory Data Analysis**

# In[ ]:


train.head()


# In[ ]:


test.head()


# The train set contains 891 rows and 12 columns.
# The test set contains 418 rows and 11 columns.

# 
# The train and test set has some missing values.
# Missing values are in the Age, Embarked and cabin in the train set whereas in test set some of the Age, Fare and Cabin values are missing.

# In[ ]:


train.info(), test.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# **Visualization**

# How many people have not survived from the Titanic ship tragedy?

# In[ ]:


sns.countplot('Survived', data=train)


# In[ ]:


sns.countplot(x='Survived', hue='Sex', data = train)


# It is clearly evident from the graph, survived female passengers is more when compared to male passengers.

# Now lets see how Pclass category affects survived and non survived passengers.

# In[ ]:


sns.countplot(x='Survived', hue='Pclass', data=train)


# 
# Obvisously, most of the people from the Pclass category 3 had not been survived. More priority to survive had been given to people from category 1 .
# 

# In[ ]:


sns.catplot(x='Sex', hue='Pclass',col='Survived', data=train, kind='count')


# Most of the female passengers irrespective of the category had been survived except category 3.
# It is evident from the graph, the survival rate is too less for male passengers. Massive number of people had died who travelled in category 3 class.

# **Feature Engineering**

# First combine train and test data into one list

# In[ ]:


train_test_data = [train, test]

for dataset in train_test_data:
    dataset['Title'] = dataset["Name"].str.extract( '([A-za-z]+)\.', expand=False )


# In[ ]:


train['Title'].value_counts()


# In[ ]:


test['Title'].value_counts()


# In[ ]:





# 1. Title
# 
# Map Title:
#     Mr -> 0
#     Miss ->1
#     Mrs -> 2
#     Others -> 3

# In[ ]:


title_mapping = { 'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Dr':3, 'Rev':3, 'Mlle':3, 'Col':3, 'Major':3, 'Sir':3, 'Ms':3,
                 'Capt':3, 'Don':3, 'Jonkheer':3,'Countess':3,'Lady':3, 'Mme':3, 'Dona':3
    
}


# In[ ]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


#delete unnecessary columns from the dataset.

train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info(), test.info()


# 2.Sex
# 
#     Male -> 1
#     Female -> 0

# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelEncoder_Sex = LabelEncoder()

for dataset in train_test_data:
    dataset['Sex'] = labelEncoder_Sex.fit_transform(dataset['Sex'])


# In[ ]:


train.head(), test.head()


# 3. Age
# 
# 

# In[ ]:


#Filling missing values with median age for each title (Mr,Mrs, Miss, Others)

train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# Converting numerical Age to categorical variable
# 
# Child ->0
# Young ->1
# Adult ->2
# Mid-Men ->3
# Senior ->4

# In[ ]:


for dataset in train_test_data:
    dataset.loc[dataset['Age']<= 16 , 'Age'] =0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <=26), 'Age' ] =1,
    dataset.loc[(dataset['Age'] >26) & (dataset['Age'] <= 36),'Age'] =2,
    dataset.loc[(dataset['Age'] >36 & (dataset['Age'] <=62),'Age')]=3,
    dataset.loc[(dataset['Age'] >62),'Age']=4


# In[ ]:


train.head()


# In[ ]:


test.head()


# 4.Embarkation

# In[ ]:


#Fill the missing values with the 'mode' of Embarkation

for dataset in train_test_data:
    mode_embarked = dataset['Embarked'].mode()[0]
    dataset['Embarked'].fillna(mode_embarked, inplace=True)


# In[ ]:


train.info(), test.info()


# In[ ]:


# Convert the Embarkation to categorical variables.

for dataset in train_test_data:
    dataset['Embarked'] = labelEncoder_Sex.fit_transform(dataset['Embarked'])


# In[ ]:


train.head()


# In[ ]:


test.head()


# 5. Fare

# In[ ]:


# Fill missing values of fare using median fare of each class.

test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace=True)


# In[ ]:


test.info()


# In[ ]:


#Convert Fare to categorical values

for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <=17, 'Fare'] =0,
    dataset.loc[(dataset['Fare'] >17) & (dataset['Fare'] <= 30), 'Fare'] =1,
    dataset.loc[(dataset['Fare'] >30) & (dataset['Fare'] <= 100), 'Fare'] =2,
    dataset.loc[dataset['Fare'] >100, 'Fare'] =3
    
    
    


# In[ ]:


train.head()


# In[ ]:


test.head()


# 6.Cabin
# 

# In[ ]:


#Fill out the missing values of cabin.

train['Cabin'].value_counts()


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[ ]:


cabin_mapping = {'A': 0 , 'B':0.4 , 'C':0.8, 'D':1.2, 'E':1.6, 'F':2, 'G':2.4, 'T':2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[ ]:


#fill out the missing values of cabin 

train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'), inplace=True)


# In[ ]:


train.info(),test.info()


# In[ ]:


train.head()


# In[ ]:


train['Cabin'].value_counts()


# In[ ]:


test.head()


# In[ ]:


test['Cabin'].value_counts()


# 7.Family Size

# In[ ]:


train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1


# In[ ]:


family_mapping = {1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6 , 6:2.0, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4.0}

for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


#Drop unnecessary columns from train and test data.
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop('PassengerId', axis=1)


# In[ ]:


train_data = train.drop('Survived', axis =1)
target = train['Survived']


# In[ ]:


train_data.head()


# In[ ]:


target.head()


# In[ ]:


train.info(), test.info()


# ### **Modelling**

# In[ ]:



# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


train.info()


# Cross Validation Fold
# 

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# KNN

# In[ ]:


clf = KNeighborsClassifier(n_neighbors=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, scoring = scoring, n_jobs=1)
print(score)


# In[ ]:


#knn score

round(np.mean(score)*100,2)


# Decision Tree

# In[ ]:


clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, scoring=scoring,n_jobs=1)
print(score)


# In[ ]:


#Decision Tree score
round(np.mean(score)*100,2)


# Random Forest

# In[ ]:


clf = RandomForestClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, scoring=scoring, n_jobs=1, cv=k_fold)
print(score)


# In[ ]:


#Random forest classifier score

round(np.mean(score)*100,2)


# Naive Bayes

# In[ ]:


clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, scoring=scoring, n_jobs=1)
print(score)


# In[ ]:


#Naive bayes score
round(np.mean(score)*100,2)


# SVM

# In[ ]:


clf = SVC(gamma='auto')
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


#SVM score
round(np.mean(score)*100,2)


# ### **Testing** 

# In[ ]:


clf = SVC(gamma='auto')
clf.fit(train_data, target)

test_data = test.drop('PassengerId', axis=1).copy()
prediction = clf.predict(test_data)


# In[ ]:


submission = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':prediction
})

submission.to_csv('/kaggle/working/submission.csv', index=False)


# In[ ]:


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.head()


# In[ ]:


print(os.listdir("/kaggle/working/"))


# In[ ]:




