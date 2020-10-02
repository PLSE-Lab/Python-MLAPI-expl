#!/usr/bin/env python
# coding: utf-8

# # 1 Collecting the data
# Download from kaggle directly https://www.kaggle.com/c/3136/download-all

# # load train and test data sets using pandas

# In[ ]:


import pandas as pd
import numpy as np
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")


# # 2 Exploratory Data Analysis
# first five rows of train dataset

# In[ ]:


train.head()


# # Data Dictionary
# 
# Survived 0= Dead, 1 = Alive
# Pclass = (Ticket) 1st class, 2nd class or 3rd class 
# SibSp = sibling or spouse
# Parch = parents or children with passenger
# Embarked = Port of loadnig passenger(S= Southampton,C = Cheroburg,Q=Queenstown)

# In[ ]:


test.head()


# In[ ]:


train.shape


# Trainig data has 891 rows and 12 columns

# In[ ]:


test.shape


# Test data has 418 rows and 11 columns

# In[ ]:


train.info


# # There are some missing fields in test data

# In[ ]:


test.info


# # There are some missing fields in test data

# # Total count of Null values in train data is done below

# In[ ]:


train.isnull().sum()


# # Similarly count of null values in test data is done below

# In[ ]:


test.isnull().sum()


# # Now we need to visualise the data

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # default for plots


# Bar chart for categorical features

# In[ ]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df =pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


bar_chart('Sex')


# # This chart confirms Women more likely survived than Men

# In[ ]:


bar_chart('Pclass')


# # This chart confirms that passengers having 3rd class tickets were dead in large quantity 

# In[ ]:


bar_chart('SibSp')


# # This chart confirms that Passengers with more than 2 siblings or spouse more likely survived

# In[ ]:


bar_chart('Parch')


# # The Chart confirms a person aboarded alone are more likely dead

# We will now begin the most important part of Machine Learning 

# #  3 Feature Engineering

# Feature Engineering is the process of using domain knowledge of the data
# to create features( vectors ) where ML algorithms work
# 
# # Features can be said as columns of the data!

# # Now  we should know that How Titanic Sank?
# Actually Titanic sank from the part where Passengers of 3rd class category were located

# In[ ]:


train.head(10)


# #  *Now I am going to combine train and test data set and create a Title Feature/Column and extract whether Mr/Mrs using regular Expression.*

# # 4 Name

# In[ ]:


train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train['Title'].value_counts()


# In[ ]:


test['Title'].value_counts()


# # Here there are different types of titles but I am going to use Mr ,Miss Mrs and other categories to Others

# Assign using these values
# Mr : 0
# Miss : 1
# Mrs: 2
# Others: 3 

# In[ ]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


train.head()


# # Here last column is created for Title and values are assigned
# 

# In[ ]:


test.head()


# In[ ]:


bar_chart('Title')


# # This chart cannot specify any data because other titles may be maleor female

# # Now we should delete unnecessary data

# In[ ]:


# delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[ ]:


train.head()


# # We now need to change the values of Sex because ML algorithms cannot classify with "male" or "female"

# # male= 0, female = 1

# In[ ]:


sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


train.head()


# In[ ]:


bar_chart('Sex')


# # Now we know that some Age columns have missing values
# Statistically we can use Title Median age

# # 5 Age

# In[ ]:


# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.tail()


# In[ ]:


test.tail()


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
 
plt.show()


# # Until 16 yrs of age there is high chance to survive
# we can use limits in above code to see the exact age range of survival

# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(0, 20)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(20, 30)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(30, 40)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(40, 80)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(60)


# In[ ]:


train.info()


# In[ ]:


test.info()


# # Now we need to convert Age to categorical variable
# child: 0
# young: 1
# adult: 2
# mid-age: 3
# senior: 4

# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4


# In[ ]:


train.head()


# In[ ]:


bar_chart('Age')


# # 6 Embarked / Boarding Place
# Find the missing values
# 

# In[ ]:


Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# more than 50% of 1st class are from S embark
# 
# more than 50% of 2nd class are from S embark
# 
# more than 50% of 3rd class are from S embark

# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


train.head()


# In[ ]:


embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# In[ ]:


train.head()


# # 7 Fare/ Ticket Price

# In[ ]:


# fill missing Fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
train.head(20)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
 
plt.show()


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.xlim(0, 20)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.xlim(0, 30)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.xlim(0)


# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[ ]:


train.head()


# # 7 Cabin

# In[ ]:


train.Cabin.value_counts()


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[ ]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


# Cabin mapping
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# # I have given diff values above you can give your values
# This is called feature Scaling
# these above values are convinient because
# ML Classifiers use Euclidian distances

# In[ ]:


# fill missing Fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# # 8 Family Size

# In[ ]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)


# In[ ]:


family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# Drop unwanted features
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)


# In[ ]:


train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape


# In[ ]:


train_data.head()


# # Modelling
# This is the most important step in ML

# In[ ]:


# We need to import Classifier modules/Packages

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


train.info()


# # 9 Cross Validation (K-fold)
# In ML we cannot fit the model on training data also we cannot say model will work on final data
# We must assure our model is not too noisy and got correct pattern

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# # 10 K- Nearest Neighbours

# In[ ]:


clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# # Above we got the acuracy

# # Score

# In[ ]:


#KNN Score
round(np.mean(score)*100, 2)


# # 11 Descison Tree

# In[ ]:


clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# # Score

# In[ ]:


# decision tree Score
round(np.mean(score)*100, 2)


# # 12 Random Forest

# In[ ]:


clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# # Score

# In[ ]:


# Random Forest Score
round(np.mean(score)*100, 2)


# # 13 Naive Bayes

# In[ ]:


clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# # Score

# In[ ]:


# Naive Bayes Score
round(np.mean(score)*100, 2)


# # 14 SVM

# In[ ]:


clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# # Testing 
# Testing our Model is the most importnt step because it tests the accuracy/prescision of the model

# In[ ]:


clf = SVC()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)


# In[ ]:


# Submit the data 
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)


# In[ ]:


submission = pd.read_csv('submission.csv')
submission.head()


# # This is the accuracy and the data we have obtained from the Dirty Data 

# In[ ]:




