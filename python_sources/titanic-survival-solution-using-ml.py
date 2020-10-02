#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# COLLECTING  TRAIN DATA AND TEST DATA USING PANDAS
import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # EXPLORATORY DATA ANALYSIS

# In[ ]:


train.head()


# In[ ]:


test.head(10)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


# IMPORT PYTHON LIBRARY FOR VISUALISATION
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def bar_chart(feature):
    survived = train[train['Survived'] == 1 ] [feature].value_counts()
    dead = train[train['Survived'] == 0] [feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind = 'bar',stacked = True ,figsize = (10,5))


# In[ ]:


bar_chart('Pclass')              


# # The Chart confirms 1st class more likely survived than other classes
# # The Chart confirms 3rd class more likely dead than other classes

# In[ ]:


bar_chart('Sex')    # FEMALE ARE MORE LIKELY TO SURVIVE THAN MEN


# In[ ]:


bar_chart('SibSp')


# # The Chart confirms a person aboarded with more than 2 siblings or spouse more likely survived
# # The Chart confirms a person aboarded without siblings or spouse more likely dead

# In[ ]:


bar_chart('Parch')


# # The Chart confirms a person aboarded with more than 2 parents or children more likely survived
# # The Chart confirms a person aboarded alone more likely dead

# In[ ]:


bar_chart('Embarked')


# # The Chart confirms a person aboarded from C slightly more likely survived
# # The Chart confirms a person aboarded from Q more likely dead
# # The Chart confirms a person aboarded from S more likely dead

# # FEATURE ENGINEERING

# In[ ]:


train.head(10)


# In[ ]:


# COMBINING TRAIN DATA AND TEST DATA 
train_test_data = [train,test]
for data in train_test_data :
    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.',expand = False) # EXTRACTING TITLE FROM NAME


# In[ ]:


train['Title'].value_counts()


# In[ ]:


test['Title'].value_counts()


# # TITLE MAPPING 
# # Mr = 0
# # Miss = 1
# # Mrs = 2 
# # others = 3

# In[ ]:


title_mapping = { 'Mr' : 0 ,'Miss': 1,  'Mrs' : 2,'Master' : 3  ,'Dr' : 3 ,'Rev' : 3, 'Col' : 3 ,
                 'Major' :3 , 'Mlle' : 3 ,'Mme' : 3 ,'Countess' : 3  , 'Sir' : 3  ,'Dona' : 3,'Don' : 3,
                 'Ms' : 3 ,  'Capt' : 3 ,  'Lady' : 3 ,'Jonkheer' : 3 }     
for data in train_test_data:
    data['Title'] = data['Title'].map(title_mapping)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


bar_chart('Title')


# In[ ]:


# delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[ ]:


train.head()


# # SEX MAPPING
#  # male = 0
# # female = 1

# In[ ]:


sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


bar_chart('Sex')


# # AGE

# In[ ]:


train.info() # age are missing so fill the missing age


# In[ ]:


train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


train.head(30)
train.groupby("Title")["Age"].transform("median")


# In[ ]:


train.info()


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
 
plt.show()


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
plt.xlim(40, 60)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(60)


# # converting numerical age to categorical variable

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


# # EMBARKED 

# In[ ]:


Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


# filling missing value in Embarked with S
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


# EMBARKED MAPPING
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# # FARE ( filling missing value in fare)

# In[ ]:


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


# # CABIN ( filling missing value of Cabin)

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


# cabin mapping
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[ ]:


# fill missing Fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# In[ ]:


train.info()


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


unecessary_feature = ['Ticket', 'SibSp', 'Parch']
train = train.drop(unecessary_feature, axis=1)
test = test.drop(unecessary_feature, axis=1)
train = train.drop(['PassengerId'], axis=1)


# In[ ]:


train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape


# In[ ]:


train_data.head(5)


# In[ ]:


# MODELLING


# In[ ]:


# IMPORTING CLASSIFIERS MODULES
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np


# In[ ]:


# CROSS VALIDATION
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[ ]:


# kNN
knn = KNeighborsClassifier(n_neighbors= 13)
scoring = 'accuracy'
score = cross_val_score(knn,train_data,target,cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# In[ ]:


# DECISION TREE CLASSIFIER
dtc = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(dtc, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# In[ ]:


Rfc = RandomForestClassifier( n_estimators= 13)
scoring = 'accuracy'
score = cross_val_score(Rfc, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# In[ ]:


# NAIVE BAYES
NB = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(NB, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# In[ ]:


# SUPPORT VECTOR MACHINE
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# In[ ]:


# model testing            SVM  IS BEST APPROACH HERE AS THIS METHOD GIVES HIGH S


# In[ ]:


test.info()


# In[ ]:


test_data = test.drop("PassengerId", axis=1).copy()


# In[ ]:


test_data


# In[ ]:


clf = SVC()
clf.fit(train_data,target)
prediction = clf.predict(test_data)


# In[ ]:


prediction


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)


# In[ ]:


submission = pd.read_csv('submission.csv')
submission.head()


# In[ ]:




