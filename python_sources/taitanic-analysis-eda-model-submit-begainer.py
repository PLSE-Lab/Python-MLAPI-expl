#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# # Data Load

# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train_df.head(10)


# In[ ]:


test_df.tail(10)


# # First submission
# Mark everyone as surviving and see the Accuracy.

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": 1
    })
submission.to_csv('submission.csv', index=False)


# # Test data load again

# In[ ]:


test_df2 = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


test_df2.head()


# In[ ]:


test_df['sur'] = 0


# In[ ]:


test_df.head()


# In[ ]:


test_df = test_df.drop(['sur'], axis=1)


# In[ ]:


test_df.head()


# In[ ]:


test_df2['sur'] = 0


# In[ ]:


test_df2.head()


# In[ ]:


# Mark all female had been survived thats why sur=1
test_df2.loc[test_df2.Sex == 'female', 'sur'] = 1


# In[ ]:


test_df2.head()


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_df2["sur"]
    })


# # 3rd submission

# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:


# fare less than 20 and who belong Pclass = 3 and who was female, mark then and died.
test_df2.loc[(test_df2.Fare > 20) & (test_df2['Pclass'] == 3) & (test_df['Sex']== 'female') , 'sur'] = 0


# In[ ]:


test_df2.head()


# In[ ]:


# Update submission file 
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_df2["sur"]
    })


# # 4th submission

# In[ ]:


submission.to_csv('submission.csv', index=False)


# # Start again for 5th Prediction and submission

# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train_df.shape


# In[ ]:


test_df.shape


# In[ ]:


train_df.info()
# observe some missing data in Age, Cabin, Embarked


# In[ ]:


test_df.info()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


#define a function, so that we can make bar chart for every feature. 
def barchart(feature):
    g = sns.barplot(x=feature,y="Survived",data=train_df)
    g = g.set_ylabel("Survival Probability")


# In[ ]:


# For sex feature. And see most of Feamale passenger had beed survived.
barchart('Sex')


# In[ ]:


barchart('Pclass')


# In[ ]:


barchart('SibSp')


# In[ ]:


barchart('Parch')


# In[ ]:


barchart('Embarked')


# # Feature Engineering and increase your accuracy

# In[ ]:


# Marged train and test data set.
train_test_data = [train_df, test_df]
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand = False)


# In[ ]:


train_df['Title'].value_counts()


# In[ ]:


test_df['Title'].value_counts()


# In[ ]:


# extract example 
s2 = pd.Series(['a_b, Dr. c', 'c_d001. e', 'ADR, Mr1. Sajal', 'f_g8.h','as, Miss. Angel'])
s2.str.extract('([A-Za-z]+[0-9]+)\.')


# In[ ]:


s2.str.extract('([A-Za-z]+)\.')


# In[ ]:


#Mapping the unnecessary title with 0,1,2,3
title_mapping = {"Mr": 0,"Miss": 1,"Mrs": 2,"Master": 3,"Dr": 3,"Rev": 3,"Mlle": 3,"Countess": 3,"Ms": 3,"Lady": 3,"Jonkheer": 3,"Don": 3,"Dona": 3,"Mme": 3,"Capt": 3,"Sir": 3,"Col":3,"Major":3 }

for dataset in train_test_data:
    dataset['Title']  = dataset['Title'].map(title_mapping)


# In[ ]:


test_df['Title'].value_counts()


# In[ ]:


train_df['Title'].value_counts()


# In[ ]:


train_df.info()


# In[ ]:


# Delete unnecessary feature from dataset
train_df.drop('Name',axis=1,inplace=True)
test_df.drop('Name',axis=1,inplace=True)


# In[ ]:


test_df.head()


# In[ ]:


#Mapping Male and Female in number 
sex_mapping = {"male": 0,"female": 1 }

for dataset in train_test_data:
    dataset['Sex']  = dataset['Sex'].map(sex_mapping)


# In[ ]:


test_df.head()


# In[ ]:


barchart('Sex')


# In[ ]:


# FIll missing age with measian age of passengers 
train_df["Age"].fillna(train_df.groupby("Title")["Age"].transform("median"), inplace=True)
test_df["Age"].fillna(test_df.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


# See -> Age are now not NULL
train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


# For better understanding we make some chart for age 

facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0,train_df['Age'].max()))
facet.add_legend()

plt.show()


# In[ ]:


facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0,train_df['Age'].max()))
facet.add_legend()

plt.xlim(0,20)
# plt.show()


# In[ ]:


facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0,train_df['Age'].max()))
facet.add_legend()

plt.xlim(20,30)


# In[ ]:


facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0,train_df['Age'].max()))
facet.add_legend()

plt.xlim(30,40)


# In[ ]:


facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0,train_df['Age'].max()))
facet.add_legend()

plt.xlim(40,80)


# In[ ]:


# Make category for age in five as child=0, young=1, adult=2, mid_age=3, senior=4
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <=16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] >16) & (dataset['Age'] <=26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] >26) & (dataset['Age'] <=36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] >36) & (dataset['Age'] <=62), 'Age'] = 3,
    dataset.loc[dataset['Age'] >62, 'Age'] = 4


# In[ ]:


train_df.head()


# In[ ]:


barchart('Age')


# In[ ]:


# filling missing value of Embarked
for dataset in train_test_data:
    dataset['Embarked']  = dataset['Embarked'].fillna('S')


# In[ ]:


train_df.info()


# In[ ]:


embarked_map = {"S":0, "C":1, "Q":2}
for dataset in train_test_data:
    dataset['Embarked']  = dataset['Embarked'].map(embarked_map)


# In[ ]:


train_df.head()


# In[ ]:


test_df.info()


# In[ ]:


# FIll missing Fare with measian age of passengers 
train_df["Fare"].fillna(train_df.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test_df["Fare"].fillna(test_df.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test_df.info()


# In[ ]:


# For better understanding we make some chart for Fare

facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0,train_df['Fare'].max()))
facet.add_legend()

plt.show()


# In[ ]:


facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0,train_df['Fare'].max()))
facet.add_legend()
plt.xlim(0,20)


# In[ ]:


facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0,train_df['Fare'].max()))
facet.add_legend()
plt.xlim(20,30)


# In[ ]:


facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0,train_df['Fare'].max()))
facet.add_legend()
plt.xlim(30,100)


# In[ ]:


# Make category for FARE in four 
for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <=7.5, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] >7.5) & (dataset['Fare'] <=15), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] >15) & (dataset['Fare'] <=30), 'Fare'] = 2,
    dataset.loc[(dataset['Fare'] >30) & (dataset['Fare'] <=100), 'Fare'] = 3,
    dataset.loc[dataset['Fare'] >100, 'Fare'] = 4


# In[ ]:


train_df.head(20)


# In[ ]:


# work with Cabin 
train_df.Cabin.value_counts()


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[ ]:


train_df.Cabin.value_counts()


# In[ ]:


pclass1 = train_df[train_df['Pclass'] == 1]['Cabin'].value_counts()
pclass2 = train_df[train_df['Pclass'] == 2]['Cabin'].value_counts()
pclass3 = train_df[train_df['Pclass'] == 3]['Cabin'].value_counts()
df = pd.DataFrame([pclass1,pclass2,pclass3])
df.index = ['1st class','2nd class','3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


# Cabin Mapping 
cabin_mapping = {"A":0,"B":0.4,"C":0.8,"D":1.2,"E":1.6,"F":2,"G":2.4,"T":2.8}
for dataset in train_test_data:
    dataset['Cabin']  = dataset['Cabin'].map(cabin_mapping)


# In[ ]:


# filling missing Fare with median fare for each Pclass
train_df["Cabin"].fillna(train_df.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test_df["Cabin"].fillna(test_df.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


# For family size
train_df["FamilySize"] = train_df["SibSp"]+ train_df["Parch"]+1


# In[ ]:


test_df["FamilySize"] = test_df["SibSp"]+ test_df["Parch"]+ 1


# In[ ]:


facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'FamilySize', shade=True)
facet.set(xlim=(0,train_df['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)


# In[ ]:


# Family Mapping 
family_mapping = {1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2,7:2.4,8:2.8,9:3.2,10:3.6,11:4}
for dataset in train_test_data:
    dataset['FamilySize']  = dataset['FamilySize'].map(family_mapping)


# In[ ]:


train_df.head()


# In[ ]:


# Dropping the unnecessary feature
frdp = ['Ticket','SibSp','Parch']
train_df = train_df.drop(frdp, axis=1)
test_df = test_df.drop(frdp, axis=1)
train_df = train_df.drop(['PassengerId'], axis=1)


# In[ ]:


train_df.head()


# # Machine Learning modeling 

# In[ ]:


# Importing Classifier Modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# cross validatin with KFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
K_fold = KFold(n_splits=10,shuffle=True,random_state =0)


# # DecisionTree

# In[ ]:


clf = DecisionTreeClassifier()
scoring = 'accuracy'
x = train_df.drop('Survived',axis=1)
y = train_df['Survived']
score = cross_val_score(clf ,x ,y , cv=K_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


#decision tree Score
round(np.mean(score)*100,2)


# # RandomForest

# In[ ]:


clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf ,x ,y , cv=K_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


#Random Forest Score
round(np.mean(score)*100,2)


# **Make Submission file**

# In[ ]:


clf = RandomForestClassifier(n_estimators=13)
clf.fit(x, y)

test_data = test_df.drop("PassengerId", axis=1).copy()


# In[ ]:


test_data.info()


# In[ ]:


prediction = clf.predict(test_data)


# In[ ]:


# Update submission file 
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": prediction
    })
submission.to_csv('submission.csv',index=False)


# In[ ]:


submission = pd.read_csv('submission.csv')
submission.head()


# In[ ]:




