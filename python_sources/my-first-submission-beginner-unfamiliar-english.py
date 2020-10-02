#!/usr/bin/env python
# coding: utf-8

# # My First Submission(beginner, unfamiliar English)
# 
# Hello, This is my first kaggle challenge and the first entry into data science. i will focus on feature engineering , ensemble model and feature analysis.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

sns.set(style='white', context='notebook', palette='deep')


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# **Let's take a quick look at train data and test data.**

# In[ ]:


train.head()


# In[ ]:


test.head()


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


# test.isnull().sum()

# In[ ]:


test.isnull().sum()


# Results
# 1. Train data has 891 data and 12 features. you have to fill in lost data(Age, Cabin, Embarked).
# 2. Test data has 418 data and 11 features. you have to fill in lost data(Age, Cabin, Fare).
# 
# Lost data should be replaced with meaningful data through feature engineeing. just applying the model can cause problems. 
#  
#  Of course, the test data has 11 features because there is no survival feature.

# In[ ]:


def bar_chart(features):
    survived = train[train['Survived']==1][features].value_counts()
    dead = train[train['Survived']==0][features].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))


# * Prepare fuctions to easily determine the ratio of survival and death according to features.

# In[ ]:


bar_chart('Sex')


# In[ ]:


bar_chart('Pclass')


# In[ ]:


bar_chart('SibSp')


# In[ ]:


bar_chart('Parch')


# In[ ]:


bar_chart('Embarked')


# These are the information i can get from the charts.
# 
# 1. 'Sex' chart shows that the survival rate of women is relatively higher than men.
# 2. 'Pclass' chart shows that 1st class passengers have a higher survival rate than in other spaces.
# 3. 'SibSp' chart shows that passengers with realtive or spouse have a higher survival rate than alone. too many 'SibSp' are not.
# 4. 'Parch' chart shows that passengers with family or child have a higher survival rate than alone. too many 'Parch' are not.
# 5. 'Embarked' chart shows that passengers in the C dock have a higher survival rate than other dock.
# 

# In[ ]:


train_test_data = [train,test]

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-za-a]+)\.',expand=False)


# * The important information you can get from the 'Name' feature is the title.
# * extract the title from the name and add a new feature('Title').
# * Except for the three titles with many numbers, the remains are mapping '3'.

# In[ ]:


train['Title'].value_counts()


# In[ ]:


test['Title'].value_counts()


# In[ ]:


title_mapping = {"Mr":0, "Miss":1, "Mrs":2, "Master":3, "Dr":3, "Rev":3, "Col":3, "Major":3, "Mile":3, "Jonkheer":3, "Countess":3, "Sir":3, "Ms":3, "Lady":3, "Mme":3, "Don":3, "Capt":3, "Dona":3}


# In[ ]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


train.head()


# In[ ]:


train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[ ]:


sex_mapping = {'male':0, 'female':1}

for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


train.head()


# In[ ]:


train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)


# * The way to fill in the lost data of the 'Age' feature could be the average or median of all passengers ages, but i thought that 'Title' was the basis for gender and filled it with a median value according to 'Title'. 

# In[ ]:


train.head(20)


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
plt.xlim(0,20)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(20,40)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(40,60)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(60,100)


# In[ ]:


for dataset in train_test_data:
    dataset.loc[dataset['Age']<=16, 'Age'] = 0
    dataset.loc[(dataset['Age']>16)& (dataset['Age']<=23), 'Age'] =1
    dataset.loc[(dataset['Age']>23)& (dataset['Age']<=34), 'Age'] =2
    dataset.loc[(dataset['Age']>34)& (dataset['Age']<=42), 'Age'] =3
    dataset.loc[(dataset['Age']>42)& (dataset['Age']<=58), 'Age'] =4
    dataset.loc[dataset['Age']>58, 'Age'] = 5


# In[ ]:


train.head(10)


# In[ ]:


Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,5))


# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Title'] = dataset['Title'].fillna(0)


# * Passengers embarked at the S dock occupy more than 70 percent of all class.
# * so the two lost data are filled with S dock.

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


embarked_mapping = {'S': 0, "C": 1, "Q": 2}

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# In[ ]:


train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# * The way to fill in the lost data of the 'Fare' feature could be the average or median of all passengers fare, but i thought that 'Fare' was the basis for class and filled it with a median value according to 'Pclass'.  

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
plt.xlim(0,20)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.xlim(20,40)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.xlim(40,60)


# In[ ]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.xlim(60,100)


# In[ ]:


for dataset in train_test_data:
    dataset.loc[dataset['Fare']<=17, 'Fare'] = 0
    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=29), 'Fare']= 1
    dataset.loc[(dataset['Fare']>29) & (dataset['Fare']<=100), 'Fare']= 2
    dataset.loc[dataset['Fare']>100, 'Fare'] = 3


# In[ ]:


train.head(10)


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
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind ='bar', stacked=True, figsize=(10,5))


# In[ ]:


cabin_mapping = {"A":0, "B":0.4, "C":0.8, "D":1.2, "E":1.6, "F":2.0, "G":2.4, "T":2.8}
for dataset in train_test_data:
    dataset['Cabin']=dataset['Cabin'].map(cabin_mapping)


# In[ ]:


train['Cabin'].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test['Cabin'].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# * To distinguish the 'Cabin', we extract only the first letter. because 'Cabin' is related to class, we
#  divide 'Cabin' by class. 
# * As a result, we can confirm that cabin belongs to 1st class only.
# * The way to fill in the lost data of the 'Cabin' was the basis for class and filled it with a median value according to 'Pclass'.

# In[ ]:


train.head(20)


# In[ ]:


train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1


# In[ ]:


train['FamilySize'].max()


# In[ ]:


family_mapping = {1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2.0, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4.0}
for dataset in train_test_data:
    dataset['FamilySize']=dataset['FamilySize'].map(family_mapping)


# In[ ]:


train.head(10)


# In[ ]:


features_drop = ['SibSp', 'Ticket', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)


# In[ ]:


train_data = train.drop(['Survived'], axis=1)
target = train['Survived']


# In[ ]:


train_data.head(10)


# In[ ]:


train.info()


# In[ ]:


test.info()


# I studied 6 popular classifiers and evaluate the mean accuracy of each of them by a stratified kfold cross validation procedure.
# 
# 1. SVC
# 2. DecisionTree
# 3. AdaBoost
# 4. RandomForest
# 5. ExtraTree
# 6. GradientBoosting
# 

# In[ ]:


k_fold = StratifiedKFold(n_splits=10)


# In[ ]:


random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, train_data, y = target, scoring = "accuracy", cv = k_fold, n_jobs=-1))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# I decided to choose the SVC, AdaBoost, RandomForest , ExtraTrees and the GradientBoosting classifiers for the ensemble modeling.

# In[ ]:


DTC = DecisionTreeClassifier(max_depth=5)

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,10,20,30,40,50],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=k_fold, scoring="accuracy", n_jobs= -1, verbose = 1,return_train_score = True)

gsadaDTC.fit(train_data, target)

ada_best = gsadaDTC.best_estimator_


# In[ ]:


gsadaDTC.best_score_


# In[ ]:


ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [5],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[500,1000],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=k_fold, scoring="accuracy", n_jobs= -1, verbose = 1)

gsExtC.fit(train_data,target)

ExtC_best = gsExtC.best_estimator_


# In[ ]:


gsExtC.best_score_


# In[ ]:


RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [2],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,500],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=k_fold, scoring="accuracy", n_jobs=-1, verbose = 1)

gsRFC.fit(train_data,target)

RFC_best = gsRFC.best_estimator_


# In[ ]:


gsRFC.best_score_


# In[ ]:


GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [1000,2000,3000],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [19],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=k_fold, scoring="accuracy", n_jobs= -1, verbose = 1)

gsGBC.fit(train_data,target)

GBC_best = gsGBC.best_estimator_


# In[ ]:


gsGBC.best_score_


# In[ ]:


SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [ 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=k_fold, scoring="accuracy", n_jobs=-1, verbose = 1)

gsSVMC.fit(train_data,target)

SVMC_best = gsSVMC.best_estimator_


# In[ ]:


gsSVMC.best_score_


# * Ensemble medeling 
# 
# We will use the voting classifier to combine the predictions from the 5 classifiers for ensemble modeling.
# 

# In[ ]:


votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=-1)

votingC = votingC.fit(train_data, target)


# * Predict and submission

# In[ ]:


test_data = test.drop("PassengerId", axis=1).copy()
prediction = votingC.predict(test_data)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })
submission.to_csv('submission.csv', index=False)


# If you have seen this notebook helpful, please upvote it. And many more advice and questions. They will be a great help to me.::))
