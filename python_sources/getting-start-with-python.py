#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # **Import data**

# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
submission = pd.read_csv("../input/titanic/gender_submission.csv")


# # **Data exploratory analysis**

# ## **Data overview**

# In[ ]:


train.head()


# In[ ]:


print('Train data shape is: ', train.shape)
print('Test data shape is: ', test.shape)


# ## **Data visualization**

# NA value:

# In[ ]:


plt.subplots(0,0, figsize = (15,3))
train.isnull().mean().sort_values(ascending = False).plot.bar(color = 'grey')
plt.axhline(y=0.1, color='r', linestyle='-')
plt.title('Missing values average per columns in TRAIN data', fontsize = 20)
plt.show()

plt.subplots(1,0, figsize = (15,3))
test.isnull().mean().sort_values(ascending = False).plot.bar(color = 'grey')
plt.axhline(y=0.1, color='r', linestyle='-')
plt.title('Missing values average per columns in TEST data', fontsize = 20)
plt.show()


# In[ ]:


fig = plt.figure(figsize = (15,10))

ax1 = plt.subplot2grid((2,3),(0,0))
sns.countplot(x = 'Survived', data = train)
plt.title('Survived')

ax1 = plt.subplot2grid((2,3),(0,1))
sns.countplot(x = 'Sex',hue = 'Survived',data = train)
plt.title('Sex')

ax1 = plt.subplot2grid((2,3),(0,2))
sns.countplot(x = 'SibSp',hue = 'Survived', data = train)
plt.title('SibSp')

ax1 = plt.subplot2grid((2,3),(1,0))
sns.countplot(x = 'Parch',hue = 'Survived', data = train)
plt.title('Parch')

ax1 = plt.subplot2grid((2,3),(1,1))
sns.countplot(x = 'Pclass',hue = 'Survived', data = train)
plt.title('Pclass')

ax1 = plt.subplot2grid((2,3),(0,1))
sns.countplot(x = 'Embarked',hue = 'Survived', data = train)
plt.title('Embarked')


# Combine train & test data to pre-processing

# In[ ]:


df = train.drop('Survived',axis = 1).append(test)


# In[ ]:


df['Cabin'] = df['Cabin'].str[0]
df['Cabin'] = df['Cabin'].fillna("noCabin")
df['Cabin'] = df['Cabin'].replace("T","C")
#mapping = {'noCabin':0,'G':1,'F':2,'E':3,'D':4,'C':5,'B':6,'A':7}
#df['Cabin'] = df['Cabin'].map(mapping)

df['Embarked'] = df['Embarked'].fillna("S")

#mapping = {'C':3,'Q':2,'S':1}
#df['Embarked'] = df['Embarked'].map(mapping)


# In[ ]:


sns.countplot(x = 'Cabin', data = df)
plt.title('Cabin barplot')


# In[ ]:


df['FamilySize'] = df['SibSp'] + df['Parch'] + 1


# In[ ]:


#df['Size'] = df['FamilySize']
#df['Size'] = df['Size'].replace(1,1)
#df['Size'] = df['Size'].replace([2,3,4],2)
#df['Size'] = df['Size'].replace([5,6,7],3)
#df['Size'] = df['Size'].replace([8,10,11], 4)
df['IsSolo'] = 0
df['IsSolo'][df['FamilySize'] == 1] = 1
df['SmallGroup'] = 0
df['SmallGroup'][df['FamilySize'].isin([2,3,4])] = 1


# In[ ]:


df.head()


# In[ ]:


df['Title'] = df['Name'].str.replace('(.*, )|(\..*)',"")


# In[ ]:


df['Title'] = df['Title'].replace(['Don','Mme','Lady','Mlle','Dona','Miss','Ms'],'lady')
df['Title'] = df['Title'].replace(['Rev','Dr','Major','Sir','Col','Capt','Capt','the Countess','Jonkheer'],'other')
df['Title'].unique().tolist()


# In[ ]:


# fillna of age
for tit in df['Title'].unique().tolist():
    df['Age'][df['Title'] == tit] = df['Age'][df['Title'] == tit].fillna(df['Age'][df['Title'] == tit].mean())


# In[ ]:


sns.set_style("whitegrid")
fig = plt.figure(figsize = (12,6))

ax1 = plt.subplot2grid((1,2),(0,0))
sns.boxplot(x = 'Title',y='Age',data = df)
plt.title('Age boxplot by title')

ax1 = plt.subplot2grid((1,2),(0,1))
plt.hist(df['Age'])
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('frequency')


# In[ ]:


#df['Title'][(df['Title'] == 'lady') & (df['Age'] < 10)] = 'Master'
#df['Title'][(df['Title'] == 'Mr') & (df['Age'] < 15)] = 'Master'
#mapping = {'other':0,'Mr':1,'lady':2,'Master':3}
#df['Title'] = df['Title'].map(mapping)


# In[ ]:


df.Sex = df.Sex.replace('male','0')
df.Sex = df.Sex.replace('female','1')
df.Sex = df.Sex.astype(int)


# In[ ]:


#df['Pclass'] = df['Pclass'].astype(object)


# In[ ]:


df['Fare'].plot(kind = 'hist')
plt.title('Fare histogram')
plt.xlabel('Fare')


# In[ ]:


tickets = set(df['Ticket'])
ticket_numbers = []
fares = []
for ticket in tickets:
    ticket_numbers.append(len(df[df['Ticket'] == ticket]))
    fares.append(df['Fare'][df['Ticket'] == ticket].mean())


# In[ ]:


tickets_sum = pd.DataFrame()
tickets_sum['Ticket'] = list(tickets)
tickets_sum['Count'] = ticket_numbers
tickets_sum['Fare'] = fares
tickets_sum['FarePP'] = tickets_sum['Fare']/tickets_sum['Count']
tickets_sum = tickets_sum.drop(['Count','Fare'], axis = 1)

# concat to df
df = pd.merge(df, tickets_sum,'left','Ticket')


# In[ ]:


tickets_sum.head()


# In[ ]:


df['FarePP'] = df['FarePP'].fillna(df['FarePP'].median())


# # **Feature selection**

# In[ ]:


df.info()


# In[ ]:


df_2 = df.loc[:,['Pclass','Age',
                 #'Cabin',
                 'Embarked',
                 #'FamilySize',
                 'IsSolo','SmallGroup',
                 'Title','FarePP']]

df_2 = pd.get_dummies(df_2)
train_x = df_2.iloc[:891,:]
test_x = df_2.iloc[891:len(df)+1,:]

#train_x.to_csv('train_x.csv', index = False)
#test_x.to_csv('test_x.csv', index = False)

print('train: ',train_x.shape)
print('test: ',test_x.shape)


# In[ ]:


train_y = train['Survived']


# # **Modeling**

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# ## **Average bagging**

# **Architecture:**
# 
# * For each i in range(10):
#     
#     * seed = i
#     
#     * GridSearchCV for RandomForest()

# In[ ]:


submission.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# split to training data to 2 parts: training and validation
x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size = 0.2,random_state=123)

models = []
bag = 10
y_test_bags = np.zeros((test_x.shape[0]))

for i in range(bag):
    params = {'n_estimators':[100,300],
             'max_depth': [2,3,4,5],
             'max_features':[3,5,7]}
    rf = RandomForestClassifier(random_state = i)
    grid = GridSearchCV(rf, params,cv=5)
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    models.append(model)
    
    y_hat = model.predict(x_valid)
    y_hat_test = model.predict_proba(test_x)[:,1]
    y_test_bags += y_hat_test
    
    score = accuracy_score(y_valid, y_hat)
    print('Seed: {} - AUC: {}'.format(i,score))

final_test_y = y_test_bags/bag


# In[ ]:


submission['Survived'] = np.where(final_test_y > 0.5,1,0)
submission.to_csv('submission.csv',index=False)


# ## **KFold**

# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

param_grid = [{'max_depth':[3,4,5],
              'min_child_weight':[1,2,3],
              'n_estimators':[100,200],
              'learning_rate':[0.1,0.01]}] #set of trial values for min_child_weight
i=1
kf = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)

# for each fold in kfold:
for train_index,test_index in kf.split(train_x, train_y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
     
    # split to train and test
    x_train, x_valid = train_x.loc[train_index],train_x.loc[test_index]
    y_train, y_valid = train_y[train_index],train_y[test_index]
    
    # GridSearch
    model = GridSearchCV(XGBClassifier(), param_grid, cv=10, scoring= 'accuracy')
    model.fit(x_train, y_train)
    print(model.best_params_)
    y_hat = model.predict(x_valid)
    print('accuracy_score',accuracy_score(y_valid,y_hat))
    i+=1


# In[ ]:


op=pd.DataFrame(data={'PassengerId':test['PassengerId'],'Survived':model.predict(test_x)})
op.to_csv('KFold_XGB_GridSearchCV_submission.csv',index=False)


# ## **Stacking**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[ ]:


#ExtraTrees 
ExtC = ExtraTreesClassifier()

## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

# define kfold cv
kfold = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)

gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(train_x, train_y)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_


# In[ ]:


# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(train_x, train_y)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_


# In[ ]:


submission['Survived'] = gsRFC.predict(test_x)
submission.to_csv("submission_rf.csv",index=False)


# In[ ]:


# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(train_x, train_y)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_


# In[ ]:


submission['Survived'] = gsGBC.predict(test_x)
submission.to_csv("submission_GBC.csv",index=False)


# In[ ]:


### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(train_x, train_y)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_


# **Plot learning curve**

# In[ ]:


from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",train_x, train_y,cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",train_x, train_y,cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",train_x, train_y,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",train_x, train_y,cv=kfold)


# **Ensemble by Voting**

# In[ ]:


votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(train_x, train_y)


# In[ ]:


submission['Survived'] = votingC.predict(test_x)
submission.to_csv("ensemble_voting.csv",index=False)

