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
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train.corr()


# In[ ]:


sns.heatmap(train.corr(), annot=True)


# We can see that 'Survived' has high correlation with 'Pclass'(0.34) and 'Fare'(0.26) categories...
# 
# We can concluded that;
# 
# - the more money passengers pay, the higher probability of survival they have
# - the lower class passengers belongs to, the lowest probability of survival they have

# **MISSING VALUES**

# In[ ]:


print(train.isnull().sum())
print(test.isnull().sum())


# We can see that 'Age', 'Cabin', 'Fare' and 'Embarked' features have null values,
# - It is easy to fill Fare and Embarked because there are only 3 null values
# - I think that Age is important feature for predict so we have to fill it logically
# - I am not sure Cabin feature is important or not and there are many null values so we can explore it.

# In[ ]:


# fill 'Fare' and 'Embarked'

print(train['Embarked'].value_counts())
train[train['Embarked'].isnull()]

# We can see that we can fill two null values with 'S'


# In[ ]:


train['Embarked'].fillna('S', inplace=True)
print(train['Embarked'].value_counts())
print(train['Embarked'].isnull().sum())


# In[ ]:


print(test['Fare'].value_counts())
test[test['Fare'].isnull()]

# we can fill with Fare(mean) in Pclass = 3


# In[ ]:


test['Fare'].fillna(test.loc[(test['Pclass'] == 3),'Fare'].mean(), inplace=True)
print([test['Fare'].isnull().sum()])


# # Now we can focus on null values in 'Age' feature

# In[ ]:


train['Age'].describe()


# In[ ]:


train['Age'].value_counts()


# In[ ]:


train.groupby(['Survived','Sex'])[['Age']].agg([np.mean,'std','count'])

# I want to fill Age values randomly, we can see from table that females who survived ages between (mean-std) (mean+std) 
# so we can fill null values randomly between two numbers 


# In[ ]:


# To learn how many null values in these groups I am doing this

train["Age"] = train["Age"].fillna(-0.5)

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train["Age_categories"] = pd.cut(train["Age"],cut_points,labels=label_names)

train.groupby(['Survived','Sex','Age_categories'])[['Age']].agg([np.mean,'count'])


# In[ ]:


rand_age1 = np.random.randint(25.046875 - 13.618591, 25.046875 + 13.618591, size = 17)
rand_age2 = np.random.randint(31.618056 - 14.056019, 31.618056 + 14.056019, size = 108)
rand_age3 = np.random.randint(28.847716 - 14.175073, 28.847716 + 14.175073, size = 36)
rand_age4 = np.random.randint(27.276022 - 16.504803, 27.276022 + 16.504803, size = 16)

rand_age1 = list(rand_age1)
rand_age2 = list(rand_age2)
rand_age3 = list(rand_age3)
rand_age4 = list(rand_age4)


# In[ ]:


train.loc[(train['Sex'] == 'female') & (train['Survived'] == 0) & (train['Age'] == -0.5),'Age'] = rand_age1
train.loc[(train['Sex'] == 'male') & (train['Survived'] == 0) & (train['Age'] == -0.5), 'Age'] = rand_age2
train.loc[(train['Sex'] == 'female') & (train['Survived'] == 1) & (train['Age'] == -0.5),'Age'] = rand_age3
train.loc[(train['Sex'] == 'male') & (train['Survived'] == 1) & (train['Age'] == -0.5), 'Age'] = rand_age4


# In[ ]:


test.groupby(['Sex'])[['Age']].agg([np.mean,'std','count'])


# In[ ]:


test["Age"] = test["Age"].fillna(-0.5)
cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
test["Age_categories"] = pd.cut(test["Age"],cut_points,labels=label_names)
test.groupby(['Sex','Age_categories'])[['Age']].agg([np.mean,'count'])


# In[ ]:


rand_age1 = np.random.randint(30.272362 - 15.428613, 30.272362 + 15.428613, size = 25)
rand_age2 = np.random.randint(30.272732 - 13.389528, 30.272732 + 13.389528, size = 61)

rand_age1 = list(rand_age1)
rand_age2 = list(rand_age2)

test.loc[(test['Sex'] == 'female') & (test['Age'] == -0.5), 'Age'] = rand_age1
test.loc[(test['Sex'] == 'male') & (test['Age'] == -0.5), 'Age'] = rand_age2


# In[ ]:


# now we can drop age_categories

train.drop('Age_categories', axis=1, inplace=True)
test.drop('Age_categories', axis=1, inplace=True)


# ## After filling AGE values we can focus CABIN

# In[ ]:


train["Cabin_type"] = train["Cabin"].str[0]
train["Cabin_type"] = train["Cabin_type"].fillna("Unknown")

test["Cabin_type"] = test["Cabin"].str[0]
test["Cabin_type"] = test["Cabin_type"].fillna("Unknown")


# In[ ]:


sns.barplot(x = 'Cabin_type',y ='Survived',data=train)


# ### After looking plot I am thinking that 'Cabin' feature is not important to predict Survived I want to drop it

# In[ ]:


train.drop(['Cabin','Cabin_type'], axis=1, inplace=True)
test.drop(['Cabin','Cabin_type'], axis=1, inplace=True)


# In[ ]:


# finally end of fill null values

print(train.isnull().sum())
print(test.isnull().sum())


# In[ ]:


train.head()


# ## Now we can check the other columns

# In[ ]:


# I think Ticket feature is not necessary for us so we can drop them.

train.drop('Ticket', axis=1, inplace=True)
test.drop('Ticket', axis=1, inplace=True)


# We can create new fature from NAME

# In[ ]:


train["Title"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
test["Title"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train['Title'] = train['Title'].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')


# In[ ]:


test['Title'] = test['Title'].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')


# In[ ]:


# now we can drop NAME

train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[ ]:


train.head()


# we can creat a new feature about family size because SibSp and Parch feature are about same meaning so if we merge them we can creat family size feature

# In[ ]:


train['Family_Size'] = train['SibSp'] + train['Parch'] + 1
test['Family_Size'] = test['SibSp'] + test['Parch'] + 1


# In[ ]:


# Now we dont need SibSp and Parch and drop them

train.drop(['SibSp','Parch'], axis=1, inplace=True)
test.drop(['SibSp','Parch'], axis=1, inplace=True)


# In[ ]:


train.head()


# ## Now we have to focus categorical features because for ML algorithms we can not use it, Pclass, Sex, Embarked, Title
# 
# ## There are two options for categorical values to numerical, one hot encoding and label encoding, one hot is good for nominal variables and label is good for ordinal variables
# 
# 
# ## All of them is nominal here so we can apply one hot encoding but sex has two categori so we have to apply label encoder for it

# In[ ]:


# We can apply one hot encoding method to ['Pclass','Sex','Embarked','Title'] 

train = pd.get_dummies(train, columns = ['Pclass','Embarked','Title'],prefix=['Pc','',''])
test = pd.get_dummies(test, columns = ['Pclass','Embarked','Title'],prefix=['Pc','',''])


# In[ ]:


from sklearn import preprocessing

lbe = preprocessing.LabelEncoder()
train["Sex"] = lbe.fit_transform(train["Sex"])
test["Sex"] = lbe.fit_transform(test["Sex"])


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# we can see there is no ROYAL column in test data so we have to add 

test['_Royal'] = 0


# In[ ]:


test.head()


# **MACHINE LEARNING**

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier


# ## This dataset is not big so to split train and validation set is not good solution, to aplly cross validation is necessary but to see difference I will do both of them

# ### 1) split data

# In[ ]:


def base_models(df):
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X = df.drop(['Survived', 'PassengerId'], axis=1)
    Y = df["Survived"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size = 0.20, 
                                                    random_state = 42)
    
    #results = []
    
    names = ["LogisticRegression","GaussianNB","KNN","LinearSVC","SVC",
             "CART","RF","GBM","XGBoost","LightGBM","CatBoost"]
    
    
    classifiers = [LogisticRegression(),GaussianNB(), KNeighborsClassifier(),LinearSVC(),SVC(),
                  DecisionTreeClassifier(),RandomForestClassifier(), GradientBoostingClassifier(),
                  XGBClassifier(), LGBMClassifier(), CatBoostClassifier(verbose = False)]
    
    
    for name, clf in zip(names, classifiers):

        model = clf.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        msg = "%s: %f" % (name, acc)
        print(msg)


# In[ ]:


base_models(train)


# ### 2) cross validation

# In[ ]:


def base_models_cv(df):
    
    from sklearn.model_selection import cross_val_score, KFold 
    
    X = df.drop(['Survived', 'PassengerId'], axis=1)
    Y = df["Survived"]
    
    results = []
    A = []
    
    names = ["LogisticRegression","GaussianNB","KNN","LinearSVC","SVC",
             "CART","RF","GBM","XGBoost","LightGBM","CatBoost"]
    
    
    classifiers = [LogisticRegression(),GaussianNB(), KNeighborsClassifier(),LinearSVC(),SVC(),
                  DecisionTreeClassifier(),RandomForestClassifier(), GradientBoostingClassifier(),
                  XGBClassifier(), LGBMClassifier(), CatBoostClassifier(verbose = False)]
    
    
    for name, clf in zip(names, classifiers):
        
        kfold = KFold(n_splits=10, random_state=1001)
        cv_results = cross_val_score(clf, X, Y, cv = kfold, scoring = "accuracy")
        results.append(cv_results)
        A.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean() , cv_results.std())
        print(msg)


# In[ ]:


base_models_cv(train)


# ## Finally I select to use GradientBoostingClassifier()

# In[ ]:


X = train.drop(['Survived', 'PassengerId'], axis=1)
y = train["Survived"]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.20, 
                                                    random_state = 42)


# In[ ]:


models = [LogisticRegression(),
          GaussianNB(), 
          KNeighborsClassifier(),
          SVC(probability=True),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          GradientBoostingClassifier(),
          XGBClassifier(), 
          LGBMClassifier(),
          CatBoostClassifier(verbose = False)]

names = ["LogisticRegression",
         "GaussianNB",
         "KNeighborsClassifier",
         "SVC",
         "DecisionTreeClassifier",
         "RandomForestClassifier",
         "GradientBoostingClassifier",
         "XGBClassifier",
         "LGBMClassifier",
         "CatBoostClassifier"]
    
logreg_params = {"C":np.logspace(-1, 1, 10),
                 "penalty": ["l1","l2"],
                 "solver":['newton-cg','lbfgs','liblinear','sag','saga'],
                 "max_iter":[1000]}
    
NB_params = {'var_smoothing': np.logspace(0,-9, num=100)}

knn_params = {"n_neighbors":np.linspace(1,19,10, dtype = int).tolist(),
                 "weights":["uniform","distance"],
                 "metric":["euclidean","manhattan"]}

svc_params = {"kernel":["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1, 5, 10 ,50 ,100],
                 "C": [1,10,50,100,200,300,1000]}

dtree_params = {"min_samples_split" : range(10,500,20),
                    "max_depth": range(1,20,2)}

rf_params = {"max_features": ["log2","Auto","None"],
                 "min_samples_split":[2,3,5],
                 "min_samples_leaf":[1,3,5],
                 "bootstrap":[True,False],
                 "n_estimators":[50,100,150],
                 "criterion":["gini","entropy"]}

gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],
                  "n_estimators": [100,500,100],
                  "max_depth": [3,5,10],
                  "min_samples_split": [2,5,10]}

xgb_params = {'n_estimators': [50, 100, 200],
                 'subsample': [ 0.6, 0.8, 1.0],
                 'max_depth': [1,2,3,4],
                 'learning_rate': [0.1,0.2, 0.3, 0.4, 0.5],
                 "min_samples_split": [1,2,4,6]}

lgbm_params = {'n_estimators': [100, 500, 1000, 2000],
               'subsample': [0.6, 0.8, 1.0],
               'max_depth': [3, 4, 5,6],
               'learning_rate': [0.1,0.01,0.02,0.05],
               "min_child_samples": [5,10,20]}
    
catb_params = {'depth':[2, 3, 4],
                   'loss_function': ['Logloss', 'CrossEntropy'],
                   'l2_leaf_reg':np.arange(2,31)}

classifier_params = [logreg_params, NB_params, knn_params, svc_params, dtree_params,
                         rf_params, gbm_params, xgb_params, lgbm_params, catb_params]


# In[ ]:


cv_result = {}
best_estimators = {}
best_params = {}
    
for name, model,classifier_param in zip(names, models,classifier_params):
    clf = GridSearchCV(model, param_grid=classifier_param, cv =10, scoring = "accuracy", n_jobs = -1,verbose = False)
    clf.fit(X_train,y_train)
    cv_result[name]=clf.best_score_
    best_estimators[name]=clf.best_estimator_
    best_params[name]=clf.best_params_
    print(name,'cross validation accuracy : %.3f'%cv_result[name])

accuracies={}

print('Validation accuracies of the tuned models for the train data:', end = "\n\n")
    
for name, model_tuned in zip(best_estimators.keys(),best_estimators.values()):
    y_pred =  model_tuned.fit(X_train,y_train).predict(X_test)
    accuracy=accuracy_score(y_pred, y_test)
    print(name,':', "%.3f" %accuracy)
    accuracies[name]=accuracy
sorted_accuracies=sorted(accuracies, reverse=True, key= lambda k:accuracies[k])
    
print(sorted_accuracies)
    
for i in list(best_estimators):
    if i == sorted_accuracies[0]:
        model_tuned = best_estimators[sorted_accuracies[0]]
        
predictions = model_tuned.fit(X_train,y_train).predict(test.drop('PassengerId', axis=1))


# In[ ]:


ids = test['PassengerId']
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions.astype(int)})
output.to_csv('submission.csv', index=False)

