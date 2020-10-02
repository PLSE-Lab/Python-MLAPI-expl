#!/usr/bin/env python
# coding: utf-8

# In this kernel I am going to show some simple ideas about extracting important features from data, imputing missing values and choosing suitable ML models, using Titanic dataset.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn.model_selection
import math
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("/kaggle/input/titanic"))

# Any results you write to the current directory are saved as output.


# Reading the input data:

# In[ ]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
y = train_data['Survived']
test_id = test_data['PassengerId']
train_id = train_data['PassengerId']


# Some insights into training and test set:

# In[ ]:


train_data.head(10)


# In[ ]:


train_data.info()


# In[ ]:


test_data.head(10)


# In[ ]:


test_data.info()


# Quite a lot of data about the cabin is missing:

# In[ ]:


train_data.Cabin.unique()


# The letter the cabin starts with might matter, however a lot of them are missing, and imputing them with the same value does not make much sense. Hence we can safely drop the cabin (and ticket) column: 

# In[ ]:


train_data.drop(['Cabin', 'Ticket'], axis = 1, inplace = True)
test_data.drop(['Cabin', 'Ticket'], axis = 1, inplace = True)


# We should also drop passenger ID:

# In[ ]:


test_data = test_data.drop(['PassengerId'], axis = 1)
train_data = train_data.drop(['PassengerId'], axis = 1)


# In[ ]:


train_data.info()


# A useful feature we might extract is the title:

# In[ ]:


train_data['Title'] = train_data['Name'].str.extract("([A-za-a]+)\.", expand = False) 
test_data['Title'] = test_data['Name'].str.extract("([A-za-a]+)\.", expand = False)


# In[ ]:


train_data.Title.unique()


# In[ ]:


test_data.Title.unique()


# Now we can drop the name column:

# In[ ]:


train_data.drop(['Name'], axis = 1, inplace = True)
test_data.drop(['Name'], axis = 1, inplace = True)


# Now we can explore how many of each title there are:

# In[ ]:


train_data.Title.value_counts()


# In[ ]:


test_data.Title.value_counts()


# Some titles, like Col, Capt, etc. are minority, hence can be classified as "Rare". We make corresponding replacements:

# In[ ]:


train_data['Title'] = train_data['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev', 
                                                   'Countess', 'Don', 'Dona', 'Jonkheer', 
                                                  'Lady',  'Sir'], 'Rare')

test_data['Title'] = test_data['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev', 
                                                 'Countess', 'Don', 'Dona', 'Jonkheer', 
                                                  'Lady',  'Sir'], 'Rare')

train_data['Title'] = train_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs') 

test_data['Title'] = test_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')


# Now we need to fill in age and Embarked in train data. Embarked can just be filled with most frequently occuring value:

# In[ ]:


mfembarked = train_data['Embarked'].value_counts().idxmax()
train_data['Embarked'] = train_data['Embarked'].fillna(mfembarked)
print(mfembarked)


# In[ ]:


mfembarked = test_data['Embarked'].value_counts().idxmax()
print(mfembarked)


# In both training and test data S is the most frequent value for embarked, so it can be imputed in both sets:

# In[ ]:


test_data['Embarked'] = test_data['Embarked'].fillna(mfembarked)


# Other useful features might be family size and whether a person is travelling alone.

# In[ ]:


train_data['FSize'] = 1 + train_data['SibSp'] + train_data['Parch']
test_data['FSize'] = 1 + test_data['SibSp'] + test_data['Parch']

train_data['IsAlone'] = train_data['FSize'].apply(lambda x: 1 if x==1 else 0)
test_data['IsAlone'] = test_data['FSize'].apply(lambda x: 1 if x==1 else 0)


# In[ ]:


ga = sns.heatmap(train_data.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# Now we need to fill in the age, which might be a bit tricky. Just filling with median age may not always be correct. For example, someone who has title "Master", can not be assigned the median age of 28 (which is the median age of the training set). That is why it makes sense to replace the missing age with median grouped by title and class:

# In[ ]:


train_data[np.isnan(train_data['Age'])]


# From the heatmap, it is clear that age correlates with Pclass more than with any other numerical feature. Moreover, age should be filled in according to Title, as well as to Pclass (e.g. someone with a title Master can not be 27/28 y.o. which is the median age)

# In[ ]:


all_data = pd.concat([train_data, test_data])

grouped = all_data.iloc[:len(all_data)].groupby(['Pclass', 'Title'])
grouped_median = grouped.median()
grouped_median = grouped_median.reset_index()[['Pclass','Title', 'Age']]

#print(grouped_median)

def fill_age(row):
    condition = (
        (grouped_median['Pclass'] == row['Pclass']) & 
        (grouped_median['Title'] == row['Title'])
    ) 
    return grouped_median[condition]['Age'].values[0]

train_data['Age'] = train_data.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
test_data['Age'] = test_data.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)


# We only need to fill in the fare for test data, which we can easily do by replacing with median value grouped by class, since fare correlates with class more than with anything else (see heatmap):

# In[ ]:


grouped_fare = all_data.groupby(['Pclass'])
grouped_median_fare = grouped_fare.median()
grouped_median_fare = grouped_median_fare.reset_index()[['Pclass', 'Fare']]

def fill_fare_test(row):
    condition = (
        (grouped_median_fare['Pclass'] == row['Pclass'])
    ) 
    return grouped_median_fare[condition]['Fare'].values[0]

test_data['Fare'] = test_data.apply(lambda row: fill_fare_test(row) if np.isnan(row['Fare']) else row['Fare'], axis=1)


# Now we have all the data imputed. The next important step would be to actually select the features to train the model. First, we convert some categorical features to numerical, or, more precisely, logical (we can not just assign some number to title like 1,2,3,4 since it might affect the accuracy of classification algorithm).

# In[ ]:


train_data['Mr'] = train_data['Title'].map(lambda x : 1 if x == 'Mr' else 0)
train_data['Miss'] = train_data['Title'].map(lambda x : 1 if x == 'Miss' else 0)
train_data['Mrs'] = train_data['Title'].map(lambda x : 1 if x == 'Mrs' else 0)
train_data['Master'] = train_data['Title'].map(lambda x : 1 if x == 'Master' else 0)
train_data['Rare'] = train_data['Title'].map(lambda x : 1 if x == 'Rare' else 0)

train_data['S'] = train_data['Embarked'].map(lambda x : 1 if x == 'S' else 0)
train_data['C'] = train_data['Embarked'].map(lambda x : 1 if x == 'C' else 0)
train_data['Q'] = train_data['Embarked'].map(lambda x : 1 if x == 'Q' else 0)

test_data['Mr'] = test_data['Title'].map(lambda x : 1 if x == 'Mr' else 0)
test_data['Miss'] = test_data['Title'].map(lambda x : 1 if x == 'Miss' else 0)
test_data['Mrs'] = test_data['Title'].map(lambda x : 1 if x == 'Mrs' else 0)
test_data['Master'] = test_data['Title'].map(lambda x : 1 if x == 'Master' else 0)
test_data['Rare'] = test_data['Title'].map(lambda x : 1 if x == 'Rare' else 0)

test_data['S'] = test_data['Embarked'].map(lambda x : 1 if x == 'S' else 0)
test_data['C'] = test_data['Embarked'].map(lambda x : 1 if x == 'C' else 0)
test_data['Q'] = test_data['Embarked'].map(lambda x : 1 if x == 'Q' else 0)


# Another thing we can do is to bin continuous numerical data using qcut, i.e. splitting into buckets of equal size. Then each Fare, Age, etc. value is assigned the corresponding bin:

# In[ ]:


pd.qcut(train_data['Age'], 6, duplicates = 'drop')


# Age distribution:

# In[ ]:


sns.distplot(train_data['Age'])


# Binning the Fare:

# In[ ]:


pd.qcut(train_data['Fare'], 4)


# Fare distribution:

# In[ ]:


sns.distplot(train_data['Fare'])


# After that, we add extra columns, which correspond to Fare bin, Age bin and Family Size category (i.e. single, small family or large family):

# In[ ]:


train_data['F1'] = train_data['Fare'].map(lambda x: 1 if x <= 7.91 else 0)
train_data['F2'] = train_data['Fare'].map(lambda x: 1 if (x > 7.91 and x <= 14.454) else 0)
train_data['F3'] = train_data['Fare'].map(lambda x: 1 if (x > 14.454 and x <= 31.0) else 0)
train_data['F4'] = train_data['Fare'].map(lambda x: 1 if (x > 31.0) else 0)

train_data['FSize'] = 1 + train_data['SibSp'] + train_data['Parch']
train_data['Single'] = train_data['FSize'].map(lambda x : 1 if x == 1 else 0)
train_data['SmallF'] = train_data['FSize'].map(lambda x : 1 if (x > 1 and x < 5) else 0)
train_data['LargeF'] = train_data['FSize'].map(lambda x : 1 if (x > 4) else 0)

train_data['Male'] = train_data['Sex'].map(lambda x: 1 if x == 'male' else 0)
train_data['Female'] = train_data['Sex'].map(lambda x: 1 if x == 'female' else 0)

train_data['Class1'] = train_data['Pclass'].map(lambda x: 1 if x == 1 else 0)
train_data['Class2'] = train_data['Pclass'].map(lambda x: 1 if x == 2 else 0)
train_data['Class3'] = train_data['Pclass'].map(lambda x: 1 if x == 3 else 0)

train_data['A1'] = train_data['Age'].map(lambda x : 1 if x <= 18 else 0) 
train_data['A2'] = train_data['Age'].map(lambda x : 1 if (x <= 24 and x > 18) else 0)
train_data['A3'] = train_data['Age'].map(lambda x : 1 if (x <= 26 and x > 24) else 0)
train_data['A4'] = train_data['Age'].map(lambda x : 1 if (x <= 32.167 and x > 26) else 0)
train_data['A5'] = train_data['Age'].map(lambda x : 1 if (x <= 42 and x > 32.167) else 0)
train_data['A6'] = train_data['Age'].map(lambda x : 1 if x > 42 else 0)

test_data['F1'] = test_data['Fare'].map(lambda x: 1 if x <= 7.91 else 0)
test_data['F2'] = test_data['Fare'].map(lambda x: 1 if (x > 7.91 and x <= 14.454) else 0)
test_data['F3'] = test_data['Fare'].map(lambda x: 1 if (x > 14.454 and x <= 31.0) else 0)
test_data['F4'] = test_data['Fare'].map(lambda x: 1 if (x > 31.0) else 0)

test_data['FSize'] = 1 + test_data['SibSp'] + test_data['Parch']
test_data['Single'] = test_data['FSize'].map(lambda x : 1 if x == 1 else 0)
test_data['SmallF'] = test_data['FSize'].map(lambda x : 1 if (x > 1 and x < 5) else 0)
test_data['LargeF'] = test_data['FSize'].map(lambda x : 1 if (x > 4) else 0)

test_data['Male'] = test_data['Sex'].map(lambda x: 1 if x == 'male' else 0)
test_data['Female'] = test_data['Sex'].map(lambda x: 1 if x == 'female' else 0)

test_data['Class1'] = test_data['Pclass'].map(lambda x: 1 if x == 1 else 0)
test_data['Class2'] = test_data['Pclass'].map(lambda x: 1 if x == 2 else 0)
test_data['Class3'] = test_data['Pclass'].map(lambda x: 1 if x == 3 else 0)

test_data['A1'] = test_data['Age'].map(lambda x : 1 if x <= 18 else 0) 
test_data['A2'] = test_data['Age'].map(lambda x : 1 if (x <= 24 and x > 18) else 0)
test_data['A3'] = test_data['Age'].map(lambda x : 1 if (x <= 26 and x > 24) else 0)
test_data['A4'] = test_data['Age'].map(lambda x : 1 if (x <= 32.167 and x > 26) else 0)
test_data['A5'] = test_data['Age'].map(lambda x : 1 if (x <= 42 and x > 32.167) else 0)
test_data['A6'] = test_data['Age'].map(lambda x : 1 if x > 42 else 0)


# Exploratory data analysis:

# In[ ]:


sns.barplot(x = 'Sex', y = 'Survived',  data=train_data)


# In[ ]:


sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=train_data)


# In[ ]:


sns.barplot(x = 'FSize', y = 'Survived', data = train_data)


# Survival based on the fare and age bin:

# In[ ]:


def farebin(row):
    if row['F1'] == 1:
        x = 0
    elif row['F2'] == 1:
        x = 1;
    elif row['F3'] == 1:
        x = 2;
    elif row['F4'] == 1:
        x = 3;
    else:
        x = 4
        
    return x

train_data['Fare_bin'] = train_data.apply(lambda row: farebin(row), axis = 1)
test_data['Fare_bin'] = test_data.apply(lambda row: farebin(row), axis = 1)


# In[ ]:


sns.barplot(x = 'Fare_bin', y = 'Survived', data = train_data)


# Very clear trend of survival vs fare bin (with 5 bins for fare it does not work so well).

# In[ ]:


sns.barplot(x = 'SibSp', y = 'Survived', data = train_data)


# In[ ]:


sns.barplot(x = 'Parch', y = 'Survived', data = train_data)


# In[ ]:


def agebin(row):
    if row['Age'] <= 18:
        x = 0
    elif row['Age'] <= 24:
        x = 1
    elif row['Age'] <= 26:
        x = 2
    elif row['Age'] <= 32.167:
        x = 3
    elif row['Age'] <= 42:
        x = 4
    else:
        x = 5
        
    return x

train_data['Age_bin'] = train_data.apply(lambda row: agebin(row), axis = 1)
test_data['Age_bin'] = test_data.apply(lambda row: agebin(row), axis = 1)

sns.barplot(x = 'Age_bin', y = 'Survived', data = train_data)


# Clearly, the chances of survival for people age 24-26 are the lowest. This fact might not be captured if we use less age bins, so it might even be reasonable to use age as a continuous variable (i.e. without bins). We might also want to explore survival based on family size, but we encode it as if it was categorical variable:

# In[ ]:


def FamSize(row):
    if row['FSize'] == 1:
        x = '1'
    elif row['FSize'] == 2:
        x = '2'
    elif row['FSize'] == 3:
        x = '3'
    elif row['FSize'] == 4:
        x = '4'
    elif row['FSize'] == 5:
        x = '5'
    else:
        x = '>=6'
    return x

train_data['FamSize'] = train_data.apply(lambda row: FamSize(row), axis = 1)
test_data['FamSize'] = test_data.apply(lambda row: FamSize(row), axis = 1)

sns.barplot(x = 'FamSize', y = 'Survived', data = train_data)


# Survival depending on whether a person travels on their own, with a small family or with a large family:

# In[ ]:


def fbin(row):
    if row['Single'] == 1:
        x = 0
    elif row['SmallF'] == 1:
        x = 1;
    else:
        x = 2
        
    return x

train_data['F_bin'] = train_data.apply(lambda row: fbin(row), axis = 1)
sns.barplot(x = 'F_bin', y = 'Survived', data = train_data)


# Survival dependence on port of departure:

# In[ ]:


sns.barplot(x = 'Embarked', y = 'Survived', data = train_data)


# Survival dependence on title:

# In[ ]:


sns.barplot(x = 'Title', y = 'Survived', data = train_data)


# An interesting thing to look at would be the dependence of survival on whether a person's title is Rare or not:

# In[ ]:


def Title_cat(row):
    if row['Mr'] == 1 or row['Mrs'] == 1 or row['Miss'] == 1:
        x = 'Normal'
    elif row['Master'] == 1:
        x = 'Child';
    else:
        x = 'Powerful';
    return x
    
train_data['Title_cat'] = train_data.apply(lambda row: Title_cat(row), axis = 1)
test_data['Title_cat'] = test_data.apply(lambda row: Title_cat(row), axis = 1)

sns.barplot(x = 'Title_cat', y = 'Survived', data = train_data[train_data['Male'] == 1])


# Clearly, someone with a 'Rare' title has a greater chance of survival than someone called 'Mr' but lower chance than children (for males). For females, someone with a rare title has a higher chance of survival as well. This might be due to the fact that someone with Rare title belongs to upper class, buys expensive ticket or other people simply care about them more (as a consequence of the first two :( )

# In[ ]:


sns.barplot(x = 'Title_cat', y = 'Survived', data = train_data[train_data['Female'] == 1])


# We can now see some trends, e.g. medium family size is more likely to survive or more expensive fare increases chances of survival. Now we can start making models.

# In[ ]:


train_data.info()


# In[ ]:


train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)


# We can now try various combinations of features for our model. For example, I try the combination below, but it is totally possible to use any other combination of features. In fact I believe it is possible to achieve better results (on public leaderboard) using some other combination of features/hyperparameters, so everything I show below is just an example.

# In[ ]:


feats = ['Age',
          'Class1', 'Class2', 'Class3', 
          'Male', 'Female', 'Fare', 
          'Single', 'SmallF', 'LargeF', 'SibSp', 'Parch',
         'S', 'C', 'Q', 'Mr', 'Mrs', 'Miss', 'Master', 'Rare']


# In[ ]:


train = train_data[feats]
test = test_data[feats]


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# Let us start with something simple, like Random forest:

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state = 3)
print(cross_val_score(rf, train, y, scoring = 'accuracy', cv = 5).mean())


# Accuracy is already quite high, and can be improved using grid search. We also use RobustScaler in order to avoid errors due to different scales for different variables.

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler

pipeline_rf = Pipeline(
                    [ 
                     ('sca', RobustScaler()),
                     ('rf', RandomForestClassifier(random_state = 3))
                     
])

parameters = {}
parameters['rf__min_samples_split'] = [2,3,4,5] 
parameters['rf__max_depth'] = [4, 6, 8, None] 
parameters['rf__n_estimators'] = [10, 25, 50, 100] 

#CV = GridSearchCV(pipeline_rf, parameters, scoring = 'accuracy', n_jobs= 4, cv = 5)
#CV.fit(train, y)   

#print('Best score and parameter combination = ')

#print(CV.best_score_)    
#print(CV.best_params_) 

#Best score and parameter combination = 
#0.8338945005611672
#{'rf__max_depth': 4, 'rf__min_samples_split': 2, 'rf__n_estimators': 100}


# In[ ]:


pipeline_rf = Pipeline(
                    [('sca', RobustScaler()),
                     ('rf', RandomForestClassifier(random_state = 3, max_depth = 4, min_samples_split = 2, n_estimators = 100))
                     
])


# Trying with XGBoost:

# In[ ]:


import xgboost as xgb

pipeline_xgb = Pipeline(
                    [ 
                     ('sca', RobustScaler()),
                     ('xgb', xgb.XGBClassifier(random_state = 3))
                     
])

parameters = {}
parameters['xgb__min_child_weight'] = [0, 0.5, 1.0] 
parameters['xgb__max_depth'] = [4,6,8] 
parameters['xgb__eta'] = [0.3, 0.1, 0.05] 
parameters['xgb__subsample'] = [0.5, 0.75, 1.0] 
parameters['xgb__n_estimators'] = [15, 25, 45] 
parameters['xgb__gamma'] = [0, 0.5, 1.0]
parameters['xgb__colsample_by'] = [0.2, 0.6, 1.0]

#CV = GridSearchCV(pipeline_xgb, parameters, scoring = 'accuracy', n_jobs= 4, cv = 5)
#CV.fit(train, y)   

#print('Best score and parameter combination = ')

#print(CV.best_score_)    
#print(CV.best_params_) 

#Best score and parameter combination = 
#0.8462401795735129
#{'xgb__colsample_by': 0.2, 'xgb__eta': 0.3, 'xgb__gamma': 1.0, 'xgb__max_depth': 6, 'xgb__min_child_weight': 0, 'xgb__n_estimators': 15, 'xgb__subsample': 0.75, 'xgb__tree_method': 'auto'}


# In[ ]:


pipeline_xgb = Pipeline(
                    [('sca', RobustScaler()),
                     ('xgb', xgb.XGBClassifier(random_state = 3, max_depth = 6, gamma = 1.0, min_child_weight = 0, subsample = 0.75, n_estimators = 15, eta = 0.3, colsample_by = 0.2, tree_method = 'auto'))
                     
])


# Now we can try SVC:

# In[ ]:


from sklearn.svm import SVC

pipeline_svc = Pipeline(
                    [
                     ('sca', RobustScaler()),
                     ('svc', SVC(random_state = 3))
                     
])

parameters = {}
parameters['svc__kernel'] = ['rbf', 'poly', 'sigmoid', 'linear'] 
parameters['svc__C'] = [0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] 

#CV = GridSearchCV(pipeline_svc, parameters, scoring = 'accuracy', n_jobs= 4, cv = 5)
#CV.fit(train, y)   

#print('Best score and parameter combination = ')

#print(CV.best_score_)    
#print(CV.best_params_) 

#Best score and parameter combination = 
#0.8338945005611672
#{'svc__C': 3.0, 'svc__kernel': 'poly'}


# In[ ]:


pipeline_svc = Pipeline(
                    [('sca', RobustScaler()),
                     ('svc', SVC(random_state = 3, kernel = 'poly', C = 3.0))
                     
])


# Logistic Regression:

# In[ ]:


from sklearn.linear_model import LogisticRegression

pipeline_lr = Pipeline(
                    [
                     ('sca', RobustScaler()),
                     ('lr', LogisticRegression(random_state = 3))
                     
])

parameters = {}
parameters['lr__l1_ratio'] = [0, 0.5, 1.0]  

#CV = GridSearchCV(pipeline_lr, parameters, scoring = 'accuracy', n_jobs= 4, cv = 5)
#CV.fit(train, y)   

#print('Best score and parameter combination = ')

#print(CV.best_score_)    
#print(CV.best_params_) 

#Best score and parameter combination = 
#0.8271604938271605
#{'lr__l1_ratio': 0}


# In[ ]:


pipeline_lr = Pipeline(
                    [('sca', RobustScaler()),
                     ('lr', LogisticRegression(random_state = 3, l1_ratio = 0))
                     
])


# AdaBoost:

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

pipeline_ada = Pipeline(
                    [
                     ('sca', RobustScaler()),
                     ('ada', AdaBoostClassifier(random_state = 3))
                     
])

parameters = {}
parameters['ada__learning_rate'] = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]  
parameters['ada__n_estimators'] = [5, 10, 15, 20, 25, 35, 45, 60, 80, 100, 120] 
parameters['ada__base_estimator'] = [DecisionTreeClassifier(max_depth = 1), DecisionTreeClassifier(max_depth = 3), 
                                    DecisionTreeClassifier(max_depth = 5), DecisionTreeClassifier(max_depth = None)]

#CV = GridSearchCV(pipeline_ada, parameters, scoring = 'accuracy', n_jobs= 4, cv = 5)
#CV.fit(train, y)   

#print('Best score and parameter combination = ')

#print(CV.best_score_)    
#print(CV.best_params_)

#Best score and parameter combination = 
#0.8305274971941639
#{'ada__base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
#                       max_features=None, max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, presort=False,
#                       random_state=None, splitter='best'), 'ada__learning_rate': 0.05, 'ada__n_estimators': 15}


# In[ ]:


pipeline_ada = Pipeline(
                    [('sca', RobustScaler()),
                     ('ada', AdaBoostClassifier(random_state = 3, n_estimators = 15, learning_rate = 0.05, base_estimator = DecisionTreeClassifier(max_depth = 3)))
                     
])


# Gradient Boosting Classifier:

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

pipeline_gb = Pipeline(
                    [
                     ('sca', RobustScaler()),
                     ('gb', GradientBoostingClassifier(random_state = 3))
                     
])

parameters = {}
parameters['gb__learning_rate'] = [0.05, 0.1, 0.15, 0.2]  
parameters['gb__n_estimators'] = [30, 45, 60, 80, 100, 120] 
parameters['gb__max_depth'] = [1,2,3,4,5,6] 
parameters['gb__subsample'] = [0.2, 0.6, 1.0]
parameters['gb__min_samples_split'] = [2,4,6]
parameters['gb__min_samples_leaf'] = [1,2,3]

#CV = GridSearchCV(pipeline_gb, parameters, scoring = 'accuracy', n_jobs= 4, cv = 5)
#CV.fit(train, y)   

#print('Best score and parameter combination = ')

#print(CV.best_score_)    
#print(CV.best_params_)

#Best score and parameter combination = 
#0.8451178451178452
#{'gb__learning_rate': 0.1, 'gb__max_depth': 4, 'gb__min_samples_leaf': 1, 'gb__min_samples_split': 6, 'gb__n_estimators': 30, 'gb__subsample': 0.6}


# In[ ]:


pipeline_gb = Pipeline(
                    [('sca', RobustScaler()),
                     ('gb', GradientBoostingClassifier(random_state = 3, learning_rate = 0.1, max_depth = 4, n_estimators = 30, min_samples_leaf = 1, min_samples_split = 6, subsample = 0.6))
                     
])


# Ridge Classifier:

# In[ ]:


from sklearn.linear_model import RidgeClassifier

pipeline_rc = Pipeline([('sca', RobustScaler()), ('rc', RidgeClassifier(random_state = 1))])

cross_val_score(pipeline_rc, train, y, scoring = 'accuracy', cv = 5).mean()


# Naive Bayes:

# In[ ]:


from sklearn.naive_bayes import GaussianNB

pipeline_nb = Pipeline([('sca', RobustScaler()), ('nb', GaussianNB())])

cross_val_score(pipeline_nb, train, y, scoring = 'accuracy', cv = 5).mean()


# LightGBM:

# In[ ]:


import lightgbm as lgb

pipeline_lgb = Pipeline(
                    [
                     ('sca', RobustScaler()),
                     ('lgb', lgb.LGBMClassifier(random_state = 3))
                     
])

parameters = {}
parameters['lgb__n_estimators'] = [10, 25, 50, 100, 250] 
parameters['lgb__learning_rate'] = [0.25, 0.1, 0.05, 0.01]
parameters['lgb__subsample'] = [0.2, 0.6, 1.0]
parameters['lgb__colsample_bytree'] = [0.2, 0.6, 1.0]
parameters['lgb__min_child_samples'] = [10, 20, 30]

#CV = GridSearchCV(pipeline_lgb, parameters, scoring = 'accuracy', n_jobs= 4, cv = 5)
#CV.fit(train, y)   

#print('Best score and parameter combination = ')

#print(CV.best_score_)    
#print(CV.best_params_)

#Best score and parameter combination = 
#0.8428731762065096
#{'lgb__colsample_bytree': 1.0, 'lgb__learning_rate': 0.1, 'lgb__min_child_samples': 30, 'lgb__n_estimators': 100, 'lgb__subsample': 0.2}


# In[ ]:


pipeline_lgb = Pipeline(
                    [('sca', RobustScaler()),
                     ('lgb', lgb.LGBMClassifier(random_state = 3, learning_rate = 0.1, subsample = 0.2, n_estimators = 100, min_child_samples = 30, colsample_bytree = 1.0))
                     
])


# K-nearest neighbors classifier:

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

pipeline_knn = Pipeline(
                    [
                     ('sca', RobustScaler()),
                     ('knn', KNeighborsClassifier())
                     
])

parameters = {}
parameters['knn__n_neighbors'] = [2,3,4,5,6,7,8,9,10]
parameters['knn__algorithm'] = ['auto', 'ball_tree', 'kd_tree', 'brute']
parameters['knn__weights'] = ['uniform', 'distance']

#CV = GridSearchCV(pipeline_knn, parameters, scoring = 'accuracy', n_jobs= 4, cv = 5)
#CV.fit(train, y)   

#print('Best score and parameter combination = ')

#print(CV.best_score_)    
#print(CV.best_params_)

#Best score and parameter combination = 
#0.8125701459034792
#{'knn__algorithm': 'ball_tree', 'knn__n_neighbors': 5, 'knn__weights': 'uniform'}


# Some base models have really good scores (XGBClassifier being the best so far). We can try is stacking models: in this example we do it with KNN, SVM and Naive Bayes, using Random Forest as meta-model.

# In[ ]:


from mlxtend.classifier import StackingCVClassifier

sc = Pipeline( [ ('sca', RobustScaler()), ('sc', StackingCVClassifier(classifiers = [KNeighborsClassifier(),
                                                                                     SVC(random_state = 1),
                                                                                     GradientBoostingClassifier(random_state = 3)],
                                                                      meta_classifier = RandomForestClassifier(random_state = 2))) ])

parameters = {}


parameters['sc__meta_classifier__min_samples_split'] = [2] 
parameters['sc__meta_classifier__max_depth'] = [8] 
parameters['sc__meta_classifier__n_estimators'] = [15] 

parameters['sc__kneighborsclassifier__n_neighbors'] = [5]

parameters['sc__svc__kernel'] = ['rbf', 'poly', 'sigmoid', 'linear'] 
parameters['sc__svc__C'] = [0.1, 0.5, 1.0, 2.0, 3.0] 

parameters['sc__gradientboostingclassifier__subsample'] = [0.5, 0.75, 1.0] 
parameters['sc__gradientboostingclassifier__max_depth'] = [4, 6, 8]

#CV = GridSearchCV(sc, parameters, scoring = 'accuracy', n_jobs= 4, cv = 5, verbose = 10)
#CV.fit(train, y)   

#print('Best score and parameter combination = ')

#print(CV.best_score_)    
#print(CV.best_params_)

#Best score and parameter combination = 
#0.8406285072951739
#{'sc__gradientboostingclassifier__max_depth': 4, 'sc__gradientboostingclassifier__subsample': 0.5,
#'sc__kneighborsclassifier__n_neighbors': 5, 'sc__meta_classifier__max_depth': 8, 
#'sc__meta_classifier__min_samples_split': 2, 'sc__meta_classifier__n_estimators': 15, 
#'sc__svc__C': 2.0, 'sc__svc__kernel': 'poly'}


# In[ ]:


sc = Pipeline( [ ('sca', RobustScaler()), ('sc', StackingCVClassifier(classifiers = [KNeighborsClassifier(),
                                                                                     SVC(random_state = 1, kernel = 'poly', C = 2.0),
                                                                                     GradientBoostingClassifier(random_state = 3, max_depth = 4, subsample = 0.5)],
                                                                      meta_classifier = RandomForestClassifier(random_state = 2, n_estimators = 15, max_depth = 8))) ])


# In this example, the cross-validation accuracy is actually higher for stacked model than for each of the models separately.

# Overall, there are a few conclusions I would like to say:
# 
# 1) What makes a model robust is primarily the features we select/extract. Even fine tuning or using sophisticated algorithm does not improve the result dramatically, and in this work it was done only for learning purposes. 
# 
# 2) There are many ways to improve performance, such as extracting informations about which group each passenger travels with (e.g. https://www.kaggle.com/cdeotte/titantic-mega-model-0-84210), but this goes beyond the simpler model I am considering. In general, one can experiment trying different features.
# 
# 3) Titanic dataset itself is random to some degree - there are many factors that no model can take into account (such as physical/psychological condition of people at the moment of the catastrophe), so 80% is already a good score.

# Thank you for your time, I hope you found some of the ideas useful! p.s. Please upvote :)
