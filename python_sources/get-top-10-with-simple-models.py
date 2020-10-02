#!/usr/bin/env python
# coding: utf-8

# ## 1. Import data & useful libraries
# ## 2. Analyse and prepare data
# ### A. Dataset analyse
# ### B. Outliers
# ### C. Missing data
# ### D. More feature engineering
# ### E. Encoding of categorical features
# ## 3. Applying classification algorithms
# ### A. Validation method
# ### B. Comparing models
# ### C. Create submitting file

# # 1. Import data & useful libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# # 2. Analyse and prepare data 

# ## A. Dataset analyse

# In[ ]:


train.shape


# In[ ]:


test.shape


# Ok, so we have 12 features (including the one to predict). We have data about 891 people in the train set and 418 in the test set.

# In[ ]:


train.head()


# In[ ]:


#Save the 'Id' column
train_ID = train['PassengerId']
test_ID = test['PassengerId']

# Remove 'Id' for analysis
train = train.drop('PassengerId', axis=1)
test = test.drop('PassengerId', axis=1)


# In[ ]:


#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# ## B. Outliers

# In[ ]:


train.describe()


# In[ ]:


train['Sex'].value_counts()


# In[ ]:


train['Embarked'].value_counts()


# It seems that there is no outliers !

# ## C. Missing data

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.Survived.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['Survived'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# In[ ]:


#display top missing data ratio
total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data


# * **Cabin**

# The huge majority of data for this feature is missing and it does not bring a lot of information for our predictions. So, I decide to remove this column.

# In[ ]:


all_data.drop(['Cabin'], axis=1, inplace=True)


# * **Age**

# We could replace the 263 missing values by mean or median. But let's check if we can do something a little bit smarter. On the correlation matrix, we can notice that Pclass and SibSp are strongly correlated to the age. Let's explore this idea.

# In[ ]:


#box plot Pclass/Age
var = 'SibSp'
data = pd.concat([all_data['Age'], all_data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 12))
fig = sns.boxplot(x=var, y="Age", data=data)


# In[ ]:


#box plot Pclass/Age
var = 'Pclass'
data = pd.concat([all_data['Age'], all_data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 12))
fig = sns.boxplot(x=var, y="Age", data=data)


# Waouh ! It seems that from 2 sibling and spouse, the average age is becoming very lower ! It seems normal, you don't usely travel that much with your whole brothers and sisters when you are adult.
# Let's first fill in the blanks by the median of the given SibSp, for SibSp > 1.
# For the others, I fill in the blanks by the median of the given Pclass.

# In[ ]:


all_data['SibSp'].loc[np.isnan(all_data['Age'])].value_counts()


# In[ ]:


all_data["Age"] = all_data.loc[all_data["SibSp"]>1].groupby("SibSp")["Age"].transform(
    lambda x: x.fillna(x.median()))


# In[ ]:


all_data["Age"] = all_data.groupby("Pclass")["Age"].transform(
    lambda x: x.fillna(x.median()))


# * **Embarked**

# In[ ]:


all_data["Embarked"].value_counts()


# I fill in the blank by the most common value

# In[ ]:


all_data["Embarked"] = all_data["Embarked"].fillna("S")


# * **Fare**

# The fare mainly depends on pclass

# In[ ]:


all_data["Fare"] = all_data.groupby("Pclass")["Fare"].transform(
    lambda x: x.fillna(x.median()))


# ## D. More feature engineering

# In[ ]:


all_data["TotalRelatives"] = all_data['SibSp'] + all_data['Parch']

all_data['IsAlone'] = 1 #initialize to yes/1 is alone
all_data['IsAlone'].loc[all_data["TotalRelatives"] > 0] = 0 # now update to no/0 if family size is greater than 1

#quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
all_data['Title'] = all_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


#Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
#Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
all_data['FareBin'] = pd.qcut(all_data['Fare'], 4)

#Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
all_data['AgeBin'] = pd.cut(all_data['Age'].astype(int), 5)


# ## E. Encoding of categorical features

# In[ ]:


final_features = pd.get_dummies(all_data).reset_index(drop=True)
final_features.shape


# # 3. Applying classification algorithms

# In[ ]:


train = final_features[:ntrain]
test = final_features[ntrain:]


# ## A. Validation method

# In[ ]:


#Validation function
n_folds = 5

def score_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    score = cross_val_score(model, train.values, y_train, cv = kf)
    return("score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# ## B. Comparing models

# * **GaussianNB**

# In[ ]:


score_cv(GaussianNB())


# * **LogisticRegression**

# In[ ]:


params = {'logisticregression__C' : [0.001,0.01,0.1,1,10,100,1000]}
pipe = make_pipeline(RobustScaler(), LogisticRegression())
gridsearch_logistic = GridSearchCV (pipe, params, cv=10)
gridsearch_logistic.fit(train, y_train)
print ("Meilleurs parametres: ", gridsearch_logistic.best_params_)


# In[ ]:


score_cv(gridsearch_logistic.best_estimator_)


# * **KNeighborsClassifier**

# In[ ]:


params = {'kneighborsclassifier__n_neighbors' : [3,4,5,6,7],
         'kneighborsclassifier__weights' : ['uniform','distance'],
         'kneighborsclassifier__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']}
pipe = make_pipeline(RobustScaler(), KNeighborsClassifier())
gridsearch_KNC = GridSearchCV (pipe, params, cv=5)
gridsearch_KNC.fit(train, y_train)
print ("Meilleurs parametres: ", gridsearch_KNC.best_params_)


# In[ ]:


score_cv(gridsearch_KNC.best_estimator_)


# * **XGBClassifier**

# In[ ]:


score_cv(XGBClassifier())


# * **GradientBoostingClassifier**

# In[ ]:


gradient = GradientBoostingClassifier()
gradient.fit(train, y_train)


# In[ ]:


score_cv(GradientBoostingClassifier())


# In[ ]:


params = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 4),
    "min_samples_leaf": np.linspace(0.1, 0.5, 4),
    "max_depth":[3,5,8],
    "max_features":["auto","log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[100]
    }
gridsearch_gradient = RandomizedSearchCV (GradientBoostingClassifier(), params, n_iter = 500, cv=5)
gridsearch_gradient.fit(train, y_train)
print ("Meilleurs parametres: ", gridsearch_gradient.best_params_)


# In[ ]:


score_cv(gridsearch_gradient.best_estimator_)


# ## C. Create submitting file

# In[ ]:


pred = gridsearch_logistic.best_estimator_.predict(test)
sub = pd.DataFrame()
sub['PassengerID'] = test_ID
sub['Survived'] = pred
sub.to_csv('submission.csv',index=False)

