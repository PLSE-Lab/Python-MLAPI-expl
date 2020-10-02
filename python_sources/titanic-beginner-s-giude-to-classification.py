#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Let us put both the training and test data into respective dataframes. Pandas is already exported for us.

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")


# # Analysis and Preprocessing.
# 1. Look for missing values, drop them or fix them.
# 2. Feature Selection/Rejection
# 3. Dummify the categorical variables.

# In[ ]:


# Shall we have a peek inside train data ??

train_df.head(3)


# In[ ]:


train_df.isnull().sum()


# **Missing value Analysis : **Age & Embarked have a few missing values, Cabin has a lot of missing values.

# In[ ]:


# peek inside test data..

test_df.head(3)


# In[ ]:


test_df.isnull().sum()


# **Missing value Analysis : **Few missing values in Age & Fare, Cabin has a lot of it.

# ## 1. Fixing missing values.

# ### Train Set

# In[ ]:


train_df.isnull().sum()


# In[ ]:


# Drop the Cabin column as it has close to 80% missing data. Will not make sense to Impute it.
train_df.drop('Cabin', axis=1, inplace=True)

# Status update...
train_df.isnull().sum()


# In[ ]:


# Only 2 missing values for Embarked, drop them too..
train_df.dropna(subset=['Embarked'], inplace=True)
train_df.isnull().sum()


# In[ ]:


# Let us deal with age now. Since people who booked the tickets ranged from adults as well as oldies, i am gonna go and fill the empty ones with average of data.
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
train_df.isnull().sum()


# In[ ]:


train_df.info()


# So no null values, let's proceed to test data.

# ### Test Set

# In[ ]:


test_df.info()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


# Let's get rid of Cabin, 327 out of 418 are missing.
test_df.drop('Cabin', axis=1, inplace=True)
test_df.isnull().sum()


# In[ ]:


# Fare, only a single peice missing, but can't remove from test data
test_df['Fare'] = test_df['Fare'].fillna(train_df['Fare'].median())
test_df.isnull().sum()


# In[ ]:


# Average the hell out of Age..
test_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
test_df.isnull().sum()


# In[ ]:


test_df.info()


# ## 2. Feature Selection/Rejection

# There could be some features that we could identify from the first sight and say - "this is random, or maybe not useful". I am just gonna do that, will use all the charts and gizmos later as i am yet to learn them. 
# Moreover, you can apply relation techniqes and improve. No hard feelings.

# In[ ]:


train_df.head(2)


# Here is what the features mean..
# ![image.png](attachment:image.png)

# 1. pid - Index. Will remove it.
# 2. Survival - Its a target variable, so its gonna go anyways.
# 3. **pclass** - It will be relevant, remmember in the movie how the first class passengers were being taken to the boat first. Hell they took the dog too.
# 4. **Sex** - 70% of survivors were women. So its relevant. Remmember - Ladies first.
# 5. **Age** - The old and mothers were lifted first. So will keep it too.
# 6. **sibsp/parch** - will keep them. Later we will do feature engineering with them.
# 7. ticket - Looks to me like a alphanumeric random set. Cut it.
# 8. fare - I will remove it as we already have class. That was more relevant.
# 9. cabin - Already removed. Too many missing values.
# 10. **embarked** - Will keep it. 

# In[ ]:


# Dropping irrelaveant features from test and train sets. Also alloting dependent var "y" and independent vars "X".

X = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Fare'], axis=1)
y = train_df['Survived']

n_test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Fare'], axis=1)


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


n_test_df.head()


# ## 3. Dummifying Categorical variables.
# 

# Categorical variables are ones which are not numerical. Like in our fresh dataframe X, Sex is one categorical variable containing values "Male" & "Female". Embarked is another.
# We will use pandas get_dummies to convert them. Read more about it from google.

# In[ ]:


# Train set.
X = pd.get_dummies(X, drop_first=True)


# In[ ]:


# Test set.
n_test_df = pd.get_dummies(n_test_df, drop_first=True)


# We are now ready for model building.

# # Model Training & Prediction.

# We will train with bunch of models. They will be kept adding from time to time.
# For optimization, we will use hyperparameter tuning.

# ## 1. Train Test Split

# Before we start, we need to divide the training data into a training set and a testing set. This will ensure that our model predict on data it has never seen before, which will give its true sense of strength.

# In[ ]:


# train_test_split from sklearn helps arbitararily split the data between test and train sets. Here i have choose 0.3 (30%) as test data size.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## 2. Modelling with hyperparameter tunining for best results.

# In[ ]:


# Here we will use the XGBClassifier M/L algorithm.
# To find out the best parameter values to call this algorithm, we will use RandomizedSearchCV.

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


parameters = {"learning_rate": [0.1, 0.01, 0.001],
               "gamma" : [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
               "max_depth": [2, 4, 7, 10],
               "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
               "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
               "reg_alpha": [0, 0.5, 1],
               "reg_lambda": [1, 1.5, 2, 3, 4.5],
               "min_child_weight": [1, 3, 5, 7],
               "n_estimators": [100, 250, 500, 1000]}

xgb_rscv = RandomizedSearchCV(XGBClassifier(), param_distributions = parameters, cv = 7, verbose = 3, random_state = 40)

model_rscv = xgb_rscv.fit(X_train, y_train)
model_rscv.best_params_


# In[ ]:


# Best parameter values
tuned_model = XGBClassifier(booster='gbtree', subsample= 0.7,
 reg_lambda= 3,
 reg_alpha= 1,
 n_estimators= 100,
 min_child_weight= 3,
 max_depth= 10,
 learning_rate = 0.001,
 gamma= 0.01,
 colsample_bytree= 0.6)

tuned_model.fit(X_train, y_train)


# In[ ]:


# Score - For this, we now use the 30% test data.
tuned_model.score(X_test, y_test)


# # Submission

# In[ ]:


predictions = tuned_model.predict(n_test_df)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

