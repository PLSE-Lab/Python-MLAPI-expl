#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# - In this challenge we need to predict whether a passenger survived or did not survived on the Titanic
# - This is a <b>Classification</b> problem
# - URL of the dataset https://www.kaggle.com/c/titanic

# ### Importing Libraries

# In[ ]:


# For linear algebra
import numpy as np  

# For EDA and cleaning the data
import pandas as pd

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# For building a model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm
import xgboost
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

import warnings
warnings.filterwarnings('ignore')


# ### Loading the data

# In[ ]:


train_df = pd.read_csv('../input/train.csv')


# ### Miscellaneous

# In[ ]:


train_df.head()


# In[ ]:


train_df.shape


# There are <b>891</b> rows and <b>12</b> columns in the data

# In[ ]:


train_df.info()


# There are <b>null</b> values in the columns <b>Age</b>, <b>Cabin</b> and <b>Embarked</b><br>
# There are <b>seven numerical</b> columns<br>
# There are <b>five categorical</b> columns<br>

# In[ ]:


train_df.describe()


# <b>Statistics</b> about the numerical columns present in the data

# Before moving on to <b>EDA</b>, ley's clean the data.

# ### Cleaning the data

# In[ ]:


train_df.isnull().sum()


# <b>Age column has 177 null values</b><br>
# <b>Cabin column has 687 null values</b><br>
# <b>Embarked column has 2 null values</b><br>

# In[ ]:


# replacing null values with median in Age column
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# replacing null values with mode in Embarked column
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# we will drop Cabin column from our data
train_df.drop('Cabin', axis=1, inplace=True)


# In[ ]:


print("Columns with their count of null values: ")
train_df.isnull().sum()


# Our data is clean now

# ## EDA

# In[ ]:


train_df.head()


# In[ ]:


print(train_df.Survived.value_counts())
sns.countplot(x='Survived', data=train_df, palette='rainbow')


# ***0 -> did not survived<br>
# ***1 -> survived<br>
# <b>549</b> people <b> did not survived</b><br>
# <b>342</b> people <b>survived<b><br>

# In[ ]:


sns.countplot(x='Pclass',hue='Survived',data=train_df)


# ***Pclass -> Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd***<br>
# People who were in <b>Pclass 1</b> survived the most<br>
# Most of the people from <b>Pclass 3</b> did not survived<br>
# 
# <b>Reason:</b><br>
# - One reason could be more priority was given to Pclass 1 people than Pclass 3 people<br>

# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train_df)


# <b>Females survived</b> the most<br>
# <b>Reason:</b><br>
# 
# - Usually, females are given more priority than males. Because, naturally males are stronger than females.

# In[ ]:


sns.violinplot(x='Age', data=train_df, palette='Greens_r')


# <b>Most of the people in the dataset are between the ages 20 and 40</b><br>

# In[ ]:


sns.barplot(x='Survived', y='Age', data=train_df, palette='rocket_r')


# In[ ]:


sns.countplot(x='SibSp', hue='Survived',data=train_df, palette='binary_r')


# ***SibSp -> Number of siblings / spouses aboard the Titanic*** <br>
# Most of the people who <b>survived had no Siblings/Spouses</b><br>
# People with more Siblings/Spouses were not able to survive<br>

# In[ ]:


sns.catplot(x='Parch', data=train_df, kind='count', col='Survived')


# ***Parch -> Number of parents / children aboard the Titanic***<br>
# People with <b>no Parents/Children</b> Survived the most<br>

# In[ ]:


sns.barplot(x='Embarked', y='Survived', data=train_df, palette='Spectral')


# ***Embarked -> Port of Embarkation C = Cherbourg(France), Q = Queenstown(New Zealand), S = Southampton(England)***<br>
# Most of the people who boarded from <b>Cherbourg</b> survived the most

# In[ ]:


train_df.head()


# ### Removing redundant features

# Features going to be removed
# - PassengerId
# - Name
# - Ticket

# In[ ]:


train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


# In[ ]:


train_df.head()


# ### Converting categorical features to numerical features

# <b>Most machine learning algorithms can't work on text data. So, we need to convert text to numbers.</b><br>

# There are many techniques, mostly used ones are described below:
# - Use <b>Label Encoding</b> for ordinal(Which has order) columns
# - Use <b>One-Hot Encoding</b> for nominal(which doesn't have any order in them) columns<br>
# Here's a link to learn about ordinal and nominal columns:
# https://sciencing.com/difference-between-nominal-ordinal-data-8088584.html

# Nominal columns: <b>Sex</b><br>
# Ordinal columns: <b>Embarked</b><br>

# In[ ]:


lb = LabelEncoder()


# In[ ]:


lb.fit(train_df.Embarked)


# In[ ]:


train_df['Embarked'] = lb.transform(train_df.Embarked)


# In[ ]:


train_df.head()


# In[ ]:


lb.classes_


# <b>Label Encoder replaced 'C' with 0, 'Q' with 1 and 'S' with 2</b><br>

# In[ ]:


train_df = pd.get_dummies(train_df) # One-Hot Encoding is also called dummy encoding, we use pd.get_dummies func


# In[ ]:


train_df.head()


# How does One-Hot Encoding work?<br>
# - It extracts the all the <b>categories</b> and makes them columns. In our case Male and Female<br>
# - Whenever there occurs a <b>Female</b> in the Sex column, it places <b>1</b> in the <b>Sex_female</b> column and <b>0</b> in the <b>Sex_male</b> column<br>
# Link to learn more about One-Hot Encoding: https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science

# ### Splitting the data

# In[ ]:


# X contains all the columns except the Survived columns, becuase predictions will be made on Survived column
# Y contains only the Survived column
# Note: the column we are going to predict is also called target

X = train_df.drop('Survived', axis=1)
y = train_df.Survived


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


print(f"Training size: {X_train.shape[0]}")
print(f"Testing size: {X_test.shape[0]}")


# We will train our model on <b>712</b> rows<br>
# We will test our model on <b>179</b> rows</b>

# ### Training the model

# - We'll use many models to train on and choose the one which gives the best accuracy

# #### GradientBoostingClassifier

# In[ ]:


gbm = GradientBoostingClassifier(n_estimators=1000)


# In[ ]:


gbm.fit(X_train, y_train)


# In[ ]:


gbm_preds = gbm.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test, gbm_preds)


# #### RandomForestClassifier

# In[ ]:


rfc = RandomForestClassifier(n_jobs=2, n_estimators=500, oob_score=True)


# In[ ]:


rfc.fit(X_train, y_train)


# In[ ]:


rfc_preds = rfc.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test, rfc_preds)


# #### LGBMClassifier

# In[ ]:


lgbm = lightgbm.LGBMClassifier()


# In[ ]:


lgbm.fit(X_train, y_train)


# In[ ]:


lgbm_preds = lgbm.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test, lgbm_preds)


# #### XGBClassifier

# In[ ]:


xgb = xgboost.XGBClassifier(n_jobs=2, n_estimators=500, base_score=0.7)


# In[ ]:


xgb.fit(X_train, y_train)


# In[ ]:


xgb_preds = xgb.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test, xgb_preds)


# #### ExtraTreeClassifier

# In[ ]:


etc = ExtraTreesClassifier(n_jobs=2, bootstrap=True, oob_score=True, verbose=2, n_estimators=1000)


# In[ ]:


etc.fit(X_train, y_train)


# In[ ]:


etc_preds = etc.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test, etc_preds)


# #### AdaBoostClassifier

# In[ ]:


adbc = AdaBoostClassifier(n_estimators=500, learning_rate=0.04)


# In[ ]:


adbc.fit(X_train, y_train)


# In[ ]:


adbc_preds = adbc.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test, adbc_preds)


# #### DecisionTreeClassifier

# In[ ]:


dtc = DecisionTreeClassifier()


# In[ ]:


dtc.fit(X_train, y_train)


# In[ ]:


dtc_preds = dtc.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test, dtc_preds)


# #### LogisticRegression

# In[ ]:


lg = LogisticRegression(max_iter=1000, verbose=4, n_jobs=3, dual=True)


# In[ ]:


lg.fit(X_train, y_train)


# In[ ]:


lg_preds = lg.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test, lg_preds)


# ### Predicting on Test Data using LGBM

# In[ ]:


test_df = pd.read_csv('../input/test.csv')


# In[ ]:


test_df.head()


# In[ ]:


PassengerId = test_df.PassengerId


# In[ ]:


test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# In[ ]:


test_df.head()


# In[ ]:


# replacing null values with median in Age column
test_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# replacing null values with mode in Embarked column
test_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)


# In[ ]:


lb.fit(test_df.Embarked)


# In[ ]:


test_df.Embarked = lb.transform(test_df.Embarked)


# In[ ]:


test_df = pd.get_dummies(test_df)


# In[ ]:


test_df.head()


# In[ ]:


predictions = lgbm.predict(test_df)


# In[ ]:


predictions


# In[ ]:


submit_df = pd.DataFrame()


# In[ ]:


submit_df['PassengerId'] = PassengerId
submit_df['Survived'] = predictions


# In[ ]:


submit_df.head()


# In[ ]:


submit_df.to_csv('submission.csv', index=False)

