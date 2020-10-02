#!/usr/bin/env python
# coding: utf-8

# # Business Understanding / Problem Definition

# # Data Understanding (Exploratory Data Analysis)

# ## Importing Librarires

# In[ ]:


# data analysis libraries:
import numpy as np
import pandas as pd

# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# to display all columns:
pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split, GridSearchCV


# ## Loading Data

# In[ ]:


# Read train and test data with pd.read_csv():
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


# copy data in order to avoid any change in the original:
train = train_data.copy()
test = test_data.copy()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# ## Analysis and Visualization of Numeric and Categorical Variables

# ### Basic summary statistics about the numerical data

# In[ ]:


train.describe().T


# ### Classes of some categorical variables

# In[ ]:


train['Pclass'].value_counts()


# In[ ]:


train['Sex'].value_counts()


# In[ ]:


train['SibSp'].value_counts()


# In[ ]:


train['Parch'].value_counts()


# In[ ]:


train['Ticket'].value_counts()


# In[ ]:


train['Cabin'].value_counts()


# In[ ]:


train['Embarked'].value_counts()


# ### Visualization

# In general, barplot is used for categorical variables while histogram, density and boxplot are used for numerical data.

# #### Pclass vs survived:

# In[ ]:


sns.barplot(x = 'Pclass', y = 'Survived', data = train);


# #### SibSp vs survived:

# In[ ]:


sns.barplot(x = 'SibSp', y = 'Survived', data = train);


# #### Parch vs survived:

# In[ ]:


sns.barplot(x = 'Parch', y = 'Survived', data = train);


# #### Sex vs survived:

# In[ ]:


sns.barplot(x = 'Sex', y = 'Survived', data = train);


# # Data Preparation

# ## Deleting Unnecessary Variables

# In[ ]:


train.head()


# ### Ticket & Cabin

# In[ ]:


# We can drop the Ticket feature since it is unlikely to have useful information
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

train.head()


# In[ ]:


train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
train.head()


# ## Outlier Treatment

# In[ ]:


train.describe().T


# In[ ]:


# It looks like there is a problem in Fare max data. Visualize with boxplot.
sns.boxplot(x = train['Fare']);


# In[ ]:


Q1 = train['Fare'].quantile(0.05)
Q3 = train['Fare'].quantile(0.95)
IQR = Q3 - Q1

lower_limit = Q1- 1.5*IQR
lower_limit

upper_limit = Q3 + 1.5*IQR
upper_limit


# In[ ]:


# observations with Fare data higher than the upper limit:

train['Fare'] > (upper_limit)


# In[ ]:


train.sort_values("Fare", ascending=False).head()


# In[ ]:


# In boxplot, there are too many data higher than upper limit; we can not change all. Just repress the highest value -512- 
train['Fare'] = train['Fare'].replace(512.3292, 300)


# In[ ]:


train.sort_values("Fare", ascending=False).head()


# In[ ]:


test.sort_values("Fare", ascending=False)


# In[ ]:


test['Fare'] = test['Fare'].replace(512.3292, 300)


# In[ ]:


test.sort_values("Fare", ascending=False)


# ## Missing Value Treatment

# In[ ]:


train.isnull().sum()


# ### Age

# In[ ]:


train["Age"] = train["Age"].fillna(train["Age"].mean())


# In[ ]:


test["Age"] = test["Age"].fillna(test["Age"].mean())


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# ### Embarked

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train["Embarked"].value_counts()


# In[ ]:


# Fill NA with the most frequent value:
train["Embarked"] = train["Embarked"].fillna("S")


# In[ ]:


test["Embarked"] = test["Embarked"].fillna("S")


# ### Fare

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


test[test["Fare"].isnull()]


# In[ ]:


test[["Pclass","Fare"]].groupby("Pclass").mean()


# In[ ]:


test["Fare"] = test["Fare"].fillna(12)


# In[ ]:


test["Fare"].isnull().sum()


# ## Variable Transformation

# ### Embarked

# In[ ]:


# Map each Embarked value to a numerical value:

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)


# In[ ]:


train.head()


# ### Sex

# In[ ]:


# Convert Sex values into 1-0:

from sklearn import preprocessing

lbe = preprocessing.LabelEncoder()
train["Sex"] = lbe.fit_transform(train["Sex"])
test["Sex"] = lbe.fit_transform(test["Sex"])


# In[ ]:


train.head()


# ### Name - Title

# In[ ]:


train["Title"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
test["Title"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train.head()


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


train.head()


# In[ ]:


test.head()


# In[ ]:


train[["Title","PassengerId"]].groupby("Title").count()


# In[ ]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


# Map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5}

train['Title'] = train['Title'].map(title_mapping)


# In[ ]:


train.isnull().sum()


# In[ ]:


test['Title'] = test['Title'].map(title_mapping)


# In[ ]:


test.head()


# In[ ]:


train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# In[ ]:


train.head()


# ### AgeGroup

# In[ ]:


bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
mylabels = ['1', '2', '3', '4', '5', '6', '7']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)


# In[ ]:


train.head()


# In[ ]:


#dropping the Age feature for now, might change:
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)


# In[ ]:


train.head()


# ### Fare

# In[ ]:


# Map Fare values into groups of numerical values:
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])


# In[ ]:


# Drop Fare values:
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# In[ ]:


train.head()


# ## Feature Engineering

# ### Family Size

# In[ ]:


train.head()


# In[ ]:


train["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1


# In[ ]:


test["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1


# In[ ]:


# Create new feature of family size:

train['Single'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)
train['SmallFam'] = train['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
train['MedFam'] = train['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
train['LargeFam'] = train['FamilySize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


train.head()


# In[ ]:


# Create new feature of family size:

test['Single'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)
test['SmallFam'] = test['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
test['MedFam'] = test['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
test['LargeFam'] = test['FamilySize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


test.head()


# ### Embarked & Title

# In[ ]:


# Convert Title and Embarked into dummy variables:

train = pd.get_dummies(train, columns = ["Title"])
train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")


# In[ ]:


train.head()


# In[ ]:


test = pd.get_dummies(test, columns = ["Title"])
test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")


# In[ ]:


test.head()


# ### Pclass

# In[ ]:


# Create categorical values for Pclass:
train["Pclass"] = train["Pclass"].astype("category")
train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")


# In[ ]:


test["Pclass"] = test["Pclass"].astype("category")
test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:





# # Modeling, Evaluation and Model Tuning

# ## Spliting the train data

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 0)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_logreg)


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_randomforest)


# ## Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)


# In[ ]:


xgb_params = {
        'n_estimators': [200, 500],
        'subsample': [0.6, 1.0],
        'max_depth': [2,5,8],
        'learning_rate': [0.1,0.01,0.02],
        "min_samples_split": [2,5,10]}


# In[ ]:


xgb = GradientBoostingClassifier()

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)


# In[ ]:


xgb_cv_model.fit(x_train, y_train)


# In[ ]:


xgb_cv_model.best_params_


# In[ ]:


xgb = GradientBoostingClassifier(learning_rate = xgb_cv_model.best_params_["learning_rate"], 
                    max_depth = xgb_cv_model.best_params_["max_depth"],
                    min_samples_split = xgb_cv_model.best_params_["min_samples_split"],
                    n_estimators = xgb_cv_model.best_params_["n_estimators"],
                    subsample = xgb_cv_model.best_params_["subsample"])


# In[ ]:


xgb_tuned =  xgb.fit(x_train,y_train)


# In[ ]:


y_pred = xgb_tuned.predict(x_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)


# # Deployment

# In[ ]:


test


# In[ ]:


#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = logreg.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)


# In[ ]:


output.head()


# In[ ]:




