#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pwd


# In[ ]:


import pandas as pd


# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train = train_data.copy()
test = test_data.copy()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


train['Pclass'].value_counts()


# In[ ]:


train['Age'].value_counts()


# In[ ]:


train['SibSp'].value_counts()


# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


train['Cabin'].value_counts()


# In[ ]:


train['Ticket'].value_counts()


# In[ ]:


train['Parch'].value_counts()


# In[ ]:


train['Sex'].value_counts()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.barplot(x = 'Pclass', y = 'Survived', data= train);


# In[ ]:


sns.barplot(x = 'Sex', y = 'Survived', data= train);


# In[ ]:


sns.barplot(x = 'Embarked', y = 'Survived', data= train);


# In[ ]:


sns.barplot(x = 'Age', y = 'Survived', data= train);


# In[ ]:


train.head()


# In[ ]:


train = train.drop(['Ticket'], axis = 1)


# In[ ]:


test = test.drop(['Ticket'], axis = 1)


# In[ ]:


train.head()


# In[ ]:


train.describe().T


# In[ ]:


sns.boxplot(x = train['Fare']);


# In[ ]:


Q1 = train['Fare'].quantile(0.25)


# In[ ]:


Q3 = train['Fare'].quantile(0.75)


# In[ ]:


IQR = Q3 - Q1


# In[ ]:


lower_limit = Q1- 1.5*IQR
lower_limit


# In[ ]:


upper_limit = Q3 + 1.5*IQR
upper_limit


# In[ ]:


train['Fare'] > (upper_limit)


# In[ ]:


train.sort_values("Fare", ascending=False).head()


# In[ ]:


train["Fare"]=train["Fare"].replace(512.3292,300)


# In[ ]:


train.sort_values("Fare", ascending=False).head()


# In[ ]:


train.isnull().sum()


# In[ ]:


train["Age"]=train["Age"].fillna(train["Age"].mean())


# In[ ]:


test["Age"]=test["Age"].fillna(test["Age"].mean())


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train["Embarked"].value_counts()


# In[ ]:


train["Embarked"]=train["Embarked"].fillna("S")


# In[ ]:


test["Embarked"].value_counts()


# In[ ]:


test["Embarked"]=test["Embarked"].fillna("S")


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


test[test["Fare"].isnull()]


# In[ ]:


test[["Pclass","Fare"]].groupby("Pclass").mean()


# In[ ]:


test["Fare"]=test["Fare"].fillna(12)


# In[ ]:


test["Fare"].isnull()


# In[ ]:


test["Fare"].isnull().sum()


# In[ ]:


train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))


# In[ ]:


train = train.drop(['Cabin'], axis = 1)


# In[ ]:


test  = test.drop(["Cabin"], axis = 1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


embarked_mapping = {"S": 1, "C": 2, "Q": 3}


# In[ ]:


train["Embarked"]=train["Embarked"].map(embarked_mapping)


# In[ ]:


test["Embarked"]=test["Embarked"].map(embarked_mapping)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


from sklearn import preprocessing

lbe = preprocessing.LabelEncoder()
train["Sex"] = lbe.fit_transform(train["Sex"])
test["Sex"] = lbe.fit_transform(test["Sex"])


# In[ ]:


train.head()


# In[ ]:


train["Title"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
test["Title"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train.head()


# In[ ]:


train["Title"].value_counts()


# In[ ]:


train['Title'] = train['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Lady', 'Capt', 'Jonkheer', 'Dona','Don'], 'Rare')


# In[ ]:


train['Title'] = train['Title'].replace(['Countess', 'Sir'], 'Royal')


# In[ ]:


train['Title'] = train['Title'].replace('Mlle', 'Miss')


# In[ ]:


train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')


# In[ ]:


train.head()


# In[ ]:


test['Title'] = test['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')


# In[ ]:


test.head()


# In[ ]:


train[["Title","PassengerId"]].groupby("Title").count()


# In[ ]:


train[["Title","Survived"]].groupby(["Title"], as_index=False).mean()


# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5}

train['Title'] = train['Title'].map(title_mapping)


# In[ ]:


train.head()


# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5}

test['Title'] = test['Title'].map(title_mapping)


# In[ ]:


test.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train=train.drop(["Name"],axis=1)


# In[ ]:


test=test.drop(["Name"],axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


import numpy as np


# In[ ]:


bins = [0, 5, 12, 18, 24, 35, 60,  np.inf] 


# In[ ]:


mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)


# In[ ]:


age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)


# In[ ]:


train.head()


# In[ ]:


train=train.drop(["Age"],axis=1)


# In[ ]:


test=test.drop(["Age"],axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])


# In[ ]:


train=train.drop(["Fare"],axis=1)


# In[ ]:


test=test.drop(["Fare"],axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train["FareBand"].value_counts()


# In[ ]:


train["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1


# In[ ]:


test["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1


# In[ ]:


train.head()


# In[ ]:


train['Single'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)
train['SmallFam'] = train['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
train['MedFam'] = train['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
train['LargeFam'] = train['FamilySize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


train.head()


# In[ ]:


test['Single'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)
test['SmallFam'] = test['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
test['MedFam'] = test['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
test['LargeFam'] = test['FamilySize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


test.head()


# In[ ]:


train = pd.get_dummies(train, columns = ["Title"])
train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")


# In[ ]:


train.head()


# In[ ]:


test = pd.get_dummies(test, columns = ["Title"])
test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")


# In[ ]:


test.head()


# In[ ]:


train["Pclass"] = train["Pclass"].astype("category")
train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")


# In[ ]:


train.head()


# In[ ]:


test["Pclass"] = test["Pclass"].astype("category")
test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X = train.drop(['Survived', 'PassengerId'], axis=1)
Y = train["Survived"]


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 67)


# In[ ]:


x_train.head()


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_logreg)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_randomforest)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)


# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV


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


# In[ ]:


ids = test['PassengerId']
predictions = xgb_tuned.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submissiongul.csv', index=False)


# In[ ]:


output.head()


# In[ ]:




