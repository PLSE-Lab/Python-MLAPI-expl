#!/usr/bin/env python
# coding: utf-8

# **Importing  necessary libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dt_train = pd.read_csv("../input/train.csv")
dt_test = pd.read_csv("../input/test.csv")


# In[ ]:


dt_train.head()


# In[ ]:


dt_train.info()


# In[ ]:


dt_train.describe()


# Biding the datasets together and then looking for Null values:

# In[ ]:


#Survived = dt_train['Survived']
#dt_train.drop('Survived', axis=1, inplace=True)
dt = pd.concat([dt_train.drop('Survived', axis=1), dt_test])
dt.info()


# In[ ]:


plt.figure(figsize=(14,7))
sbn.heatmap(dt.isnull(), yticklabels=False, cbar=False)


# In[ ]:


print(dt['Embarked'].isnull().sum())
print(dt['Fare'].isnull().sum())


# In[ ]:


dt['Age'].hist()


# Null values at the "Age" column will take the median value of the data set. And for the "Fare" column, the mean value will be given.

# In[ ]:


dt_train['Age'].fillna(dt['Age'].median(skipna=True), inplace=True)
dt_test['Age'].fillna(dt['Age'].median(skipna=True), inplace=True)


# In[ ]:


dt_train['Fare'].fillna(dt['Fare'].mean(skipna=True), inplace=True)
dt_test['Fare'].fillna(dt['Fare'].mean(skipna=True), inplace=True)


# Looking at the different values of the column "Embarked".

# In[ ]:


print(dt['Embarked'].value_counts())
print(pd.crosstab(dt_train.Survived, dt_train.Embarked))


# Since most of the observations have "S" for this variable, we'll fill the Null values with it.

# In[ ]:


dt_train['Embarked'].fillna('S', inplace=True)
dt_test['Embarked'].fillna('S', inplace=True)


# In[ ]:


dt = pd.concat([dt_train.drop('Survived', axis=1), dt_test])
dt.info()


# Now all the observations are complete.

# In[ ]:


dt_train.head()


# In[ ]:


plt.figure(figsize=(12,6))
sbn.countplot(x="Sex", data=dt_train, hue="Survived", palette="Set1")
plt.title('Sex x Survived')


# In[ ]:


plt.figure(figsize=(12,6))
sbn.countplot(x="Embarked", data=dt_train, hue="Survived", palette="Set1")
plt.title('Embarked x Survived')


# We are going to create 3 categories for the "Age" column: Under the quantile 0.08,  between the 0.08 and the 0.75 quantile, over the 0.75 quantile.

# In[ ]:


intervals = (0, dt['Age'].quantile(q=0.08), dt['Age'].quantile(q=0.75), 150)
cats = ["under_ages", "between_ages", "upper_ages"]

dt_train["Age_cat"] = pd.cut(dt_train.Age, intervals, labels=cats)
dt_test["Age_cat"] = pd.cut(dt_test.Age, intervals, labels=cats)

dt_train.drop('Age', axis=1, inplace=True)
dt_test.drop('Age', axis=1, inplace=True)


# In[ ]:


plt.figure(figsize=(12,6))
sbn.countplot(x="Age_cat", data=dt_train, hue="Survived", palette="Set1")
plt.title('Age_cat x Survived')


# In[ ]:


dt['Fare'].hist(bins=16)
print(dt['Fare'].quantile(q=[0.5,0.75]))


# Also, we are going to create 3 categories for the "Fare" column: Under the median value, between the median value and the 0.75 quantile, over the 0.75 quantile.

# In[ ]:


intervals = (dt['Fare'].min(), dt['Fare'].quantile(q=0.5), dt['Fare'].quantile(q=0.75), dt['Fare'].max())
cats = ["cheap", "expensive", "millionaire"]

dt_train["Fare_cat"] = pd.cut(dt_train.Fare, intervals, labels=cats)
dt_test["Fare_cat"] = pd.cut(dt_test.Fare, intervals, labels=cats)

dt_train.drop('Fare', axis=1, inplace=True)
dt_test.drop('Fare', axis=1, inplace=True)


# In[ ]:


plt.figure(figsize=(12,6))
sbn.countplot(x="Fare_cat", data=dt_train, hue="Survived", palette="Set1")
plt.title('Fare_cat x Survived')


# In[ ]:


dt_train.head()


# The columns "Name", "Cabin" and "Ticket" are not going to help with anything, so we're going to drop them.

# In[ ]:


dt_train.drop(['Name','Cabin', 'Ticket'], axis=1, inplace=True)
dt_test.drop(['Name','Cabin', 'Ticket'], axis=1, inplace=True)


# The "SibSp" and "Parch" columns are going to form the "Family" column.

# In[ ]:


dt_train['Family'] = dt_train['SibSp']+dt_train['Parch']+1
dt_test['Family'] = dt_test['SibSp']+dt_test['Parch']+1

dt_train.drop(['SibSp', 'Parch'], axis=1, inplace=True)
dt_test.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# In[ ]:


print(dt_train['Family'].value_counts())


# In[ ]:


intervals = (0, 1, 2, 11)
cats = ["alone", "couple", "family"]

dt_train["Fam_cat"] = pd.cut(dt_train.Family, intervals, labels=cats)
dt_test["Fam_cat"] = pd.cut(dt_test.Family, intervals, labels=cats)

dt_train.drop('Family', axis=1, inplace=True)
dt_test.drop('Family', axis=1, inplace=True)


# In[ ]:


plt.figure(figsize=(12,6))
sbn.countplot(x="Fam_cat", data=dt_train, hue="Survived", palette="Set1")
plt.title('Fam_cat x Survived')


# In[ ]:


dt_train = pd.get_dummies(dt_train, columns=['Pclass', 'Sex', 'Embarked', 'Age_cat', 'Fare_cat', 'Fam_cat'], drop_first=True)
dt_test = pd.get_dummies(dt_test, columns=['Pclass', 'Sex', 'Embarked', 'Age_cat', 'Fare_cat', 'Fam_cat'], drop_first=True)


# In[ ]:


dt_train.head(10)


# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


classifier =  XGBClassifier(n_estimators=1000, learning_rate=0.05,n_jobs=-1)


# In[ ]:


X_train = dt_train.drop(['Survived', 'PassengerId'], axis=1)
y_train = dt_train['Survived']


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


predictions = classifier.predict(dt_test.drop('PassengerId', axis=1))
output = pd.DataFrame()
output['PassengerId'] = dt_test['PassengerId']
output['Survived'] = predictions


# In[ ]:


output.head()


# In[ ]:


output.to_csv("output.csv", index=False)


# In[ ]:




