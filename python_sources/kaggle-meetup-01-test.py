#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import  seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


sns.heatmap(train.isnull())


# In[ ]:


sns.countplot(x="Survived", data=train)


# In[ ]:


sns.countplot(x="Survived", hue="Sex", data= train)


# In[ ]:


sns.countplot(x="Survived", hue="Sex", data=train[train["Age"]  < 16])


# In[ ]:


sns.distplot(train["Fare"])


# In[ ]:


sns.countplot(x="Survived", hue="Pclass", data=train)


# In[ ]:


sns.distplot(train["Age"].dropna(),bins=50)


# In[ ]:


sns.boxplot(x="Pclass", y="Age", data=train)


# In[ ]:


train[train["Pclass"] == 1]["Age"].median()


# In[ ]:


train[train["Pclass"] == 2]["Age"].median()


# In[ ]:


train[train["Pclass"] == 3]["Age"].median()


# In[ ]:


def replace_age(row):
    Age = row[0]
    Pclass = row[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        if Pclass == 2:
            return 29
        if Pclass == 3:
            return 24
    else:
        return Age


# In[ ]:


train["Age"] = train[["Age", "Pclass"]].apply(replace_age, axis=1)


# In[ ]:


train = train.drop("Cabin", axis=1)


# In[ ]:


train.head()


# In[ ]:


train.drop(["Name","Ticket"], axis=1, inplace=True)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


train["Embarked"] = train["Embarked"].dropna()


# In[ ]:


sns.heatmap(test.isnull())


# In[ ]:


test["Age"] = test[["Age","Pclass"]].apply(replace_age, axis=1)


# In[ ]:


test["Fare"] = test["Fare"].fillna(test["Fare"].median())


# In[ ]:


test.drop(["Name", "Cabin", "Ticket"], axis=1, inplace=True)


# In[ ]:


test.head()


# In[ ]:


train.head(2)


# In[ ]:


emark = pd.get_dummies(train["Embarked"], drop_first= True)
sex = pd.get_dummies(train["Sex"], drop_first= True)
train = pd.concat([train, emark, sex], axis=1)
train.drop(["Sex", "Embarked"], axis=1, inplace=True)

emark = pd.get_dummies(test["Embarked"], drop_first= True)
sex = pd.get_dummies(test["Sex"], drop_first= True)
test = pd.concat([test, emark, sex], axis=1)
test.drop(["Sex", "Embarked"], axis=1, inplace=True)


# In[ ]:


x_train = train.drop("Survived", axis=1)


# In[ ]:


y_train = train["Survived"]


# In[ ]:


rf = RandomForestClassifier()


# In[ ]:


rf.fit(x_train, y_train)


# In[ ]:


prediction = rf.predict(test)


# In[ ]:


output = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": prediction})


# In[ ]:


output.to_csv("titanic.csv", index=False)

