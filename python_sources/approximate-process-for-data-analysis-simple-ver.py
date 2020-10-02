#!/usr/bin/env python
# coding: utf-8

# **Column description**
# * survival : 0 = No, 1 = Yes
# * pclass : ticket class / 1 = 1st, 2 = 2nd, 3 = 3rd
# * sex : male, female
# * Age : age in years
# * sibsp : # of siblings / spouses aboard the Titanic
# * parch : # of parents / children aboard the Titanic
# * ticket : ticket number(strings)
# * fare : passenger fare
# * cabin : cabin number
# * embarked = port of embarkation / C = Cherbourg(France), Q = Queenstown(UK), S = Southampton(UK)

# **Load Data**

# In[ ]:


import os
print(os.listdir("./"))

import pandas as pd # data processing
train = pd.read_csv('../input/train.csv',index_col = "PassengerId")
print(train.shape)
train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv',index_col = "PassengerId")
print(test.shape)
test.head()


# **EDA : Exploratory Data Analysis**
# 1. 1 Variable
#     * Sex
#     * Pclass
#     * Embarked
# 
# 2. 2 Variable
#     * Age & Fare
#     * SibSp & Parch

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt


# **Sex**

# In[ ]:


sns.countplot(data = train, x = "Sex", hue = "Survived")


# **Conclusion** : Analysis shows that female passengers are overwhelmingly more likely to survive than male passengers.

# In[ ]:


pd.pivot_table(train, index = "Sex", values = "Survived")


# **Pclass**

# In[ ]:


sns.countplot(data = train, x = "Pclass", hue = "Survived")


# **Conclusion** : Analysis shows that the higher the Pclass, the higher the probability of survival.

# In[ ]:


pd.pivot_table(train, index = "Pclass", values = "Survived")


# **Embarked**

# In[ ]:


sns.countplot(data = train, x = "Embarked", hue = "Survived")


# **Conclusion** : Analysis shows that the more you board in Cherbourg (C), the more likely you are to survive, and the more likely you are to die in Southampton (S).

# In[ ]:


pd.pivot_table(train, index = "Embarked", values = "Survived")


# **Preprocessing**
# 1. Encoding Sex
# 2. One-Hot-Encoding Embarked
# 3. Fill in missing value - Fare
# 4. Name
# 5. Age
# 6. Family Size

# **Encoding Sex**

# In[ ]:


train.loc[train["Sex"] == "male", "enc_sex"] = 0
train.loc[train["Sex"] == "female", "enc_sex"] = 1

print(train.shape)

train[["Sex","enc_sex"]].head()


# In[ ]:


test.loc[test["Sex"] == "male", "enc_sex"] = 0
test.loc[test["Sex"] == "female", "enc_sex"] = 1

print(test.shape)

test[["Sex","enc_sex"]].head()


# **One-Hot-Encoding Embarked**

# In[ ]:


train["Emb_C"] = train["Embarked"] == "C"
train["Emb_S"] = train["Embarked"] == "S"
train["Emb_Q"] = train["Embarked"] == "Q"

print(train.shape)

train[["Embarked","Emb_C","Emb_S","Emb_Q"]].head()


# In[ ]:


test["Emb_C"] = test["Embarked"] == "C"
test["Emb_S"] = test["Embarked"] == "S"
test["Emb_Q"] = test["Embarked"] == "Q"

print(test.shape)

test[["Embarked","Emb_C","Emb_S","Emb_Q"]].head()


# **Fill in missing value - Fare**

# In[ ]:


train[train["Fare"].isnull()]


# In[ ]:


test[test["Fare"].isnull()]


# In[ ]:


train["fillinFare"] = train["Fare"]

print(train.shape)

train[["Fare","fillinFare"]].head()


# In[ ]:


test["fillinFare"] = test["Fare"]

print(test.shape)

test[["Fare","fillinFare"]].head()


# In[ ]:


test.loc[test["Fare"].isnull(), "fillinFare"] = 0
test.loc[test["Fare"].isnull(), ["Fare", "fillinFare"]]


# **Name**

# In[ ]:


train["Name"].head()


# In[ ]:


def title(Name):
    Ans = Name.split(', ')[1].split(', ')[0]
    return Ans

train["Name"].apply(title).unique()


# In[ ]:


train.loc[train["Name"].str.contains("Mr"), "title"] = "Mr"
train.loc[train["Name"].str.contains("Miss"), "title"] = "Miss"
train.loc[train["Name"].str.contains("Mrs"), "title"] = "Mrs"
train.loc[train["Name"].str.contains("Master"), "title"] = "Master"

print(train.shape)

train[["Name", "title"]].head(10)


# In[ ]:


sns.countplot(data=train, x="title", hue="Survived")


# In[ ]:


pd.pivot_table(train, index="title", values="Survived")


# **Conclusion** : The analysis shows that although the survival rate of the passenger named Mr. is only 15.8%, the survival rate of the passenger named Miss is 70%, the survival rate of the passenger with Mrs is 79%, and, crucially, the survival rate of the master is 57.5%.

# In[ ]:


train["Master"] = train["Name"].str.contains("Master")
print(train.shape)
train[["Name", "Master"]].head(20)


# In[ ]:


test["Master"] = test["Name"].str.contains("Master")
print(test.shape)
test[["Name", "Master"]].head(20)


# **Age**

# In[ ]:


train["Child"] = train["Age"] < 14

print(train.shape)

train[["Age", "Child"]].head(10)


# In[ ]:


test["Child"] = test["Age"] < 14

print(test.shape)

test[["Age", "Child"]].head(10)


# ** FamilySize **

# In[ ]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

print(train.shape)

train[["SibSp", "Parch", "FamilySize"]].head()


# In[ ]:


test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

print(test.shape)

test[["SibSp", "Parch", "FamilySize"]].head()


# In[ ]:


train["Single"] = train["FamilySize"] == 1

train["Middle"] = (train["FamilySize"] > 1) & (train["FamilySize"] < 5)

train["Big"] = train["FamilySize"] >= 5

print(train.shape)

train[["FamilySize", "Single", "Middle", "Big"]].head(10)


# In[ ]:


test["Single"] = test["FamilySize"] == 1
test["Middle"] = (test["FamilySize"] > 1) & (test["FamilySize"] < 5)
test["Big"] = test["FamilySize"] >= 5

print(test.shape)

test[["FamilySize", "Single", "Middle", "Big"]].head(10)


# ** Training **

# **Feature Selection**

# In[ ]:


feature = ["Pclass", "enc_sex", "Emb_C", "Emb_S", "Emb_Q","fillinFare",
                 "Master","Child", "Single", "Middle", "Big"]
feature


# In[ ]:


label = "Survived"
label


# In[ ]:


X_train = train[feature]

print(X_train.shape)

X_train.head()


# In[ ]:


X_test = test[feature]

print(X_test.shape)

X_test.head()


# **Label allocate**

# In[ ]:


y_train = train[label]

print(y_train.shape)

y_train.head()


# **Decision Tree - Scikit-learn**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=9, random_state=0)
model


# In[ ]:


model.fit(X_train, y_train)


# **Visualization**

# In[ ]:


import graphviz
from sklearn.tree import export_graphviz

tree = export_graphviz(model,
                           feature_names=feature,
                           class_names=["Perish", "Survived"],
                           out_file=None)

graphviz.Source(tree)


# **Prediction**

# In[ ]:


prediction = model.predict(X_test)

print(prediction.shape)

prediction[0:9]


# **Submission**

# In[ ]:


submission = pd.read_csv('../input/gender_submission.csv',index_col = "PassengerId")

print(submission.shape)

submission.tail(10)


# In[ ]:


submission["Survived"] = prediction

print(submission.shape)

submission.tail(10)


# In[ ]:


submission.to_csv("tree.csv")


# In[ ]:




