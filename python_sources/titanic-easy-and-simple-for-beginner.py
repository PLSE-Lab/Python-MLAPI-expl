#!/usr/bin/env python
# coding: utf-8

# # **1. Load Dataset**

# In[ ]:


import pandas as pd
train = pd.read_csv('../input/train.csv', index_col="PassengerId")
print(train.shape)    
train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv', index_col="PassengerId")
print(test.shape)    
test.head()


# # **2. Data explore**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ## 2-1. Sex

# In[ ]:


sns.countplot(data=train, x='Sex', hue='Survived')


# In[ ]:


pd.pivot_table(train, index="Sex", values="Survived")


# ### Accoding to the result, Female's survival rate is higher than male. Female's survival rate is 74.2% and Male's survival rate is 18.9% . It means 'Sex' is very important in this data. 
# ### Plus, I think I should explore how to find living men. Because most of men died. So, From below I will concentrate on finding men's survival as well as important value.

# ## 2-2. Pclass

# In[ ]:


sns.countplot(data=train, x="Pclass", hue="Survived")


# In[ ]:


pd.pivot_table(train, index="Pclass", values='Survived')


# ### From this data, the higher class is higher survival rate.

# ## 2-3. Embarked

# In[ ]:


sns.countplot(data=train, x="Embarked", hue='Survived')


# In[ ]:


pd.pivot_table(train, index="Embarked", values="Survived")


# ### If get on C, It's most likely to be survived. On the other hand, if get on S, the highest number of passengers from S. But also have the highest death rate.

# ## 2-4. Age&Fare

# In[ ]:


sns.lmplot(data=train, x="Age", y='Fare', hue="Survived", fit_reg=False)


# In[ ]:


low_fare = train[train["Fare"] < 500]  # Remove Outlier
train.shape, low_fare.shape


# In[ ]:


sns.lmplot(data=low_fare, x="Age", y="Fare", hue="Survived", fit_reg=False)


# In[ ]:


low_low_fare = train[train["Fare"] < 100]  # Remove Outlier


# In[ ]:


train.shape, low_fare.shape, low_low_fare.shape   #Compare the number of rows


# In[ ]:


sns.lmplot(data=low_low_fare, x="Age", y="Fare", hue="Survived", fit_reg=False)


# ### I can find that passengers under the age of 15 are more likely to survive, especially those who have paid less than $20 for fares.

# ## 2-5. SibSp, Parch

# In[ ]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1   #Including self
print(train.shape)
train[["SibSp", "Parch", "FamilySize"]].head()


# In[ ]:


sns.countplot(data=train, x="FamilySize", hue="Survived")


# ### If on board alone (FamilySize = 1) or too large(FamilySize>5), survival rate is very low.
# ### If suitable family member (2 < = FamilySize <= 4) get on, the probability of survival is relatively high.

# In[ ]:


# Grouping by FamilySize
train.loc[train["FamilySize"] == 1, "FamilyType"] = "Single"
train.loc[(train["FamilySize"] > 1) & (train["FamilySize"] < 5), "FamilyType"] = "Nuclear"
train.loc[train["FamilySize"] >= 5, "FamilyType"] = "Big"
print(train.shape)
train[["FamilySize", "FamilyType"]].head()


# In[ ]:


sns.countplot(data=train, x="FamilyType", hue="Survived")


# In[ ]:


pd.pivot_table(data=train, index="FamilyType", values="Survived")


# ### If 'Nuclear', the survival rate is relatively high with 57.8%.

# ## 2-6. Name

# In[ ]:


train["Name"].head()


# In[ ]:


def get_title(name):
    return name.split(", ")[1].split('. ')[0]
train["Name"].apply(get_title).unique()


# In[ ]:


train.loc[train["Name"].str.contains("Mr"), "Title"] = "Mr"
train.loc[train["Name"].str.contains("Miss"), "Title"] = "Miss"
train.loc[train["Name"].str.contains("Mrs"), "Title"] = "Mrs"
train.loc[train["Name"].str.contains("Master"), "Title"] = "Master"
print(train.shape)
train[["Name", "Title"]].head(10)


# In[ ]:


sns.countplot(data=train, x="Title", hue="Survived")


# In[ ]:


pd.pivot_table(train, index="Title", values="Survived")


# ### Althogh 'Master' is a male title, Survival rate is 57.5%.

# # **3. Preprocessing**

# ## 3-1. Encode Sex

# In[ ]:


train.loc[train["Sex"] == "male", "Sex_encode"] = 0
train.loc[train["Sex"] == "female", "Sex_encode"] = 1
print(train.shape)
train[["Sex", "Sex_encode"]].head()


# In[ ]:


test.loc[test["Sex"] == "male", "Sex_encode"] = 0
test.loc[test["Sex"] == "female", "Sex_encode"] = 1
print(test.shape)
test[["Sex", "Sex_encode"]].head()


# ## 3-2. Fill in missing fare

# In[ ]:


train[train["Fare"].isnull()]


# In[ ]:


test[test["Fare"].isnull()]


# In[ ]:


train["Fare_fillin"] = train["Fare"]
print(train.shape)
train[["Fare", "Fare_fillin"]].head()


# In[ ]:


test["Fare_fillin"] = test["Fare"]
print(test.shape)
test[["Fare", "Fare_fillin"]].head()


# In[ ]:


train["Fare"].mean()


# In[ ]:


test.loc[test["Fare"].isnull(), "Fare_fillin"] = 32
test.loc[test["Fare"].isnull(), ["Fare", "Fare_fillin"]]


# ## 3-3. Encode Embarked

# In[ ]:


train["Embarked_C"] = train["Embarked"] == "C"
train["Embarked_S"] = train["Embarked"] == "S"
train["Embarked_Q"] = train["Embarked"] == "Q"
print(train.shape)
train[["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]].head()


# In[ ]:


test["Embarked_C"] = test["Embarked"] == "C"
test["Embarked_S"] = test["Embarked"] == "S"
test["Embarked_Q"] = test["Embarked"] == "Q"
print(test.shape)
test[["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]].head()


# ## 3-4. Age

# In[ ]:


train["Child"] = train["Age"] < 15
print(train.shape)
train[["Age", "Child"]].head()


# In[ ]:


test["Child"] = test["Age"] < 15
print(test.shape)
test[["Age", "Child"]].head()


# ## 3-5. Family Size

# In[ ]:


train[["SibSp", "Parch", "FamilySize"]].head()


# In[ ]:


test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
print(test.shape)
test[["SibSp", "Parch", "FamilySize"]].head()


# In[ ]:


train["Single"] = train["FamilySize"] == 1
train["Nuclear"] = (train["FamilySize"] > 1) & (train["FamilySize"] < 5)
train["Big"] = train["FamilySize"] >= 5
print(train.shape)
train[["FamilySize", "Single", "Nuclear", "Big"]].head()


# In[ ]:


test["Single"] = test["FamilySize"] == 1
test["Nuclear"] = (test["FamilySize"] > 1) & (test["FamilySize"] < 5)
test["Big"] = test["FamilySize"] >= 5
print(test.shape)
test[["FamilySize", "Single", "Nuclear", "Big"]].head()


# ## 3-6. Name

# In[ ]:


train["Master"] = train["Name"].str.contains("Master")
print(train.shape)
train[["Name", "Master"]].head(10)


# In[ ]:


test["Master"] = test["Name"].str.contains("Master")
print(test.shape)
test[["Name", "Master"]].head(10)


# # **4. Train**

# In[ ]:


feature_names = ["Pclass", "Sex_encode", "Fare_fillin",
                 "Embarked_C", "Embarked_S", "Embarked_Q",
                 "Child", "Single", "Nuclear", "Big", "Master"]
feature_names


# In[ ]:


label_name = "Survived"
label_name


# In[ ]:


X_train = train[feature_names]
print(X_train.shape)
X_train.head()


# In[ ]:


X_test = test[feature_names]
print(X_test.shape)
X_test.head()


# In[ ]:


y_train = train[label_name]
print(y_train.shape)
y_train.head()


# # ** 5.Use Decision Tree **

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=8, random_state=0)
model


# In[ ]:


model.fit(X_train, y_train)


# # ** 6. Visuallize**

# In[ ]:


import graphviz
from sklearn.tree import export_graphviz
dot_tree = export_graphviz(model,
                           feature_names=feature_names,
                           class_names=["Perish", "Survived"],
                           out_file=None)
graphviz.Source(dot_tree)


# # **7. Predict** 

# In[ ]:


predictions = model.predict(X_test)
print(predictions.shape)
predictions[0:10]


# # **8. Submit**

# In[ ]:


submission = pd.read_csv("../input/gender_submission.csv")
print(submission.shape)
submission.head()


# In[ ]:


submission["Survived"] = predictions
print(submission.shape)
submission.head()


# In[ ]:


submission = pd.DataFrame({"PassengerId": submission.PassengerId, "Survived": submission.Survived})
submission.to_csv("submission.csv", index=False)


# In[ ]:




