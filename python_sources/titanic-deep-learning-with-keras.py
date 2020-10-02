#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Welcome to my first Deep Learning Kernel!
# 
# In this notebook we'll do some easy data visualizations and feature engineering before building a deep neural network to predict if passengers of the titanic survived or died. If you have any suggestions on how to improve this notebook please let me know!
# 
# ---
# 
# # History
# RMS Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters. RMS Titanic was the largest ship afloat at the time she entered service and was the second of three Olympic-class ocean liners operated by the White Star Line. She was built by the Harland and Wolff shipyard in Belfast.
# 
# ---
# 
# # Goal
# It is our job to predict if a passenger survived the sinking of the Titanic or not. For each person in the test set, we must predict a 0 or 1 value for the variable. Our score is the percentage of passengers we correctly predict.

# ## 1. Import modules

# In[ ]:


import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from matplotlib.pyplot import plot

style.use("seaborn-whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.regularizers import l1, l2

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## 2. Import data

# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# ## 3. Exploratory Data Analysis [EDA]

# ### 3.0 Data Dictionary
# 
# * survival - Survival	: 0 = No, 1 = Yes
# * pclass - Ticket class :	1 = 1st, 2 = 2nd, 3 = 3rd
# * sex - Gender	
# * Age - Age in years	
# * sibsp - # of siblings / spouses aboard the Titanic	
# * parch - # of parents / children aboard the Titanic	
# * ticket - Ticket number	
# * fare - Passenger fare	
# * cabin - Cabin number	
# * embarked - Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# ### 3.1 Structure and NaN values

# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


print(train.info())


# In[ ]:


print(test.info())


# In[ ]:


plt.figure(figsize = (12,8))
sns.heatmap(train.isnull(), cbar = False, cmap = "Blues")
plt.title("Missing Values (train)", fontsize = 14)
plt.show()


# In[ ]:


plt.figure(figsize = (12,8))
sns.heatmap(test.isnull(), cbar = False, cmap = "Blues")
plt.title("Missing Values (test)", fontsize = 14)
plt.show()


# ### 3.2 Basic Visualizations

# #### 3.2.1 Target Variable

# In[ ]:


plt.figure(figsize = (14,3))
sns.countplot(data = train, y = "Survived", palette = ["Salmon", "Lightblue"])
plt.title("Survived", fontsize = 14)
plt.show()


# #### 3.2.2 Class

# In[ ]:


plt.figure(figsize = (14,5))
sns.countplot(data = train, y = "Pclass", hue = "Survived", palette = ["Salmon", "Lightblue"])
plt.title("Class", fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., prop = {"size": 13})
plt.show()


# #### 3.2.3 Gender

# In[ ]:


plt.figure(figsize = (14,4))
sns.countplot(data = train, y = "Sex", hue = "Survived", palette = ["Salmon", "Lightblue"])
plt.title("Gender", fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., prop = {"size": 13})
plt.show()


# #### 3.2.4 Embarked

# In[ ]:


plt.figure(figsize = (14,5))
sns.countplot(data = train, y = "Embarked", hue = "Survived", palette = ["Salmon", "Lightblue"])
plt.title("Embarked", fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., prop = {"size": 13})
plt.show()


# #### 3.2.5 Fare

# In[ ]:


plt.figure(figsize = (14,8))
sns.kdeplot(train.Fare, shade = True, color = "Salmon")
plt.title("Fare Distribution", fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., prop = {"size": 13})
plt.show()


# #### 3.2.6 Age

# In[ ]:


plt.figure(figsize = (14,8))
sns.kdeplot(train.Age, shade = True, color = "Salmon")
plt.title("Age Distribution", fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., prop = {"size": 13})
plt.show()


# ### 3.3 Feature Engineering

# In[ ]:


train.head(3)


# #### 3.3.1 Family Size

# In[ ]:


train["FamilySize"] = (train["SibSp"] + train["Parch"])
test["FamilySize"] = (test["SibSp"] + test["Parch"])


# In[ ]:


plt.figure(figsize = (14,9))
sns.countplot(data = train, y = "FamilySize", hue = "Survived", palette = ["Salmon", "Lightblue"])
plt.title("Family Size", fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., prop = {"size": 13})
plt.show()


# #### 3.3.2 Name Titles

# In[ ]:


train["NameTitle"] = train["Name"].str.split(", ", expand = True)[1].str.split(". ", expand = True)[0]
test["NameTitle"] = test["Name"].str.split(", ", expand = True)[1].str.split(". ", expand = True)[0]

print(train["NameTitle"].unique())
print("-----------------------" * 3)
print(test["NameTitle"].unique())


# In[ ]:


min_titles = (train["NameTitle"].value_counts() < 10)
train["NameTitle"] = train["NameTitle"].apply(lambda x: "Misc" if min_titles.loc[x] == True else x)
print(train["NameTitle"].unique())


# In[ ]:


min_titles = (test["NameTitle"].value_counts() < 10)
test["NameTitle"] = test["NameTitle"].apply(lambda x: "Misc" if min_titles.loc[x] == True else x)
print(test["NameTitle"].unique())


# In[ ]:


plt.figure(figsize = (14,6))
sns.countplot(data = train, y = "NameTitle", hue = "Survived", palette = ["Salmon", "Lightblue"])
plt.title("Name Titles", fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., prop = {"size": 13})
plt.show()


# #### 3.3.3 Age Groups

# In[ ]:


ages_train = train.groupby("NameTitle")["Age"].mean().to_dict()
train.loc[train["Age"].isnull(), "Age"] = train.loc[train["Age"].isnull(), "NameTitle"].map(ages_train)

ages_test = test.groupby("NameTitle")["Age"].mean().to_dict()
test.loc[test["Age"].isnull(), "Age"] = test.loc[test["Age"].isnull(), "NameTitle"].map(ages_test)


# In[ ]:


train["AgeGroup"] = ""
train.loc[train["Age"] <= 14, "AgeGroup"] = "A"
train.loc[(train["Age"] > 14) & (train["Age"] <= 21), "AgeGroup"] = "B"
train.loc[(train["Age"] > 21) & (train["Age"] <= 40), "AgeGroup"] = "C"
train.loc[(train["Age"] > 40) & (train["Age"] <= 65), "AgeGroup"] = "D"
train.loc[train["Age"] > 65, "AgeGroup"] = "E"

test["AgeGroup"] = ""
test.loc[test["Age"] <= 14, "AgeGroup"] = "A"
test.loc[(test["Age"] > 14) & (test["Age"] <= 21), "AgeGroup"] = "B"
test.loc[(test["Age"] > 21) & (test["Age"] <= 40), "AgeGroup"] = "C"
test.loc[(test["Age"] > 40) & (test["Age"] <= 65), "AgeGroup"] = "D"
test.loc[test["Age"] > 65, "AgeGroup"] = "E"


# In[ ]:


plt.figure(figsize = (14,6))
sns.countplot(data = train, y = "AgeGroup", hue = "Survived", palette = ["Salmon", "Lightblue"])
plt.title("Age", fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., prop = {"size": 13})
plt.show()


# #### 3.3.4 Fare Groups

# In[ ]:


train["FareGroup"] = ""
train.loc[train["Fare"] <= 100, "FareGroup"] = "D"
train.loc[(train["Fare"] > 100) & (train["Fare"] <= 200), "FareGroup"] = "C"
train.loc[(train["Fare"] > 200) & (train["Fare"] <= 300), "FareGroup"] = "B"
train.loc[train["Fare"] > 300, "FareGroup"] = "A"

test["Fare"].fillna(test["Fare"].mean(), inplace = True)
test["FareGroup"] = ""
test.loc[test["Fare"] <= 100, "FareGroup"] = "D"
test.loc[(test["Fare"] > 100) & (test["Fare"] <= 200), "FareGroup"] = "C"
test.loc[(test["Fare"] > 200) & (test["Fare"] <= 300), "FareGroup"] = "B"
test.loc[test["Fare"] > 300, "FareGroup"] = "A"


# In[ ]:


plt.figure(figsize = (14,5))
sns.countplot(data = train, y = "FareGroup", hue = "Survived", palette = ["Salmon", "Lightblue"])
plt.title("Fare", fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., prop = {"size": 13})
plt.show()


# #### 3.3.5 Cabins / Deck

# In[ ]:


train["Cabin"].fillna("Missing", inplace = True)
test["Cabin"].fillna("Missing", inplace = True)


# In[ ]:


train["Deck"] = ""
train.loc[train["Cabin"] == "Missing", "Deck"] = "Missing"
train.loc[train["Cabin"] != "Missing", "Deck"] = train["Cabin"].apply(lambda x: list(x)[0])

test["Deck"] = ""
test.loc[test["Cabin"] == "Missing", "Deck"] = "Missing"
test.loc[test["Cabin"] != "Missing", "Deck"] = test["Cabin"].apply(lambda x: list(x)[0])


# In[ ]:


train["HasCabin"] = ""
train.loc[train["Cabin"] == "Missing", "HasCabin"] = 0
train.loc[train["Cabin"] != "Missing", "HasCabin"] = 1

test["HasCabin"] = ""
test.loc[test["Cabin"] == "Missing", "HasCabin"] = 0
test.loc[test["Cabin"] != "Missing", "HasCabin"] = 1


# In[ ]:


plt.figure(figsize = (14,9))
sns.countplot(data = train, y = "Deck", hue = "Survived", palette = ["Salmon", "Lightblue"])
plt.title("Deck", fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., prop = {"size": 13})
plt.show()


# In[ ]:


plt.figure(figsize = (14,4))
sns.countplot(data = train, y = "HasCabin", hue = "Survived", palette = ["Salmon", "Lightblue"])
plt.title("HasCabin", fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., prop = {"size": 13})
plt.show()


# #### 3.3.6 Alone

# In[ ]:


train["Alone"] = 0
train.loc[train["FamilySize"] == 0, "Alone"] = 1
train.loc[train["FamilySize"] != 0, "Alone"] = 0

test["Alone"] = 0
test.loc[test["FamilySize"] == 0, "Alone"] = 1
test.loc[test["FamilySize"] != 0, "Alone"] = 0


# In[ ]:


plt.figure(figsize = (14,4))
sns.countplot(data = train, y = "Alone", hue = "Survived", palette = ["Salmon", "Lightblue"])
plt.title("Alone", fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., prop = {"size": 13})
plt.show()


# #### 3.3.7 Class

# In[ ]:


train["3rdClass"] = 0
train.loc[train["Pclass"] == 3, "3rdClass"] = 1
train.loc[train["Pclass"] != 3, "3rdClass"] = 0

test["3rdClass"] = 0
test.loc[test["Pclass"] == 3, "3rdClass"] = 1
test.loc[test["Pclass"] != 3, "3rdClass"] = 0


# In[ ]:


plt.figure(figsize = (14,4))
sns.countplot(data = train, y = "3rdClass", hue = "Survived", palette = ["Salmon", "Lightblue"])
plt.title("Class", fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., prop = {"size": 13})
plt.show()


# #### 3.3.8 Data overview

# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


train["Embarked"].fillna(train["Embarked"].mode()[0], inplace = True)
train.isnull().sum().sort_values(ascending = False).head(5)


# In[ ]:


test.isnull().sum().sort_values(ascending = False).head(5)


# In[ ]:


train.info()


# In[ ]:


test.info()


# ### 3.4 Correlation

# In[ ]:


corr = train.corr()
plt.figure(figsize = (10,8))
sns.heatmap(corr, cmap = "Blues", linewidth = 4, linecolor = "white")
plt.title("Correlation", fontsize = 14)
plt.show()


# In[ ]:


upper = corr.where(np.triu(np.ones(corr.shape), k = 1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
print(to_drop)


# ### 3.5 Preprocessing

# In[ ]:


train.drop(["Name", "Cabin", "Ticket", "Fare"], axis = 1, inplace = True)
test.drop(["Name", "Cabin", "Ticket", "Fare"], axis = 1, inplace = True)


# In[ ]:


train["Pclass"] = train["Pclass"].astype("object")
test["Pclass"] = test["Pclass"].astype("object")


# In[ ]:


train = pd.get_dummies(train)
test = pd.get_dummies(test)


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.head(3)


# In[ ]:


drop_values = ["Pclass_3", "Sex_female", "Embarked_S", "NameTitle_Misc", 
               "AgeGroup_E", "FareGroup_D", "HasCabin_1"]

train.drop(drop_values, axis = 1, inplace = True)
test.drop(drop_values, axis = 1, inplace = True)


# In[ ]:


X = train.drop(["Survived", "PassengerId", "Deck_T"], axis = 1)
y = train["Survived"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.3)


# ## 4. Neural Network
# 
# 

# In[ ]:


model = Sequential()

model.add(Dense(512, activation = "relu", kernel_regularizer = l2(0.0001), input_shape = (31,)))
model.add(BatchNormalization())
model.add(Dense(512, activation = "relu", kernel_regularizer = l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(256, activation = "relu", kernel_regularizer = l2(0.0001)))
model.add(BatchNormalization())
model.add(Dense(256, activation = "relu", kernel_regularizer = l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation = "relu", kernel_regularizer = l2(0.0001)))
model.add(BatchNormalization())
model.add(Dense(128, activation = "relu", kernel_regularizer = l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(64, activation = "relu", kernel_regularizer = l2(0.0001)))
model.add(BatchNormalization())
model.add(Dense(64, activation = "relu", kernel_regularizer = l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = 0.001, 
                                                      centered = True, 
                                                      momentum = 0), 
              loss = "binary_crossentropy", 
              metrics = ["accuracy"])

history = model.fit(X_train, 
                    y_train, 
                    validation_split = 0.3,
                    epochs = 90, 
                    batch_size = 8, 
                    shuffle = True,
                    verbose = 0)

print("Test score :" + str(model.evaluate(X_test, y_test)))
print("")
print("Train score :" + str(model.evaluate(X_train, y_train)))
print("")

print(model.summary())


# ### 4.1 Confusion Matrix

# In[ ]:


y_predicted = model.predict(X_test)
y_predicted = (y_predicted.ravel() > 0.5).astype(int)


# In[ ]:


confusion_matrix(y_test, y_predicted)


# ### 4.2 Accuracy and Loss

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (20,8))
ax[0].plot(history.history["accuracy"], label = "Train", color = "Lightblue", linewidth = 3)
ax[0].plot(history.history["val_accuracy"], label = "Test", color = "Salmon", linewidth = 3)
ax[1].plot(history.history["loss"], label = "Train", color = "Lightblue", linewidth = 3)
ax[1].plot(history.history["val_loss"], label = "Test", color = "Salmon", linewidth = 3)
ax[0].set_title("Accuracy", fontsize = 14)
ax[1].set_title("Loss", fontsize = 14)
ax[0].set_xlabel("Epoch")
ax[1].set_xlabel("Epoch")
ax[0].set_ylabel("Accuracy")
ax[1].set_ylabel("Loss")
ax[0].legend()
ax[1].legend()
plt.show()


# ## 5. Submission

# In[ ]:


y_predicted_sub = model.predict(test.drop(["PassengerId"], axis = 1))
y_predicted_sub = (y_predicted_sub.ravel() > 0.5).astype(int)

submission = pd.DataFrame({"PassengerId" : test.PassengerId, "Survived" : y_predicted_sub})
submission.to_csv("submission.csv", index = False)

submission.head()

