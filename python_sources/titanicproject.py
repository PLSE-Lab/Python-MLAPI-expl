#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
#     In this kernel I will analyse the data, visualise it and then predict some values.
# 
# <font color = "purple">
# Content:
# 
# 1. [Load and Check Data](#1)
# 1. [Variable Description](#2)
#     * [Univariate Variable Analysis](#3)
#         * [Categorical Variable Analysis](#4)
#         * [Numerical Variable Analysis](#5)
# 1. [Basic Data Analysis](#6)
# 1. [Outlier Detection](#7)
# 1. [Missing Values](#8)
#     * [Finding Missing Value](#9)
#     * [Filling Missing Value](#10)
# 1. [Visualisation](#11)
#     * [Correlation Between Sibsp -- Parch -- Age -- Fare -- Survived](#12)
#     * [Sibsp -- Survived](#13)
#     * [Parch -- Survived](#14)
#     * [Pclass-- Survived](#15)
#     * [Age -- Survived](#16)
#     * [Pclass Age -- Survived](#17)
#     * [Embarked -- Sex -- Pclass -- Survived](#18)
#     * [Embarked -- Sex -- Fare -- Survived](#19)
#     * [Fill Missing: Age Feature](#20)
# 1. [Feature Engineering](#21)
#     * [Name -- Title](#22)
#     * [Family Size](#23)
#     * [Embarked](#24)
#     * [Ticket](#25)
#     * [Pclass](#26)
#     * [Sex](#27)
#     * [Drop Passenger ID and Cabin](#28)
# 1. [Modeling](#29)
#     * [Train - Test Split](#30)
#     * [Simple Logistic Regression](#31)
#     * [Hyperparameter Tuning -- Grid Search -- Cross Validation](#32)
#     * [Ensemble Modeling](#33)
#     * [Prediction and Submission](#34)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import seaborn as sns
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id = "1"></a>
# # Load and Check Data
# 
# 

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassenngerId = test_df["PassengerId"]


# In[ ]:


# kinda unnecessary when .columns is used
train_df.head()


# In[ ]:


train_df.columns


# In[ ]:


test_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


# <a id = "2"></a>
# 
# # Variable Description
# 1. PassengerId: Unique id number for each passenger
# 2. Survived: Displays 1 if the passenger survived, 0 if the passenger died
# 3. Pclass: Indicates passenger's class
# 4. Name: Passenger's name
# 5. Sex: Passenger's sex, female or male
# 6. Age: AGE
# 7. SibSp: Number of siblings or spouse
# 8. Parch: Number of parents or children
# 9. Ticket: Ticket number
# 10. Fare: Amount of money paid for the ticket
# 11. Cabin: The cabin category passengers had stayed
# 12. Embarked: Port where passengers embarked, C for Cherbourg, Q for Queenstown and S for Southampton

# In[ ]:


train_df.info()


# * float64(2): Fare and Age
# * int64(5): PassengerId, Survived, Pclass, SipSp and Parch
# * object(5): Name, Sex, Ticket, Cabin and Embarked

# <a id = "3"></a>
# # Univariate Variable Analysis
# * Categorical Variables: Survived, Sex, Pclass, Embarked, Name, Ticket, Sipsp and Parch
# * Numerical Variables: Age, PassengerId, Fare

# <a id = "4"></a>
# ## Categorical Variable Analysis

# In[ ]:


def bar_plot(variable):
    """
        input = variable, e.g. sex
        output = bar plot & value count
    """
    var = train_df[variable]
    varValue = var.value_counts()
    
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable, varValue))


# In[ ]:


cat1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]
for c in cat1:
    bar_plot(c)


# In[ ]:


cat2 = ["Cabin", "Name", "Ticket"]
for c in cat2:
    print("{} \n".format(train_df[c].value_counts()))


# <a id = "5"></a>
# ## Numerical Variable Analysis

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with histogram" .format(variable))
    plt.show


# In[ ]:


cat3 = ["Fare", "Age", "PassengerId"]
for c in cat3:
    plot_hist(c)


# <a id = 6></a>
# # Basic Data Analysis
# 
# * Pclass - Survived
# * Sex - Survived
# * SipSp - Survived
# * Parch - Survived

# In[ ]:


train_df[["Pclass", "Survived"]]


# In[ ]:


#pclass vs survived

train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# In[ ]:


#sex and survived

train_df[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# In[ ]:


#sipsp and survived

train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# In[ ]:


#parch vs survived

train_df[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# <a id = 7></a>
# # Outlier Detection

# In[ ]:


def detectOutlier(df, features):
    outlierIndices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c], 25)
        # 3rd quartile
        Q3 = np.percentile(df[c], 75)
        #IQR
        IQR = Q3 - Q1
        #outlier step
        outlierStep = IQR * 1.5
        #detect outlier and indices
        outlierListCol = df[(df[c] < Q1 - outlierStep) | (df[c] > Q3 + outlierStep)].index
        #store indices
        outlierIndices.extend(outlierListCol)
        
    outlierIndices = Counter(outlierIndices) #counter shows how many of a single element exists
    multipleOutliers = list(i for i, v in outlierIndices.items() if v > 2)
    
    return multipleOutliers


# In[ ]:


train_df.loc[detectOutlier(train_df, ["Age", "SibSp", "Parch", "Fare"])]


# In[ ]:


#dropping outliers
train_df = train_df.drop(detectOutlier(train_df, ["Age", "SibSp", "Parch", "Fare"]), axis = 0).reset_index(drop = True)


# <a id = 8></a>
# # Missing Values

# In[ ]:


train_df_len = len(train_df)
train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)


# In[ ]:





# In[ ]:





# <a id = 9></a>
# ## Finding Missing Values

# In[ ]:


train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# In[ ]:





# <a id = 10></a>
# ## Filling Missing Values
# * Embarked: 2
# * Fare: 1

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df.boxplot(column = "Fare", by = "Embarked")
plt.show()


# Most likely: C port

# In[ ]:


train_df["Embarked"] = train_df["Embarked"].fillna("C")


# In[ ]:


train_df[train_df["Fare"].isnull()]


# In[ ]:


train_df.boxplot(column = "Fare", by = "Pclass")
plt.show()


# In[ ]:


np.mean(train_df[train_df["Pclass"] == 3]["Fare"])


# In[ ]:


train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))


# <a id = "11"></a>
# # Visualisation
# 
# 

# <a id = 12></a>
# # Correlations

# In[ ]:


listo = ["SibSp", "Parch", "Age", "Fare", "Survived"]

sns.heatmap(train_df[listo].corr(), annot = True, fmt = ".2f")
plt.show()


# <a id = "13"></a>
# ## SipSp -- Survived

# In[ ]:


g = sns.factorplot(x= "SibSp", y = "Survived", data = train_df, kind = "bar", size = 6)
g.set_ylabels("survived prob sibsp")
plt.show()


# <a id = 14></a>
# ## Parch - Survived

# In[ ]:


g = sns.factorplot(x = "Parch", y = "Survived", data = train_df, kind = "bar", size = 9)
g.set_ylabels("survived prob")
plt.show()


# * small families are more likely to survive.

# <a id = 15></a>
# 
# ## pclass - survived

# In[ ]:


g = sns.factorplot(x = "Pclass", y = "Survived", data = train_df, kind = "bar", size = 5)
plt.show()


# * first class is more likely to survive

# <a id = 16></a>
# 
# ## Age - Survived

# In[ ]:


g = sns.FacetGrid(train_df, col = "Survived")
g.map(sns.distplot, "Age", bins = 25)
plt.show()


# * most of the survivors are children
# * most of the dead people are between 20-40 years
# * this can be used to fill the age data

# <a id = 17></a>
# 
# ## Pclass - Age - Survived

# In[ ]:


g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass")
g.map(plt.hist, "Age", bins = 25)
plt.show()


# <a id = 18></a>
# 
# ## Embarked - Pclass - Sex - Survived

# In[ ]:


g = sns.FacetGrid(train_df, row= "Embarked")
g.map(sns.pointplot, "Pclass", "Survived", "Sex")
g.add_legend()
plt.show()


# <a id = 19></a>
# ## Embarked - Sex - Fare - Survived

# In[ ]:


g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived")
g.map(sns.barplot, "Sex", "Fare")
g.add_legend()
plt.show()


# <a id = 20></a>
# 
# # Filling Age Value

# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:


sns.factorplot(x = "Sex", y = "Age", data = train_df, kind = "box")
plt.show()


# In[ ]:


sns.factorplot(x = "Sex", y = "Age", hue= "Pclass", data = train_df, kind = "box")
plt.show()


# In[ ]:


sns.factorplot(x = "Parch", y = "Age", data = train_df, kind = "box")
sns.factorplot(x = "SibSp", y = "Age", data = train_df, kind = "box")
plt.show()


# In[ ]:


train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]


# In[ ]:


sns.heatmap(train_df[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot = True)


# In[ ]:


indexes = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in indexes:
    ageP = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & (train_df["Parch"] == train_df.iloc[i]["Parch"]) & (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    ageM = train_df["Age"].median()
    if not np.isnan(ageP):
        train_df["Age"].iloc[i] = ageP
    else: 
        train_df["Age"].iloc[i] = ageM


# In[ ]:


train_df[train_df["Age"].isnull()]


# <a id = 21></a>
# # Feature Engineering

# <a id = 22></a>
# ## Name -- Title

# In[ ]:


train_df["Name"].head(10)


# In[ ]:


name = train_df["Name"]
train_df["title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]


# In[ ]:


sns.countplot(x = "title", data = train_df)
plt.xticks(rotation = 60)
plt.show()


# In[ ]:


train_df["title"] = train_df["title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"], "other")
train_df["title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["title"]]
train_df["title"].head(20)


# In[ ]:


sns.countplot(x = "title", data = train_df)
plt.xticks(rotation = 60)
plt.show()


# In[ ]:


g = sns.factorplot(x = "title", y = "Survived", data = train_df, kind = "bar")
g.set_xticklabels(["Master", "Miss-Mrs", "Mr", "Other"])
g.set_ylabels("Survival")
plt.show()


# In[ ]:


train_df.drop(columns = ["Name"], inplace = True)


# In[ ]:


train_df = pd.get_dummies(train_df, columns = ["title"])
train_df.head()


# <a id = 23></a>
# ## Family Size

# In[ ]:


train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1


# In[ ]:


g = sns.factorplot(x = "Fsize", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Survival")
plt.show()


# In[ ]:


train_df["famsize"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]


# In[ ]:


train_df.head()


# In[ ]:


sns.countplot(x = "famsize", data = train_df)
plt.show()


# In[ ]:


g = sns.factorplot(x = "famsize", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Survival")
plt.show()


# *Small families have more chanceto survive.*

# In[ ]:


train_df = pd.get_dummies(train_df, columns = ["famsize"])


# <a id = 24></a>
# ## Embarked

# In[ ]:


sns.countplot(x = "Embarked", data = train_df)
plt.show()


# In[ ]:


train_df = pd.get_dummies(train_df, columns=["Embarked"])
train_df.head()


# <a id = 25></a>
# ## Ticket

# In[ ]:


train_df["Ticket"].head(30)


# In[ ]:


tickets = []
for i in (train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".", " ").replace("/", " ").strip().split(" ")[0])
    else:
        tickets.append("x")
train_df["Ticket"] = tickets


# In[ ]:


train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix = "T")
train_df.head(10)


# <a id = 26></a>
# ## PClass

# In[ ]:


sns.countplot(x = "Pclass", data = train_df)
plt.show()


# In[ ]:


train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df, columns= ["Pclass"])
train_df.head()


# <a id = 27></a>
# ## Sex

# In[ ]:


train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df, columns=["Sex"])
train_df.head()


# <a id = 28></a>
# ## Drop Passenger ID and Cabin

# In[ ]:


train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)


# <a id = 29></a>
# # Modeling

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# <a id = 30></a>
# ## Train Test Split

# In[ ]:


train_df_len


# In[ ]:


test = train_df[train_df_len:]
test.drop(labels = ["Survived"],axis = 1, inplace = True)


# In[ ]:


test.head()


# In[ ]:


train = train_df[:train_df_len]
x_train = train.drop(labels = "Survived", axis = 1)
y_train = train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.33, random_state = 42)
print("X_train",len(x_train))
print("X_test",len(x_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(test))


# <a id = 31></a>
# ## Simple Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)

acctrain = round(logreg.score(x_train, y_train) * 100, 2)
acctest = round(logreg.score(x_test, y_test) * 100, 2)
print("train data accuracy: %", acctrain)
print("test data accuracy: %", acctest)


# <a id = 32></a>
# ## Hyperparameter Tuning -- Grid Search -- Cross Validation

# We will compare 5 ml classifier and evaluate mean accuracy of each of them by stratified cross validation.
# 
# * Decision Tree
# * SVM
# * Random Forest
# * KNN
# * Logistic Regression

# In[ ]:


random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state), SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state), 
             KNeighborsClassifier()]


# In[ ]:


dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]


# In[ ]:


cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


# In[ ]:


cv_results = pd.DataFrame({"Cross Validation Means": cv_result, "ML Models": ["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression", "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")


# <a id = 33></a>
# ## Ensemble Modeling

# In[ ]:


votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)
votingC = votingC.fit(x_train, y_train)
print(accuracy_score(votingC.predict(x_test),y_test))


# <a id = 34></a>
# ## Prediction and Submission

# In[ ]:


test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)
results = pd.concat([test_PassenngerId, test_survived],axis = 1)
results.to_csv("titanic.csv", index = False)

