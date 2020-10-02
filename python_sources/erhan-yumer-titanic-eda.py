#!/usr/bin/env python
# coding: utf-8

# # Introduction
# RMS Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean in the early morning hours of April 15, 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters.
# 
# <font color = blue>
# Contents
# 
# 1. [Load and Check the Data](#1)
# 1. [Variable Description](#2)
#    * [Univariate Variable Analysis](#3)
#        * [Categorical Variable](#4)
#        * [Numerical Variable](#5)
# 1. [Basic Data Analysis](#6)
# 1. [Outlier Detection](#7)
# 1. [Missing Values](#8)
#    * [Find Missing Value](#9)
#    * [Fill Missing Vaule](#10)
# 1. [Visualization](#11)
#    * [Correlation Between Sibsp -- Parch -- Age-- Fare -- Survived](#12)
#    * [Analyze SibSp -- Survived](#13)
#    * [Analyze Parch -- Survived](#14)
#    * [Analyze Pclass -- Survived](#15)
#    * [Analyze Age -- Survived](#16)
#    * [Analyze Pclass -- Age -- Survived](#17)
#    * [Analyze Embarked -- Sex -- Pclass -- Survived](#18)
#    * [Analyze Embarked -- Sex -- Fare  -- Survived](#19)
#    * [Fill Missing Age Feature](#20)
# 1. [Feature Engineering](#21)
#     * [Name -- Title](#22)
#     * [Family Size](#23)
#     * [Embarked](#24)
#     * [Ticket](#25)
#     * [PClass](#26)
#     * [Sex](#27)
#     * [Drop Passenger ID and Cabin](#28)
# 1. [Modelling](#29)
#     * [Train Test Split](#30)
#     * [Simple Logistic Regression Model](#31)
#     * [Hyperparameter Tuning -- Grid Search -- Cross Validation](#32)
#     * [Ensemble Modelling](#33)
# 1. [Prediction and Submission](#34)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid") #for available styles; use plt.style.available
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


# <a id = "1"></a><br>
# ## 1. Load and Check the Data

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# <a id = "2"></a><br>
# ## 2. Variable Description
# 1. Passenger ID
# 2. Survived (0 is dead, 1 is live)
# 3. Pclass (Class ticket: 1st class, 2nd class etc.)
# 4. Name
# 5. Sex
# 6. Age
# 7. SibSp (Number of siblings/spouses)
# 8. Parch (Number of parents/children)
# 9. Ticket
# 10. Fare (Ticket Money)
# 11. Cabin
# 12. Embarked (Port that passenger board to Titanic. S: Southampton(GB), C: Cherbourg(France) Q: Queenstown(Ireland))

# In[ ]:


train_df.info()


# * float64(2) :Fare, Age
# * int64(5) :Pclass, SibSp. Parch, PassengerID, Survival
# * object(5) :Name, Sex, Ticket, Cabin, Embarked

# <a id = '3'></a><br>
# # Univariate Variable Analysis
#   * Categorical Variable :Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, SibSp, Parch
#   * Numerical Variable :Age, PassengerID, Fare

# <a id = '4'></a><br>
# ## Categorical Variable

# In[ ]:


def bar_plot(variable):
        # get feature
        var = train_df[variable]
        # count number of categorical variable(value/sample)
        varValue = var.value_counts()
        
        #visualize
        plt.figure(figsize = (9,3))
        plt.bar(varValue.index, varValue)
        plt.xticks(varValue.index, varValue.index.values)
        plt.ylabel("Frequency")
        plt.title(variable)
        plt.show()
        print("{}: \n {}".format(variable,varValue))


# In[ ]:


category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]
for c in category1:
    bar_plot(c)


# In[ ]:


category2 = ["Cabin", "Name", "Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))


# <a id = '5'></a><br>
# ## Numerical Variable

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist.".format(variable))
    plt.show()


# In[ ]:


numericVar = ["Fare", "Age", "PassengerId"]
for n in numericVar:
    plot_hist(n)


# <a id = "6"></a><br>
# # Basic Data Analysis
# * Pclass - Survived
# * Sex - Survived
# * Parch - Survived
# * SibSp - Survived

# In[ ]:


#Pclass vs Survived
train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived", ascending = False)


# In[ ]:


#Sex vs Survived
train_df[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived", ascending = False)


# In[ ]:


#Parch vs Survived
train_df[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived", ascending = False)


# In[ ]:


#SibSp vs Survived
train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived", ascending = False)


# <a id = "7"></a><br>
# > # Outlier Detection

# In[ ]:


def detect_outliers(df, features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier Step
        outlier_step = IQR * 1.5
        # Detect outlier and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # Store indices
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return  multiple_outliers


# In[ ]:


train_df.loc[detect_outliers(train_df,["Age", "SibSp", "Parch", "Fare"])]


# In[ ]:


# drop outliers
train_df = train_df.drop(detect_outliers(train_df,["Age", "SibSp", "Parch", "Fare"]),axis = 0).reset_index(drop = True)


# <a id = "8"></a><br>
# # Missing Value
#  * Find Missing Value
#  * Fill Missing Value

# In[ ]:


train_df_len = len(train_df)
train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)


# <a id = "9"></a><br>
# ## Find Missing Value

# In[ ]:


train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# <a id = "10"></a><br>
# ## Fill Missing Vaule
# * Embarked has 2 missing values
# * Fare has only 1

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df.boxplot(column = "Fare",by = "Embarked")
plt.show()


# In[ ]:


train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df[train_df["Fare"].isnull()]


# In[ ]:


np.mean(train_df[train_df["Pclass"]==3]["Fare"])


# In[ ]:


train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))


# In[ ]:


train_df[train_df["Fare"].isnull()]


# <a id = "11"></a><br>
# # Visualization
# 

# <a id = "12"></a><br>
# ## Correlation Between Sibsp -- Parch -- Age -- Fare -- Survived

# In[ ]:


list1 = ["SibSp",  "Parch",  "Age", "Fare",  "Survived"]
sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f")
plt.show()


# * Fare feature seems to have correlation with survived feature (0.26)

# <a id = "13"></a><br>
# ## Sibsp -- Survived

# In[ ]:


g = sns.factorplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", size = 7)
g.set_ylabels("Survived Probability")
plt.show()


# * Having a lot of siblings or spouses have less chance to survive.

# <a id = "14"></a><br>
# ## Analyze Parch -- Survived

# In[ ]:


g = sns.factorplot(x = "Parch", y = "Survived", kind = "bar", data = train_df, size = 7)
g.set_ylabels("Survived Probability")
plt.show()


# * Having more parent or children is an advantage for survivalability until having 3.

# <a id = "15"></a><br>
# ## Analyze Pclass -- Survived

# In[ ]:


g = sns.factorplot(x = "Pclass", y = "Survived", kind = "bar", data = train_df, size = 7)
g.set_ylabels("Survived Probability")
plt.show()


# * Pclass and survivalability has inverse proportion between them.

# <a id = "16"></a><br>
# ## Analyze Age -- Survived

# In[ ]:


g = sns.FacetGrid(train_df, col = "Survived")
g.map(sns.distplot, "Age", bins = 25)
plt.show()


# * Age <= 30 has a high survivalability rate.
# * Large number of 20 years old did not survive.

# <a id = "17"></a><br>
# ## Analyze Pclass -- Age -- Survived

# In[ ]:


g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass", size = 2)
g.map(plt.hist, "Age", bins = 25)
g.add_legend()
plt.show()


# * Pclass will play a big role of our prediction.

# <a id = "18"></a><br>
# ## Analyze Embarked -- Sex -- Pclass -- Survived

# In[ ]:


g = sns.FacetGrid(train_df, row = "Embarked", size = 2)
g.map(sns.pointplot, "Pclass", "Survived", "Sex")
g.add_legend()
plt.show()


# * Female passenger have much better survavilability than males.

# <a id = "19"></a><br>
# ## Analyze Embarked -- Sex -- Fare -- Survived

# In[ ]:


g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived", size = 2)
g.map(sns.barplot, "Sex", "Fare")
g.add_legend()
plt.show()


# * More money paid for ticket means higher survivalability rate.

# <a id = "20"></a><br>
# ## Fill Missing Age Feature

# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:


sns.factorplot(x = "Sex", y = "Age", data = train_df, kind = "box")
plt.show()


# * Sex is not informative for age prediction. Because it seems to be same.

# In[ ]:


sns.factorplot(x = "Sex", y = "Age", hue = "Pclass", data = train_df, kind = "box")
plt.show()


# * Pclass is a good variable to use in age prediction. 1st class passengers are older than 2nd and 2nd is older than 3rd class.

# In[ ]:


sns.factorplot(x = "Parch", y = "Age", data = train_df, kind = "box")
sns.factorplot(x = "SibSp", y = "Age", data = train_df, kind = "box")
plt.show()


# * 

# In[ ]:


sns.heatmap(train_df[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(),annot = True)
plt.show()


# * Age is not correlated with sex but it is correlated with Parch, SibSp and Pclass.

# In[ ]:


index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) 
    &(train_df["Parch"] == train_df.iloc[i]["Parch"])
    & (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    age_med = train_df["Age"].median()
    
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med


# In[ ]:


train_df[train_df["Age"].isnull()]


# <a id = "21"></a><br>
# # Feature Engineering

# <a id = "22"></a><br>
# ## [Name -- Title]

# In[ ]:


train_df["Name"].head(10)


# In[ ]:


name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]


# In[ ]:


train_df["Title"].head(10)


# In[ ]:


sns.countplot(x="Title", data = train_df)
plt.xticks(rotation = 60)
plt.show()


# In[ ]:


# convert to categorical
train_df["Title"] = train_df["Title"].replace(["Mme", "Ms", "Mlle", "Lady", "the Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "others")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]


# In[ ]:


sns.countplot(x="Title", data = train_df)
plt.xticks(rotation = 60)
plt.show()


# In[ ]:


g = sns.factorplot(x = "Title", y = "Survived", data = train_df, kind = "bar")
g.set_xticklabels(["Master", "Mrs", "Mr", "Other"])
g.set_ylabels("Survival Probability")


# In[ ]:


train_df.drop(labels = ["Name"], axis = 1, inplace = True)


# In[ ]:


train_df.head()


# In[ ]:


train_df = pd.get_dummies(train_df, columns = ["Title"])
train_df.head()


# <a id = "23"></a><br>
# ## Family Size

# In[ ]:


train_df.head()


# In[ ]:


train_df["FSize"] = train_df["SibSp"] + train_df["Parch"] + 1


# In[ ]:


train_df.head()


# In[ ]:


g = sns.factorplot(x = "FSize", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Survival")
plt.show()


# In[ ]:


train_df["family_size"] = [1 if i<5 else 0 for i in train_df["FSize"]]


# In[ ]:


train_df.head(10)


# In[ ]:


sns.countplot(x = "family_size", data = train_df)


# In[ ]:


g = sns.factorplot(x = "family_size", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Survival")
plt.show() 


# Small families like have 2 or 3 members has more chance to survive rather than bigger families like 5 or 6 members.

# In[ ]:


train_df = pd.get_dummies(train_df, columns = ["family_size"])
train_df.head()


# <a id = "24"></a><br>
# ## Embarked

# In[ ]:


train_df["Embarked"].head()


# In[ ]:


sns.countplot(x = "Embarked", data = train_df)
plt.show()


# In[ ]:


train_df = pd.get_dummies(train_df, columns = ["Embarked"])
train_df.head()


# <a id = "25"></a><br>
# ## Ticket

# In[ ]:


train_df["Ticket"].head(20)


# In[ ]:


tickets = []
for i in list(train_df.Ticket):
    if i.isdigit():
        tickets.append(i.replace(".", " ").replace("/", " ").strip().split(" ")[0])
    else:
        tickets.append("x")
train_df["Ticket"] = tickets


# In[ ]:


train_df["Ticket"].head(20)


# In[ ]:


train_df.head()


# In[ ]:


train_df = pd.get_dummies(train_df, columns = ["Ticket"], prefix = "T")
train_df.head(10)


# <a id = "26"></a><br>
# ## PClass

# In[ ]:


sns.countplot(x = "Pclass", data = train_df)


# In[ ]:


train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df, columns = ["Pclass"])
train_df.head()


# <a id = "27"></a><br>
# ## Sex

# In[ ]:


train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df, columns = ["Sex"])
train_df.head()


# <a id = "28"></a><br>
# ## Drop Passenger ID and Cabin

# In[ ]:


train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)


# In[ ]:


train_df.columns


# <a id = "29"></a><br>
# # Modelling

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# <a id = "30"></a><br>
# ## Train Test Split

# In[ ]:


train_df_len


# In[ ]:


test = train_df[train_df_len:]
test.drop(labels = ["Survived"], axis = 1, inplace = True)


# In[ ]:


test.head()


# In[ ]:


train = train_df[:train_df_len]
X_train = train.drop(labels = "Survived", axis = 1)
y_train = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)
print("X_train", len(X_train))
print("X_test", len(X_test))
print("y_train", len(y_train))
print("y_test", len(y_test))
print("test", len(test))


# <a id = "31"></a><br>
# ## Simple Logistic Regression Model

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_log_train = round(logreg.score(X_train, y_train) * 100, 2)
acc_log_test = round(logreg.score(X_test, y_test) * 100, 2)
print("Training Accuracy :% {}".format(acc_log_train))
print("Testing Accuracy :% {}".format(acc_log_test))


# <a id = "32"></a><br>
# ## Hyperparameter Tuning -- Grid Search -- Cross Validation
# * Comparing 5 machine learning classifier and evaluate mean accuracy of each of them by stratified cross validation.
# 
# 
# * Decision Tree
# * SVM
# * Random Forest
# * KNN
# * Logistic Regression

# In[ ]:


random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

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
    clf.fit(X_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


# In[ ]:


cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")


# <a id = "33"></a><br>
# ## Ensemble Modelling

# In[ ]:


votingC = VotingClassifier(estimators = [("dt", best_estimators[0]),
                                        ("rfc", best_estimators[2]),
                                        ("lr", best_estimators[3])],
                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)
print(accuracy_score(votingC.predict(X_test), y_test))


# <a id = "34"></a><br>
# # Prediction and Submission

# In[ ]:


test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)
results = pd.concat([test_PassengerId, test_survived], axis = 1)
results.to_csv("titanic.csv", index = False)

