#!/usr/bin/env python
# coding: utf-8

# # Introduction
# RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters.
# 
# <font color="green">
#     Content:
#     
# 1.     [Load and Check Data](#1)
# 1. [Variable Description](#2)
#     * [Univariate Variable Analysis](#3)
#         * [Categorical Variable Analysis](#4)
#         * [Numerical Variable Analysis](#5)
# 1. [Basic Data Analysis](#6)
# 1. [Outlier Detection](#7)
# 1. [Missing Value](#8)
#     * [Find Missing Value](#9)
#     * [Fill Missing Value](#10)
# 1. [Visualization](#11)
#     * [Correlation Between SibSp - Age - Fare - Parch - Survived](#12)
#     * [SibSp - Survived](#13)
#     * [Parch - Survived](#14)
#     * [Pclass - Survived](#15)
#     * [Age - Survived](#16)
#     * [Pclass - Survived - Age](#17)
#     * [Embarked - Sex - Pclass - Survived](#18)
#     * [Embarked - Sex - Fare - Survived](#19)
#     * [Fill Missing : Age feature](#20)
# 1. [Feature Engineering](#21)
#     * [Name - Title](#22)
#     * [Family Size](#23)
#     * [Embarked](#24)
#     * [Ticket](#25)
#     * [Pclass](#26)
#     * [Sex](#27)
#     * [Drop PassengerId and Cabin](#28)
# 1. [Modeling](#29)
#     * [Train Test Split](#30)
#     * [Simple Logistic Regression](#31)
#     * [Hyperparameter Tuning - Grid Search - Cross Validation](#32)
#     * [Ensemble Modeling](#33)
#     * [Prediction and Submission](#34)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-whitegrid")

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


# # Load and Check Data <a id=1></a>

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


# # Variable Description <a id=2></a>
# 
# 1. PassengerId : Unique ID for each passenger
# 2. Survived : Did the passenger survive? Yes (1), No (0)
# 3. Pclass : Passenger class
# 4. Name : Name of passenger
# 5. Sex : Sex of passenger
# 6. Age : Age of passenger
# 7. SibSp : Number of passengers siblings/spouses on the ship
# 8. Parch : Number of passengers parents/children on the ship
# 9. Ticket : Ticket number
# 10. Fare : Amount of money spent on ticket
# 11. Cabin : Cabin category
# 12. Embarked : The port that passenger was embarked (C = Cherbourg,Q = Queenstown,S = Southampton)

# In[ ]:


train_df.info()


# * float64(2) : Age and Fare
# * int64(5) : PassengerId, Survived, Pclass, SibSp and Parch
# * object(5) : Name, Sex, Ticket, Cabin and Embarked

# # Univariate Variable Analysis <a id=3></a>
# * Categorical Variable: Variables which has over 2 categories. (Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, SibSp and Parch)
# * Numerical Variable: Numerical variables. (Fare, Age and PassengerId)

# ## Categorical Variable Analysis <a id=4></a>

# In[ ]:


def barplot(variable):
    """
    input : variable example: "Sex"
    output : barplot & value count
        
    """
    # get feature
    var = train_df[variable]
    # count categories
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Count")
    plt.title(variable)
    plt.show()
    print(f"{variable}: \n {varValue}")


# In[ ]:


category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]
for i in category1:
    barplot(i)


# In[ ]:


category2 = ["Cabin","Name","Ticket"]
for i in category2:
    print(f"{train_df[i].value_counts()} \n")


# ## Numerical Variable Analysis <a id=5></a>

# In[ ]:


def plothist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable])
    plt.xlabel(variable)
    plt.ylabel("Sample count")
    plt.title(f"{variable} distribution with histogram")
    plt.show()


# In[ ]:


numericVar = ["Fare","Age","PassengerId"]
for i in numericVar:
    plothist(i)


# # Basic Data Analysis <a id=6></a>
# * Pclass - Survived
# * Sex - Survived
# * SibSp - Survived
# * Parch - Survived

# In[ ]:


# Pclass vs Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index = False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


# Sex vs Survived
train_df[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


# SibSp vs Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index = False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


# Parch vs Survived
train_df[["Parch","Survived"]].groupby(["Parch"],as_index = False).mean().sort_values(by="Survived",ascending=False)


# # Outlier Detection <a id=7></a>

# In[ ]:


def detect_outliers(df,features):
    outlier_indices = []
    for c in features:
        # 1st quartile
        q1 = np.percentile(df[c],25)
        # 3rd quartile
        q3 = np.percentile(df[c],75)
        # IQR
        IQR = q3 - q1
        # Outlier step
        outlier_step = IQR * 1.5
        # Detect outlier and their indices
        outlier_list_col = df[(df[c] < q1 - outlier_step) | (df[c] > q3 + outlier_step)].index
        # Store indices
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    return multiple_outliers


# In[ ]:


train_df.loc[detect_outliers(train_df,["Age","Parch","SibSp","Fare"])]


# In[ ]:


train_df = train_df.drop(detect_outliers(train_df,["Age","Parch","SibSp","Fare"]),axis=0).reset_index(drop = True)


# # Missing Value <a id=8></a>
#    * Find missing value
#    * Fill missing value

# In[ ]:


train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop=True)


# ## Find missing value <a id=9></a>

# In[ ]:


train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# ## Fill missing value <a id=10></a>
# * Embarked has 2 missing value.
# * Fare has 6 missing value.

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df.boxplot(column="Fare",by="Embarked")
plt.show()
plt.show()


# In[ ]:


train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df[train_df["Fare"].isnull()]


# In[ ]:


train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))


# In[ ]:


train_df[train_df["Fare"].isnull()]


# # Visualization <a id=11></a>

# ## Correlation Between SibSp - Age - Fare - Parch - Survived <a id=12></a>

# In[ ]:


list1 = ["SibSp","Age","Fare","Parch","Survived"]
sns.heatmap(train_df[list1].corr(),annot=True,fmt=".2f")
plt.show()


# Fare feature seems to have correlation with Survived feature. (0.26)

# ## SibSp - Survived  <a id=13></a>

# In[ ]:


g = sns.factorplot(x="SibSp",y="Survived",data=train_df,kind="bar",size=6)
g.set_ylabels("Survived Probability")
plt.show()


# * The more passenger has the SibSp feature,we have the less chance to survive.
# * If SibSp == 0, 1 or 2, passenger has more chance to survive.
# * We can consider a new feature describing these categories.

# ## Parch - Survived <a id=14></a>

# In[ ]:


g = sns.factorplot(x="Parch",y="Survived",data=train_df,kind="bar",size=6)
g.set_ylabels("Survived Probability")
plt.show()


# * SibSp and Parch can be used for feature extraction with threshold = 3
# * The passengers with Parch = 4 has no chance to survive.
# * Small families have more chance to survive.
# * There is a standard in survival of passenger with Parch = 3

# ## Pclass - Survived <a id=15></a>

# In[ ]:


g = sns.factorplot(x="Pclass",y="Survived",data=train_df,kind="bar",size=6)
g.set_ylabels("Survived Probability")
plt.show()


# * The less passengers class is, the less chance to survive.

# ## Age - Survived <a id=16></a>

# In[ ]:


g = sns.FacetGrid(train_df,col="Survived")
g.map(sns.distplot,"Age",bins=25)
plt.show()


# * Age <= 10 has a high survival rate.
# * Oldest passengers (80) survived.
# * Large part of 20 years old passengers couldn't survive.
# * Most passengers are in 15-35 Age range.
# * Use Age feature in training.
# * Use age distribution for missing values of Age.

# ## Pclass - Survived - Age <a id=17></a>

# In[ ]:


g = sns.FacetGrid(train_df,col="Survived",row="Pclass")
g.add_legend()
g.map(plt.hist,"Age",bins = 25)
plt.show()


# * Pclass is an important feature for model training.

# ## Embarked - Sex - Pclass - Survived <a id=18></a>

# In[ ]:


g = sns.FacetGrid(train_df,row="Embarked",size=2)
g.map(sns.pointplot,"Pclass","Survived","Sex")
g.add_legend()
plt.show()


# * Survival rate of female passenger is greater than male passengers survival rate.
# * Male passengers have better survival rate when Pclass is equal to 3 in C.
# * Embarked and Sex will be used in training.

# ## Embarked - Sex - Fare - Survived <a id=19></a>

# In[ ]:


g = sns.FacetGrid(train_df,row="Embarked",col="Survived",size=2.5)
g.map(sns.barplot,"Sex","Fare")
g.add_legend()
plt.show()


# * Passengers who paid more have more chance to survive.
# * Fare can be used in categorical for training.

# ## Fill Missing : Age feature <a id=20></a>

# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:


sns.factorplot(x = "Sex",y = "Age",data = train_df, kind = "box")
plt.show()


# * Sex feature is not effective in age prediction, age distribution seems to be the same.

# In[ ]:


sns.factorplot(x = "Sex",y = "Age",hue="Pclass",data = train_df, kind = "box")
plt.show()


# * Age raises backwards Pclass.

# In[ ]:


sns.factorplot(x = "Parch",y = "Age",data = train_df, kind = "box")
sns.factorplot(x = "SibSp",y = "Age",data = train_df, kind = "box")
plt.show()


# In[ ]:


train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot=True)
plt.show()


# * Age is not correlated with Sex, but it is correlated with Parch, SibSp, Pclass.

# In[ ]:


index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"])& (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    age_med = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med


# In[ ]:


train_df[train_df["Age"].isnull()]


# # Feature Engineering <a id=21></a>

# ## Name - Title <a id=22></a>

# In[ ]:


train_df["Name"].head(10)


# In[ ]:


name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]


# In[ ]:


train_df["Title"].head(10)


# In[ ]:


sns.countplot(x="Title",data=train_df)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


# Convert to categorical
train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i=="Ms" or i=="Mlle" or i=="Mrs" else 2 if i =="Mr" else 3 for i in train_df["Title"]]


# In[ ]:


g = sns.factorplot(x="Title",y="Survived",data=train_df,kind="bar")
g.set_xticklabels(["Master","Mrs","Mr","Other"])
g.set_ylabels("Survival Probability")
plt.show()


# In[ ]:


train_df.drop(labels=["Name"],axis=1,inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


train_df = pd.get_dummies(train_df,columns=["Title"])
train_df.head()


# ## Family Size <a id=23></a>

# In[ ]:


train_df.head()


# In[ ]:


train_df["FSize"] = train_df["SibSp"] + train_df["Parch"] + 1


# In[ ]:


train_df.head()


# In[ ]:


g = sns.factorplot(x="FSize",y="Survived",data=train_df,kind="bar")
g.set_ylabels("Survival Probability")
plt.show()


# In[ ]:


train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["FSize"]]


# In[ ]:


train_df.head(20)


# In[ ]:


sns.countplot(x="family_size",data=train_df)
plt.show()


# In[ ]:


g = sns.factorplot(x="family_size",y="Survived",data=train_df,kind="bar")
g.set_ylabels("Survival Probability")
plt.show()


# * Small families have more chance to survive than large families.

# In[ ]:


train_df = pd.get_dummies(train_df,columns=["family_size"])


# In[ ]:


train_df.head()


# ## Embarked <a id=24></a>

# In[ ]:


train_df["Embarked"].head()


# In[ ]:


sns.countplot(x="Embarked",data=train_df)
plt.show()


# In[ ]:


train_df = pd.get_dummies(train_df,columns=["Embarked"])
train_df.head()


# ## Ticket <a id="25"></a>

# In[ ]:


train_df["Ticket"].head(20)


# In[ ]:


tickets = []
for i in list(train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        tickets.append("x")
train_df["Ticket"] = tickets


# In[ ]:


train_df["Ticket"].head(20)


# In[ ]:


train_df = pd.get_dummies(train_df,columns=["Ticket"],prefix="T")
train_df.head(10)


# ## Pclass <a id="26"></a>

# In[ ]:


sns.countplot(x="Pclass",data=train_df)
plt.show()


# In[ ]:


train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df, columns=["Pclass"])
train_df.head()


# ## Sex <a id="27"></a>

# In[ ]:


train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df,columns=["Sex"])
train_df.head()


# ## Drop PassengerId and Cabin <a id="28"></a>

# In[ ]:


train_df.drop(labels=["PassengerId","Cabin"],axis=1,inplace=True)


# In[ ]:


train_df.columns


# # Modeling <a id="29"></a>

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ## Train Test Split <a id="30"></a>

# In[ ]:


train_df_len


# In[ ]:


test = train_df[train_df_len:]
test.drop(labels=["Survived"],axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:


train = train_df[:train_df_len]
x_train = train.drop(labels="Survived",axis=1)
y_train = train["Survived"]
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.33,random_state=42)
print("x_train",len(x_train))
print("x_test",len(x_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(test))


# ## Simple Logistic Regression <a id="31"></a>

# In[ ]:


lr = LogisticRegression()
lr.fit(x_train,y_train)
acc_lr_train = round(lr.score(x_train,y_train) * 100,2)
acc_lr_test = round(lr.score(x_test,y_test) * 100,2)
print(f"Train accuracy: %{acc_lr_train}")
print(f"Test accuracy: %{acc_lr_test}")


# ## Hyperparameter Tuning - Grid Search - Cross Validation <a id="32"></a>
# We will compare 5 ML Classifiers and evaluate mean accuracy of each of them by Stratified Cross Validation.
# 
# * Decision Tree
# * SVM
# * Random Forest
# * KNN
# * Logistic Regression

# In[ ]:


random_state = 42
classifier = [DecisionTreeClassifier(random_state=random_state),
              SVC(random_state=random_state),
             RandomForestClassifier(random_state=random_state),
             LogisticRegression(random_state=random_state),
             KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth" : range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma" : [0.001,0.01,0.1,1],
                 "C" : [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features" : [1,3,10],
                "min_samples_split" : [2,3,10],
                "min_samples_leaf" : [1,3,10],
                "bootstrap" : [False],
                "n_estimators" : [100,300],
                "criterion" : ["gini"]}

lr_param_grid = {"C" : np.logspace(-3,3,7),
                "penalty" : ["l1","l2"]}

knn_param_grid = {"n_neighbors" : np.linspace(1,19,10,dtype=int).tolist(),
                 "weights" : ["uniform","distance"],
                 "metric" : ["euclidean","manhattan"]}

classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   lr_param_grid,
                   knn_param_grid]


# In[ ]:


cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i],param_grid = classifier_param[i],cv=StratifiedKFold(n_splits = 10),scoring = "accuracy",n_jobs = -1,verbose = 1)
    clf.fit(x_train,y_train)
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


# ## Ensemble Modeling <a id="33"></a>

# In[ ]:


votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],voting = "soft",n_jobs = -1)

votingC = votingC.fit(x_train,y_train)
print(accuracy_score(votingC.predict(x_test),y_test))


# ## Prediction and Submission <a id="34"></a>

# In[ ]:


test_survived = pd.Series(votingC.predict(test),name="Survived").astype(int)
results = pd.concat([test_PassengerId,test_survived],axis=1)
results.to_csv("titanic.csv",index = False)

