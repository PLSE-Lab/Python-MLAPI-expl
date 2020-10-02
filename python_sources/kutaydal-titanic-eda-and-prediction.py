#!/usr/bin/env python
# coding: utf-8

# # Introduction
# RMS Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean in the early morning hours of April 15, 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters.
# 
# <font color = 'orange'>
# Content:
# 1. [Load and Check Data](#1)
# 1. [Variable Description](#2)
#     * [Univarite Variable Analysis](#3)
#         * [Categorical Variable](#4)
#         * [Numerical Variable](#5)
# 1. [Basic Data Analysis](#6)
# 1. [Outlier Detection](#7)
# 1. [Missing Value](#8)
#     * [Find Missing Value](#9)
#     * [Fill Missing Value](#10)
# 1. [Visualization](#11)
#     * [Correlation between Sibsp - Parch - Age -- Fare - Survived](#12)
#     * [Pclass - Survived](#13)
#     * [Age - Survived](#14)
#     * [Pclass - Survived - Age](#15)
#     * [Embarked - Sex - Pclass - Survived](#16)
#     * [Embarked - Sex - Fare - Survived](#17)
#     * [Fill missing age value](#18)
# 1. [Machine Learning](#19)
#     * [Name -- Title](#20)
#     * [Embarked](#21)
#     * [Ticket](#22)
#     * [Pclass](#23)
#     * [Sex](#24)
#     * [Drop Passenger ID and Cabin](#25)
# 1. [Modeling](#26)
#     * [Train - Test Split](#27)
#     * [Simple Logistic Regression](#28)
#     * [Hyperparameter Tuning -- Grid Search -- Cross Validation](#29) 
#     * [Ensemble Modeling](#30)
#     * [Prediction and Submission](#31)
#     

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("whitegrid")

from collections import Counter

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id = '1'></a>
# # Load and Check Data
# 
# Read data files.

# In[ ]:


train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
test_PassengerId = test_df['PassengerId']


# Looked for the columns in the data set.

# In[ ]:


train_df.columns


# A small sample of the table

# In[ ]:


train_df.head()


# Values in the table. such as total, average, standard deviation

# In[ ]:


train_df.describe()


# <a id = '2'></a>
# # Variable Description
# 1. PassengerId : Unique id to each passenger
# 1. Survived : Survived = 1, Died = 0
# 1. Pclass : Passenger class
# 1. Name : Name of passenger
# 1. Sex : Gender of passenger
# 1. Age : Age of passenger
# 1. SibSp : Number of Siblings/Spouses
# 1. Parch : Number of Parents/Children
# 1. Ticket : Ticket number of passenger
# 1. Fare : Amount of money spent on ticket
# 1. Cabin : Cabin category
# 1. Embarked : Port where passenger embarked (C = Cherbourg, Q = Queenstown, S = Southampten)

# In[ ]:


train_df.info()


# * float64(2) : Age and Fare
# * int64(5) : PassengerId, Survived, Pclass, SibSp and Parch
# * object(5) : Name, Sex, Ticket, Cabin, Embarked

# <a id = '3'></a>
# # Univarite Variable Analysis
#  * Categorical Variable
#  * Numerical Variable

# <a id = '4'></a>
# ## Categorical Variable
# *Survived, Pclass, Sex, Embarked, Cabin, Name, Ticket, SibSp and Parch*

# The function I prepared to visually see some of the values in the table.

# In[ ]:


def bar_plot(variable):
    """
        input: variable example: 'Survived'
        output: bar plot & value count
    """
    # get variable
    var = train_df[variable]
    # count number of categorical variable
    varValue = var.value_counts()
    
    # visualization
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel('Frequency')
    plt.title(variable)
    plt.show()
    print("{}:\n{}".format(variable, varValue))


# In[ ]:


category1 = ['Survived', 'Pclass', 'Sex', 'Embarked', 'Cabin', 'Name', 'Ticket', 'SibSp', 'Parch']
for c in category1:
    bar_plot(c)


# In[ ]:


category2 = ['Cabin', 'Name', 'Ticket']
for c in category2:
    print('{}\n'.format(train_df[c].value_counts()))


# <a id = '5'></a>
# ## Numerical Variable
# *PassengerId, Age and Fare*

# In[ ]:


def hist_plot(variable):
    # visualization
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.title('{} distribution with histogram'.format(variable))
    plt.show()


# I created these tables to see the age and ticket price distribution on the ship.
# 
# Avg age is like ~25,
# Avg fare is ~8

# In[ ]:


numericVar = ['Age', 'Fare']
for n in numericVar:
    hist_plot(n)


# <a id = '6'></a>
# # Basic Data Analysis
# Probability of survival:
# * Pclass - Survived
# * Sex - Survived
# * SibSp - Survived
# * Parch - Survived

# In[ ]:


# Pclass - Survived
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


# Sex - Survived
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


# SibSp - Survived
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


# Parch - Survived
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# <a id = '7'></a>
# # Outlier Detection

# In[ ]:


def detect_outlier(df, features):
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
        # Outlier and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step )].index
        # Store indices
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v >2)
    
    return(multiple_outliers)
    
    


# In[ ]:


train_df.loc[detect_outlier(train_df, ['Age', 'SibSp', 'Parch', 'Fare'])]


# In[ ]:


# Drop outliers
tran_df = train_df.drop(detect_outlier(train_df, ['Age', 'SibSp', 'Parch', 'Fare']), axis = 0).reset_index(drop = True)


# <a id = '8'></a>
# # Missing Value
#    * Find Missing Value
#    * Fill Missing Value

# In[ ]:


train_df_len = len(train_df)
train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)


# <a id = '9'></a>
# ## Find Missing Value

# In[ ]:


train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# <a id = '10'></a>
# ## Fill Missing Value

# In[ ]:


train_df[train_df['Embarked'].isnull()]


# In[ ]:


train_df.boxplot(column = 'Fare', by = 'Embarked')
plt.show()


# In[ ]:


train_df['Embarked'] = train_df['Embarked'].fillna('C')
train_df[train_df['Embarked'].isnull()]


# In[ ]:


train_df[train_df['Fare'].isnull()]


# In[ ]:


train_df['Fare'] = train_df['Fare'].fillna(np.mean(train_df[train_df['Pclass'] == 3]['Fare']))
train_df[train_df['Fare'].isnull()]


# <a id = '11'></a>
# # Visualization

# <a id = '12'></a>
# ### Correlation between Sibsp -- Parch -- Age -- Fare -- Survived

# In[ ]:


list1 = ['SibSp', 'Parch', 'Age', 'Fare', 'Survived']
sns.heatmap(train_df[list1].corr(), annot = True, fmt = '.2f')


# Fare seems to have correlation  with survival.

# <a id = '13'></a>
# ### Pclass - Survived
# 
# I created this table to see if the ticket class has anything to do with the chance of survival, and as you can see, someone in the higher class is more likely to survive.

# In[ ]:


y = sns.factorplot(x= "Pclass", y="Survived",data = train_df, kind='bar', size = 6)
y.set_ylabels("Survived Probability")
plt.show


# <a id = '14'></a>
# ### Age - Survived
# 
# I created this table to see the average age of survivors and deceased.

# In[ ]:


g = sns.FacetGrid(train_df, col = "Survived")
g.map(sns.distplot, "Age", bins = 20)
plt.show()


# <a id = '15'></a>
# ### Pclass - Survived - Age
# 
# I combined the two tables above and looked at the average age of survivors and deceased for each class.

# In[ ]:


g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass")
g.map(plt.hist, "Age", bins=20)
g.add_legend()
plt.show()


# <a id = '16'></a>
# ### Embarked - Sex - Pclass - Survived
# 
# I divided these tables into groups according to the port from which people boarded.

# In[ ]:


g = sns.FacetGrid(train_df, row = "Embarked")
g.map(sns.pointplot, "Pclass", "Survived", "Sex")
g.add_legend()
plt.show()


# Female passenger have much more survival rate than male pasengers.

# <a id = '17'></a>
# ### Embarked - Sex - Fare - Survived

# In[ ]:


g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived")
g.map(sns.barplot, "Sex", "Fare")
g.add_legend()
plt.show()


# Higher fare have more survival chance

# <a id = '18'></a>
# ### Fill missing age value

# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:


train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]


# Convert sex to numeric

# In[ ]:


sns.heatmap(train_df[["Age","Sex","Parch","SibSp","Pclass"]].corr(), annot = True)


# Age is not correlated with sex but it's correlated with parch, sibsp and pclass

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


# <a id = '19'></a>
# # Machine Learning

# <a id = '20'></a>
# ## Name - Title

# In[ ]:


train_df["Name"].head(5)


# In[ ]:


name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]


# In[ ]:


train_df["Title"].head(10)


# In[ ]:


sns.countplot(x= "Title", data = train_df)
plt.xticks(rotation= 90)
plt.show()


# In[ ]:


#category
train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]
train_df["Title"].head(10)


# In[ ]:


sns.countplot(x="Title", data = train_df)
plt.show()


# In[ ]:


train_df.drop(labels = ["Name"], axis = 1, inplace = True)
train_df.head()


# In[ ]:


train_df = pd.get_dummies(train_df,columns=["Title"])
train_df.head()


# <a id = "21"></a><br>
# ## Embarked

# In[ ]:


train_df["Embarked"].head()


# In[ ]:


sns.countplot(x = "Embarked", data = train_df)
plt.show()


# In[ ]:


train_df = pd.get_dummies(train_df, columns=["Embarked"])
train_df.head()


# <a id = "22"></a><br>
# ## Ticket

# In[ ]:


train_df["Ticket"].head(10)


# In[ ]:


tickets = []
for i in list(train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        tickets.append("x")
train_df["Ticket"] = tickets


# In[ ]:


train_df["Ticket"].head(10)


# In[ ]:


train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix = "T")
train_df.head(10)


# <a id = "23"></a><br>
# ## Pclass

# In[ ]:


sns.countplot(x = "Pclass", data = train_df)
plt.show()


# In[ ]:


train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df, columns= ["Pclass"])
train_df.head()


# <a id = "24"></a><br>
# ## Sex

# In[ ]:


train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df, columns=["Sex"])
train_df.head()


# <a id = "25"></a><br>
# ## Drop Passenger ID and Cabin 

# In[ ]:


train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)


# In[ ]:


train_df.columns


# <a id = "26"></a><br>
# # Modeling

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# <a id = "27"></a><br>
# ## Train - Test Split

# In[ ]:


train_df_len


# In[ ]:


test = train_df[train_df_len:]
test.drop(labels = ["Survived"],axis = 1, inplace = True)


# In[ ]:


test.head()


# In[ ]:


train = train_df[:train_df_len]
X_train = train.drop(labels = "Survived", axis = 1)
y_train = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(test))


# <a id = "28"></a><br>
# ## Simple Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_log_train = round(logreg.score(X_train, y_train)*100,2) 
acc_log_test = round(logreg.score(X_test,y_test)*100,2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))


# <a id = "29"></a><br>
# ## Hyperparameter Tuning -- Grid Search -- Cross Validation
# * Decision Tree
# * SVM
# * Random Forest
# * KNN
# * Logistic Regression
# 

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


# <a id = "30"></a><br>
# ## Ensemble Modeling

# In[ ]:


votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)
votingC = votingC.fit(X_train, y_train)
print(accuracy_score(votingC.predict(X_test),y_test))


# <a id = "31"></a><br>
# ## Prediction and Submission

# In[ ]:


test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)
results = pd.concat([test_PassengerId, test_survived],axis = 1)
results.to_csv("titanic.csv", index = False)

