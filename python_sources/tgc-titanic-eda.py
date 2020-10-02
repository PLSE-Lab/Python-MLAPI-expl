#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The Titanic was a British steamship that sank in the early hours of April 15, 1912 after striking an iceberg,leading to the deaths of more than 1,500 passengers and crew.
# <font color='blue'/>
# Content:
# 1. [Load and Check Data](#1)
# 2. [Variable Description](#2)
#    * [Univariate Variable Analysis](#3)
#        * [Categorical Variable](#4)
#        * [Numerical Variable](#5)
# 3. [Basic Data Analysis](#6)
# 4. [Outlier Detection](#7)
# 5. [Missing Value](#8)
#     * [Find Missing Value](#9)
#     * [Fill Missing Value](#10)
# 6. [Visualization](#11)
#     * [Correlation Between SibSp -- Parch -- Age -- Fare -- Survived](#12)
#     * [SibSp -- Survived](#13)
#     * [Parch -- Survived](#14)
#     * [Pclass -- Survived](#15)
#     * [Age -- Survived](#16)
#     * [Pclass -- Survived -- Age](#17)
#     * [Embarked -- Sex -- Pclass -- Survived](#18)
#     * [Embarked -- Sex -- Fare -- Survived](#19)
#     * [Fill Mising: Age Feature](#20)
# 7. [Feature Engineering](#21)
#     * [Name -- Title](#22)
#     * [Family Size](#23)
#     * [Embarked](#24)
#     * [Ticket](#25)
#     * [Pclass](#26)
#     * [Sex](#27)
#     * [Drop PassengerId and Cabin](#28)
# 8. [Modeling](#29)
#     * [Train Test Split](#30)
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
plt.style.use("seaborn-pastel")
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


# <a id="1"></a><br>
# # Load and Check Data

# In[ ]:


train_df=pd.read_csv("/kaggle/input/titanic/train.csv")
test_df=pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId=test_df["PassengerId"]


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# <a id="2"></a><br>
# # Variable Description
# 1. PassengerId: Unique id number to each passenger
# 1. Survived: Passenger survive(1) or died(0)
# 1. Pclass: Passenger class
# 1. Name: Name of passenger
# 1. Sex: Gender of passenger
# 1. Age: Age of passenger
# 1. SibSp: Number of siblings/spouses
# 1. Parch: Number of parents/children
# 1. Ticket: Ticket number
# 1. Fare: Amount of money spent on ticket
# 1. Cabin: Cabin category
# 1. Embarked: Port where passenger embarked (C: Cherbourg, Q: Queenstown, S: Southampton

# In[ ]:


train_df.info()


# 
# * float64(2): Fare and Age
# * int64(5): Pclass, SibSp, Parch, PassengerId and Survived
# * object(5): Cabin, Embarked, Ticket, Name and Sex
# 

# <a id="3"></a><br>
# # Univariate variable Analysis
# * Categorical Variable: Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, SibSp, Parch
# * Numerical Variable: Fare, Age and PassengerId

# <a id="4"></a><br>
# ## Categorical Variable

# In[ ]:


def bar_plot(variable):
    """
        input: variable ex: "Sex"
        output: bar plot & value count
    """
    # Get Feature
    var=train_df[variable]
    # Count Number of Categorical Variable(Value/Sample)
    varValue=var.value_counts()
    # Visualize
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))


# In[ ]:


category1=["Survived","Sex","Pclass","Embarked","SibSp","Parch"]
for c in category1:
    bar_plot(c)


# In[ ]:


category2=["Cabin","Name","Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))


# <a id="5"></a><br>
# ## Numerical Variable

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable],bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist ".format(variable))
    plt.show()


# In[ ]:


numericVar=["Fare", "Age", "PassengerId"]
for n in numericVar:
    plot_hist(n)


# <a id="6"></a><br>
# # Basic Data Analysis
# * Pclass - Survived
# * Sex - Survived
# * SibSp - Survived
# * Parch - Survived

# In[ ]:


# Pclass vs Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


# Sex vs Survived
train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


# SibSp vs Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


# Parch vs Survived
train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending=False)


# <a id="7"></a><br>
# # Outlier Detection 

# In[ ]:


def detect_outliers(df,features):
    outlier_indices=[]
    
    for c in features:
        #1st quartile
        Q1=np.percentile(df[c],25)
        #3rd quartile
        Q3=np.percentile(df[c],75)
        #IQR
        IQR=Q3-Q1
        #Outlier Step
        outlier_step=IQR*1.5
        #Detect outlier and their indices
        outlier_list_col=df[(df[c]<Q1-outlier_step) | (df[c]>Q3+outlier_step)].index
        #Store indices
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices=Counter(outlier_indices)
    multiple_outliers=list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


# In[ ]:


train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]


# In[ ]:


#Drop Outliers
train_df=train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)


# <a id="8"></a><br>
# # Missing Value
# * Find Missing Value
# * Fill Missing Value

# In[ ]:


train_df_len=len(train_df)
train_df=pd.concat([train_df,test_df],axis=0).reset_index(drop=True)


# In[ ]:


train_df.head()


# <a id="9"></a><br>
# ## Find Missing Value

# In[ ]:


train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# <a id="10"></a><br>
# ## Fill Missing Value
# * Embarked has 2 missing value
# * Fare has only 1

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df.boxplot(column="Fare",by="Embarked")
plt.show()


# In[ ]:


train_df["Embarked"]=train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df[train_df["Fare"].isnull()]


# In[ ]:


train_df["Fare"]=train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))


# In[ ]:


train_df[train_df["Fare"].isnull()]


# <a id="11"></a><br>
# # Visualization

# <a id="12"></a><br>
# ## Correlation Between SibSp -- Parch -- Age -- Fare -- Survived

# In[ ]:


list1=["SibSp","Parch","Age","Fare","Survived"]
sns.heatmap(train_df[list1].corr(),annot=True,fmt=".2f")
plt.show()


# * Fare feature seems to have correlation with survived feature (0.26)

# <a id="13"></a><br>
# ## SibSp -- Survived

# In[ ]:


g=sns.factorplot(x="SibSp",y="Survived",data=train_df,kind="bar",size=6)
g.set_ylabels("Survived Probability")
plt.show()


# * Having a lot of SibSp have less chance to survive
# * if SibSp == 0 or 1 or 2, passenger has more chance to survive
# * We can consider a new feature describing these categories

# <a id="14"></a><br>
# ## Parch -- Survived

# In[ ]:


g=sns.factorplot(x="Parch",y="Survived",kind="bar",data=train_df,size=6)
g.set_ylabels("Survived Probability")
plt.show()


# * Black line indicates standard deviation
# * SibSp and Parch can be used for new feature extraction with th=3
# * Small families have more chance to survive
# * There is a std in survival of passenger with Parch=3

# <a id="15"></a><br>
# ## Pclass -- Survived

# In[ ]:


g=sns.factorplot(x="Pclass",y="Survived",data=train_df,kind="bar",size=6)
g.set_ylabels("Survived Probability")
plt.show()


# <a id="16"></a><br>
# ## Age -- Survived

# In[ ]:


g=sns.FacetGrid(train_df,col="Survived")
g.map(sns.distplot,"Age",bins=25)
plt.show()


# * Age <= 10 has a high survival rate
# * Oldest passenger (about 80 years old) survived
# * Large number of 20 years old did not survive
# * Most passengers are in 15-35 age range
# * Use age feature in training
# * Use age distribution for missing value of age

# <a id="17"></a><br>
# ## Pclass -- Survived -- Age

# In[ ]:


g=sns.FacetGrid(train_df, col="Survived",row="Pclass",size=3)
g.map(plt.hist,"Age",bins=25)
g.add_legend()
plt.show()


# * Pclass is important featyre for model training

# <a id="18"></a><br>
# ## Embarked -- Sex -- Pclass -- Survived
# 

# In[ ]:


g=sns.FacetGrid(train_df, row="Embarked", size=3)
g.map(sns.pointplot, "Pclass", "Survived", "Sex")
g.add_legend()
plt.show()


# * Female passengers have much better survival rate than male passengers
# * Males have better survival rate in Pclass 3 in C
# * Embarked and Sex will be used in training

# <a id="19"></a><br>
# ## Embarked -- Sex -- Fare -- Survived

# In[ ]:


g=sns.FacetGrid(train_df,row="Embarked", col="Survived", size=3.5)
g.map(sns.barplot,"Sex", "Fare")
g.add_legend()
plt.show()


# * Passengers who pay higher Fare have better survival
# * Fare can be used as categorical for training

# <a id="20"></a><br>
# ## Fill Mising: Age Feature

# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:


sns.factorplot(x="Sex",y="Age",data=train_df,kind="box")
plt.show()


# * Sex is not informative for age prediction, age distribution seems to be same

# In[ ]:


sns.factorplot(x="Sex",y="Age", hue="Pclass",data=train_df,kind="box")
plt.show()


# * First class passenger are older than 2nd, and 2nd is older than 3rd class

# In[ ]:


sns.factorplot(x="Parch",y="Age",data=train_df,kind="box")
sns.factorplot(x="SibSp",y="Age",data=train_df,kind="box")
plt.show()


# In[ ]:


train_df["Sex"]=[1 if i=="male" else 0 for i in train_df["Sex"]] 


# In[ ]:


sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot=True)
plt.show()


# * Age is not correlated with Sex but it is correlated with Parch, SibSp and Pclass

# In[ ]:


index_nan_age=list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred=train_df["Age"][((train_df["SibSp"]==train_df.iloc[i]["SibSp"])& (train_df["Parch"]==train_df.iloc[i]["Parch"])& (train_df["Pclass"]==train_df.iloc[i]["Pclass"]))].median()
    age_med=train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i]=age_pred
    else:
        train_df["Age"].iloc[i]=age_med


# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:


age_pred


# <a id="21"></a><br>
# # Feature Engineerig

# <a id="22"></a><br>
# ## Name -- Title

# In[ ]:


train_df["Name"].head(10)


# In[ ]:


name=train_df["Name"]
train_df["Title"]=[i.split(".")[0].split(",")[-1].strip() for i in name]


# In[ ]:


train_df["Title"].head(10)


# In[ ]:


sns.countplot(x="Title",data=train_df)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


# Convert to categorical
train_df["Title"]=train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"Other")
train_df["Title"]=[0 if i=="Master" else 1 if i=="Miss" or i=="Ms" or i=="Mlle" or i=="Mrs" else 2 if i=="Mr" else 3 for i in train_df["Title"]]
train_df["Title"].head(20)


# In[ ]:


sns.countplot(x="Title",data=train_df)
plt.xticks(rotation=360)
plt.show()


# In[ ]:


g=sns.factorplot(x="Title",y="Survived",data=train_df,kind="bar")
g.set_xticklabels(["Master","Miss-Mrs","Mr","Other"])
g.set_ylabels("Survival Probability")
plt.show()


# In[ ]:


train_df.drop(labels=["Name"],axis=1,inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


train_df=pd.get_dummies(train_df,columns=["Title"])
train_df.head()


# <a id="23"></a><br>
# ## Family Size

# In[ ]:


train_df["Fsize"]=train_df["SibSp"]+train_df["Parch"]+1


# In[ ]:


train_df.head()


# In[ ]:


g=sns.factorplot(x="Fsize",y="Survived",data=train_df,kind="bar")
g.set_ylabels("Survival")
plt.show()


# In[ ]:


train_df["family_size"]=[1 if i<5 else 0 for i in train_df["Fsize"]]


# In[ ]:


train_df.head(20)


# In[ ]:


sns.countplot(x="family_size",data=train_df)
plt.show()


# In[ ]:


g=sns.factorplot(x="family_size",y="Survived",data=train_df,kind="bar")
g.set_ylabels("Survival")
plt.show()


# * Small families have more chance to survive than large families

# In[ ]:


train_df=pd.get_dummies(train_df, columns=["family_size"])

train_df.head()


# <a id="24"></a><br>
# ## Embarked

# In[ ]:


sns.countplot(x="Embarked",data=train_df)
plt.show()


# In[ ]:


train_df=pd.get_dummies(train_df,columns=["Embarked"])
train_df.head()


# <a id="25"></a><br>
# ## Ticket

# In[ ]:


train_df["Ticket"].head(20)


# In[ ]:


tickets=[]
for i in list(train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    
    else:
        tickets.append("x")
train_df["Ticket"]=tickets


# In[ ]:


train_df["Ticket"].head(20)


# In[ ]:


train_df=pd.get_dummies(train_df,columns=["Ticket"],prefix="T")
train_df.head(10)


# <a id="26"></a><br>
# ## Pclass

# In[ ]:


sns.countplot(x="Pclass",data=train_df)
plt.show()


# In[ ]:


train_df["Pclass"]=train_df["Pclass"].astype("category")
train_df=pd.get_dummies(train_df,columns=["Pclass"])
train_df.head()


# <a id="27"></a><br>
# ## Sex

# In[ ]:


train_df["Sex"]=train_df["Sex"].astype("category")
train_df=pd.get_dummies(train_df,columns=["Sex"])
train_df.head()


# <a id="28"></a><br>
# ## Drop PassengerId and Cabin

# In[ ]:


train_df.drop(labels=["PassengerId","Cabin"],axis=1,inplace=True)


# In[ ]:


train_df.columns


# <a id="29"></a><br>
# # Modeling

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn. ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# <a id="30"></a><br>
# ## Train Test Split

# In[ ]:


train_df_len


# In[ ]:


test=train_df[train_df_len:]
test.drop(labels=["Survived"],axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:


train=train_df[:train_df_len]
x_train=train.drop(labels="Survived",axis=1)
y_train=train["Survived"]
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.33,random_state=42)
print("x_train",len(x_train))
print("x_test",len(x_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(test))


# <a id="31"></a><br>
# ## Simple Logistic Regression

# In[ ]:


logreg=LogisticRegression()
# x_train has features, y_train has survived value
logreg.fit(x_train,y_train)
acc_log_train=round(logreg.score(x_train,y_train)*100,2)
acc_log_test=round(logreg.score(x_test,y_test)*100,2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))


# <a id="32"></a><br>
# ## Hyperparameter Tuning -- Grid Search -- Cross Validation
# We will compare 5 machine learning classifier and evaluate mean accuracy of each of them by stratified cross validation
# * Decision Tree
# * SVM
# * Random Forest
# * KNN
# * Logistic Regression

# In[ ]:


random_state=42
classifier=[DecisionTreeClassifier(random_state=random_state),
           SVC(random_state=random_state),
           RandomForestClassifier(random_state=random_state),
           LogisticRegression(random_state=random_state),
           KNeighborsClassifier()]

dt_param_grid={"min_samples_split":range(10,500,20),
              "max_depth":range(1,10,2)}

svc_param_grid={"kernel":["rbf"],
               "gamma":[0.001,0.01,0.1,1],
               "C":[1,10,50,100,200,300,1000]}

rf_param_grid={"max_features":[1,3,10],
              "min_samples_split":[2,3,10],
              "min_samples_leaf":[1,3,10],
              "bootstrap":[False],
              "n_estimators":[100,300],
              "criterion":["gini"]}

logreg_param_grid={"C":np.logspace(-3,3,7),
                  "penalty":["l1","l2"]}

knn_param_grid={"n_neighbors":np.linspace(1,19,10, dtype=int).tolist(),
               "weights":["uniform","distance"],
               "metric":["euclidean","manhattan"]}

classifier_param=[dt_param_grid,
                 svc_param_grid,
                 rf_param_grid,
                 logreg_param_grid,
                 knn_param_grid]


# In[ ]:


cv_result=[]
best_estimators=[]
for i in range(len(classifier)):
    clf=GridSearchCV(classifier[i],param_grid=classifier_param[i],cv=StratifiedKFold(n_splits=10),scoring="accuracy",n_jobs=-1,verbose=1)
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


# In[ ]:


cv_results=pd.DataFrame({"Cross Validation Means":cv_result,"ML Models":["DecisionTreeClassifier","SVM","RandomForestClassifier","LogisticRegression","KNeighborsClassifier"]})

g=sns.barplot("Cross Validation Means","ML Models",data=cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")
plt.show()


# <a id="33"></a><br>
# ## Ensemble Modeling
# 

# In[ ]:


votingC=VotingClassifier(estimators=[("dt",best_estimators[0]),
                                    ("rfc",best_estimators[2]),
                                     ("lr",best_estimators[3])],
                                     voting="soft",n_jobs=-1)
votingC=votingC.fit(x_train,y_train)
print(accuracy_score(votingC.predict(x_test),y_test))
# x_test is for prediction, y_test is for test


# <a id="34"></a><br>
# ## Prediction and Submission

# In[ ]:


test_survived=pd.Series(votingC.predict(test),name="Survived").astype(int)
results=pd.concat([test_PassengerId,test_survived],axis=1)
results.to_csv("titanic.csv",index=False)

