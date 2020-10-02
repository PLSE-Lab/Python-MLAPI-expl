#!/usr/bin/env python
# coding: utf-8

# # Introduction
#   Titanic is a ship which sank in 1912 beacuse of an iceberg and had been grave fo hundreds of people.
#   
# <font color = "blue">
# Content:
# 
# 1. [Load and Check Data](#1)
# 2. [Variable Desription](#2)
#    * [Univariate Variable Analysis](#3)
#         * [Categorical Variable](#4)
#         * [Numerical Variable](#5)
# 3. [Basic Data Analysis](#6)
# 4. [Outlier Detection](#7)
# 5. [Missing Value](#8)
#     * [Find Missing Value](#9)
#     * [Fill Missing Value](#10)
# 6. [Visualization](#11)
#     * [Correlation Between SibSp -- Parch -- Age -- Fare -- Pclass -- Survived](#12)
#     * [SibSp -- Survived Correlation Analysis](#13)
#     * [Parch -- Survived Correlation Analysis](#14)
#     * [Pclass -- Survived Correlation Analysis](#15)
#     * [Age -- Survived Correlation Analysis](#16)
#     * [Pclass -- Age -- Survived Correlation Analysis](#17)
#     * [Embarked -- Sex -- Pclass -- Survived Correlation Analysis](#18)
#     * [Embarked -- Sex -- Fare -- Survived Correlation Analysis](#19)
#     * [Fill Mising Value: Age](#20)
# 7. [Feature Engineering](#21)
#     * [Name - Title](#22)
#     * [Family Size](#23)
#     * [Embarked](#24)
#     * [Ticket](#25)
#     * [Pclass](#26)
#     * [Sex](#27)
#     * [Drop Passenger ID and Cabin](#28)
# 8. [Modeling](#29)
#     * [Train - Test Split](#30)
#     * [Simple Logistic Regression](#31)
#     * [Hyperparameter Tuning -- Grid Search -- Cross Validation](#32)
#     * [Ensemble Modeling](#33)
#     * [Prediction and Submission](#34)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <a id="1" ></a><br>
# # Load and Check Data

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]


# In[ ]:


train_df.columns


# <a id="2" ></a><br>
# # Variable Description
# 1. PassengerId: Unique ID number for each passenger
# 1. Survived: Have passenger survived(1) or died(0)
# 1. Pclass: Passenger class
# 1. Name: Name of the passenger
# 1. Sex: Sex of the passenger
# 1. Age: Age of the passenger
# 1. SibSp: Number of siblings or spouses of passenger
# 1. Parch: Number of parents or children of passenger
# 1. Ticket: Ticket number of passenger
# 1. Fare: The cost of ticket
# 1. Cabin: The name of the cabin of the passenger
# 1. Embarked: The port which passengers embarked

# In[ ]:


train_df.info()


# * float64(2): Fare and Age
# * int64(5): Pclass, SibSp, Parch, PassengerId and Survived
# * object(5): Cabin, Embarked, Ticket, Name and Sex

# <a id="3" ></a><br>
# ## Unvariate Variable Analysis
# * Categorical Variable
# * Numerical Variable

# <a id="4" ></a><br>
# ### Categorical Variable

# In[ ]:


def bar_plot(variable):
    """
    input: Variable, ex: "Sex"
    output: Bar plot & Variable count
    
    """
    #get feature
    var = train_df[variable]
    #count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))


# In[ ]:


category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]
for c in category1:
    bar_plot(c)


# In[ ]:


category2 = ["Cabin", "Name", "Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))


# <a id="5" ></a><br>
# ### Numerical Variable

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable], bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} Distribution with Histogram".format(variable))
    plt.show()


# In[ ]:


NumericVar = ["Fare","Age", "PassengerId"]
for c in NumericVar:
    plot_hist(c)


# <a id="6" ></a><br>
# # Basic Data Analysis
# * Pclass - Survived
# * Sex - Survived 
# * SibSp - Survived
# * Parch - Survived

# In[ ]:


# Pclass - Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived", ascending= False)


# The values below "Survived" column are the percentages of the possibilty of survival of each class.
# 
# * Survival Possibility of a First Class Survival Possibility: 62.96%
# * Survival Possibility of a Second Class Survival Possibility: 47.28%
# * Survival Possibility of a Third Class Passenger: 24.23%

# In[ ]:


# Sex - Survived
train_df[["Sex","Survived"]].groupby(["Sex"], as_index= False).mean().sort_values(by= "Survived",ascending=False)


# The values below "Survived" column are the percentages of the possibilty of survival of each gender.
# 
# * Survival Possibility of a Male: 18.89%
# * Survival Possibility of a Female: 74.20%

# In[ ]:


# SibSp - Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by = "Survived", ascending=False)


# This chart shows the relation between Sibling or Spouse number and survival possibility(percentage).

# In[ ]:


# Parch - Survived
train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending=False)


# This chart shows the relation between Parent or Children number and survival possibility(percentage).

# <a id="7" ></a><br>
# # Outlier Detection

# In[ ]:


def detect_outliers(df,features):
    outlier_indices=[]
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # Detect outliers and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # Store indices
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i , v in outlier_indices.items() if v>2)
    
    return multiple_outliers


# In[ ]:


train_df.loc[detect_outliers(train_df, ["Age", "SibSp","Parch","Fare"])]


# In[ ]:


# Drop outliers
train_df = train_df.drop(detect_outliers(train_df, ["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)


# In[ ]:


train_df.loc[detect_outliers(train_df, ["Age", "SibSp","Parch","Fare"])]


# As we can see, there are no outliers anymore.

# <a id="8" ></a><br>
# # Missing Value
# * Find Missing Value
# * Fill Missing Value

# In[ ]:


train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)


# <a id="9" ></a><br>
# ## Find Missing Value

# In[ ]:


train_df.columns[train_df.isnull().any()]


# These are the columns which include missing values.

# In[ ]:


train_df.isnull().sum()


# <a id="10" ></a><br>
# ## Fill Mising Value

# * Embarked has 2 missing values.
# * Fare has 1 missing value.
# * Cabin and Age will be filled later.
# * Missing values below "Survived" column are from test data frame.

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# These passengers' values might be filled by using "Fare" values. Because "Fare" are same for each. It should be checked by a boxplot.

# In[ ]:


train_df.boxplot(column = "Fare", by = "Embarked")
plt.show()


# The passengers whose "Embarked" values are missing paid 80.0 for tickets and according to boxplot,the median of "Fare" of the passengers embarked from C port is the closest to 80, so we can say that these passengers embarked from C. This possibility is high enough to make a change on "Embarked" values of these passengers.

# In[ ]:


train_df["Embarked"] = train_df["Embarked"].fillna("C")


# In[ ]:


train_df[train_df["Embarked"].isnull()]


# The filling of the values is successful.

# In[ ]:


train_df[train_df["Fare"].isnull()]


# If the passenger is in class 3 we can fill the "Fare" value with average of "Fare" of class 3 tickets.

# In[ ]:


train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))


# In[ ]:


train_df[train_df["Fare"].isnull()]


# Filling of the value is successfull.

# <a id="11" ></a><br>
# # Visualization

# In[ ]:


train_df


# <a id="12" ></a><br>
# ## Correlation Between SibSp -- Parch -- Age -- Fare -- Pclass -- Survived

# In[ ]:


list1=["SibSp", "Parch", "Age", "Fare", "Pclass", "Survived"]
f, ax=plt.subplots(figsize=(11,9))
sns.heatmap(train_df[list1].corr(), annot=True, fmt=" .2f", ax=ax)
plt.show()


# * There is a correlation between "Fare" and "Survived" (0.26).
# * There is a negative correlaation between "Pclass" and "Survived" (-0.33).

# <a id="13" ></a><br>
# ## SibSp -- Survived Correlation Analysis

# In[ ]:


g=sns.factorplot(x="SibSp", y="Survived", data=train_df, size=6, kind="bar")
g.set_ylabels("Possibility of Survival")
plt.show()


# According to chart it is possible to consider the passengers whose "SibSp" value is higher than 2 have a lower chance of survival.
# 
# SibSp < 2 --> High Chance of Survival

# <a id="14" ></a><br>
# ## Parch -- Survived Correlation Analysis

# In[ ]:


g=sns.factorplot(data=train_df, x="Parch", y="Survived", kind="bar", size=6)
g.set_ylabels("Possibility of Survival")
plt.show()


# According to this chart, the passengers whose "Parch" value is higher than 3 have a low chance of survival. Also whose value is 0 have a lower chance than the ones with 1,2,3 values.

# <a id="15" ></a><br>
# ## Pclass -- Survived Correlation Analysis

# In[ ]:


g=sns.factorplot(data=train_df, kind="bar", x="Pclass", y="Survived", size=6)
g.set_ylabels("Possibility of Survival")
plt.show()


# There is a clear negative correlation between "Pclass" and "Survived".

# <a id="16" ></a><br>
# ## Age -- Survived Correlation Analysis

# In[ ]:


g=sns.FacetGrid(train_df, col="Survived", height=5)
g.map(sns.distplot, "Age", bins=25)
plt.show()


# * It is possible to say that children are mostly survived
# * Passengers over 80 survived
# * Most of passengers around 20 and 35 didn't survive
# * A big part of passengers are 15-35 years old

# <a id="17" ></a><br>
# ## Pclass -- Age -- Survived Correlation Analysis

# In[ ]:


g=sns.FacetGrid(train_df, col="Survived", row="Pclass", size=3)
g.map(plt.hist, "Age", bins=25)
g.add_legend()
plt.show()


# It is clear that, "Pclass" quality has correlation with survival possibility.

# <a id="18" ></a><br>
# ## Embarked -- Sex -- Pclass -- Survived Correlation Analysis

# In[ ]:


g=sns.FacetGrid(train_df, "Embarked", size=3)
g.map(sns.pointplot, "Pclass", "Survived", "Sex")
g.add_legend()
plt.show()


# * Female passengers have higher rate than males
# * Male passengers' highest chance of survival is in Embarked=C ,Pclass=3

# <a id="19" ></a><br>
# ## Embarked -- Sex -- Fare -- Survived Correlation Analysis

# In[ ]:


g=sns.FacetGrid(train_df, col="Survived", row="Embarked", size=3)
g.map(sns.barplot, "Sex", "Fare")
g.add_legend()
plt.show()


# There is a correlation between "Fare" and survival possibility, especially in C and Q.

# <a id="20" ></a><br>
# ## Fill Mising Value: Age

# In[ ]:


train_df[train_df.Age.isnull()]


# In[ ]:


sns.factorplot(data=train_df, x="Sex", y="Age", kind="box")
plt.show()


# Sex feature is not useful for age prediction because medians are very close.

# In[ ]:


sns.factorplot(data=train_df, x="Pclass", y="Age", hue="Sex", kind="box")
plt.show()


# The class number and age is clearly negative correlated. (Age of 1st Class Passengers > 2nd > 3rd)

# In[ ]:


sns.factorplot(data=train_df, x="Parch", y="Age", kind="box")
sns.factorplot(data=train_df, x="SibSp", y="Age", kind="box")
plt.show()


# * According to the first chart, it is possible to seperate passengers about Parch in two: Parch =< 2 is Approximate Average=25, Parch > 2 Approximate Average=45
# * According to the second chart, it is possible to seperate passengers about SibSp in two: SibSp =< 2 is Approximate Average=25, SibSp > 2 Approximate Average=10

# In[ ]:


train_df["Sex01"]=[1 if i=="male" else 0 for i in train_df.Sex]


# In[ ]:


sns.heatmap(train_df[["Age", "Sex01", "Pclass", "SibSp", "Parch"]].corr(), annot=True)
plt.show()


# Age is negatively correlated with Pclass, SibSp, Parch, not correlated with sex.

# In[ ]:


index_nan_age=list(train_df[train_df.Age.isnull()].index)
for i in index_nan_age:
    age_pred=train_df.Age[((train_df.SibSp==train_df.iloc[i].SibSp)&(train_df.Pclass==train_df.iloc[i].Pclass)&(train_df.Parch==train_df.iloc[i].Parch))].median()
    age_med=train_df.Age.median()
    if not np.isnan(age_pred):
        train_df.Age.iloc[i]=age_pred
    else:
        train_df.Age.iloc[i]=age_med


# In[ ]:


train_df[train_df.Age.isnull()]


# Age values are totally filled now.

# <a id="21" ></a><br>
# # Feature Engineering

# <a id="22" ></a><br>
# ## Name - Title

# In[ ]:


train_df.Name


# In[ ]:


Title=[i.split(".")[0].split(",")[-1].strip() for i in train_df.Name]
train_df["Title"]=Title


# In[ ]:


f,ax=plt.subplots(figsize=(18,7))
sns.countplot(x=train_df.Title)
plt.show()


# In[ ]:


train_df.Title=train_df.Title.replace(["Don", "Rev", "Dr", "Mme", "Major", "Lady", "Sir", "Col", "Capt", "the Countess", "Jonkheer", "Dona"], "Other")


# In[ ]:


train_df.Title.unique()


# In[ ]:


new_title=[]
for i in train_df.Title:
    if i=="Master":
        new_title.append(0)
    elif i=="Miss" or i=="Mrs" or i=="Ms" or i=="Mlle":
        new_title.append(1)
    elif i=="Mr":
        new_title.append(2)
    elif i=="Other":
        new_title.append(3)
set(new_title)


# In[ ]:


train_df.Title=new_title


# In[ ]:


f,ax=plt.subplots(figsize=(18,7))
sns.countplot(x=train_df.Title)
plt.show()


# In[ ]:


g=sns.factorplot(x="Title", y="Survived", data=train_df, kind="bar")
g.set_xticklabels(["Master", "Mrs", "Mr", "Other"])
g.set_ylabels("Survival Possibility")
plt.show()


# In[ ]:


train_df=pd.get_dummies(train_df, columns=["Title"])


# In[ ]:


train_df.head()


# <a id="23" ></a><br>
# ## Family Size

# In[ ]:


train_df["Fsize"]=train_df.Parch+train_df.SibSp+1


# In[ ]:


g=sns.factorplot(x="Fsize", y="Survived", data=train_df, kind="bar")
g.set_ylabels("Survival Possibility")
plt.show()


# In[ ]:


new_fsize=[]
for each in train_df.Fsize:
    if each<=4:
        new_fsize.append(1)
    elif each>4:
        new_fsize.append(0)
train_df["family_size"]=new_fsize


# In[ ]:


train_df


# In[ ]:


g=sns.factorplot(x="family_size", y="Survived", data=train_df, kind="bar")
g.set_ylabels("Survival Possibility")
plt.show()


# In[ ]:


train_df=pd.get_dummies(train_df, columns=["family_size"])


# <a id="24" ></a><br>
# ## Embarked

# In[ ]:


train_df=pd.get_dummies(train_df, columns=["Embarked"])


# <a id="25" ></a><br>
# ## Ticket

# In[ ]:


train_df.Ticket


# In[ ]:


new_ticket=[]
for each in train_df.Ticket:
    if not each.isdigit():
        new_ticket.append(each.replace(".", "").replace("/", "").strip().split(" ")[0])
    else:
        new_ticket.append("x")
train_df["Ticket"]=new_ticket


# In[ ]:


train_df=pd.get_dummies(train_df, columns=["Ticket"], prefix="T")


# <a id="26" ></a><br>
# ## Pclass

# In[ ]:


sns.countplot(x="Pclass", data=train_df)
plt.show()


# In[ ]:


train_df=pd.get_dummies(train_df, columns=["Pclass"])


# <a id="27" ></a><br>
# ## Sex

# In[ ]:


train_df=pd.get_dummies(train_df, columns=["Sex"])


# In[ ]:


train_df.drop("Sex01", axis=1, inplace=True)


# <a id="28" ></a><br>
# ## Drop Passenger ID and Cabin

# In[ ]:


train_df


# In[ ]:


train_df.drop(["PassengerId", "Cabin"], axis=1, inplace=True)


# In[ ]:


train_df


# <a id="29" ></a><br>
# # Modeling

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# <a id="30" ></a><br>
# ## Train - Test Split

# In[ ]:


train_df_len


# In[ ]:


train_data=train_df[:train_df_len]


# In[ ]:


len(train_data)


# In[ ]:


test_data=train_df[train_df_len:]


# In[ ]:


len(test_data)


# In[ ]:


x=train_data.drop(["Survived", "Name", "SibSp", "Parch"], axis=1)
y=train_data["Survived"]
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=42)


# In[ ]:


x_train


# <a id="31" ></a><br>
# ## Simple Logistic Regression

# In[ ]:


logreg=LogisticRegression()
logreg.fit(x_train, y_train)
train_acc=round(logreg.score(x_train, y_train)*100,3)
test_acc=round(logreg.score(x_test, y_test)*100,3)
print("Training Accuracy:", train_acc, "%")
print("Test Accuracy:", test_acc, "%")


# <a id="32" ></a><br>
# ## Hyperparameter Tuning -- Grid Search -- Cross Validation
# * Decision Trees
# * SVM
# * Random Forest
# * KNN
# * Logistic Regression

# In[ ]:


rs=42
classifier=[DecisionTreeClassifier(random_state=rs),
           SVC(random_state=rs),
           RandomForestClassifier(random_state=rs),
           KNeighborsClassifier(),
           LogisticRegression(random_state=rs)]

dt_param_grid={"min_samples_split": range(10,500,20), 
               "max_depth": range(1,20,2)}

svc_param_grid={"kernel": ["rbf"], 
               "gamma": [0.001, 0.01, 0.1, 1],
               "C": [1,10,50,100,200,300,1000]}

rf_param_grid={"max_features": [1,3,10],
              "min_samples_split": [2,3,10],
              "min_samples_leaf": [1,3,10],
              "bootstrap": [False],
              "n_estimators": [100,300],
              "criterion": ["gini"]}

knn_param_grid={"n_neighbors": np.linspace(1,19,10, dtype=int).tolist(),
               "weights": ["distance", "uniform"],
               "metric": ["euclidean", "manhattan"]}

logreg_param_grid={"C": np.logspace(-3,3,7),
                  "penalty": ["l1", "l2"]}

classifier_param=[dt_param_grid, svc_param_grid, rf_param_grid, knn_param_grid, logreg_param_grid]


# In[ ]:


cv_result=[]
best_estimators=[]
for i in range(len(classifier)):
    clf=GridSearchCV(classifier[i], param_grid=classifier_param[i], cv=StratifiedKFold(n_splits=10), scoring="accuracy", n_jobs=-1, verbose=1)
    clf.fit(x_train, y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


# In[ ]:


cv_result=[100*each for each in cv_result]


# In[ ]:


results=pd.DataFrame({"Cross Validation Best Scores": cv_result, "ML Models": ["DecisionTreeClassifier", "SVM", "RandomForestClassifier", "KNeighborsClassifier", "LogisticRegression"]})
f,ax=plt.subplots(figsize=(12,7))
g = sns.barplot(data=results, y="ML Models", x="Cross Validation Best Scores")
g.set_ylabel("")
g.set_xlabel("Accuracy %")
plt.show()
for i in range(len(results)):
    print(results["ML Models"][i], "Accuracy:", results["Cross Validation Best Scores"][i], "%")


# <a id="33" ></a><br>
# ## Ensemble Modeling

# In[ ]:


voting_c=VotingClassifier(estimators=[("dt", best_estimators[0]), ("rf", best_estimators[2]), ("lr", best_estimators[4])],
                         voting="soft", n_jobs=-1)
voting_c=voting_c.fit(x_train, y_train)
print("Accuracy:", 100*accuracy_score(voting_c.predict(x_test), y_test), "%")


# <a id="34" ></a><br>
# ## Prediction and Submission

# In[ ]:


test=test_data.drop(["Survived", "Name", "Parch", "SibSp"], axis=1)
test_survived=pd.Series(voting_c.predict(test), name="Survived").astype(int)
results=pd.concat([test_PassengerId, test_survived], axis=1)


# In[ ]:


results.to_csv("submission.csv", index=False)

