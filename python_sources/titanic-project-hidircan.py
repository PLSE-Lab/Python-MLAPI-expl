#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The sinking of Titanic is one of the most notorious shipwrecks in the history. In 1912, during her voyage, the Titanic sank after colliding with an iceberg then killed 1502 out of 2224 passengers and crew.
#  And now I'm observing that the sink of the titanic ship's machine learning algorithm to understand how people survived.
#    
# <font color = "yellow">
# Content:
#     
# 1. [Load and Check Data](#1)
# 1. [Variable Descriptions](#2)
#    * [Univariate Variable Analysis](#3)
#      * [Numerical Variable Analysis](#4)
#      * [Categorical Variable Analysis](#5)
#    
# 1. [Basic Data Analysis](#6)
# 1. [Outlier Detection](#7)
# 1. [Missing Value](#8)
#  * [Find missing Value](#9)
#  * [Fill Missing Value](#10)
# 1. [Visualization](#11)
#  * [Correlation Between SibSp -- ParCh -- Age -- Fare -- Survived](#12)
#  * [SibSp -- Survived](#13)
#  * [Parch -- Survived](#14)
#  * [Pclass -- Survived](#15)
#  * [Age -- Survive](#16)
#  * [Pclass -- Survived -- Age](#17)
#  * [Embarked -- Sex -- Pclass --Survived](#18)
#  * [Embarked -- Sex -- Fare -- Survived](#19)
#  * [Filling Missing Age Value](#20)
#  
# 1. [Feauture Engineering](#21)
#  * [Name -- Title](#22)
#  * [Family Size](#23)
#  * [Embarked](#24)
#  * [Ticket](#25)
#  * [Pclass](#26)
#  * [Sex](#27)
#  * [Drop Passenger Id and Cabin](#28)
# 1. [Modeling](#29)
#  * [Train Test Split](#30)
#  * [Simple Logistic Regresiion](#31)
#  * [Hyperparameter Tuning -- Grid Search -- Cross Validation](#32)
#  * [Ensemble Modeling](#33)
#  * [Prediction and Submission](#34)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

plt.style.use("seaborn-colorblind")
plt.style.use("seaborn-whitegrid")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id ="1"></a><br>
# # 1-Load and Check Data

# In[ ]:


#=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_passengerId = test_df["PassengerId"]


# <a id = "2" ></a><br>
# # 2- Variable Descriptions
# 
# 1. PassengerId : id of passenger
# 1. Survived: if died :0 ,lived :1
# 1. Pclass: passenger class
# 1. Name: name of passenger
# 1. Sex: gender of passenger
# 1. Age: age of passenger
# 1. SibSp: Sib means siblings, Sp means spouses
# 1. Parch: par means parents, ch means children
# 1. Ticket: ticket number of passenger
# 1. Fare: amount of money spent on ticket
# 1. Cabin: cabin category
# 1. Embarked: port where passenger embarked (C = Cherburg, S =Southhampton, Q=Queenstown
# 

# In[ ]:


train_df.info()


# * float64(2):Age, Fare
# * int64(5):PassengerID,Survived,PClass,SibSp,Parch
# * object(5):Name,Sex,Ticket,Cabin,Embarked

# <a id="3" ></a><br>
# # 2-a)Univarite Variable Analysis
# * Categorical Variable Analysis : Survived, Sex, Pclass,Embarked, Cabin, Name, Ticket, Sibsp and Parch
# * Numerical Variable Analysis : Fare, age and passengerId

# <a id="4" ></a><br>
# # 2-a-i)Categorical Variable Analysis

# In[ ]:


def bar_plot(variable):
    
    """
    input: varible exmple: "sex"
    output: bar plot& value count
    """
    
    # get feature
    var = train_df[variable]
    
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("The {} is: \n {}".format(variable,varValue))    
    


# In[ ]:


category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]
for c in category1:
    bar_plot(c)
    


# In[ ]:


category2 = ["Cabin","Name","Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))


# <a id="5"></a><br>
# # 2-a-ii)Numerical Variable Analysis

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable],bins=80)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


numVar = ["Fare","Age","PassengerId"]
for n in numVar:
    plot_hist(n)
    


# <a id="6"></a><br>
# # 3-Basic Data Analysis
# * Pclass - Survived
# * Sex - Survived
# * SibSp - Survived
# * Parch - Survived

# In[ ]:


# Pclass - Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index = False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


# Sex -Survived

train_df[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean().sort_values(by="Survived",ascending= False)


# In[ ]:


# SibSp - Survived

train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index =False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


#Parc-Survived
train_df[["Parch","Survived"]].groupby(["Parch"],as_index = False).mean().sort_values(by="Survived",ascending = False)


# In[ ]:


#Parch-SibSp -Survived
train_df[["Parch","SibSp","Survived"]].groupby(["Parch","SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


#Fare and Survived correlation
train_df[["Fare","Survived"]].groupby(["Fare"],as_index=False).mean().sort_values(by= "Survived",ascending= False)


# In[ ]:


train_df


# <a id="7"></a><br>
# # 4-Outlier Detection

# In[ ]:


def detectOutliers(df,features):
    outlier_indices = []
    for i in features:
        #1st quartile
        Q1 = np.percentile(df[i],25)
        #3rd quartile
        Q3 = np.percentile(df[i],75)
        #IQR
        IQR = Q3 - Q1
        #Outlier_step
        outlierStep = IQR * 1.5
        # detect oulier and their indeces
        outlier_listCol = df[((df[i] < Q1 - outlierStep)| (df[i] > Q3 + outlierStep))].index
        outlier_indices.extend(outlier_listCol)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i,k in outlier_indices.items() if k>2)
    
    return multiple_outliers


# In[ ]:


train_df.loc[detectOutliers(train_df,["Age","SibSp","Parch","Fare"])]
train_df = train_df.drop(detectOutliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop=True)
train_df


# <a id="8" ></a><br>
# # 5- Missing Value
#    * Find Missing Value
#    * Fill Missing Value

# In[ ]:


train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)


# <a id="9"></a><br>
# ## 5-a)Find Missing Value

# In[ ]:


train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# <a id="10"></a><br>
# ## 5-b)Fill Missing Value
# * Embarked has 2 missing value
# * Fare has only 1
# 

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df.boxplot(column = "Fare",by="Embarked")
plt.show()


# In[ ]:


train_df["Embarked"] = train_df["Embarked"].fillna("C")


# In[ ]:


train_df[train_df["Fare"].isnull()]


# In[ ]:


train_df["Fare"] = train_df["Fare"].fillna(train_df[train_df["Pclass"]==3]["Fare"].mean())


# In[ ]:


train_df[train_df["Fare"].isnull()]


# <a id="11" ></a><br>
# # 6- Visualization

# <a id="12" ></a><br>
# ## 6-a) Correlation Between SibSp -- ParCh -- Age -- Fare -- Survived

# In[ ]:


#seaborn
list1 = ["SibSp","Parch","Age","Fare","Survived"]
sns.heatmap(train_df[list1].corr(),annot = True,fmt = ".3f")


# Fare feature seems to have correlation with survived feature (0.265)

# <a id = "13"></a><br>
# ## 6-b) SibSp -- Survived

# In[ ]:


g = sns.factorplot(x="SibSp",y = "Survived",data = train_df,kind = "bar",size = 6)
g.set_ylabels("Survived Probabilty")
plt.show()


# * Having a lot of SibSp have less chance to survive
# * If sibsp == 0 or 1 or 2, passenger has more chance to survive
# * we can consider a new feature describing these categories

# <a id = "14"></a><br>
# ## 6-c) ParCh -- Survived

# In[ ]:


g = sns.factorplot(x= "Parch",y= "Survived",data = train_df,kind="bar",size=5)
g.set_ylabels("Survied Probability")
plt.show()


# * Sibsp and parch can be used for new feature extraction with th = 3
# * small families have more chance to survive
# * there is a standard deviation in survival of passenger with parch=3

# <a id="15"></a><br>
# ## 6-d) Pclass -- Survived

# In[ ]:


g = sns.factorplot(x= "Pclass",y = "Survived",data =train_df,kind = "bar",size= 5)
g.set_ylabels("Survived Probability")
plt.show()


# <a id = "16" ></a><br>
# ## 6-e) Age -- Survived

# In[ ]:


g = sns.FacetGrid(train_df,col = "Survived")
g.map(sns.distplot,"Age",bins = 25)
plt.show()


# * age <=10 has a high surviva rate
# * the oldest passengers (80) survived
# * large number of 20 years old didnt survive
# * most passengers are in 15-35 age range,
# * use age feature in training
# * use age distribution for missing value of age

# <a id="17"></a><br>
# ## 6-f) Pclass -- Survived -- Age

# In[ ]:


g = sns.FacetGrid(train_df,col = "Survived",row="Pclass",size = 2)
g.map(plt.hist,"Age",bins=25)
g.add_legend()
plt.show()


# <a id="18"></a><br>
# ## 6-g)Embarked -- Sex -- Pclass -- Survived
# 

# In[ ]:


g = sns.FacetGrid(train_df,row = "Embarked",size = 3)
g.map(sns.pointplot, "Pclass","Survived","Sex")
g.add_legend()
plt.show()


# * Female passengers have much better survival rate than males
# * males have better survival rate in pclass 3 in C
# * embarked ans sex will be used in training

# <a id = "19" ></a> <br>
# ## 6-h) Embarked -- Sex -- Fare -- Survived

# In[ ]:


g = sns.FacetGrid(train_df,col = "Embarked",row = "Survived",size = 2.3)
g.map(sns.barplot,"Sex","Fare")
g.add_legend()
plt.show()


# * Passengers who pay higher fare have better survival. Fare can be used as categorical for training

# <a id ="20"></a><br>
# ## 6-i)Filling Missing Age Value

# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:


sns.factorplot(x = "Sex",y="Age",data = train_df,kind="box",size = 4)
plt.show()


# Sex is not informative for age prediction, age distribution seems to be same

# In[ ]:


sns.factorplot(x = "Sex",y = "Age",hue= "Pclass",data = train_df ,kind = "box",size = 5)
plt.show()


# First class passengers are older than second . and seconds are older than third class passengers

# In[ ]:


sns.factorplot(x = "Parch",y = "Age",data = train_df,kind = "box",size = 7)
sns.factorplot(x = "SibSp",y = "Age",data= train_df,color = "red",kind = "box",size = 7)
plt.show()


# In[ ]:


train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]


# In[ ]:


sns.heatmap(train_df[["Age","Sex","Parch","SibSp","Pclass"]].corr(),annot = True)
plt.show()


# Age is not correlated with sex but correalated with Parch, SibSp, and Pclass

# In[ ]:


index_nan_age = list(train_df[train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][(train_df["Parch"]==train_df.iloc[i]["Parch"]) & (train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & (train_df["Pclass"] == train_df.iloc[i]["Pclass"])].median()
    age_med = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med


# In[ ]:


train_df["Age"][train_df["Age"].isnull()]


# <a id ="21"></a><br>
# ## 7) Feature Engineering

# <a id ="22"></a><br>
# ## 7-a) Name -- Title

# In[ ]:


train_df["Name"].head()


# In[ ]:


name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
train_df["Title"]


# In[ ]:


sns.countplot(x = "Title",data = train_df)
plt.xticks(rotation = 40)
plt.show()


# In[ ]:


# convert to categorical
train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Jonkheer","Dona","Major","Dr","Rev","Don","Sir","Col"],"other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Mrs" or i== "Mlle" or i == "Ms" else 2 if i == "Mr" else 3 for i in train_df["Title"]]


# In[ ]:


train_df["Title"].head(13)


# In[ ]:


g = sns.factorplot(x = "Title" , y = "Survived" , data = train_df,kind = "bar",size=5)
g.set_xticklabels(["Master","Miss-Mrs-Mlle-Ms","Mr","other"])
g.set_ylabels("Survived Probabilty")
plt.show()


# In[ ]:


train_df.drop(labels = ["Name"],axis = 1,inplace = True)


# In[ ]:


train_df = pd.get_dummies(train_df,columns = ["Title"])
train_df.head()


# <a id = "23" ></a><br>
# ## 7-b) Family Size

# In[ ]:


train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1


# In[ ]:


train_df.head()


# In[ ]:


g = sns.factorplot(x= "Fsize", y = "Survived", data = train_df, kind = "bar", size = 5)
g.set_ylabels("Survival Posibility")
plt.show()


# In[ ]:


train_df["family_size"] = [1 if i<5 else 0 for i in train_df["Fsize"]]


# In[ ]:


train_df.head(15)


# In[ ]:


sns.countplot(x = "family_size",data = train_df)
plt.show()


# In[ ]:


g = sns.factorplot(x = "family_size",y = "Survived", data = train_df,kind="bar",size=5)


# Small families have more chances to survive than large families

# In[ ]:


train_df = pd.get_dummies(train_df,columns=["family_size"])
train_df.head()


# <a id = "24"></a><br>
# ## 7-c) Embarked

# In[ ]:


g = sns.countplot(x = "Embarked",data = train_df)
plt.show()


# In[ ]:


train_df = pd.get_dummies(train_df,columns = ["Embarked"])
train_df.head()


# <a id ="25"></a><br>
# ## 7-d) Ticket

# In[ ]:


train_df["Ticket"].head(20)


# In[ ]:


liste = []
for i in train_df["Ticket"]:
    if not i.isdigit():
        liste.append(i.replace("."," ").replace("/"," ").strip().split(" ")[0])
    else:
        liste.append("X")
train_df["Ticket"] = liste


# In[ ]:


train_df = pd.get_dummies(train_df,columns = ["Ticket"],prefix = "T")


# In[ ]:


train_df.head()


# <a id="26"></a><br>
# ## 7-e) Pclass

# In[ ]:


g = sns.countplot(x = "Pclass",data = train_df)
plt.show()


# In[ ]:


train_df = pd.get_dummies(train_df, columns = ["Pclass"])
train_df.head()


# <a id ="27" ></a><br>
# ## 7-f) Sex

# In[ ]:


train_df["Sex"] = train_df [ "Sex"].astype("category")
train_df = pd.get_dummies(train_df,columns = ["Sex"])


# In[ ]:


train_df.head()


# <a id = "28"></a><br>
# ## 7-g)Drop Passenger ID and Cabin

# In[ ]:


train_df.drop(labels = ["PassengerId","Cabin"],axis = 1,inplace = True)


# In[ ]:


train_df.head()


# <a id= "29"></a><br>
# # 8)Modeling

# In[ ]:


from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# <a id ="30"></a><br>
# ## 8-a)Train Test Split

# In[ ]:


train_df_len


# In[ ]:


test = train_df[train_df_len:]
test.drop(labels = ["Survived"],axis = 1,inplace = True)


# In[ ]:


test.head()


# In[ ]:


train = train_df[:train_df_len]
X_train = train.drop(labels= "Survived",axis = 1)
y_train = train["Survived"]
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size = 0.33,random_state =42)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(test))


# <a id = "31"></a><br>
# ## 8-b) Simple Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train,y_train)
acc_log_train = round(logreg.score(X_train,y_train)* 100,2)
acc_log_test = round(logreg.score(X_test,y_test)*100,2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))


# <a id="32"></a><br>
# ## 8-c) Hyperparameter Tuning - GridSearch - CrossValidation
# 
#  We Will compare 5 machine learning classifier and evaluate mean accuracy of each of them by stratified Cross validation
# * DECISION TREE
# * SWM 
# * RANDOM FOREST
# * KNN
# * LOGISTIC REGRESSION

# In[ ]:


random_state = 42
# All the classifiers are listed in classifier[] like this:
classifier = [DecisionTreeClassifier( random_state = random_state),
             SVC(random_state=random_state),
             RandomForestClassifier(random_state=random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]
#Decision Tree Classifier parameters are Min Samples split, and Max Depth
dt_param_grid = {"min_samples_split": range(10,500,20),
                "max_depth": range(1,20,2)}
#SVC parameters are: Kernel, Gamma, and C
svc_param_grid = {"kernel":["rbf"],
                 "gamma": [ 0.001,0.01,0.1,1],
                 "C":[1,10,50,100,200,300,1000]}
#Random Forest Classifier parameters are max_feature,min_samples_split,min_samples_leaf,bootstrap,n_estimators,criterion:
rf_param_grid = {"max_features":[1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,200],
                "criterion":["gini"]}
# Logistic Regression Parameters are C, Penalty
logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty":["l1","l2"]}
# KNNeighbors parameters are n_neighbors,weights,metric
knn_param_grid = {"n_neighbors": np.linspace(1,19,10,dtype = int).tolist(),
                 "weights": [ "uniform","distance"],
                 "metric":["euclidean","manhattan"]}

# All of below parameters grids are listed on classifier_param
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                    knn_param_grid]


# In[ ]:


cv_result = []
best_estimator = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i],param_grid = classifier_param[i],cv = StratifiedKFold(n_splits = 10),scoring = "accuracy",n_jobs = -1,verbose = 1)
    clf.fit(X_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimator.append(clf.best_estimator_)
    print(cv_result[i])


# In[ ]:


#Visualization of the cross validation results
cv_results = pd.DataFrame({"Cross Validation Means": cv_result,"Ml Models": [ "DecisionTreeClassifier","SVC","RandomForestClassifier","LogisticRegression","KNeighborsClassifier"]})
g = sns.barplot("Cross Validation Means","Ml Models",data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")


# <a id = "33" ></a><br>
# ## 8-d) Emsemble Modeling

# In[ ]:


votingC =  VotingClassifier(estimators = [("dt",best_estimator[0]),
                                         ("rfc",best_estimator[2]),
                                         ("lr",best_estimator[3])],voting = "soft",n_jobs = -1)
votingC = votingC.fit(X_train,y_train)
print(accuracy_score(votingC.predict(X_test),y_test))


# <a id = "34" ></a><br>
# ## 8-e) Prediction and Submission

# In[ ]:


test_survived = pd.Series(votingC.predict(test),name = "Survived").astype(int)
results = pd.concat([test_passengerId, test_survived],axis = 1)
results.to_csv("titanic_hidircan.csv",index = False)

