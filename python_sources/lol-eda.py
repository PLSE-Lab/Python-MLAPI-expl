#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
#    Firstly I want to say that I am playing LoL for 5 years and the first time I see this data I said to myself that I should analys this data. In this data I just tried to predict win chance of the blue team according to their inputs.I've shocked beacuse of the thruth that there is no any correlation between win and number of wards you placed or destroyed. I talked about it in Wins -- WardsDestroyed part under Visualization part. Finally, I am actually happy to work on two things that I'm loving to do. 
# 
# 1. [Entering and Seperating Data](#1)
# 1. [Visualization](#2)
#      * [Wins -- Kills](#10)
#      * [Wins -- AvgLevel](#11)
#      * [Wins -- Deaths](#12)
#      * [Wins -- FirstBlood](#13)
#      * [Wins -- EliteMonsters](#14)
#      * [Wins -- Dragons](#15)
#      * [Wins -- Heralds](#16)
#      * [Wins -- Assists](#17)
#      * [Wins -- TowersDestroyed](#18)
#      * [Wins -- RoundedCSPerMin](#19)
#      * [Wins -- WardsPlaced](#20)
#      * [Wins -- WardsDestroyed](#21)
#      * [Wins -- TotalGold](#22)
#      * [Wins -- TotalExperience](#23)
#      * [Wins -- GoldPerMin](#24)
#      * [Wins -- ExperienceDiff](#25)
# 1. [Outlier Detection](#40)
# 1. [Feature Engineering](#3)
#      * [Wins -- Kills](#26)
#      * [Wins -- AvgLevel](#27)
#      * [Wins -- Deaths](#28)
#      * [Wins -- FirstBlood](#29)
#      * [Wins -- EliteMonsters](#30)
#      * [Wins -- Dragons](#31)
#      * [Wins -- Heralds](#32)
#      * [Wins -- Assists](#33)
#      * [Wins -- TowersDestroyed](#34)
#      * [Wins -- RoundedCSPerMin](#35)
#      * [Wins -- TotalGold](#36)
#      * [Wins -- TotalExperience](#37)
#      * [Wins -- GoldPerMin](#38)
#      * [Wins -- ExperienceDiff](#39)
# 1. [Drop Other Elements](#4)
# 1. [Modelling](#5)
#      * [Train Test Split](#6)
#      * [Simple Logistic Regression](#7)
#      * [Hyperparameter Tuning -- Grid Search -- Cross Validation](#8)
#      * [Ensemble Modelling](#9)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings
warnings.filterwarnings("ignore")

from collections import Counter

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <a id="1"></a><br>
# # Entering and Seperating Data

# In[ ]:


data=pd.read_csv("/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


id_list=["Wins","WardsPlaced","WardsDestroyed","FirstBlood","Kills","Deaths","Assists","EliteMonsters","Dragons","Heralds","TowersDestroyed","TotalGold","AvgLevel","TotalExperience","TotalMinionsKilled","TotalJungleMinionsKilled","GoldDiff","ExperienceDiff","CSPerMin","GoldPerMin"]
blue_data=[]
blue_data=pd.DataFrame(blue_data)
for i in id_list:
    blue_data[i]=data["blue"+i]


# In[ ]:


blue_data.head()


# In[ ]:


id_list=id_list[1:]


# In[ ]:


print(id_list)


# In[ ]:


red_data=[]
red_data=pd.DataFrame(red_data)
wins=[0 if each == 1 else 1 for each in blue_data["Wins"]]
red_data["Wins"]=wins
for i in id_list:
    red_data[i]=data["red"+i]


# In[ ]:


red_data.head()


# <a id="2"></a><br>
# # Visualization

# In[ ]:


corr_list=["Wins","WardsPlaced","WardsDestroyed","TotalGold","TotalExperience","TotalMinionsKilled","GoldDiff","ExperienceDiff","CSPerMin","GoldPerMin","Kills","AvgLevel","Deaths","FirstBlood","EliteMonsters","Dragons","Heralds","Assists","TowersDestroyed"]

f,ax=plt.subplots(figsize=(12,12))
sns.heatmap(blue_data[corr_list].corr(),annot=True,linewidths=.5,fmt=".1f",ax=ax)
plt.show()


# <a id="10"></a><br>
# ## Wins -- Kills

# In[ ]:


g = sns.factorplot(x="Kills",y="Wins", data=blue_data,kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# * From this graph we can say that for a player who had 12 or more kills has really high chance to win.
# * The player with between 6 and 12 kills player has nearly same cahnce to win and lose.
# * The others are most probably lost their games.

# <a id="11"></a><br>
# ## Wins -- AvgLevel

# In[ ]:


g = sns.factorplot(x="AvgLevel",y="Wins", data=blue_data, kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# * From this graph we can say that if the average level is higher than 7.2 they have more cahnce to others.

# <a id="12"></a><br>
# ## Wins -- Deaths

# In[ ]:


g = sns.factorplot(x="Deaths",y="Wins", data=blue_data,kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# * According to graph we can say that a player with 4 or less has more chance to win a ranked game.

# <a id="13"></a><br>
# ## Wins -- FirstBlood

# In[ ]:


g = sns.factorplot(x="FirstBlood",y="Wins", data=blue_data,kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# * In this kinda situations there is just 2 probability. If we look at the graph we can see that if the player have taken the first blood player has more chance to win.

# <a id="14"></a><br>
# ## Wins -- EliteMonsters

# In[ ]:


g = sns.factorplot(x="EliteMonsters",y="Wins",data=blue_data,kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# * According to the graph we can say that if you kill more elite monsters you have higher chance to win.

# <a id="15"></a><br>
# ## Wins -- Dragons

# In[ ]:


g = sns.factorplot(x="Dragons",y="Wins",data=blue_data,kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# * As we expected from the past graph killing more dragons makes you closer to being a winner.

# <a id="16"></a><br>
# ## Wins -- Heralds

# In[ ]:


g = sns.factorplot(x="Heralds",y="Wins",data=blue_data,kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# * The explanaton that I did in Wins -- Dragons part is also applies for this part. 

# <a id="17"></a><br>
# ## Wins -- Assists

# In[ ]:


g = sns.factorplot(x="Assists",y="Wins",data=blue_data,kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# * If the player's assists are greater or equal to 15 has more chance to other situations.
# * If the player's assists are between 7 and 15 has nearly same chance to win or lose.
# * The last probability has more chance to lose than win.

# <a id="18"></a><br>
# ## Wins -- TowersDestroyed

# In[ ]:


g = sns.factorplot(x="TowersDestroyed",y="Wins",data=blue_data,kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# * From the graph we can say that the more you destroy tower the more you ave chance to win.

# <a id="19"></a><br>
# ## Wins -- RoundedCSPerMin

# In[ ]:


blue_data["RoundedCSPerMin"]=np.round(blue_data["CSPerMin"])


# In this part I wanted to say that I firstly looked at graphs of TotalMinionsKilled and TotalJungleMinionsKilled and after that I saw classifiaction is really hard because of the variation of the numbers. Then I decided to look CSPerMin. "CS" is shorthand for "Creep Score", and is the number of creeps (neutral jungle creeps or the enemy's creeps) that you have scored the last hit on and obtained gold from. After look at te CSPerMin I saw same problem with others but after that I thought that I can round it and then use it.
# 
# CS information from: https://gaming.stackexchange.com/questions/58190/what-does-cs-mean-and-how-do-you-increase-it-the-best-you-can

# In[ ]:


g = sns.factorplot(x="RoundedCSPerMin",y="Wins",data=blue_data,kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# * According to the graph we can say that if your creep score is 25 or higher you have really more cahce to others.

# <a id="20"></a><br>
# ## Wins -- WardsPlaced

# Actually if we looked at the heatmap we can see that there is no correlation between Wins and Wardsplaced. But as I said in the introduction part I think that wards are one of the most important part of a LoL game. Because of that I wanted to examine. For examining I looked at it in 5 parts.

# In[ ]:


print("max:",np.max(blue_data.WardsPlaced),"min:",np.min(blue_data.WardsPlaced))


# In[ ]:


i=0
while i<5:
    data1=blue_data[blue_data.WardsPlaced<=(i+1)*50]
    data2=data1[data1.WardsPlaced>i*50]
    g=sns.factorplot(x="WardsPlaced",y="Wins",data=data2,kind="bar",height=12)
    g.set_ylabels("Win Probability")
    plt.show()
    i=i+1


# <a id="21"></a><br>
# ## Wins -- WardsDestroyed

# In[ ]:


print("max:",np.max(blue_data.WardsDestroyed),"min:",np.min(blue_data.WardsDestroyed))


# In[ ]:


g=sns.factorplot(x="WardsDestroyed",y="Wins",data=blue_data,kind="bar",height=12)
g.set_ylabels("Win Probability")
plt.show()


# This markdown is for WardsPlaced and WardsDestroyed part:
# * After I looked their graphs I also decided that win rates are randomly scatterd into the data. Because of that I decided to drop this par tin [Drop Other Elements](#4) part.

# <a id="22"></a><br>
# ## Wins -- TotalGold

# After this part I have the same problem with wards part and I classified them according to their average values.

# In[ ]:


print(np.mean(blue_data.TotalGold))


# In[ ]:


liste=[1 if each>=16503 else 0 for each in blue_data.TotalGold]
blue_data["totalgold"]=liste
g=sns.factorplot(x="totalgold",y="Wins",data=blue_data,kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# <a id="23"></a><br>
# ## Wins -- TotalExperience

# In[ ]:


print("mean:",np.mean(blue_data.TotalExperience),"max:",np.max(blue_data.TotalExperience),"min:",np.min(blue_data.TotalExperience))


# In[ ]:


liste=[1 if each>=20000 else 0 for each in blue_data.TotalExperience]
blue_data["totalexp"]=liste
g=sns.factorplot(x="totalexp",y="Wins",data=blue_data,kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# <a id="24"></a><br>
# ## Wins -- GoldPerMin

# In[ ]:


print("mean:",np.mean(blue_data.GoldPerMin),"max:",np.max(blue_data.GoldPerMin),"min:",np.min(blue_data.GoldPerMin))


# In[ ]:


liste=[1 if each>=2000 else 0 for each in blue_data.GoldPerMin]
blue_data["goldpermin"]=liste
g=sns.factorplot(x="goldpermin",y="Wins",data=blue_data,kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# <a id="25"></a><br>
# ## Wins -- ExperienceDiff

# In[ ]:


print("mean:",np.mean(blue_data.ExperienceDiff),"max:",np.max(blue_data.ExperienceDiff),"min:",np.min(blue_data.ExperienceDiff))


# In[ ]:


liste=[1 if each>=0 else 0 for each in blue_data.ExperienceDiff]
blue_data["expdiff"]=liste
g=sns.factorplot(x="expdiff",y="Wins",data=blue_data,kind="bar",height=7)
g.set_ylabels("Win Probability")
plt.show()


# <a id="40"></a><br>
# # Outlier Detection

# In[ ]:


def detect_outliers(df,features):
    outlier_indices=[]
    
    for c in features:
        #1st quartile
        Q1 = np.percentile(df[c],25)
        #3rd quartile
        Q3= np.percentile(df[c],75)
        # IQR
        IQR= Q3-Q1
        # Outlier step
        outlier_step=IQR*1.5
        #detect outliers and their indices
        outlier_list_col=df[(df[c]<Q1-outlier_step) |( df[c]>Q3+outlier_step)].index
        # Store indices
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices=Counter(outlier_indices)
    multiple_outliers=list(i for i, v in outlier_indices.items() if v>2)
    
    return multiple_outliers


# In[ ]:


blue_data.loc[detect_outliers(blue_data,["Kills","AvgLevel","Deaths","FirstBlood","EliteMonsters","Dragons","Heralds","Assists","TowersDestroyed","RoundedCSPerMin","totalgold","totalexp","goldpermin","expdiff"])]


# In[ ]:


blue_data=blue_data.drop(detect_outliers(blue_data,["Kills","AvgLevel","Deaths","FirstBlood","EliteMonsters","Dragons","Heralds","Assists","TowersDestroyed","RoundedCSPerMin","totalgold","totalexp","goldpermin","expdiff"]),axis=0).reset_index(drop = True)


# <a id="3"></a><br>
# # Feature Engineering

# <a id="26"></a><br>
# ## Wins -- Kills

# In[ ]:


blue_data.Kills=[2 if i>=12 else 1 if i>=6 and i<12 else 0 for i in blue_data.Kills]


# In[ ]:


blue_data.Kills.unique()


# In[ ]:


sns.countplot(x="Kills",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["Kills"])
blue_data.head()


# <a id="27"></a><br>
# ## Wins -- AvgLevel

# In[ ]:


blue_data.AvgLevel=[1 if i>=7.2 else 0 for i in blue_data.AvgLevel]


# In[ ]:


blue_data.AvgLevel.unique()


# In[ ]:


sns.countplot(x="AvgLevel",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["AvgLevel"])
blue_data.head()


# <a id="28"></a><br>
# ## Wins -- Deaths

# In[ ]:


blue_data.Deaths=[1 if i<5 else 0 for i in blue_data.Deaths]


# In[ ]:


blue_data.Deaths.unique()


# In[ ]:


sns.countplot(x="Deaths",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["Deaths"])
blue_data.head()


# <a id="29"></a><br>
# ## Wins -- FirstBlood

# In[ ]:


sns.countplot(x="FirstBlood",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["FirstBlood"])
blue_data.head()


# <a id="30"></a><br>
# ## Wins -- EliteMonsters

# In[ ]:


sns.countplot(x="EliteMonsters",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["EliteMonsters"])
blue_data.head()


# <a id="31"></a><br>
# ## Wins -- Dragons

# In[ ]:


sns.countplot(x="Dragons",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["Dragons"])
blue_data.head()


# <a id="32"></a><br>
# ## Wins -- Heralds

# In[ ]:


sns.countplot(x="Heralds",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["Heralds"])
blue_data.head()


# <a id="33"></a><br>
# ## Wins -- Assists

# In[ ]:


blue_data.Assists=[1 if i>=6 and i<15 else 2 if i>=15 else 0 for i in blue_data.Assists]


# In[ ]:


blue_data.Assists.unique()


# In[ ]:


sns.countplot(x="Assists",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["Assists"])
blue_data.head()


# <a id="34"></a><br>
# ## Wins -- TowersDestroyed

# In[ ]:


sns.countplot(x="TowersDestroyed",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["TowersDestroyed"])
blue_data.head()


# <a id="35"></a><br>
# ## Wins -- RoundedCSPerMin

# In[ ]:


blue_data.RoundedCSPerMin=[1 if i>=24 else 0 for i in blue_data.RoundedCSPerMin]


# In[ ]:


blue_data.RoundedCSPerMin.unique()


# In[ ]:


sns.countplot(x="RoundedCSPerMin",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["RoundedCSPerMin"])
blue_data.head()


# <a id="36"></a><br>
# ## Wins -- TotalGold

# In[ ]:


sns.countplot(x="totalgold",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["totalgold"])
blue_data.head()


# <a id="37"></a><br>
# ## Wins -- TotalExperience

# In[ ]:


sns.countplot(x="totalexp",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["totalexp"])
blue_data.head()


# <a id="38"></a><br>
# ## Wins -- GoldPerMin

# In[ ]:


sns.countplot(x="goldpermin",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["goldpermin"])
blue_data.head()


# <a id="39"></a><br>
# ## Wins -- ExperienceDiff

# In[ ]:


sns.countplot(x="expdiff",data=blue_data)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


blue_data=pd.get_dummies(blue_data,columns=["expdiff"])
blue_data.head()


# <a id="4"></a><br>
# # Drop Other Elements

# In[ ]:


blue_data.columns


# In[ ]:


blue_data=blue_data.drop(["TotalGold","TotalExperience","TotalMinionsKilled","TotalJungleMinionsKilled","GoldDiff","ExperienceDiff","CSPerMin","GoldPerMin"],axis=1)
blue_data.head()


# In[ ]:


blue_data.columns


# <a id="5"></a><br>
# # Modelling

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# <a id="6"></a><br>
# ## Train Test Split

# In[ ]:


length=6000
test=blue_data[length:]
test.drop(["Wins"],axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:


train=blue_data[:length]
X_train=train.drop(["Wins"],axis=1)
Y_train=train.Wins

x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.3,random_state=42)
print("x_train:",len(x_train))
print("x_test:",len(x_test))
print("y_train:",len(y_train))
print("y_test:",len(y_test))
print("test:",len(test))


# <a id="7"></a><br>
# ## Simple Logistic Regression

# In[ ]:


logreg=LogisticRegression()
logreg.fit(x_train,y_train)
acc_log_train=round(logreg.score(x_train,y_train)*100,2)
acc_log_test=round(logreg.score(x_test,y_test)*100,2)
print("Training Accuracy: %{}".format(acc_log_train))
print("Test Accuracy: %{}".format(acc_log_test))


# <a id="8"></a><br>
# ## Hyperparameter Tuning -- Grid Search -- Cross Validation

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
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


# In[ ]:


cv_results=pd.DataFrame({"Cross Validation Means":cv_result,"ML Models":["DecisionTreeClassifier","SVC","RandomForestClassifier","LogisticRegression","KNeighborsClassifier"]})

g= sns.barplot("Cross Validation Means","ML Models",data=cv_results)
g.set_xlabel("Mean Acc.")
g.set_title("Cross Validation Scores")
plt.show()


# <a id="9"></a><br>
# ## Ensemble Modelling

# In[ ]:


votingC=VotingClassifier(estimators=[("dt",best_estimators[0]),
                                    ("rf",best_estimators[2]),
                                    ("lr",best_estimators[3])],
                                    voting="soft",n_jobs=-1)
votingC=votingC.fit(x_train,y_train)
print(accuracy_score(votingC.predict(x_test),y_test))

