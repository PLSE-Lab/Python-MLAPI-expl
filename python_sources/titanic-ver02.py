#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np


# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_raw = train_df.copy()
test_raw = test_df.copy()
combine = [train_df, test_df]


# EDA

# Age 177 null
# 
# Cabin 687 null
# 
# Embarked 2 null

# In[ ]:


for_info = pd.DataFrame(train_df.dtypes).T.rename(index = {0 : "dtypes"})
for_info = for_info.append(pd.DataFrame(train_df.isnull().sum()).T.rename(index = {0 : "null count"}))
for_info


# In[ ]:


for_info_test = pd.DataFrame(test_df.dtypes).T.rename(index = {0 : "dtypes"})
for_info_test = for_info_test.append(pd.DataFrame(test_df.isnull().sum()).T.rename(index = {0 : "null count"}))
for_info_test


# In[ ]:


train_df.describe()


# In[ ]:


print(train_df['Survived'].value_counts())
l=['Not Survived','Survived']
ax=train_df['Survived'].value_counts().plot.pie(autopct='%.2f%%',figsize=(6,6),labels=l)
#autopct='%.2f%%' is to show the percentage text on the plot
ax.set_ylabel('')


# In[ ]:


print(train_df['Sex'].value_counts())
l=['male','female']
ax=train_df['Sex'].value_counts().plot.pie(autopct='%.2f%%',figsize=(6,6),labels=l)
#autopct='%.2f%%' is to show the percentage text on the plot
ax.set_ylabel('')


# 20s, 30s are popular

# In[ ]:


sns.set(style = 'whitegrid')
ax = sns.kdeplot(train_df["Age"])
ax.set_title("Age")
ax.set_xlabel("Age")


# 3 class is much more than others

# In[ ]:


print(train_df['Pclass'].value_counts())
l = ["3", "1", "2"]
ax =  train_df['Pclass'].value_counts().plot.pie(autopct='%.2f%%',figsize=(6,6),labels=l)
ax.set_ylabel('')


# S is much more than other

# In[ ]:


print(train_df["Embarked"].value_counts())
l = ["S", "C", "Q"]
ax = train_df["Embarked"].value_counts().plot.pie(autopct = "%.2f%%", figsize = (6,6), labels = l)
ax.set_ylabel('')


# Survived rate of female is better than male

# In[ ]:


train_df[['Sex','Survived']].groupby('Sex').mean()


# 1 class survived rate is over 50% and female 1&2 class survived rate is over 90%

# In[ ]:


train_df[['Pclass',"Survived"]].groupby("Pclass").mean()


# In[ ]:


train_df[['Sex',"Pclass",'Survived']].groupby(['Sex','Pclass']).mean()


# 1 class most higher survived
# 3 class most lower survived

# In[ ]:


sns.countplot(x = 'Pclass', hue = 'Survived' , data = train_df)


# In[ ]:


train_df[['Embarked','Survived']].groupby('Embarked').mean()


# In[ ]:


train_df[["SibSp", "Survived"]].groupby("SibSp").mean()


# In[ ]:


train_df[["Parch", "Survived"]].groupby("Parch").mean()


# In[ ]:


ax=train_df[['Parch','Survived']].groupby('Parch').mean().plot.line(figsize=(8,4))
ax.set_ylabel('Survival')
sns.despine()


# In[ ]:


ax=train_df[['SibSp','Survived']].groupby('SibSp').mean().plot.line(figsize=(8,4))
ax.set_ylabel('Survival')
sns.despine()


# In[ ]:


a = sns.FacetGrid(train_df, col = 'Survived', hue = "Sex")
a.map(sns.distplot, "Age")


# In[ ]:


a = sns.FacetGrid(train_df, col  = 'Pclass', row ="Survived",  hue = "Sex")
a.map(sns.distplot, "Age")


# In[ ]:


a = sns.FacetGrid(train_df, col  = 'Pclass', row ="Survived",  hue = "Sex")
a.map(sns.distplot, "Fare")


# without Embarked C female survived rate is better

# In[ ]:


a = sns.FacetGrid(train_df, col = "Embarked")
a.map(sns.pointplot, "Pclass", "Survived", "Sex")
a.add_legend()


# In[ ]:


train_df.groupby(["Embarked", "Sex"])["Embarked"].count()


# In[ ]:


pd.crosstab(train_df.Pclass, train_df.Survived, margins = True). style. background_gradient(cmap = 'summer_r')


# In[ ]:


pd.crosstab([train_df.Sex,train_df.Survived],train_df.Pclass,margins=True).style.background_gradient(cmap='summer_r')


# Deal Null value

# MISS = Girl under 12 years,
# 
# MSTR = Boy under 12years,
# 
# INF = Under 2 years.
# 
# https://english.stackexchange.com/questions/320719/abbreviation-for-master
# 
# master is meaning of young boy

# In[ ]:


title = []
title = train_df.Name.str.extract('([A-Za-z]+)\.')
train_df['title'] = title


# In[ ]:


print(title[0].value_counts())


# Check without Mr, Miss, Mrs, Master
# 
# ALL of male replace to Mr
# 
# young female be Miss
# 
# old female be Mrs
# 
# Dr has one female(Mrs)

# In[ ]:


train_df[["title","Age","Sex"]].groupby(["title","Sex"]).mean()


# In[ ]:


# to train, test set
for dataset in combine:
    title = dataset.Name.str.extract('([A-Za-z]+)\.')
    dataset['title'] = title
    tt = pd.DataFrame(title[0].value_counts())
    other_list = tt.index.values[4:]
    #other_list = ['Dr', 'Rev', 'Mlle', 'Major', 'Col','Jonkheer', 'Don', 'Lady', 'Countess', 'Mme', 'Ms', 'Sir', 'Capt']

    other = pd.DataFrame()
    for i in other_list:
        other = other.append(dataset[dataset["title"] == i])
    # miss 3/4 position is 30
        for i in other.index.values:
            if other.Sex[i] == "male":
                other.title[i] = "Mr"
            elif other.Age[i] <30:
                other.title[i] = "Miss"
            else:
                other.title[i] = "Mrs"
    dataset.loc[other.index.values] = other


# In[ ]:


# to train, test set
for dataset in combine:
    dataset['Sex'].replace(["male","female"], [0,1], inplace =True)
    dataset['title'].replace(["Mr","Mrs","Miss","Master"],[0,1,2,3], inplace =True)


# Embarked

# In[ ]:


# to train, test set
for dataset in combine:
    dataset.Embarked.fillna("S",inplace = True)


# age

# In[ ]:


train_df[["title","Age","Sex"]].groupby(["title","Sex"]).agg(["mean","median","min","max"])


# In[ ]:


# to train, test set
for dataset in combine:
    for i in range(4):
        dataset.loc[(dataset.Age.isnull())&(dataset.title == i),'Age'] = dataset[dataset.title == i]["Age"].mean()


# Family, Alone

# In[ ]:


# to train, test set
for dataset in combine:
    dataset["Family"] = dataset.SibSp + dataset.Parch
    dataset['Alone']=0
    dataset.loc[dataset.Family==0,'Alone']=1


# In[ ]:


# to train, test set
for dataset in combine:
    for i in range(4):
        dataset.loc[(dataset.Age.isnull())&(dataset.title == i),'Age'] = dataset[dataset.title == i]["Age"].mean()
        


# Age band
# 
# if age band size 5 - 16~32 is 502 values 
# 
# I think it's too much 
# 
# so make age band bigger 
# 
# 6 to 7 isn't good
# 
# 8 can be cut 10/20/30/40/50/60/70
# 
# So I choose 8
# 
# 

# In[ ]:


pd.cut(train_df.Age,7).value_counts()


# In[ ]:


#choose this
pd.cut(train_df.Age,8).value_counts()


# In[ ]:


# to train, test set
for dataset in combine:
    dataset["ageband"] = pd.cut(dataset.Age,8, labels = [0,1,2,3,4,5,6,7])
    dataset.ageband = dataset.ageband.astype(int)


# what is Fare == 0
# 
# someone cheating in ship? or something data error?
# 
# The samething
# All of them are male/Alone/S
# Some of them are class 1 
# 

# In[ ]:


# to train, test set
for dataset in combine:
    for i in range(4):
        dataset.loc[(dataset.Fare.isnull())&(dataset.Pclass == i),'Fare'] = dataset[dataset.Pclass == i]["Fare"].median()


# Before think
# The Fare is different from where they are aboard and which class
# 
# And The Titanic move S->C->Q place 
# 
# but any different from place
# 
# Most of people who aboard at Q are Class 3

# In[ ]:


train_cut_fare = train_df.loc[(train_df.Fare<55)&train_df.Fare>0]


# In[ ]:


train_cut_fare[["Pclass","Fare","Embarked"]].groupby("Pclass").agg(["count","mean","median","min","max"])


# In[ ]:


train_df.loc[(train_df.Fare>80)&train_df.Fare>0].count()


# In[ ]:


sns.factorplot(x = "Embarked", y = "Fare", col = "Pclass", data = train_cut_fare, kind = "box")


# In[ ]:


sns.factorplot(x = "Pclass", y = "Fare", col = "Embarked", data = train_cut_fare, kind = "box")


# In[ ]:


train_df['fareband'] = pd.qcut(train_df.Fare,7)
train_df.groupby(['fareband'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')


# In[ ]:


train_df['fareband'] = pd.qcut(train_df.Fare,6)
train_df.groupby(['fareband'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')


# Around 50 is stq and above 1 class 3/4 fair

# In[ ]:


# to train, test set
for dataset in combine:
    dataset['fareband'] = pd.qcut(dataset.Fare,6,labels = [0,1,2,3,4,5])
    dataset.fareband = dataset.fareband.astype(int)


# In[ ]:


# to train, test set
for dataset in combine:
    dataset['Embarked'].replace(["S","C","Q"], [0,1,2], inplace =True)


# In[ ]:


# to train, test set
for dataset in combine:
    dataset.drop(["PassengerId", "Name", "Age", "SibSp", "Parch", "Ticket", "Fare","Cabin"],axis =1, inplace =True)


# In[ ]:


data = train_df.copy()
data.info()


# In[ ]:


colormap = sns.diverging_palette(220, 10, as_cmap = True)
sns.heatmap(data.corr(),annot=True,cmap=colormap,linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# In[ ]:


#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix


# In[ ]:


train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']


# In[ ]:


model = LogisticRegression()
model.fit(train_X, train_Y)
prediction3 = model.predict(test_X)
print("Accuracy of the LosgsticRegression is ", metrics.accuracy_score(prediction3, test_Y))


# In[ ]:


model=DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction4=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction4,test_Y))


# In[ ]:


model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_Y)
prediction7=model.predict(test_X)
print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction7,test_Y))


# In[ ]:


from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
kfold = KFold(n_splits =10, random_state = 22)
xyz = []
accuracy = []
std = []
classifiers = ["Logistic Regression", "Decision Tree","Random Forest"]
models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators = 100)]
for i in models:
    model = i
    cv_result = cross_val_score(model, X, Y, cv = kfold, scoring = "accuracy")
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_model_dataframe2 = pd.DataFrame({"CV Mean" : xyz, "Std": std}, index = classifiers)
new_model_dataframe2


# GridSearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV
n_estimators = range(500,1500,100)
hyper = {"n_estimators" : n_estimators}
gd = GridSearchCV(estimator = RandomForestClassifier(random_state=0), param_grid= hyper, verbose = True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)


# In[ ]:


from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf = VotingClassifier(estimators = [("RF", RandomForestClassifier(n_estimators = 900,
                                                                               random_state =0)),
                                                   ("LR", LogisticRegression(C = 0.05)),
                                                  ("DT", DecisionTreeClassifier(random_state = 0))],
                                   voting = "soft").fit(train_X, train_Y)
print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))
cross=cross_val_score(ensemble_lin_rbf,X,Y, cv = 10,scoring = "accuracy")
print('The cross validated score is',cross.mean())


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())


# In[ ]:


import xgboost as xg
xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
result=cross_val_score(xgboost,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())


# In[ ]:


f,ax=plt.subplots(2,2,figsize=(15,12))

model = LogisticRegression()
model.fit(train_X, train_Y)
model_coef = model.coef_
pd.Series(abs(model_coef.flatten()), X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])
ax[0,0].set_title('Feature Importance in LogisticRegression')

model=DecisionTreeClassifier()
model.fit(train_X,train_Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1])
ax[0,1].set_title('Feature Importance in DecisionTreeClassifier')

model=RandomForestClassifier(n_estimators=500,random_state=0)
model.fit(train_X,train_Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0])
ax[1,0].set_title('Feature Importance in Random Forests')

model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],)
ax[1,1].set_title('Feature Importance in XgBoost')
plt.show()


# 
# Reference
# Titanic Data Science Solutions
# 
# https://www.kaggle.com/startupsci/titanic-data-science-solutions
# 
# A Data Science Framework: To Achieve 99% Accuracy
# 
# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
# 
# EDA To Prediction(DieTanic)
# 
# https://www.kaggle.com/ash316/eda-to-prediction-dietanic
