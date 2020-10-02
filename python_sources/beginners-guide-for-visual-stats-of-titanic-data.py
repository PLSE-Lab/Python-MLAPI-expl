#!/usr/bin/env python
# coding: utf-8

# # TITANIC DISASTER

# ![alt text](https://vignette.wikia.nocookie.net/titanic/images/f/f9/Titanic_side_plan.png/revision/latest?cb=20180322183733)
# * On the Boat Deck there were **6** rooms labeled as **T, U, W, X, Y, Z** but only the **T** cabin is present in the dataset
# * **A**, **B** and **C** decks were only for 1st class passengers
# * **D** and **E** decks were for all classes
# * **F** and **G** decks were for both 2nd and 3rd class passengers
# * From going **A** to **G**, distance to the staircase increases which might be a factor of survival

# # INTRODUCTION
# 
# Aim of this kernel is Titanic:Machine Learning from Disaster one of popular competition among beginners.
# This is a beginner level kernel which focuses simple but effective analysis & codes. 
# First of all we must understand the data. If you work on a data without EDA you can't be a succesfull data scientist.
# Look at the picture above and imagine.
# 
# <font color='red'>
#       
# 1. [General View](#1)  
# 1. [Explotary Data Analysis](#2)
# 1. [Missing Values & Transformations](#3)
# 1. [Feature Engineering](#4)
# 1. [Outliers](#5)
# 1. [Modelling & Feature Importances](#6)
# 
# 
#      

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,GridSearchCV

import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
sns.set(style='white', context='notebook', palette='deep')


# In[ ]:


train=pd.read_csv("/kaggle/input/titanic/train.csv")
test=pd.read_csv("/kaggle/input/titanic/test.csv")


# <a id='1'></a><br>
# # 1. General View

# In[ ]:


import pandas_profiling
train.profile_report()


# In[ ]:


import pandas_profiling
test.profile_report()


# In[ ]:


train.describe().plot(kind="area",fontsize=16,figsize=(20,8),grid=True,colormap="rainbow")
plt.xlabel("Statistics")
plt.ylabel("Values")
plt.title("General Statistics of Titanic Dataset")
plt.show()
plt.savefig('graph1.png')
train.describe()


# In[ ]:


sns.pairplot(train,hue='Survived')
plt.show()


# <a id='2'></a><br>
# # 2. General Explotary Data Analysis
# Describe is the most important part of statistics. We should show these figures on charts

#    <font color='red'>
#    Categorical Variables

# In[ ]:


category1=['Pclass','Sex','SibSp','Parch','Embarked']

for c in category1:
    plt.figure(figsize=(10,5))
    sns.countplot(x=c,hue='Survived',data=train)
    plt.show()


# In[ ]:


train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


sns.factorplot(x='SibSp',y='Survived',data=train,kind='bar',size=6)
plt.show()
plt.savefig('graph3.png')


# In[ ]:


train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# <font color='red'>
#     Numerical Variables

# In[ ]:


train.boxplot(column='Fare',by='Embarked')
plt.show()


# In[ ]:


g=sns.FacetGrid(train,col='Survived')
g.map(sns.distplot,'Age',bins=25)
plt.show()


# In[ ]:


g=sns.FacetGrid(train,col='Survived',row='Pclass',size=2)
g.map(plt.hist,'Age',bins=25)
g.add_legend()
plt.show()


# In[ ]:


g=sns.FacetGrid(train,row='Embarked',size=5)
g.map(sns.pointplot,'Pclass','Survived','Sex')
g.add_legend()
plt.show()
plt.savefig('graph4.png')


# In[ ]:


g=sns.FacetGrid(train,col='Survived',row='Embarked',size=3)
g.map(sns.barplot,'Sex','Fare')
g.add_legend()
plt.show()


# In[ ]:


# convert Sex into categorical value 0 for male and 1 for female
train["Sex"] = train["Sex"].map({"male": 0, "female":1})
test["Sex"] = test["Sex"].map({"male": 0, "female":1})

#g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare","Pclass"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.heatmap(train.corr(),annot=True,cmap='cubehelix',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
plt.savefig('graph2.png')


# - Only Fare feature seems to have a significative correlation with the survival probability.
# - According to Heatmap Age negatively correlated with Pclass, Parch and SibSp so fill Age with the median age of similar rows according to Pclass, Parch and SibSp.

# <a id='3'></a><br>
# ## 3. Missing Values & Transformations

# Combine train and test

# In[ ]:


dataset =  pd.concat(objs=[train, test], axis=0,sort=False).reset_index(drop=True)


# In[ ]:


# Drop useless variables 
#dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)

# "S" is most frequent value of "Embarked"
dataset["Embarked"] = dataset["Embarked"].fillna("S")

dataset['Fare']=dataset['Fare'].fillna(np.mean(dataset[dataset['Pclass']==3]['Fare']))


## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med


# In[ ]:


# convert to indicator values Embarked 

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em",drop_first=True)


# In[ ]:


# Replace the Cabin number by the type of cabin 'X' if not
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")


# In[ ]:


# Create categorical values for Pclass
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")


# <a id='4'></a><br>
# # 4. Feature Engineering

# In[ ]:


# Create a family size descriptor from SibSp and Parch
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1


# In[ ]:


dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)
dataset = pd.get_dummies(dataset, columns = ["Title"])
dataset.drop(labels = ["Name"], axis = 1, inplace = True)


# In[ ]:


# Group family size according to factorplot
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


# Tickets with same prefixes may have a similar class and survival.

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")


# <a id='5'></a><br>
# # 5. Outliers

# First seperate train and test

# In[ ]:


train=dataset[dataset['Survived'].notnull()]
test=dataset[dataset['Survived'].isnull()]
test=test.drop(labels='Survived',axis=1)


# In[ ]:


# Outlier detection 

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# <a id='6'></a><br>
# # 6. Modelling & Feature Importances

# In[ ]:



x_train=train.drop(labels='Survived',axis=1)
#x_train=(x_train-x_train.min())/(x_train.max()-x_train.min())
y_train=train['Survived'].astype('int')
x_train.fillna(0,inplace=True)

X_train,X_test,Y_train,Y_test = train_test_split(x_train,y_train,test_size=0.25, random_state=0)

xgb=XGBClassifier(colsample_bytree=1.0,gamma=2.1,max_depth=5,min_child_weight=3,subsample=0.5,n_estimators=160,random_state=42)
xgb.fit(X_train,Y_train)
print('Score: ',xgb.score(X_train,Y_train))
y_pred=xgb.predict(X_test)
y_true=Y_test

cm=confusion_matrix(y_true,y_pred.round())
print(cm)
feature_importances = pd.DataFrame(xgb.feature_importances_,index = x_train.columns,columns=['importance']).sort_values('importance', ascending=False)
feature_importances


# In[ ]:


params = {
        'min_child_weight': [1, 2, 3],
        'gamma': [1.9, 2, 2.1, 2.2],
        'subsample': [0.4,0.5,0.6],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3,4,5]
        }
gd_sr = GridSearchCV(estimator=XGBClassifier(),
                     param_grid=params,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=1
                     )
gd_sr.fit(x_train, y_train)
best_parameters = gd_sr.best_params_
best_result = gd_sr.best_score_

print("Best result:", best_result)
pd.DataFrame({"Parameter":[i for i in best_parameters.keys()],
              "Best Values":[i for i in best_parameters.values()]})


# In[ ]:


xgb=XGBClassifier(colsample_bytree=0.8,gamma=2,max_depth=3,min_child_weight=3,subsample=0.6)
xgb.fit(x_train,y_train)
y_pred=xgb.predict(test)

submission=pd.DataFrame(columns=['PassengerId','Survived'])
submission['PassengerId']=test['PassengerId']
submission['Survived']=y_pred.astype(int)
submission.to_csv('submission.csv',header=True,index=False)

