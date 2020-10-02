#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current sessio
data_Test=pd.read_csv("../input/titanic/test.csv")
data_Train=pd.read_csv("../input/titanic/train.csv")
Test=data_Test.copy()
Train=data_Train.copy()
Test
Train
Train.head()


# **Combining all the data**

# In[ ]:


tb=[Train,Test]
Data=pd.concat(tb)


# In[ ]:


sns.barplot(x="Sex",y="Survived",data=Data)


# **Finding which predictors have missing data**

# In[ ]:


Data.isnull().sum()


# In[ ]:


Test


# **Extracting TITLE from the name. For filling the missing values in Age**

# In[ ]:


Data["Title"]=Data["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
Data


# In[ ]:


Data[["Title","Age"]].groupby("Title").sum()


# In[ ]:


Data["Title"].value_counts()


# In[ ]:


Data[["Age","Title"]].groupby("Title").mean()


# **Combining the Titles and Age assignment**

# In[ ]:


Data["Title"]=Data["Title"].replace(["Mme"],"Mrs")
Data["Title"]=Data["Title"].replace(["Ms","Mlle"],"Miss")
Data["Title"]=Data["Title"].replace(["Countess","Lady","Sir"],"Royal")
Data["Title"]=Data["Title"].replace(["Don","Dona","Jonkheer","Major","Rev","Col","Capt","Dr"],"Rare")


# In[ ]:


Data[["Age","Title"]].groupby("Title").mean()


# In[ ]:



for i in Data["Title"]:
    if i=="Master":
        Data["Age"]=Data["Age"].fillna(5)
    elif i=="Miss":
        Data["Age"]=Data["Age"].fillna(22)
    elif i=="Mr":
        Data["Age"]=Data["Age"].fillna(32)
    elif i=="Mrs":
        Data["Age"]=Data["Age"].fillna(40)
    elif i=="Rare":
        Data["Age"]=Data["Age"].fillna(40)
    else:
        Data["Age"]=Data["Age"].fillna(43)
Data["Title"].value_counts()


# In[ ]:


Data["Fare"]=Data["Fare"].fillna(12)
Data.isnull().sum()


# **Filling missing data for 'Embarked' by checking the 'Fares'**

# In[ ]:


sns.barplot(x="Embarked",y="Fare",data=Data)


# In[ ]:


Data["Embarked"]=Data["Embarked"].fillna('C')
Data.isnull().sum()


# **Dropping the useless variables**

# In[ ]:


Data=Data.drop("Ticket",axis=1)
Data=Data.drop("Cabin",axis=1)
Data=Data.drop("Name",axis=1)
Data.head()


# **Splitting the training and validation data**

# In[ ]:


#Tr=dict(tuple(Data.groupby("Survived")))
#NorTrain=pd.concat(Tr)
#NorTrain=NorTrain.sort_values(by=["PassengerId"])
#NorTrain
NorTrain=Data.loc[lambda Data:~(Data["Survived"].isnull())]
NorTrain


# In[ ]:


NorTest=Data.loc[lambda Data: Data["Survived"].isnull()]
NorTest


# In[ ]:


from sklearn import model_selection
from sklearn import svm
help(model_selection)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# In[ ]:


ySurvived=NorTrain["Survived"]
NorTrain=NorTrain.drop(["Survived"],axis=1)
pidTrain=NorTrain["PassengerId"]
NorTrain=NorTrain.drop(["PassengerId"],axis=1)
pidFinal=NorTest["PassengerId"]


# In[ ]:


NorTrain.dtypes
enTrain=pd.get_dummies(NorTrain)
enTest=pd.get_dummies(NorTest)
FinalTrain,Final=enTrain.align(enTest,join='left',axis=1)
Final["Title_Royal"]=Final["Title_Royal"].fillna(FinalTrain["Title_Royal"])
Final


# **Fitting the Data: Random Forest Classifier**

# In[ ]:


rfc=RandomForestClassifier(n_estimators=150,
    criterion='gini',
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto')
f=np.mean(cross_val_score(rfc, FinalTrain, ySurvived, cv=10))
print(f)


# **Hyperparameter Optimization: Random Search**

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[ ]:


rfc_ran=RandomizedSearchCV(RandomForestClassifier(),param_distributions = random_grid, n_iter = 100, 
                           cv = 3, verbose=2, random_state=42, n_jobs = -1)
rfc_ran.fit(FinalTrain,ySurvived)


# In[ ]:


rfc_ran.best_params_


# In[ ]:


rfc_ran=RandomForestClassifier(n_estimators=600,min_samples_split=10,min_samples_leaf=1,
                               max_features='sqrt',max_depth=110,bootstrap='True')
f=np.mean(cross_val_score(rfc_ran, FinalTrain, ySurvived, cv=10))
print(f)


# **HyperParameter Optimization: Grid Search**

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = {
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3, "sqrt"],
    'min_samples_leaf': [1, 3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [200, 300, 600, 1000]
}


# In[ ]:


rfc_grid = GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
rfc_grid.fit(FinalTrain,ySurvived)


# In[ ]:


rfc_grid.best_params_


# **Fitting the model with the best parameters found**

# In[ ]:


rfc_final=RandomForestClassifier(n_estimators=300,min_samples_split=10,min_samples_leaf=1,
                               max_features=3,max_depth=80,bootstrap='True')
f=np.mean(cross_val_score(rfc_final, FinalTrain, ySurvived, cv=10))
print(f)


# In[ ]:


rfc_final.fit(FinalTrain,ySurvived)
Predictions=rfc_final.predict(Final)


# In[ ]:


ID=pidFinal.array
Submission = pd.DataFrame({
    'PassengerId':ID,
    'Survived':Predictions
})
Submission.to_csv('Submission.csv', index = False)

