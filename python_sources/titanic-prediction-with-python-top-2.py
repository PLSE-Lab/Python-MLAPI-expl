#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score,GridSearchCV

from sklearn import ensemble
import xgboost

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Load Data

# In[ ]:


raw_train=pd.read_csv('../input/train.csv')
raw_test=pd.read_csv('../input/test.csv')


# In[ ]:


raw_train.head(1)


# In[ ]:


raw_train.tail(1)


# In[ ]:


raw_test.head(1)


# In[ ]:


raw_test.tail(1)


# In[ ]:


data=pd.concat([raw_train, raw_test], axis=0).reset_index(drop=True)


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# # Visualization & Data Cleaning

# ### Cabin

# In[ ]:


data["Cabin"].isnull().sum()


# In[ ]:


# Replace NA as X, kepp initial
data["Cabin"]=data['Cabin'].fillna('X')
data['Cabin']=data['Cabin'].str.get(0)


# In[ ]:


sns.barplot(x="Cabin", y="Survived", data=data)


# ### Embarked

# In[ ]:


data['Embarked'].value_counts()


# In[ ]:


data[data['Embarked'].isnull()]


# In[ ]:


sns.boxplot(x="Embarked", y="Fare", hue="Pclass",data=data)


# In[ ]:


data['Embarked']=data['Embarked'].fillna('C')


# ### Fare

# In[ ]:


data[data['Fare'].isnull()]


# In[ ]:


fare=data[(data['Age'] >60) & (data['Embarked'] == "S") & (data['Pclass'] == 3)].Fare
data['Fare']=data['Fare'].fillna(fare.median())


# In[ ]:


sns.kdeplot(data.loc[data['Survived'] == 0, 'Fare'], label='0')
sns.kdeplot(data.loc[data['Survived'] == 1, 'Fare'], label='1')


# ### Name

# In[ ]:


data.Name


# In[ ]:


# Get Title from Name
data["Title"] = data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
data["Title"].value_counts()


# In[ ]:


data["Title"] = data["Title"].replace(['Mlle','Ms'], 'Miss')
data["Title"] = data["Title"].replace(['Mme'], 'Mrs')
data["Title"] = data["Title"].replace(['Rev', 'Dr', 'Col', 'Major', 'Capt'], 'Officer')
data["Title"] = data["Title"].replace(['the Countess', 'Don', 'Lady', 'Sir', 'Jonkheer', 'Dona'], 'Royalty')
data["Title"].value_counts()


# In[ ]:


sns.barplot(x="Title", y="Survived", data=data)


# ### Parch & SibSp

# In[ ]:


# Create a family size descriptor from SibSp and Parch
data["Fsize"] = data["SibSp"] + data["Parch"] + 1
sns.factorplot(x="Fsize",y="Survived",data = data)


# In[ ]:


# Create new feature of family size
def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
data['FamilyLabel']=data['Fsize'].apply(Fam_label)


# In[ ]:


sns.barplot(x="FamilyLabel", y="Survived", data=data)


# ### Pclass

# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=data)


# ### Sex

# In[ ]:


sns.barplot(x="Sex", y="Survived", data=data)


# ### Ticket

# In[ ]:


Ticket_Count = dict(data['Ticket'].value_counts())
data['TicketGroup'] = data['Ticket'].apply(lambda x:Ticket_Count[x])
data['TicketGroup'].value_counts()


# In[ ]:


sns.barplot(x='TicketGroup', y='Survived', data=data)


# In[ ]:


def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

data['TicketGroup'] = data['TicketGroup'].apply(Ticket_Label)
sns.barplot(x='TicketGroup', y='Survived', data=data)


# ### Age

# In[ ]:


data.info()


# In[ ]:


age_df = data[['Age', 'Pclass','Sex','Title','Fsize']]
age_df=pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
y = known_age[:, 0]
X = known_age[:, 1:]
rfr=xgboost.XGBClassifier()
rfr.fit(X, y)
predictedAges = rfr.predict(unknown_age[:, 1::])
data.loc[(data.Age.isnull()), 'Age' ] = predictedAges 


# # Modeling

# In[ ]:


data.info()


# In[ ]:


data.head(1)


# In[ ]:


data=data[['Survived','Age','Cabin','Embarked','Fare','Pclass','Sex','TicketGroup','FamilyLabel','Title']]
data=pd.get_dummies(data)


# In[ ]:


train=data[:len(raw_train)]
test=data[len(raw_train):].drop(['Survived'],axis=1)
x = train.drop(['Survived'],axis=1)
y = train.Survived


# In[ ]:


model = ensemble.RandomForestClassifier(random_state = 10, 
                                      warm_start = True,
                                      n_estimators = 26, 
                                      max_depth = 6, 
                                      max_features = 'sqrt')
model.fit(x,y)


# In[ ]:


predictions = model.predict(test)
submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"],
                           "Survived": predictions.astype(np.int32)})
# submission.to_csv("data/submission.csv", index=False)


# In[ ]:




