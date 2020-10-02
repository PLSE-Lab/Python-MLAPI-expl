#!/usr/bin/env python
# coding: utf-8

# # Titanic: on the top with a simple model
# In this kernel I intend to use nested cross validation to choose between Random Forest and SVM. 
# 
# 
# #### <span style ='color: purple'> Please, upvote if you find useful! Also, check my other kernel about classification: </span> [Classification: Review with Python](http://www.kaggle.com/goldens/classification-review-with-python) 
# 
# 
# ### Steps:
# * 1- Preprocessing and exploring
#     * 1.1- Imports
#     * 1.2- Types
#     * 1.3 - Missing Values
#     * 1.4 - Exploring
#     * 1.5 - Feature Engineering
#     * 1.6 - Prepare for models
# * 2- Nested Cross Validation
# * 3- Submission
#     
#    
# 

# ## 1- Preprocessing and exploring

# ### 1.1 - Imports

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import warnings


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
test2=pd.read_csv("../input/test.csv")
titanic=pd.concat([train, test], sort=False)
len_train=train.shape[0]


# ### 1.2 - Types

# In[ ]:


titanic.dtypes.sort_values()


# In[ ]:


titanic.select_dtypes(include='int').head()


# In[ ]:


titanic.select_dtypes(include='object').head()


# In[ ]:


titanic.select_dtypes(include='float').head()


# ## 1.2 - Missing values

# In[ ]:


titanic.isnull().sum()[titanic.isnull().sum()>0]


# ### Fare

# In[ ]:


train.Fare=train.Fare.fillna(train.Fare.mean())
test.Fare=test.Fare.fillna(train.Fare.mean())


# ### Cabin

# In[ ]:


train.Cabin=train.Cabin.fillna("unknow")
test.Cabin=test.Cabin.fillna("unknow")


# ### Embarked

# In[ ]:


train.Embarked=train.Embarked.fillna(train.Embarked.mode()[0])
test.Embarked=test.Embarked.fillna(train.Embarked.mode()[0])


# ### Age
# Considering title
# Inspired on: https://medium.com/i-like-big-data-and-i-cannot-lie/how-i-scored-in-the-top-9-of-kaggles-titanic-machine-learning-challenge-243b5f45c8e9

# In[ ]:


train['title']=train.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
test['title']=test.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())


# In[ ]:


newtitles={
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"}


# In[ ]:


train['title']=train.title.map(newtitles)
test['title']=test.title.map(newtitles)


# In[ ]:


train.groupby(['title','Sex']).Age.mean()


# In[ ]:


def newage (cols):
    title=cols[0]
    Sex=cols[1]
    Age=cols[2]
    if pd.isnull(Age):
        if title=='Master' and Sex=="male":
            return 4.57
        elif title=='Miss' and Sex=='female':
            return 21.8
        elif title=='Mr' and Sex=='male': 
            return 32.37
        elif title=='Mrs' and Sex=='female':
            return 35.72
        elif title=='Officer' and Sex=='female':
            return 49
        elif title=='Officer' and Sex=='male':
            return 46.56
        elif title=='Royalty' and Sex=='female':
            return 40.50
        else:
            return 42.33
    else:
        return Age 


# In[ ]:


train.Age=train[['title','Sex','Age']].apply(newage, axis=1)
test.Age=test[['title','Sex','Age']].apply(newage, axis=1)


# ### 1.3 - Exploring

# In[ ]:


warnings.filterwarnings(action="ignore")
plt.figure(figsize=[12,10])
plt.subplot(3,3,1)
sns.barplot('Pclass','Survived',data=train)
plt.subplot(3,3,2)
sns.barplot('SibSp','Survived',data=train)
plt.subplot(3,3,3)
sns.barplot('Parch','Survived',data=train)
plt.subplot(3,3,4)
sns.barplot('Sex','Survived',data=train)
plt.subplot(3,3,5)
sns.barplot('Ticket','Survived',data=train)
plt.subplot(3,3,6)
sns.barplot('Cabin','Survived',data=train)
plt.subplot(3,3,7)
sns.barplot('Embarked','Survived',data=train)
plt.subplot(3,3,8)
sns.distplot(train[train.Survived==1].Age, color='green', kde=False)
sns.distplot(train[train.Survived==0].Age, color='orange', kde=False)
plt.subplot(3,3,9)
sns.distplot(train[train.Survived==1].Fare, color='green', kde=False)
sns.distplot(train[train.Survived==0].Fare, color='orange', kde=False)


# SibSp and Parch don't seem to have a clear relationship with the target, so put them together can be a good idea.
# For Ticket and Cabin a good strategie can be count the number of caracteres.

# ### 1.4 Feature Engineering

# In[ ]:


train['Relatives']=train.SibSp+train.Parch
test['Relatives']=test.SibSp+test.Parch

train['Ticket2']=train.Ticket.apply(lambda x : len(x))
test['Ticket2']=test.Ticket.apply(lambda x : len(x))

train['Cabin2']=train.Cabin.apply(lambda x : len(x))
test['Cabin2']=test.Cabin.apply(lambda x : len(x))

train['Name2']=train.Name.apply(lambda x: x.split(',')[0].strip())
test['Name2']=test.Name.apply(lambda x: x.split(',')[0].strip())


# In[ ]:


warnings.filterwarnings(action="ignore")
plt.figure(figsize=[12,10])
plt.subplot(3,3,1)
sns.barplot('Relatives','Survived',data=train)
plt.subplot(3,3,2)
sns.barplot('Ticket2','Survived',data=train)
plt.subplot(3,3,3)
sns.barplot('Cabin2','Survived',data=train)


# ### 1.4 - Prepare for model

# In[ ]:


#droping features I won't use in model
#train.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin']
train.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
test.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)


# In[ ]:


titanic=pd.concat([train, test], sort=False)


# In[ ]:


titanic=pd.get_dummies(titanic)


# In[ ]:


train=titanic[:len_train]
test=titanic[len_train:]


# In[ ]:


# Lets change type of target
train.Survived=train.Survived.astype('int')
train.Survived.dtype


# In[ ]:


xtrain=train.drop("Survived",axis=1)
ytrain=train['Survived']
xtest=test.drop("Survived", axis=1)


# # 2 - Nested Cross Validation

# ### Random Forest

# In[ ]:


RF=RandomForestClassifier(random_state=1)
PRF=[{'n_estimators':[10,100],'max_depth':[3,6],'criterion':['gini','entropy']}]
GSRF=GridSearchCV(estimator=RF, param_grid=PRF, scoring='accuracy',cv=2)
scores_rf=cross_val_score(GSRF,xtrain,ytrain,scoring='accuracy',cv=5)


# In[ ]:


np.mean(scores_rf)


# ### SVM

# In[ ]:


svc=make_pipeline(StandardScaler(),SVC(random_state=1))
r=[0.0001,0.001,0.1,1,10,50,100]
PSVM=[{'svc__C':r, 'svc__kernel':['linear']},
      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]
GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=2)
scores_svm=cross_val_score(GSSVM, xtrain.astype(float), ytrain,scoring='accuracy', cv=5)


# In[ ]:


np.mean(scores_svm)


# # 3 - Submission

# In[ ]:


model=GSSVM.fit(xtrain, ytrain)


# In[ ]:


pred=model.predict(xtest)


# In[ ]:


output=pd.DataFrame({'PassengerId':test2['PassengerId'],'Survived':pred})


# In[ ]:


output.to_csv('submission.csv', index=False)

