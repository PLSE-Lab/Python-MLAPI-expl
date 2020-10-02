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


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


Titanic =  pd.read_csv("/kaggle/input/titanic/train.csv",index_col=0)
len(Titanic)
Titanic.head()


# In[ ]:


Titanic['Survived'].value_counts()


# Overview of the Data

# In[ ]:


Titanic.info()


# In[ ]:


Titanic['Cabin'].fillna(value='Unknown',inplace=True)
Titanic['Embarked'].fillna(value='Unknown',inplace=True)


# **Exploratory Data Analysis**

# Count Plots of Categorical Variables

# In[ ]:


sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})
sns.set_context("paper", font_scale=0.9)
fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(10,6), dpi=100)
# sns.set_context('notebook',font_scale=1)
# sns.set_style('whitegrid')
sns.countplot(x='Sex',data=Titanic,ax=axes[0][0])
sns.countplot(x='Pclass',data=Titanic,ax=axes[0][1])
sns.countplot(x='Survived',data=Titanic,ax=axes[0][2])
sns.countplot(x='SibSp',data=Titanic,ax=axes[1][0])
sns.countplot(x='Parch',data=Titanic,ax=axes[1][1])
sns.countplot(x='Embarked',data=Titanic,ax=axes[1][2])


# Distribution plots of continuous variables

# In[ ]:


sns.set()
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12,4), dpi=100)
sns.distplot(Titanic[Titanic['Age'].notnull()]['Age'],ax=axes[0],kde=False)
sns.distplot(Titanic[Titanic['Fare'].notnull()]['Fare'],ax=axes[1],kde=False)


# **Analytical Graphs to Check which categorical values are making a difference**

# In[ ]:


sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})
sns.set_context("paper", font_scale=0.9)
fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(10,6), dpi=100)
# sns.set_context('notebook',font_scale=1)
# sns.set_style('whitegrid')
sns.countplot(x='Sex',data=Titanic,ax=axes[0][0],hue='Survived')
sns.countplot(x='Pclass',data=Titanic,ax=axes[0][1],hue='Survived')
sns.countplot(x='Survived',data=Titanic,ax=axes[0][2])
sns.countplot(x='SibSp',data=Titanic,ax=axes[1][0],hue='Survived')
sns.countplot(x='Parch',data=Titanic,ax=axes[1][1],hue='Survived')
sns.countplot(x='Embarked',data=Titanic,ax=axes[1][2],hue='Survived')


# In[ ]:


fig=plt.figure(figsize =(15,15),dpi=100)
g = sns.FacetGrid(Titanic, col="Pclass",  row="Sex")
g = g.map(sns.countplot, "Survived")


# In[ ]:


sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})
sns.set_context("paper", font_scale=0.9)
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,4), dpi=100)
sns.stripplot(x="Sex", y="Age", data=Titanic,jitter=True,hue='Survived',dodge=True,ax=axes[0])
sns.stripplot(x="Sex", y="Fare", data=Titanic,jitter=True,hue='Survived',dodge=True,ax=axes[1])


# In[ ]:


sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})
sns.set_context("paper", font_scale=0.9)
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,4), dpi=100)
sns.stripplot(x="Embarked", y="Age", data=Titanic,jitter=True,hue='Survived',dodge=True,ax=axes[0])
sns.stripplot(x="Embarked", y="Fare", data=Titanic,jitter=True,hue='Survived',dodge=True,ax=axes[1])


# In[ ]:


sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})
sns.set_context("paper", font_scale=0.9)
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,4), dpi=100)
sns.stripplot(x="Parch", y="Age", data=Titanic,jitter=True,hue='Survived',dodge=True,ax=axes[0])
sns.stripplot(x="Parch", y="Fare", data=Titanic,jitter=True,hue='Survived',dodge=True,ax=axes[1])


# Assuming that Age less than 1 is for child below 1 year threfore the data is real , Dropping the missing rows for Embarked and Dropping the columns Name , tiket and cabin

# In[ ]:


Titanic.drop(['Name','Ticket','Cabin','Age','Fare'],inplace=True,axis=1)
Titanic.info()


# In[ ]:


Test =  pd.read_csv("/kaggle/input/titanic/test.csv",index_col=0)


# In[ ]:


Test.drop(['Name','Ticket','Cabin','Age','Fare'],inplace=True,axis=1)
Test.info()


# We have only two missing values for Embarked column hence dropping that rows of missing values

# In[ ]:


Titanic.dropna(subset=['Embarked'],inplace=True)


# Filling the missing Age by mean for male and female seperately

# In[ ]:


Titanic[Titanic['Sex']=='male'] = Titanic[Titanic['Sex']=='male'].fillna(Titanic[Titanic['Sex']=='male']['Age'].mean())


# In[ ]:


Titanic[Titanic['Sex']=='female'] = Titanic[Titanic['Sex']=='female'].fillna(Titanic[Titanic['Sex']=='female']['Age'].mean())


# In[ ]:


Titanic.info()


# Looking for existing corelation through heat map

# In[ ]:


Corr = Titanic.corr()
fig=plt.figure(figsize =(10,10),dpi=65)
sns.heatmap(Corr,cmap='summer',linewidths=1,linecolor='black',annot=True)


# In[ ]:


fig =plt.figure(figsize=(15,15),dpi=100)
sns.set_context('notebook',font_scale=1.3)
sns.set_style('whitegrid')
g=sns.pairplot(Titanic,hue='Survived')


# **Preprocessing the Data for Model Building******

# In[ ]:


X = Titanic.iloc[:,1:].values
Y=Titanic.iloc[:,0].values


# In[ ]:


Xtest = Test.iloc[:,:].values


# In[ ]:


Xtest


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# Label encoding the non numeric columns

# In[ ]:


LabelCol = [1,4]
for i in LabelCol:
    LabelX = LabelEncoder()
    X[:,i]=LabelX.fit_transform(X[:,i])
    Xtest[:,i]=LabelX.transform(Xtest[:,i])


# In[ ]:


X


# In[ ]:


Xtest


# Building the Model Applying Logistic Rgression only however I have imported other classifier as well

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# For ROC curve
def plot_roc_curve(fpr, tpr,col,lab):
    plt.plot(fpr, tpr, color=col, label=lab)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[ ]:


X_train,X_test,Y_Train,Y_Test = train_test_split(X,Y,test_size= 0.20)


# In[ ]:


classifierLogistic = LogisticRegression()
classifierLogistic.fit(X_train, Y_Train)
y_pred_logit = classifierLogistic.predict(X_test)

cm_logit = confusion_matrix(Y_Test, y_pred_logit)
print('Confusion matrix for Logistic',cm_logit)

accuracy_logit = accuracy_score(Y_Test, y_pred_logit)
precision_logit =precision_score(Y_Test, y_pred_logit)
recall_logit =  recall_score(Y_Test, y_pred_logit)
f1_logit = f1_score(Y_Test, y_pred_logit)
print('accuracy_logistic :',accuracy_logit)
print('precision_logistic :',precision_logit)
print('recall_logistic :',recall_logit)
print('f1-score_logistic :',f1_logit)
auc_logit = roc_auc_score(Y_Test, y_pred_logit)
print('AUC_logistic : %.2f' % auc_logit)


# In[ ]:


classifierRF = RandomForestClassifier(n_estimators=50,max_depth=10,criterion='entropy')
classifierRF.fit(X, Y)
y_pred_RF = classifierRF.predict(X_test)

cm_RF = confusion_matrix(Y_Test, y_pred_RF)
print('Confusion matrix Random Forest',cm_RF)

accuracy_RF = accuracy_score(Y_Test, y_pred_RF)
precision_RF =precision_score(Y_Test, y_pred_RF)
recall_RF =  recall_score(Y_Test, y_pred_RF)
f1_RF = f1_score(Y_Test, y_pred_RF)
print('accuracy random forest :',accuracy_RF)
print('precision random forest :',precision_RF)
print('recall random forest :',recall_RF)
print('f1-score random forest :',f1_RF)
auc_RF = roc_auc_score(Y_Test, y_pred_RF)
print('AUC: %.2f' % auc_RF)


# In[ ]:


a=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
b=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
fig =plt.figure(figsize=(15,15),dpi=50)
fpr, tpr, thresholds = roc_curve(Y_Test,y_pred_RF)
plt.plot(fpr, tpr,color ='orange',label ='Logistic' )
plt.plot(a,b,color='black',linestyle ='dashed')
plt.legend(fontsize=15)
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)


# **Applying the model to our test data**

# In[ ]:


# Test =  pd.read_csv("/kaggle/input/titanic/test.csv",index_col=0)


# In[ ]:


# Test.head()


# In[ ]:


# Test.drop(['Name','Ticket','Cabin'],inplace=True,axis=1)
# Test.info()


# In[ ]:


# Test['Fare'].index[Test['Fare'].apply(np.isnan)]


# Filling the Missing value of Fare with the mean of fares for those data which has Embarked as S

# In[ ]:


# Test['Fare'][1044] =Test[Test['Embarked']=='S']['Fare'].mean()


# Filling the Age with mean for male and Female Separately

# In[ ]:


# Test[Test['Sex']=='male'] = Test[Test['Sex']=='male'].fillna(Test[Test['Sex']=='male']['Age'].mean())
# Test[Test['Sex']=='female'] = Test[Test['Sex']=='female'].fillna(Test[Test['Sex']=='female']['Age'].mean())


# In[ ]:


# Xtest = Test.iloc[:,:].values


# In[ ]:


# abelCol = [1,6]
# for i in LabelCol:
#     LabelX = LabelEncoder()
#     Xtest[:,i]=LabelX.fit_transform(Xtest[:,i])


# In[ ]:


y_pred_test = classifierRF.predict(Xtest)


# In[ ]:


output = pd.DataFrame({'PassengerId': Test.index, 'Survived': y_pred_test})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

