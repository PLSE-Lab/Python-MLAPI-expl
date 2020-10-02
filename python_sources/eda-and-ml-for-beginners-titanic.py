#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data_train= pd.read_csv("/kaggle/input/titanic/train.csv")
data_train.head()


# In[ ]:


#Printing columns
data_train.columns


# In[ ]:


data_test=pd.read_csv("/kaggle/input/titanic/test.csv")
data_test.head()


# In[ ]:


data_test.columns


# In[ ]:


#checking if the data has null values or not, if it have how many does it contains
data_train.isnull().sum()


# In[ ]:


#Minimum and Maximum age
print("Maximum age :", data_train["Age"].max())
print("Minimum age :", data_train["Age"].min())
print("Mean age :", data_train["Age"].mean())


# In[ ]:


#Minimum and Maximum fare
print("Maximum Fare", data_train["Fare"].max())
print("Minimum Fare", data_train["Fare"].min())
print("Mean Fare", data_train["Fare"].mean())


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

#How many mail and female were present
sc=data_train["Sex"].value_counts()
print(sc)
sc.plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True)


# In[ ]:


#How many from male and female survived or dead
sns.countplot("Sex",data=data_train,hue="Survived")


# In[ ]:


#How many Survived
f,ax=plt.subplots(1,2,figsize=(14,8))
surv=data_train["Survived"].value_counts()
print(surv)

surv.plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

#How many people survived or not from each class
sns.countplot("Survived",data=data_train,ax=ax[1],hue="Pclass")


# In[ ]:


sns.countplot("Survived",data=data_train,hue="Pclass")


# In[ ]:


data_train.head(10)


# In[ ]:


#preference for rescuing  according to sex and class 
sns.factorplot('Pclass','Survived',hue='Sex',data=data_train)
plt.show()


# In[ ]:


#Survived accordin to age 
f,ax=plt.subplots(1,2,figsize=(20,10))
data_train[data_train["Survived"]==0].Age.plot.hist(ax=ax[0],bins=20,color='red')
ax[0].set_title('Survived=0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
data_train[data_train["Survived"]==1].Age.plot.hist(ax=ax[1],bins=20,color='green')
ax[1].set_title('Survived=1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
data_train[data_train["Survived"]==0].Age.plot.hist(ax=ax[0],bins=20,color='red')
ax[0].set_title('Survived=0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
data_train[data_train["Survived"]==1].Age.plot.hist(ax=ax[1],bins=20,color='green')
ax[1].set_title('Survived=1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()


# In[ ]:


#finding correlations between the parameters of the dataset using heatmap from seaborn.
import seaborn as sns
corr=data_train.corr()
sns.heatmap(corr,annot=True)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show


# In[ ]:


data_train.head()


# In[ ]:


#feature engineering
#since the age have so much values, so converting it in categorical values

data_train['Age Band']=0

data_train.loc[data_train['Age']<=16,'Age Band']=0
data_train.loc[(data_train['Age']>16)&(data_train['Age']<=32), 'Age Band']=1
data_train.loc[(data_train['Age']>32)&(data_train['Age']<=48),'Age Band']=2
data_train.loc[(data_train['Age']>48)&(data_train['Age']<=64),'Age Band']=3
data_train.loc[data_train['Age']>64,'Age Band']=4

data_train.head(10)


# In[ ]:


data_train.head()
data_train['Age Band'].value_counts()


# In[ ]:


#using pd.qcut which make ranges according to the dataset and the number of ranges want here is 4 given in function.
data_train["Fare Range"]= pd.qcut(data_train["Fare"],4)

#data_train["Fare Range"].value_counts()

#data_train.groupby(['Fare Range'])['Survived'].mean()

data_train["Fare Range"].value_counts()


# In[ ]:


#Converting the fare column into categorical value, so it makes easy to understand and predict from the features.
data_train["Fare cat"]=0
data_train.loc[data_train["Fare"]<=7.91,"Fare cat"]=0
data_train.loc[(data_train["Fare"]>7.91)&(data_train["Fare"]<=14.454),"Fare cat"]=1
data_train.loc[(data_train['Fare']>14.454)&(data_train['Fare']<=31),'Fare cat']=2
data_train.loc[(data_train['Fare']>31)&(data_train['Fare']<=513),'Fare cat']=3
data_train.head()


# In[ ]:


#Visualising who has given much preference to rescue by factor plot.
sns.factorplot('Fare cat', 'Survived', hue= 'Sex', data=data_train)
plt.show()


# In[ ]:


#Converting letters to numeric, 

data_train["Sex"].replace(["male","female"],[0,1],inplace=True)
data_train["Embarked"].replace(["S","C","Q"],[0,1,2],inplace=True)
#data_train['Sex'].replace(['male','female'],[0,1],inplace=True)
#data_train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data_train.head()


# In[ ]:


data_train.drop(['PassengerId','Name','Age','Ticket','Fare','Cabin','Fare Range'], axis=1, inplace=True)


# In[ ]:


data_train.head()


# In[ ]:


#Checking null values and fill 0 at the place of NaN.
#ML models doesnt work with NaN values.
data_train.isnull().sum()

data_train.fillna(0,inplace=True)


# In[ ]:


#Splitting the dataset
X=data_train[data_train.columns[2:]]
Y=data_train['Survived']


# In[ ]:


#Importing ML libraries.
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3, random_state=0)


# In[ ]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
model_logr= LogisticRegression()
model_logr.fit(X_train,Y_train)
pred_logr=model_logr.predict(X_test)
metrics.accuracy_score(pred_logr,Y_test)


# In[ ]:


#K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
model_knn= KNeighborsClassifier()
model_knn.fit(X_train,Y_train)
pred_knn=model_knn.predict(X_test)
metrics.accuracy_score(pred_knn,Y_test)


# In[ ]:


#Naive Bayes

from sklearn.naive_bayes import GaussianNB
model_nb= GaussianNB()
model_nb.fit(X_train,Y_train)
pred_nb=model_logr.predict(X_test)
metrics.accuracy_score(pred_nb,Y_test)


# In[ ]:


#Support Vector MAchine
from sklearn import svm
model_svm= svm.SVC()
model_svm.fit(X_train,Y_train)
pred_svm=model_svm.predict(X_test)
metrics.accuracy_score(pred_svm,Y_test)


# In[ ]:


#Cross Validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict

kfold=KFold(n_splits=10, random_state=22)
xyz=[]
accuracy=[]
std=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Naive Bayes']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),GaussianNB()]
for i in models:
    model = i
    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2


# In[ ]:


###Submission for KAggle competition


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis = 1,inplace = True)
test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis = 1,inplace = True)
train.isnull().sum()
train['Age'] = train['Age'].fillna(24)
test['Age'] = test['Age'].fillna(24)
test['Fare'] = test['Fare'].fillna(7.75)
test.head()


# In[ ]:


train.head()


# In[ ]:


#Training process
Y_train=train['Survived']
X_train=train.drop('Survived',axis=1)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)


# In[ ]:


from sklearn import metrics
pred = model.predict(X_train)
metrics.accuracy_score(pred, Y_train)


# In[ ]:



pred = model.predict(test)


# In[ ]:


output = pd.DataFrame({"PassengerId":test.PassengerId , "Survived" : pred})
output.to_csv("../submission_"  + ".csv",index = False)


# In[ ]:




