#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv('../input/train.csv')
train.head(5)


# In[ ]:


test=pd.read_csv('../input/test.csv')
test.head(5)


# **encoding object data types to category and further into int data types**

# In[ ]:


train.Sex=train.Sex.astype('category').cat.codes
test.Sex=test.Sex.astype('category').cat.codes


# In[ ]:


train.isnull().sum()


# In[ ]:


train.describe()


# **hot encoding category variables here**

# In[ ]:


train_df=train.copy()
test_df=test.copy()
train_df = pd.get_dummies(train_df, columns=['Embarked', 'Pclass'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Embarked', 'Pclass'], drop_first=True)


# In[ ]:


test.isnull().sum()


# In[ ]:


train_df.info()
print(test_df.head(5))


# **computing mean age for both the test and train dataset and imputing the mean age wherever NaN value exists**

# In[ ]:


age_mean=train_df.Age.mean()
#train_df.Age=train_df.Age.fillna()


# In[ ]:


#train_df['Age'].fillna(age_mean,inplace=True)
train_df['Age'].fillna(train_df['Age'].median(),inplace=True)


# In[ ]:


train_df.info()


# In[ ]:


age_mean_test=test_df['Age'].mean()
print(age_mean_test)
#test_df['Age'].fillna(age_mean_test,inplace=True)
test_df['Age'].fillna(test_df['Age'].median(),inplace=True)


# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)


# In[ ]:


test_df.Cabin.unique()


# In[ ]:


test_df.drop(columns=['Cabin','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train_df.drop(columns=['Cabin','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


corr=train_df.corr(method='pearson')


# In[ ]:


import seaborn as sns
sns.set(context='paper', style='whitegrid', palette='muted', font='sans-serif', font_scale=1, color_codes=True, rc=None)
sns.heatmap(corr,linewidths=.5)


# In[ ]:


sns.barplot(x=corr.Survived,y=corr.columns)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
logReg=LogisticRegression(solver='liblinear')
sgdcls=SGDClassifier()
nbcls=GaussianNB()
knn=KNeighborsClassifier()
desclr=DecisionTreeClassifier()
svc=SVC()
ada=AdaBoostClassifier()
#cvs=cross_val_score()
#confusion_matrix=confusion_matrix()


# In[ ]:


# train_X=train_df.drop('Survived',axis=1)
# train_y=train_df['Survived']
# test_X=test_df
X=train_df.drop('Survived',axis=1)
y=train_df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_df_test, y_train, y_df_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


logReg.fit(X_train,y_train)
prediction=logReg.predict(X_df_test)
#pred_train=logReg.predict(train_y)
score=cross_val_score(logReg,X_train,y_train,cv=5)
print("Score:",score)
#print("Confusion_matrix:",confusion_matrix(train_y,pred_train))
acc_log = round(logReg.score(X_train,y_train) * 100, 2)
acc_log


# In[ ]:


#SGDClassifier
sgdcls.fit(X_train,y_train)
prediction_sgdcls=sgdcls.predict(X_df_test)
#pred_train=logReg.predict(train_y)
score=cross_val_score(sgdcls,X_train,y_train,cv=5)
print("Score:",score)
#print("Confusion_matrix:",confusion_matrix(train_y,pred_train))
acc_log_sgdcls = round(sgdcls.score(X_train,y_train) * 100, 2)
acc_log_sgdcls


# In[ ]:


nbcls.fit(X_train,y_train)
prediction_nbcls=nbcls.predict(X_df_test)
#pred_train=logReg.predict(train_y)
score=cross_val_score(nbcls,X_train,y_train,cv=5)
print("Score:",score)
#print("Confusion_matrix:",confusion_matrix(train_y,pred_train))
acc_log_nbcls = round(nbcls.score(X_train,y_train) * 100, 2)
acc_log_nbcls


# In[ ]:


knn.fit(X_train,y_train)
prediction_knn=knn.predict(X_df_test)
#pred_train=logReg.predict(train_y)
score=cross_val_score(knn,X_train,y_train,cv=5)
print("Score:",score)
#print("Confusion_matrix:",confusion_matrix(train_y,pred_train))
acc_log_knn = round(knn.score(X_train,y_train) * 100, 2)
acc_log_knn


# In[ ]:


#DecisionTreeClassifier
desclr.fit(X_train,y_train)
prediction_desclr=desclr.predict(X_df_test)
#pred_train=logReg.predict(train_y)
score=cross_val_score(desclr,X_train,y_train,cv=5)
print("Score:",score)
#print("Confusion_matrix:",confusion_matrix(train_y,pred_train))
acc_log_desclr = round(desclr.score(X_train,y_train) * 100, 2)
acc_log_desclr


# In[ ]:


svc.fit(X_train,y_train)
prediction_svc=svc.predict(X_df_test)
#pred_train=logReg.predict(train_y)
score=cross_val_score(svc,X_train,y_train,cv=5)
print("Score:",score)
#print("Confusion_matrix:",confusion_matrix(train_y,pred_train))
acc_log_svc = round(svc.score(X_train,y_train) * 100, 2)
acc_log_svc


# In[ ]:


ada.fit(X_train,y_train)
prediction_svc=ada.predict(X_df_test)
#pred_train=logReg.predict(train_y)
score=cross_val_score(ada,X_train,y_train,cv=5)
print("Score:",score)
#print("Confusion_matrix:",confusion_matrix(train_y,pred_train))
acc_log_ada = round(ada.score(X_train,y_train) * 100, 2)
acc_log_ada


# In[ ]:


prediction_test=logReg.predict(test_df)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": prediction_test
    })


# In[ ]:


submission.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:




