#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


data=pd.read_csv('../input/titanic.csv')


# In[ ]:


data.head()


# In[ ]:


cols=data.columns
cols


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


data.Age.isnull().sum()


# In[ ]:


m=data.Age.mean()
m


# In[ ]:


data['Age']=data['Age'].fillna(m)


# In[ ]:


data.Age.isnull().sum()


# In[ ]:


sns.pairplot(data)


# In[ ]:


corr=data.corr()


# In[ ]:


plt.figure(figsize=(20,15))
sns.heatmap(corr,vmin=-1,vmax=1,annot=True)
plt.show() 


# In[ ]:


cols


# In[ ]:


data=data.drop('PassengerId',axis=1)


# In[ ]:


data=data.drop('Name',axis=1)
data=data.drop('Ticket',axis=1)


# In[ ]:


data.info()


# In[ ]:


data


# In[ ]:


sns.countplot(data['Sex'])
plt.show()


# In[ ]:


len(data[data['Sex']=='male']) #male are 577


# In[ ]:


len(data[data['Sex']=='female']) #femmale are 314


# Male are there in the Titanic ship compared to Women

# In[ ]:


sns.countplot(data['Pclass'])
plt.show()


# More people are there in Pclass3

# In[ ]:


sns.countplot(data['Embarked'])
plt.show()


# In[ ]:


sns.lmplot('Age', 'Fare', hue ='Sex', data = data, fit_reg=True)


# In[ ]:


data.columns


# # Checking for null Values

# In[ ]:


data.Survived.isnull().sum()


# In[ ]:


data.Sex.isnull().sum()


# In[ ]:


data.SibSp.isnull().sum()


# In[ ]:


data.Parch.isnull().sum()


# In[ ]:


data.Fare.isnull().sum()


# In[ ]:


data.Cabin.isnull().sum()


# In[ ]:


data=data.drop('Cabin',axis=1)


# In[ ]:


data


# In[ ]:


y=data.Survived


# In[ ]:


y.head()


# In[ ]:


x=data.drop('Survived',axis=1)


# In[ ]:


x


# In[ ]:


x=x.drop('Embarked',axis=1)


# In[ ]:


x.head()


# In[ ]:


le=LabelEncoder()
x.Sex=le.fit_transform(x.Sex)


# In[ ]:


x.info()


# In[ ]:


x.head()


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)


# # LogisticRegression

# In[ ]:


m=LogisticRegression()
model=m.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[ ]:


print('accuracy_score:',accuracy_score(y_pred,y_test))
print()
print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))
print()
print('classification_report',classification_report(y_pred,y_test))
print()
print('confusion_matrix',confusion_matrix(y_pred,y_test))


# In[ ]:


cm=confusion_matrix(y_pred,y_test)
sns.heatmap(cm, annot=True)


# # Knn

# In[ ]:


knn=KNeighborsClassifier(n_neighbors=10)
knn_fit=knn.fit(x_train,y_train)
y_knnpred = knn_fit.predict(x_test)


# In[ ]:


print('accuracy_score:',accuracy_score(y_pred,y_test))
print()
print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))
print()
print('classification_report',classification_report(y_pred,y_test))
print()
print('confusion_matrix',confusion_matrix(y_pred,y_test))


# In[ ]:


cm=confusion_matrix(y_pred,y_test)
sns.heatmap(cm, annot=True)


# # RandomForest

# In[ ]:


model=RandomForestClassifier()
model_fit=model.fit(x_train,y_train)
y_pred=model_fit.predict(x_test)


# In[ ]:


print('accuracy_score:',accuracy_score(y_pred,y_test))
print()
print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))
print()
print('classification_report',classification_report(y_pred,y_test))
print()
print('confusion_matrix',confusion_matrix(y_pred,y_test))


# In[ ]:


cm=confusion_matrix(y_pred,y_test)
sns.heatmap(cm, annot=True)


# # SVC

# In[ ]:


model=SVC()
model_fit=model.fit(x_train,y_train)
y_pred= model_fit.predict(x_test)


# In[ ]:


print('accuracy_score:',accuracy_score(y_pred,y_test))
print()
print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))
print()
print('classification_report',classification_report(y_pred,y_test))
print()
print('confusion_matrix',confusion_matrix(y_pred,y_test))


# In[ ]:


cm=confusion_matrix(y_pred,y_test)
sns.heatmap(cm, annot=True)


# # NaiveBias

# In[ ]:


model=MultinomialNB()
mn_fit=model.fit(x_train,y_train)
y_pred=mn_fit.predict(x_test)


# In[ ]:


print('accuracy_score:',accuracy_score(y_pred,y_test))
print()
print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))
print()
print('classification_report',classification_report(y_pred,y_test))
print()
print('confusion_matrix',confusion_matrix(y_pred,y_test))


# In[ ]:


cm=confusion_matrix(y_pred,y_test)
sns.heatmap(cm, annot=True)


# # DecisionTree

# In[ ]:


model=DecisionTreeClassifier()
mn_fit=model.fit(x_train,y_train)
y_pred=mn_fit.predict(x_test)


# In[ ]:


print('accuracy_score:',accuracy_score(y_pred,y_test))
print()
print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))
print()
print('classification_report',classification_report(y_pred,y_test))
print()
print('confusion_matrix',confusion_matrix(y_pred,y_test))


# In[ ]:


cm=confusion_matrix(y_pred,y_test)
sns.heatmap(cm, annot=True)


# # Adaboost

# In[ ]:


model=AdaBoostClassifier()
mn_fit=model.fit(x_train,y_train)
y_pred=mn_fit.predict(x_test)


# In[ ]:


print('accuracy_score:',accuracy_score(y_pred,y_test))
print()
print('cohen_kappa_score',cohen_kappa_score(y_pred,y_test))
print()
print('classification_report',classification_report(y_pred,y_test))
print()
print('confusion_matrix',confusion_matrix(y_pred,y_test))


# In[ ]:


cm=confusion_matrix(y_pred,y_test)
sns.heatmap(cm, annot=True)


# # Random Forest 

# In[ ]:


model=RandomForestClassifier()
model_fit=model.fit(x_train,y_train)
y_pred=model_fit.predict(x_test)


# In[ ]:


model.feature_importances_


# In[ ]:


cm=confusion_matrix(y_pred,y_test)
sns.heatmap(cm, annot=True)


# In[ ]:




