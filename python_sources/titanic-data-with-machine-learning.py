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


import pandas as pd
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")


# In[ ]:


train.tail()


# In[ ]:


test.head()


# Performing EDA

# In[ ]:


train.info()


# In[ ]:


train.describe().T


# In[ ]:


train['Age'].isnull().value_counts()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import download_plotlyjs,init_notebook_mode,iplot,plot
import cufflinks as cf


# In[ ]:


init_notebook_mode(connected=True)


# In[ ]:


cf.go_offline()


# In[ ]:


sns.heatmap(train.isnull(),cbar=False,yticklabels=False,cmap='viridis')


# We can observe from above heatmap that Age has many missing values and also cabin has most of the values missing so we can drop cabin
# 
# We can also drop columns like Passengerid, Name,Ticket as the contain no information for passenger survival

# **Lets see Survived relationship with other Features using Graphical method**

# In[ ]:


plt.figure()
fig,axes=plt.subplots(nrows=2,ncols=2, figsize=(15,15))
sns.countplot(x='Sex',hue='Survived',data=train,ax=axes[0][0])
axes[0][0].set_title('Sex Vs Survival')
sns.countplot(x='Pclass',hue='Survived',data=train,ax=axes[0][1])
axes[0][1].set_title('Pclass Vs Survival')
sns.countplot(x='Embarked',hue='Survived',data=train,ax=axes[1][0])
axes[1][0].set_title('Embarked Vs Survival')
sns.countplot(x='SibSp',hue='Survived',data=train,ax=axes[1][1])
axes[1][1].set_title('Sibling/spouse Vs Survival')
#train.iplot(kind='bar',x='Sex')


# From Above Analysis we can conclude:
# 
# 1) Most of the people who were males and travelling by third class did not survived
# 
# 2) pepole with higher Age group are mostly travelling by First Class.
# 
# 3) People who Embarked from Southampton did not survived (it seems from the data that these people were travelling by Pclass=3
# 
# 4) Most of the people who died were travelling either alone or with one person (either child or spouse)
# 
# **So we can use Pclass as an imputer to fill the missing ages using the function implemented below**

# In[ ]:


sns.boxplot(train['Pclass'], train['Age'], hue=train['Survived'])
#train.iplot(kind='box',x='Pclass',y='Age')


# In[ ]:


def age_impute(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 45
        elif Pclass==2:
            return 32
        else:
            return 27
    else:
        return Age
    
    
 


# In[ ]:


train['Age']=train[['Age','Pclass']].apply(age_impute, axis=1)


# In[ ]:


sns.heatmap(train.isnull(),cbar=False,yticklabels=False,cmap='viridis')


# We have replaced the null values in Age with the median of Age group of Pclass.

# The cabin Feature has lot of missing values which are difficult to predict so we can drop it
# 
# Also Name, PassengerId and Ticket Features seems to have no relation with survival of the person so we are safe to drop it also

# In[ ]:


train.drop(['PassengerId','Ticket','Name','Cabin'], axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


fig1,axes1=plt.subplots(nrows=1,ncols=2,figsize=(12,12))
sns.barplot(x='Embarked',y='Fare',hue='Survived',data=train,ax=axes1[0])
sns.barplot(x='Pclass',y='Fare',hue='Survived',data=train,ax=axes1[1])


# From Above Graph we can conclude that most people who survived were from Pclass=1 and Embarked from Cherbourg

# In[ ]:


sns.heatmap(train.corr(), cmap='YlOrBr', annot=True, linecolor='black', linewidth=5)


# Preparing Data for machine Learning
# 
# Lets Prepare dummy Variables for some Catagorical Variables like Pclass, Sex, Embarked

# In[ ]:


train_dum=pd.get_dummies(data=train,columns=['Pclass','Sex','Embarked'], drop_first=True)


# In[ ]:


train_dum.drop('Fare', axis=1,inplace=True)


# Since the O/P is classification type we apply Logistic Regression, KNN, SVM, Random Forest classifiers, Boosting

# In[ ]:


#Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


X=train_dum.drop('Survived',axis=1)
y=train_dum['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


lr=LogisticRegression(solver='liblinear')


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


predictor=lr.predict(X_test)
lr_score=lr.score(X_test,y_test)
lr_score


# In[ ]:


plt.figure(figsize=(3,3))
cm=confusion_matrix(y_test,predictor,labels=[0,1])
df=pd.DataFrame(cm,index=['True_NonSur', 'True_Surv'], columns=['Predict_NonSur', 'Predict_surv'])
sns.heatmap(df,cmap='coolwarm',cbar=False, annot=True)
print(df)


# In[ ]:


print(classification_report(y_test,predictor))


# In[ ]:


#Applying KNN
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


#Prior to applying KNN we have to scale the data
from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()
s=scaler.fit_transform(X)
df1=pd.DataFrame(s,columns=X.columns)


# In[ ]:


df1.head(2)


# In[ ]:


X1_train, X1_test, y1_train, y1_test = train_test_split(df1, y, test_size=0.3, random_state=42)
knn=KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X1_train,y1_train)


# In[ ]:


predict_knn=knn.predict(X1_test)
knn_score=knn.score(X1_test,y1_test)
knn_score


# In[ ]:


plt.figure(figsize=(3,3))
cm1=confusion_matrix(y1_test,predict_knn,labels=[0,1])
df2=pd.DataFrame(cm1,index=['True_NonSur', 'True_Surv'], columns=['Predict_NonSur', 'Predict_surv'])
sns.heatmap(df2,cmap='coolwarm',cbar=False, annot=True)
print(df2)


# In[ ]:


print(classification_report(y1_test,predict_knn))


# In[ ]:


# FOR K ranging from 1-40
error_rate=[]
for i in range (1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X1_train, y1_train)
    predict_i= knn.predict(X1_test)
    error_rate.append(np.mean(predict_i != y1_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red')
plt.title('Error Rate vs K Value')
plt.xlabel('k')
plt.ylabel('Error Rate')


# In[ ]:


knn19=KNeighborsClassifier(n_neighbors=10)
knn19.fit(X1_train,y1_train)
predict_knn19=knn19.predict(X1_test)
knn1_score=knn19.score(X1_test,y1_test)
knn1_score


# In[ ]:


plt.figure(figsize=(3,3))
cm19=confusion_matrix(y1_test,predict_knn19,labels=[0,1])
df29=pd.DataFrame(cm19,index=['True_NonSur', 'True_Surv'], columns=['Predict_NonSur', 'Predict_surv'])
sns.heatmap(df29,cmap='coolwarm',cbar=False, annot=True)
print(df29)


# In[ ]:


print(classification_report(y1_test,predict_knn19))


# In[ ]:


#Navebayes Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

nb=GaussianNB()
nb.fit(X_train,y_train)


# In[ ]:


naive_predict=nb.predict(X_test)


# In[ ]:


print("Model accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,naive_predict)))


# In[ ]:


naive_score=nb.score(X_test,y_test)
naive_score


# In[ ]:


plt.figure(figsize=(4,4))
cm11=metrics.confusion_matrix(y_test, naive_predict, labels=[0,1])
df_cm1=pd.DataFrame(cm11, index=['True_NonSur', 'True_Surv'], columns=['Predict_NonSur', 'Predict_surv'])
sns.heatmap(df_cm1, annot=True, cmap='coolwarm')
df_cm1


# In[ ]:


print(classification_report(y_test, naive_predict))


# In[ ]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf=RandomForestClassifier(n_estimators=100)


# In[ ]:


rf.fit(X_train, y_train)


# In[ ]:


predict_rf=rf.predict(X_test)


# In[ ]:


rf_score=rf.score(X_test,y_test)
rf_score


# In[ ]:


print(classification_report(y_test,predict_rf))


# In[ ]:


cm_rf=metrics.confusion_matrix(y_test, predict_rf,labels=[0,1])
cm_rf1=pd.DataFrame(cm_rf,index=['True_NonSur', 'True_Surv'], columns=['Predict_NonSur', 'Predict_surv'])
sns.heatmap(cm_rf1, annot=True, cmap='coolwarm')
cm_rf1


# In[ ]:


#SVM (Support Vector Machine)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid= {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.001,0.0001]}


# In[ ]:


grid=GridSearchCV(SVC(),param_grid,verbose=2)
grid.fit(X_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


predict_grid=grid.predict(X_test)


# In[ ]:


print(classification_report(y_test, predict_grid))


# In[ ]:


svm_score=grid.score(X_test,y_test)
svm_score


# In[ ]:


cm_svm=metrics.confusion_matrix(y_test, predict_grid,labels=[0,1])
cm_svm1=pd.DataFrame(cm_svm,index=['True_NonSur', 'True_Surv'], columns=['Predict_NonSur', 'Predict_surv'])
sns.heatmap(cm_svm1, annot=True, cmap='coolwarm')
cm_svm1


# In[ ]:


Result=pd.DataFrame({'Models':['Logistic','KNN','Navebayes','RandomForest','SVM'], 'Accuracy':[lr_score,knn1_score,naive_score,rf_score,svm_score]})


# In[ ]:


Result


# Among the analysed Algos. KNN is giving good results with good Precision and F1 score.

# In[ ]:


test.head(2)


# In[ ]:


PassengerID=test['PassengerId']
test.drop(['PassengerId', 'Name','Ticket', 'Cabin'], axis=1,inplace=True)


# Replacing missing Age values in lieu of Pclass

# In[ ]:


test['Age']= test[['Age', 'Pclass']].apply(age_impute, axis=1)


# In[ ]:


test['Age'].isnull().any()


# In[ ]:


columns=['Sex', 'Pclass', 'Embarked']
test_dum=pd.get_dummies(test, columns=columns, drop_first=True)


# In[ ]:


test_dum.head()


# In[ ]:


test_dum.drop(['Fare'],axis=1, inplace=True)


# In[ ]:


test_dum.head()


# In[ ]:


s1=scaler.fit_transform(test_dum)
df2=pd.DataFrame(s1,columns=test_dum.columns)


# In[ ]:


df2.head()


# In[ ]:


df2.info()


# In[ ]:


knn_test=knn19.predict(df2)


# In[ ]:


combine=list(zip(PassengerID,knn_test))


# In[ ]:


output=pd.DataFrame(combine,columns=['PassengerId', 'Survived'])


# In[ ]:


output.head()


# In[ ]:


output.to_csv('Final_submission.csv',index=False)


# In[ ]:




