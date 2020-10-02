#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('../input/heart.csv',delimiter=',')


# In[3]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.info()


# Visualization

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True,fmt='.2g',vmax=0.4)
plt.tight_layout()


# In[ ]:


sns.pairplot(data,hue='target')


# In[ ]:


sns.countplot(x='target',data=data)
plt.xlabel('0:No heart disease,1:heart disease')


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(x='age',hue='target',data=data)
plt.legend(['No Heart Disease','Heart Disease '])


# In[ ]:


sns.countplot(x='sex',data=data)
plt.xlabel('0:Female,1:Male')


# In[ ]:


data['sex'].value_counts()


# In[ ]:


fig,axes=plt.subplots(3,2,figsize=(10,8))
a=['cp','fbs','restecg','exang','slope','ca']
for i,ax in enumerate(axes.flat):
    sns.countplot(x=data[a[i]],data=data,hue='target',ax=ax)
    ax.legend(['Not Disease','Disease'])
    plt.tight_layout()


# In[ ]:


sns.scatterplot(x='trestbps',y='thalach',data=data,hue='target')


# In[ ]:


sns.scatterplot(x='chol',y='thalach',data=data,hue='target')


# In[ ]:


plt.scatter(x=data.age[data.target==1],y=data.thalach[data.target==1],c='red')
plt.scatter(x=data.age[data.target==0],y=data.thalach[data.target==0],c='green')
plt.legend(['Disease','Not Disease'])
plt.xlabel('Age')
plt.ylabel('Maximum heart rate achieved')


# # Machine Learning

# In[ ]:


from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics  import roc_curve,roc_auc_score


# In[ ]:


X=data.drop('target',axis=1)
y=data['target']


# In[ ]:


s=StandardScaler()
X=s.fit_transform(X)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# ## Logistic Regression
# 

# In[ ]:


lr=LogisticRegression()
lr.fit(X_train,y_train)
y_predl=lr.predict(X_test)
probs=lr.predict_proba(X_test)
prob=probs[:,1]
auc=roc_auc_score(y_test,prob)
print('AUC for Logistic Regression is',auc)
fpr,tpr,thresholds=roc_curve(y_test,prob)
plt.plot([0,1],[0,1],linestyle='--')
plt.plot(fpr,tpr,marker='.')


# In[ ]:


print('Accuracy of training set',lr.score(X_train,y_train)*100)


# In[ ]:


print('Accuracy of testing set',accuracy_score(y_test,y_predl)*100)


# In[ ]:


cm=confusion_matrix(y_test,y_predl)
sns.heatmap(cm,annot=True)


# In[ ]:


print('Classification Report',classification_report(y_test,y_predl))


# In[ ]:





# ## SVM

# In[ ]:


sv=SVC()
sv.fit(X_train,y_train)
y_preds=sv.predict(X_test)


# In[ ]:


print('Accuracy of training set',sv.score(X_train,y_train)*100)


# In[ ]:


print('Accuracy of testing set',accuracy_score(y_test,y_preds)*100)


# In[ ]:


cm=confusion_matrix(y_test,y_preds)
sns.heatmap(cm,annot=True)


# In[ ]:


print('Classification Report',classification_report(y_test,y_preds))


# Using Grid search method

# In[ ]:


param=[{'C':[0.02,0.03,0.001,0.002,0.003,0.004,0.005,0.006,1],'kernel':['linear']}]
grid_search=GridSearchCV(sv,param_grid=param,scoring='accuracy',cv=10,n_jobs=-1)
grid_search.fit(X_train,y_train)
y_preds1=grid_search.predict(X_test)

print('Best score',grid_search.best_score_)
print(grid_search.best_params_)
print('Accuracy of training set',grid_search.score(X_train,y_train)*100)
print('Accuracy of testing set',accuracy_score(y_test,y_preds1)*100)

print('Classification Report',classification_report(y_test,y_preds1))




# In[ ]:


cm=confusion_matrix(y_test,y_preds1)
sns.heatmap(cm,annot=True)
#xlabel - Predicted value, ylabel-True value
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# ROC - AUC Curve

# AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.

# In[ ]:


sv1=SVC(C=0.03,kernel='linear',random_state=2,probability=True)
sv1.fit(X_train,y_train)
probs=sv1.predict_proba(X_test)
probs


# In[ ]:


#keep only probabilities of positive outcomes
prob=probs[:,1]
auc=roc_auc_score(y_test,prob)
print('AUC Score is ',auc)


# In[ ]:


fpr,tpr,thresholds=roc_curve(y_test,prob)
plt.plot([0,1],[0,1],linestyle='--')
plt.plot(fpr,tpr,marker='.')


# ## Random Forest
# 

# In[ ]:


rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_predr=rf.predict(X_test)


# In[ ]:


print('Accuracy of training set',rf.score(X_train,y_train))


# In[ ]:


print('Accuracy of testing set',accuracy_score(y_test,y_predr)*100)


# In[ ]:


cm=confusion_matrix(y_test,y_predr)
sns.heatmap(cm,annot=True)


# In[ ]:


print('Classification Report',classification_report(y_test,y_predr))


# ## KNN

# In[ ]:


kn=KNeighborsClassifier()
kn.fit(X_train,y_train)
y_predk=kn.predict(X_test)


# In[ ]:


print('Accuracy of training set',kn.score(X_train,y_train))


# In[ ]:


print('Accuracy of testing set',accuracy_score(y_test,y_predk)*100)


# In[ ]:


cm=confusion_matrix(y_test,y_predk)
sns.heatmap(cm,annot=True)


# In[ ]:


print('Classification Report',classification_report(y_test,y_predk))


# In[ ]:


#Using Grid search

param={'n_neighbors':np.arange(1,10)}
grid=GridSearchCV(kn,param_grid=param,scoring='accuracy',cv=10,n_jobs=-1)
grid.fit(X_train,y_train)
y_predk1=grid.predict(X_test)

print('Best score',grid.best_score_)
print('Best Parameter',grid.best_params_)
print('Accuracy of training set',grid.score(X_train,y_train)*100)
print('Accuracy of testing set',accuracy_score(y_test,y_predk1)*100)

print('Classification Report',classification_report(y_test,y_predk1))
print('Confusion Matrix',confusion_matrix(y_test,y_predk1))


# AUC - ROC Curve

# In[ ]:


kn1=KNeighborsClassifier(n_neighbors=4)
kn1.fit(X_train,y_train)
probs=kn1.predict_proba(X_test)
#Only probabilities of positive outcome
prob=probs[:,1]
auc=roc_auc_score(y_test,prob)
print('AUC Score for KNN is',auc)


# In[ ]:


fpr,tpr,thresholds=roc_curve(y_test,prob)
plt.plot([0,1],[0,1],linestyle='--')
plt.plot(fpr,tpr,marker='.')


# ##### Conclusion :

# Logistic Regression and Linear SVM with C=0.03 is the best model in this case.
