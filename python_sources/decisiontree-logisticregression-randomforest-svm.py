#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore')


# In[ ]:


df=pd.read_csv('Breast_cancer.csv')
df.head()


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.drop(['id','Unnamed: 32'],axis=1,inplace=True)


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


sns.distplot(df['radius_mean'])


# In[ ]:


sns.scatterplot(x=df['radius_mean'],y=df['radius_se'],hue=df['diagnosis'])


# In[ ]:


sns.scatterplot(x=df['texture_mean'],y=df['texture_se'],hue=df['diagnosis'])


# In[ ]:


sns.scatterplot(x=df['radius_mean'],y=df['radius_worst'],hue=df['diagnosis'])


# In[ ]:


plt.rcParams['figure.figsize']=(18,18)
df.hist();


# In[ ]:


corr=df.corr()


# In[ ]:


sns.heatmap(corr,fmt='.2f',annot=True,cmap=plt.cm.Blues)


# In[ ]:


from sklearn.model_selection import train_test_split
x=df.loc[:,df.columns!='diagnosis']
y=df.loc[:,'diagnosis']


# In[ ]:


x.shape,y.shape


# In[ ]:


#mapping the malignant as 1 and Benign as 0
y=y.map({'M':1,'B':0})


# In[ ]:


y.head()


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=12)


# # Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve


# In[ ]:


dt=DecisionTreeClassifier(criterion='gini',max_depth=10)
dt.fit(x_train,y_train)


# In[ ]:


dt_pred=dt.predict(x_test)
accuracy_score(dt_pred,y_test)


# In[ ]:


print(confusion_matrix(dt_pred,y_test))


# In[ ]:


print(roc_auc_score(dt_pred,y_test))


# In[ ]:


params={'max_depth':np.arange(2,10),'min_samples_leaf':np.arange(2,10)}
dt_best=GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=params,cv=5,verbose=1)
dt_best.fit(x_train,y_train)


# In[ ]:


dt_best_pred=dt_best.predict(x_test)


# In[ ]:


accuracy_score(dt_best_pred,y_test)


# In[ ]:


dt_best.best_estimator_


# In[ ]:


dt_best.best_score_,dt_best.best_params_


# # Logistic Regression Classifier

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[ ]:


lr_pred=lr.predict(x_test)
accuracy_score(lr_pred,y_test)


# In[ ]:


print(confusion_matrix(lr_pred,y_test))


# In[ ]:


lr_params={'C':[100,10,1,0.5,0.1,0.01],'penalty':['l1','l2']}
lr_best=GridSearchCV(estimator=LogisticRegression(),param_grid=lr_params,verbose=1,n_jobs=-1,cv=5)
lr_best.fit(x_train,y_train)


# In[ ]:


lr_best.best_params_,lr_best.best_score_


# In[ ]:


lr_best_pred=lr_best.predict(x_test)
accuracy_score(lr_best_pred,y_test)


# In[ ]:


print(confusion_matrix(lr_best_pred,y_test))


# In[ ]:


#Scaling the data 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


# In[ ]:


lr_best_scaled=lr_best.fit(x_train_scaled,y_train)


# In[ ]:


lr_best_scaled.best_params_,lr_best_scaled.best_score_


# In[ ]:


lr_best_scaled_pred=lr_best_scaled.predict(x_test_scaled)


# In[ ]:


means=lr_best_scaled.cv_results_['mean_test_score']
stds=lr_best_scaled.cv_results_['std_test_score']
params=lr_best_scaled.cv_results_['params']
for mean,std,param in zip(means,stds,params):
    print('%f %f in %r'%(mean,std,param))


# In[ ]:


accuracy_score(lr_best_scaled_pred,y_test)


# In[ ]:


print(confusion_matrix(lr_best_scaled_pred,y_test))


# # Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf=RandomForestClassifier()
rf.fit(x_train,y_train)


# In[ ]:


rf_pred=rf.predict(x_test)


# In[ ]:


accuracy_score(rf_pred,y_test)


# In[ ]:


#training on the scaled data
rf.fit(x_train_scaled,y_train)


# In[ ]:


rf_pred_scaled=rf.predict(x_test_scaled)


# In[ ]:


accuracy_score(rf_pred_scaled,y_test)


# In[ ]:


rf_params={'max_features':[2,4,6,8,12,15,18,20,24,30],'n_estimators':[10,100,1000]}
rf_best=GridSearchCV(estimator=RandomForestClassifier(),param_grid=rf_params,cv=5,verbose=1,n_jobs=-1)
rf_best.fit(x_train,y_train)


# In[ ]:


rf_best_pred=rf_best.predict(x_test)
accuracy_score(rf_best_pred,y_test)


# In[ ]:


rf_best.best_params_,rf_best.best_score_


# In[ ]:


means=rf_best.cv_results_['mean_test_score']
stds=rf_best.cv_results_['std_test_score']
params=rf_best.cv_results_['params']
for mean,std,param in zip(means,stds,params):
    print('%f with std %f in %r'%(mean,std,param))


# # SVM Classifier

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc=SVC()
svc.fit(x_train,y_train)


# In[ ]:


svc_pred=svc.predict(x_test)
accuracy_score(svc_pred,y_test)


# In[ ]:


svc.score(x_train,y_train)


# In[ ]:


#training on scaled data
svc.fit(x_train_scaled,y_train)
svc_pred_scaled=svc.predict(x_test_scaled)
accuracy_score(svc_pred_scaled,y_test)


# In[ ]:


svc.score(x_train_scaled,y_train)


# In[ ]:


svc.get_params().keys()


# In[ ]:


#hyperparameter tuning on scaled data
svc_params={'kernel':['linear','rbf','sigmoid','poly'],'C':[100,10,1,0.1,0.01,0.001]}
svc_best=GridSearchCV(estimator=SVC(),param_grid=svc_params,verbose=1,cv=5)
svc_best.fit(x_train_scaled,y_train)


# In[ ]:


svc_best_scaled_pred=svc_best.predict(x_test_scaled)
accuracy_score(svc_best_scaled_pred,y_test)


# In[ ]:


svc_best.best_params_,svc_best.best_score_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




