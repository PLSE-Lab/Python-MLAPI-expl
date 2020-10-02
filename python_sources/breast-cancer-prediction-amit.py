#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


from xgboost.sklearn import XGBClassifier


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


from sklearn.metrics import auc, roc_curve


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


from sklearn.metrics import precision_recall_fscore_support


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.preprocessing import StandardScaler as ss


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


import os


# In[ ]:


#os.chdir("C:\\Users\\amit\\Desktop\\ELTM")


# In[ ]:


#os.listdir()


# In[ ]:


cancer = pd.read_csv("../input/data.csv")


# In[ ]:


cancer.shape 


# In[ ]:


cancer.head


# In[ ]:


cancer.dtypes


# In[ ]:


cancer.isnull().values.any()


# In[ ]:


cancer.isnull().sum()


# In[ ]:


cancer.drop(['id','Unnamed: 32'],axis=1,inplace=True)


# In[ ]:


cancer.shape


# In[ ]:


y=cancer.pop('diagnosis')


# In[ ]:


y[:25]


# In[ ]:


X=cancer


# In[ ]:


X is cancer


# In[ ]:


X.shape


# In[ ]:


X.dtypes


# In[ ]:


y=y.map({'M':1,'B':0})


# In[ ]:


y[:25]


# In[ ]:


scale=ss()


# In[ ]:


X=scale.fit_transform(X)


# In[ ]:


pca=PCA()


# In[ ]:


TX=pca.fit_transform(X)


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


pca.explained_variance_ratio_.cumsum()


# In[ ]:


final_X=TX[:,:10]


# In[ ]:


X1=final_X


# In[ ]:


X1 is final_X


# In[ ]:


X1_train,X1_test,y_train,y_test=train_test_split(X1,y,test_size=0.2,shuffle=True)


# In[ ]:


X1_train.shape


# In[ ]:


X1_test.shape


# In[ ]:


y_test.shape


# In[ ]:


dt=DecisionTreeClassifier()


# In[ ]:


rf=RandomForestClassifier(n_estimators=100,min_samples_leaf=5)


# In[ ]:


xg=XGBClassifier(learning_rate=0.5,
                   reg_alpha= 5,
                   reg_lambda= 0.1)


# In[ ]:


gb=GradientBoostingClassifier()


# In[ ]:


et=ExtraTreesClassifier(n_estimators=100)


# In[ ]:


kn=KNeighborsClassifier()


# In[ ]:


dt1=dt.fit(X1_train,y_train)


# In[ ]:


rf1=rf.fit(X1_train,y_train)


# In[ ]:


xg1=xg.fit(X1_train,y_train)


# In[ ]:


gb1=gb.fit(X1_train,y_train)


# In[ ]:


et1=et.fit(X1_train,y_train)


# In[ ]:


kn1=kn.fit(X1_train,y_train)


# In[ ]:


y_pred_dt=dt1.predict(X1_test)


# In[ ]:


y_pred_rf=rf1.predict(X1_test)


# In[ ]:


y_pred_xg=xg1.predict(X1_test)


# In[ ]:


y_pred_gb=gb1.predict(X1_test)


# In[ ]:


y_pred_et=et1.predict(X1_test)


# In[ ]:


y_pred_kn=kn1.predict(X1_test)


# In[ ]:


y_pred_dt


# In[ ]:


y_pred_rf


# In[ ]:


y_pred_xg


# In[ ]:


y_pred_gb


# In[ ]:


y_pred_et


# In[ ]:


y_pred_kn


# In[ ]:


y_pred_dt_prob=dt1.predict_proba(X1_test)
y_pred_rf_prob=rf1.predict_proba(X1_test)
y_pred_xg_prob=xg1.predict_proba(X1_test)
y_pred_gb_prob=gb1.predict_proba(X1_test)
y_pred_et_prob=et1.predict_proba(X1_test)
y_pred_kn_prob=kn1.predict_proba(X1_test)


# In[ ]:


y_pred_dt_prob


# In[ ]:


y_pred_xg_prob


# In[ ]:


accuracy_score(y_test,y_pred_dt)


# In[ ]:


accuracy_score(y_test,y_pred_rf)


# In[ ]:


accuracy_score(y_test,y_pred_xg)


# In[ ]:


accuracy_score(y_test,y_pred_gb)


# In[ ]:


accuracy_score(y_test,y_pred_kn)


# In[ ]:


accuracy_score(y_test,y_pred_et)


# In[ ]:


confusion_matrix(y_test,y_pred_dt)


# In[ ]:


confusion_matrix(y_test,y_pred_rf)


# In[ ]:


confusion_matrix(y_test,y_pred_xg)


# In[ ]:


confusion_matrix(y_test,y_pred_et)


# In[ ]:


confusion_matrix(y_test,y_pred_kn)


# In[ ]:


confusion_matrix(y_test,y_pred_gb)


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_dt)


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_rf)


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_xg)


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_gb)


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_et)


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_kn)


# In[ ]:


fpr_dt, tpr_dt, thresholds = roc_curve(y_test,
                                 y_pred_dt_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


fpr_rf, tpr_rf, thresholds = roc_curve(y_test,
                                 y_pred_rf_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


fpr_xg, tpr_xg, thresholds = roc_curve(y_test,
                                 y_pred_xg_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


fpr_gb, tpr_gb,thresholds = roc_curve(y_test,
                                 y_pred_gb_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


fpr_et, tpr_et,thresholds = roc_curve(y_test,
                                 y_pred_et_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


fpr_kn, tpr_kn,thresholds = roc_curve(y_test,
                                 y_pred_kn_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


auc(fpr_dt,tpr_dt)


# In[ ]:


auc(fpr_rf,tpr_rf)


# In[ ]:


auc(fpr_gb,tpr_gb)


# In[ ]:


auc(fpr_xg,tpr_xg)


# In[ ]:


auc(fpr_et,tpr_et)


# In[ ]:


auc(fpr_kn,tpr_kn)


# In[ ]:


fig = plt.figure(figsize=(10,10))          
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], ls="--")
ax.set_xlabel('False Positive Rate')  
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for models')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.plot(fpr_dt, tpr_dt, label = "dt")
ax.plot(fpr_rf, tpr_rf, label = "rf")
ax.plot(fpr_xg, tpr_xg, label = "xg")
ax.plot(fpr_gb, tpr_gb, label = "gb")
ax.plot(fpr_kn, tpr_kn, label = "kn")
ax.plot(fpr_et, tpr_et, label = "et")
ax.legend(loc="lower right")
plt.show()


# In[ ]:




