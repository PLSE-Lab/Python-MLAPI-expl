#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler as ss
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# In[ ]:


import os


# In[ ]:


#os.chdir("F:\\Practice\\Machine Learning and Deep Learning\\Classes\\Assignment\\Kaggle\\2nd")


# In[ ]:


#df=pd.read_csv("breast-cancer-wisconsin-data.zip")
df = pd.read_csv("../input/data.csv")


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.isnull().sum()


# In[ ]:


df['diagnosis'].isnull()


# In[ ]:


y=df['diagnosis']


# In[ ]:


y


# In[ ]:


x=df.loc[:,'radius_mean':'fractal_dimension_worst']


# In[ ]:


x


# In[ ]:


x.isnull().sum()


# In[ ]:


df = df.drop(["id", "Unnamed: 32"], axis=1)


# In[ ]:


df.shape


# In[ ]:


x.head()


# In[ ]:


y=y.map({'M':0,'B':1})


# In[ ]:


y.head()


# In[ ]:


scale=ss()


# In[ ]:


x=scale.fit_transform(x)


# In[ ]:


x.shape


# In[ ]:


y.shape


# In[ ]:


pca=PCA()


# In[ ]:


out=pca.fit_transform(x)


# In[ ]:


out.shape


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle = True )


# In[ ]:


dt = DecisionTreeClassifier()


# In[ ]:


rf = RandomForestClassifier(n_estimators=100)


# In[ ]:


etc = ExtraTreesClassifier(n_estimators=100)


# In[ ]:


gbm = GradientBoostingClassifier()


# In[ ]:


knc = KNeighborsClassifier()


# In[ ]:


xg = XGBClassifier()


# In[ ]:


#Train data


# In[ ]:


dt1 = dt.fit(x_train,y_train)


# In[ ]:


rf1 = rf.fit(x_train,y_train)
etc1 = etc.fit(x_train,y_train)
gbm1 = gbm.fit(x_train,y_train)
knc1 = knc.fit(x_train,y_train)
xg1 = xg.fit(x_train,y_train)


# In[ ]:


#predictions


# In[ ]:


y_pred_dt = dt1.predict(x_test)


# In[ ]:


y_pred_rf = rf1.predict(x_test)
y_pred_etc= etc1.predict(x_test)
y_pred_gbm= gbm1.predict(x_test)
y_pred_knc= knc1.predict(x_test)
y_pred_xg= xg1.predict(x_test)


# In[ ]:


#probability value


# In[ ]:


y_pred_dt_prob = dt1.predict_proba(x_test)


# In[ ]:


y_pred_rf_prob = rf1.predict_proba(x_test)


# In[ ]:


y_pred_etc_prob = etc1.predict_proba(x_test)
y_pred_gbm_prob= gbm1.predict_proba(x_test)
y_pred_knc_prob = knc1.predict_proba(x_test)
y_pred_xg_prob = xg1.predict_proba(x_test)


# In[ ]:


#accuracy


# In[ ]:


accuracy_score(y_test,y_pred_dt)


# In[ ]:


accuracy_score(y_test,y_pred_rf)


# In[ ]:


accuracy_score(y_test,y_pred_etc)


# In[ ]:


accuracy_score(y_test,y_pred_knc)


# In[ ]:


accuracy_score(y_test,y_pred_xg)


# In[ ]:


accuracy_score(y_test,y_pred_gbm)


# In[ ]:


#Confusion Matrix


# In[ ]:


confusion_matrix(y_test,y_pred_dt)


# In[ ]:


confusion_matrix(y_test,y_pred_rf)


# In[ ]:


confusion_matrix(y_test,y_pred_etc)


# In[ ]:


confusion_matrix(y_test,y_pred_gbm)


# In[ ]:


confusion_matrix(y_test,y_pred_knc)


# In[ ]:


confusion_matrix(y_test,y_pred_xg)


# In[ ]:


#ROC


# In[ ]:


fpr_dt, tpr_dt, thresholds = roc_curve(y_test, y_pred_dt_prob[: , 1], pos_label= 1)


# In[ ]:


fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)
fpr_etc, tpr_etc, thresholds = roc_curve(y_test, y_pred_etc_prob[: , 1], pos_label= 1)


# In[ ]:


fpr_knc, tpr_knc, thresholds = roc_curve(y_test, y_pred_gbm_prob[: , 1], pos_label= 1)
fpr_xg, tpr_xg, thresholds = roc_curve(y_test, y_pred_xg_prob[: , 1], pos_label= 1)
fpr_gbm, tpr_gbm,thresholds = roc_curve(y_test, y_pred_gbm_prob[: , 1], pos_label= 1)


# In[ ]:


#AUC values


# In[ ]:


auc(fpr_dt,tpr_dt)


# In[ ]:


auc(fpr_rf,tpr_rf)


# In[ ]:


auc(fpr_etc,tpr_etc)


# In[ ]:


auc(fpr_knc,tpr_knc)


# In[ ]:


auc(fpr_xg,tpr_xg)


# In[ ]:


auc(fpr_gbm,tpr_gbm)


# In[ ]:


#Precision/Recall/F-score


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_dt)


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_rf)


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_etc)


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_knc)


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_xg)


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_gbm)


# In[ ]:


# ROC curve 


# In[ ]:


fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for models')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.plot(fpr_dt, tpr_dt, label = "dt")
ax.plot(fpr_rf, tpr_rf, label = "rf")
ax.plot(fpr_etc, tpr_etc, label = "etc")
ax.plot(fpr_knc, tpr_knc, label = "knc")
ax.plot(fpr_xg, tpr_xg, label = "xg")
ax.plot(fpr_gbm, tpr_gbm, label = "gbm")
ax.legend(loc="lower right")
plt.show()


# In[ ]:




