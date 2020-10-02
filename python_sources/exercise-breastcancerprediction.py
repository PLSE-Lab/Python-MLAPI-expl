#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler as ss
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# In[ ]:


# os.chdir("E:/A ML/Exercise 4")


# In[ ]:


data = pd.read_csv("../input/data.csv")


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.shape


# In[ ]:


col = data.columns


# In[ ]:


data['Unnamed: 32'].isnull().sum()


# In[ ]:


data = data.drop(['Unnamed: 32','id'],axis = 1)


# In[ ]:


data.head()


# In[ ]:


X = data.loc[:, "radius_mean" : "fractal_dimension_worst"]


# In[ ]:


y = data['diagnosis']


# In[ ]:





# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


y=y.map({'M': 1,'B': 0})


# In[ ]:


y.head()


# In[ ]:


scale = ss()
X = scale.fit_transform(X)


# In[ ]:


X


# In[ ]:


pca = PCA()
out = pca.fit_transform(X)
out.shape 


# In[ ]:


pca.explained_variance_ratio_  


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.3,
                                                    shuffle = True
                                                    )


# In[ ]:


dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)
etc = ExtraTreesClassifier(n_estimators=100)
knc = KNeighborsClassifier()
xg = XGBClassifier(learning_rate=0.5, reg_alpha= 5, reg_lambda= 0.1)
gbm = GradientBoostingClassifier()


# In[ ]:


dt1 = dt.fit(X_train,y_train)
rf1 = rf.fit(X_train,y_train)
etc1 = etc.fit(X_train,y_train)
knc1 = knc.fit(X_train,y_train)
xg1 = xg.fit(X_train,y_train)
gbm1 = gbm.fit(X_train,y_train)


# In[ ]:


y_pred_dt = dt1.predict(X_test)
y_pred_rf = rf1.predict(X_test)
y_pred_etc= etc1.predict(X_test)
y_pred_knc= knc1.predict(X_test)
y_pred_xg= xg1.predict(X_test)
y_pred_gbm= gbm1.predict(X_test)


# In[ ]:


y_pred_dt_prob = dt1.predict_proba(X_test)
y_pred_rf_prob = rf1.predict_proba(X_test)
y_pred_etc_prob = etc1.predict_proba(X_test)
y_pred_knc_prob = knc1.predict_proba(X_test)
y_pred_xg_prob = xg1.predict_proba(X_test)
y_pred_gbm_prob= gbm1.predict_proba(X_test)


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


confusion_matrix(y_test,y_pred_dt)


# In[ ]:


confusion_matrix(y_test,y_pred_rf)


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_dt)


# In[ ]:


fpr_dt, tpr_dt, thresholds = roc_curve(y_test, y_pred_dt_prob[: , 1], pos_label= 1)


# In[ ]:


auc(fpr_dt,tpr_dt)


# In[ ]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.plot(fpr_dt, tpr_dt, label = "dt")
plt.show()


# In[ ]:


fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)
auc(fpr_rf,tpr_rf)


# In[ ]:


precision_recall_fscore_support(y_test,y_pred_rf)


# In[ ]:


ax.plot(fpr_rf, tpr_rf, label = "rf")
plt.show()


# In[ ]:


fpr_etc, tpr_etc, thresholds = roc_curve(y_test, y_pred_etc_prob[: , 1], pos_label= 1)
fpr_knc, tpr_knc, thresholds = roc_curve(y_test, y_pred_knc_prob[: , 1], pos_label= 1)
fpr_xg, tpr_xg, thresholds = roc_curve(y_test, y_pred_xg_prob[: , 1], pos_label= 1)
fpr_gbm, tpr_gbm, thresholds = roc_curve(y_test, y_pred_gbm_prob[: , 1], pos_label= 1)


# In[ ]:


auc(fpr_etc,tpr_etc)


# In[ ]:


auc(fpr_knc,tpr_knc)


# In[ ]:


auc(fpr_xg,tpr_xg)


# In[ ]:


auc(fpr_gbm,tpr_gbm)


# In[ ]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.plot(fpr_dt, tpr_dt, label = "dt")
ax.plot(fpr_rf, tpr_rf, label = "rf")
ax.plot(fpr_etc, tpr_etc, label = "etc")
ax.plot(fpr_knc, tpr_knc, label = "knc")
ax.plot(fpr_xg, tpr_xg, label = "xg")
ax.plot(fpr_gbm, tpr_gbm, label = "gbm")
plt.show()

