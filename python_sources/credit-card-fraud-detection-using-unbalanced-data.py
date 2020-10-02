#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries
get_ipython().run_line_magic('reset', '-f')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OneHotEncoder as ohe
from sklearn.preprocessing import StandardScaler as ss
from sklearn.compose import ColumnTransformer as ct
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve


# In[ ]:


#reading data
#os.chdir("D:\\ml\\ex5")
#data = pd.read_csv("creditcardfraud.zip")
print(os.listdir("../input"))
data = pd.read_csv('../input/creditcard.csv')
data.head(3)


# In[ ]:


data.describe()


# In[ ]:


data.shape


# In[ ]:


data.dtypes.value_counts()


# In[ ]:


f, axes = plt.subplots(2, 2, figsize=(7, 7))
sns.boxplot(x="Class", y="Time", data=data, ax=axes[0, 0])
sns.boxplot(x="Class", y="V1", data=data, ax=axes[0, 1])
sns.boxplot(x="Class", y="V2", data=data, ax=axes[1, 0])
sns.boxplot(x="Class", y="V3", data=data, ax=axes[1, 1])
plt.show()


# In[ ]:


plt.figure(figsize = (10,10))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = data.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Greens")
plt.show()


# In[ ]:


sns.countplot(x='Class', data=data)


# In[ ]:


(data.isnull()).apply(sum, axis = 0)


# In[ ]:


y = data.iloc[:,30]
X = data.iloc[:,0:30]


# In[ ]:


X.shape


# In[ ]:


X.columns


# In[ ]:


y.head()


# In[ ]:


X_trans = ss().fit_transform(X)
X_trans.shape


# In[ ]:


X_train, X_test, y_train, y_test =   train_test_split(X_trans,
                                                      y,
                                                      test_size = 0.3,
                                                      stratify = y
                                                      )


X_train.shape    


# In[ ]:


#classifier
xg = XGBClassifier(learning_rate=0.5,
                   reg_alpha= 5,
                   reg_lambda= 0.1
                   )


# In[ ]:


#balancing data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X_train, y_train)


# In[ ]:


X_res.shape


# In[ ]:


y_res.shape


# In[ ]:


xg_res = xg.fit(X_res, y_res)


# In[ ]:


#Prediction
y_pred_xg_res = xg_res.predict(X_test)
y_pred_xg_res


# In[ ]:


y_pred_xg_res_prob = xg_res.predict_proba(X_test)
y_pred_xg_res_prob


# In[ ]:


#accuracy score
print ('Accuracy Score',accuracy_score(y_test,y_pred_xg_res))


# In[ ]:


confusion_matrix(y_test,y_pred_xg_res)


# In[ ]:


fpr_xg_res, tpr_xg_res, thresholds = roc_curve(y_test,
                                 y_pred_xg_res_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


p_xg_res,r_xg_res,f_xg_res,_ = precision_recall_fscore_support(y_test,y_pred_xg_res)


# In[ ]:


p_xg_res,r_xg_res,f_xg_res


# In[ ]:


print ('AUC using XGB and SMOTE',auc(fpr_xg_res,tpr_xg_res))


# In[ ]:


fig = plt.figure(figsize=(12,10)) 
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], ls="--")   
ax.set_xlabel('False Positive Rate')  
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for XGB and SMOTE')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.plot(fpr_xg_res, tpr_xg_res, label = "xgb")
ax.legend(loc="lower right")
plt.show()


# In[ ]:


ad = ADASYN(random_state=42)
X_ada, y_ada = sm.fit_sample(X_train, y_train)


# In[ ]:


X_ada.shape


# In[ ]:


y_ada.shape


# In[ ]:


xg_ada = xg.fit(X_ada, y_ada)


# In[ ]:


y_pred_xg_ada = xg_ada.predict(X_test)
y_pred_xg_ada


# In[ ]:


y_pred_xg_ada_prob = xg_ada.predict_proba(X_test)
y_pred_xg_ada_prob


# In[ ]:


print ('Accuracy using XGB and ADASYN',accuracy_score(y_test,y_pred_xg_ada))


# In[ ]:


confusion_matrix(y_test,y_pred_xg_ada)


# In[ ]:


fpr_xg_ada, tpr_xg_ada, thresholds = roc_curve(y_test,
                                 y_pred_xg_ada_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


p_xg_ada,r_xg_ada,f_xg_ada,_ = precision_recall_fscore_support(y_test,y_pred_xg_ada)


# In[ ]:


p_xg_ada,r_xg_ada,f_xg_ada


# In[ ]:


print ('AUC using XGB and ADASYN',auc(fpr_xg_ada,tpr_xg_ada))


# In[ ]:


fig = plt.figure(figsize=(12,10))        
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], ls="--")
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for XGB and ADASYN')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.plot(fpr_xg_ada, tpr_xg_ada, label = "xgb")
ax.legend(loc="lower right")
plt.show()

