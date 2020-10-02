#!/usr/bin/env python
# coding: utf-8

# ####  Credit Card Fraud - Working with unbalanced dataset

# __Credit Card Fraud dataset__
# 
# ================================
# 
# The datasets contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Due to confidentiality issues, original features and more background information about the data can't be provided. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# __Call libraries__

# In[ ]:


get_ipython().run_line_magic('reset', '-f')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# _Import SMOTE AND ADASYN to balance the dataset_

# In[ ]:


from imblearn.over_sampling import SMOTE, ADASYN


# _Processing data_

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OneHotEncoder as ohe
from sklearn.preprocessing import StandardScaler as ss
from sklearn.compose import ColumnTransformer as ct


# _Call Modeling module, we will be using XBG for modeling._

# In[ ]:


from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support


# _for ROC graphs & metrics_

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve


# _Read data_

# In[ ]:


os.chdir("../input")
cr = pd.read_csv("creditcard.csv")


# _Explore data_

# In[ ]:


cr.head()


# In[ ]:


cr.info()


# In[ ]:


cr.describe()


# In[ ]:


cr.shape


# In[ ]:


cr.columns.values


# In[ ]:


cr.dtypes.value_counts()


# _ploting_

# In[ ]:


f, axes = plt.subplots(2, 2, figsize=(7, 7))
sns.boxplot(x="Class", y="Time", data=cr, ax=axes[0, 0])
sns.boxplot(x="Class", y="V1", data=cr, ax=axes[0, 1])
sns.boxplot(x="Class", y="V2", data=cr, ax=axes[1, 0])
sns.boxplot(x="Class", y="V3", data=cr, ax=axes[1, 1])
plt.show()


# In[ ]:


cr.corr()
sns.heatmap(cr.corr())


# In[ ]:


sns.countplot(x='Class', data=cr)


# _Just check any columns have missing data_

# In[ ]:


(cr.isnull()).apply(sum, axis = 0)


# <font color=blue>Modelling</font>

# _Separation into target/predictors_

# In[ ]:


y = cr.iloc[:,30]
X = cr.iloc[:,0:30]


# In[ ]:


X.shape              


# In[ ]:


X.columns


# In[ ]:


y.head()


# _Fit and transform_

# In[ ]:


X_trans = ss().fit_transform(X)
X_trans.shape


# _Split data into train/test_     

# In[ ]:


X_train, X_test, y_train, y_test =   train_test_split(X_trans,
                                                      y,
                                                      test_size = 0.3,
                                                      stratify = y
                                                      )


X_train.shape        


# _XGB Classifier_

# In[ ]:


xg = XGBClassifier(learning_rate=0.5,
                   reg_alpha= 5,
                   reg_lambda= 0.1
                   )


# In[ ]:


xg_imb = xg.fit(X_train,y_train)


# In[ ]:


y_pred_xg_imb = xg_imb.predict(X_test)


# In[ ]:


y_pred_xg_imb_prob = xg_imb.predict_proba(X_test)


# In[ ]:


accuracy_score(y_test,y_pred_xg_imb)


# In[ ]:


confusion_matrix(y_test,y_pred_xg_imb)


# In[ ]:


fpr_xg_imb, tpr_xg_imb, thresholds = roc_curve(y_test,
                                 y_pred_xg_imb_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


p_xg_imb,r_xg_imb,f_xg_imb,_ = precision_recall_fscore_support(y_test,y_pred_xg_imb)


# In[ ]:


print(auc(fpr_xg_imb,tpr_xg_imb))


# In[ ]:


p_xg_imb,r_xg_imb,f_xg_imb,_ = precision_recall_fscore_support(y_test,y_pred_xg_imb)


# In[ ]:


p_xg_imb


# In[ ]:


r_xg_imb


# In[ ]:


f_xg_imb


# In[ ]:


fig = plt.figure(figsize=(12,10)) 
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], ls="--")   
ax.set_xlabel('False Positive Rate')  
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for XGB and Unbalanced data')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.plot(fpr_xg_imb, tpr_xg_imb, label = "xgb")
ax.legend(loc="lower right")
plt.show()


# <font color=blue>SMOTE-Balancing the data</font>

# In[ ]:


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X_train, y_train)


# *As we can below the X_res and y_res are having same number of rows, that means we have balanced the data*

# In[ ]:


X_res.shape


# In[ ]:


y_res.shape


# _Fit the data_

# In[ ]:


xg_res = xg.fit(X_res, y_res)


# In[ ]:


#Predict
y_pred_xg_res = xg_res.predict(X_test)
y_pred_xg_res


# In[ ]:


y_pred_xg_res_prob = xg_res.predict_proba(X_test)
y_pred_xg_res_prob


# In[ ]:


#Accuracy
print ('Accuracy using XGB and SMOTE',accuracy_score(y_test,y_pred_xg_res))


# In[ ]:


#Confusion Matrix
confusion_matrix(y_test,y_pred_xg_res)


# In[ ]:


fpr_xg_res, tpr_xg_res, thresholds = roc_curve(y_test,
                                 y_pred_xg_res_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


#Precion, Recall and F1 Score
p_xg_res,r_xg_res,f_xg_res,_ = precision_recall_fscore_support(y_test,y_pred_xg_res)


# In[ ]:


p_xg_res,r_xg_res,f_xg_res


# In[ ]:


print ('AUC using XGB and SMOTE',auc(fpr_xg_res,tpr_xg_res))


# _ROC Curve_

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


# <font color=blue>ADASYN-Balancing the data</font>

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

