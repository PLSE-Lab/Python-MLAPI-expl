#!/usr/bin/env python
# coding: utf-8

# Load libraries

# In[ ]:


get_ipython().run_line_magic('reset', '-f')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Import SMOTE AND ADASYN to balance the dataset

# In[ ]:


from imblearn.over_sampling import SMOTE, ADASYN


# Processing data

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OneHotEncoder as ohe
from sklearn.preprocessing import StandardScaler as ss
from sklearn.compose import ColumnTransformer as ct


# Call Modeling module, we will be using XBG for modeling.

# In[ ]:


from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support


# for ROC graphs & metrics

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve


# Read data

# In[ ]:


#os.chdir("F:\\Practice\\Machine Learning and Deep Learning\\Classes\\Assignment\\Kaggle\\4th")


# In[ ]:


#os.listdir()


# In[ ]:


#ccf=pd.read_csv("creditcardfraud.zip")
ccf = pd.read_csv("../input/creditcard.csv")


# Explore data

# In[ ]:


ccf.head(3)


# In[ ]:


ccf.info()


# In[ ]:


ccf.describe()


# In[ ]:


ccf.shape


# In[ ]:


ccf.columns.values


# In[ ]:


ccf.dtypes.value_counts()


# Ploting

# In[ ]:


plt.style.use('ggplot')

f, ax = plt.subplots(figsize=(11, 15))

ax.set_facecolor('#fafafa')
ax.set(xlim=(-5, 5))
plt.ylabel('Variables')
plt.title("Overview Data Set")
ax = sns.boxplot(data = ccf.drop(columns=['Amount', 'Class', 'Time']), 
  orient = 'h', 
  palette = 'Set2')


# Correlation matrix

# In[ ]:


f, (ax1, ax2) = plt.subplots(1,2,figsize =( 18, 8))
corr = ccf.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap((ccf.loc[ccf['Class'] ==1]).corr(), vmax = .8, square=True, ax = ax1, cmap = 'afmhot', mask=mask);
ax1.set_title('Fraud')
sns.heatmap((ccf.loc[ccf['Class'] ==0]).corr(), vmax = .8, square=True, ax = ax2, cmap = 'YlGnBu', mask=mask);
ax2.set_title('Normal')
plt.show()


# In[ ]:


sns.countplot(x='Class', data=ccf)


# Finding the Missing Value's

# In[ ]:


(ccf.isnull()).apply(sum, axis = 0)


# Separation into target/predictors

# In[ ]:


y = ccf.iloc[:,30]
X = ccf.iloc[:,0:30]


# In[ ]:


X.shape


# In[ ]:


X.columns


# In[ ]:


y.head()


# Fit and transform

# In[ ]:


X_trans = ss().fit_transform(X)
X_trans.shape


# Split data into train/test

# In[ ]:


X_train, X_test, y_train, y_test =   train_test_split(X_trans,
                                                      y,
                                                      test_size = 0.3,
                                                      stratify = y
                                                      )


# In[ ]:


X_train.shape


# XGB Classifier

# In[ ]:


xg = XGBClassifier(learning_rate=0.5,
                   reg_alpha= 5,
                   reg_lambda= 0.1
                   )


# SMOTE-Balancing the data

# In[ ]:


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X_train, y_train)


# As we can below the X_res and y_res are having same number of rows, that means we have balanced the data

# In[ ]:


X_res.shape


# In[ ]:


y_res.shape


# Fit the data

# In[ ]:


xg_res = xg.fit(X_res, y_res)


# Predict

# In[ ]:


y_pred_xg_res = xg_res.predict(X_test)
y_pred_xg_res


# In[ ]:


y_pred_xg_res_prob = xg_res.predict_proba(X_test)
y_pred_xg_res_prob


# Accuracy

# In[ ]:


print ('Accuracy using XGB and SMOTE',accuracy_score(y_test,y_pred_xg_res))


# Confusion Matrix

# In[ ]:


confusion_matrix(y_test,y_pred_xg_res)


# In[ ]:


fpr_xg_res, tpr_xg_res, thresholds = roc_curve(y_test,
                                 y_pred_xg_res_prob[: , 1],
                                 pos_label= 1
                                 )


# Precion, Recall and F1 Score

# In[ ]:


p_xg_res,r_xg_res,f_xg_res,_ = precision_recall_fscore_support(y_test,y_pred_xg_res)


# In[ ]:


p_xg_res,r_xg_res,f_xg_res


# In[ ]:


print ('AUC using XGB and SMOTE',auc(fpr_xg_res,tpr_xg_res))


# ROC Curve

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


# ADASYN-Balancing the data

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


# In[ ]:




