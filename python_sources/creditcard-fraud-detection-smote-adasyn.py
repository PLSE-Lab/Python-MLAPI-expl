#!/usr/bin/env python
# coding: utf-8

# #**Credit Card Fraud Detection**

# The dataset consists of credit cards transactions in September 2013 by European cardholders. It presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly imbalanced, the positive class (frauds) account for 0.172% of all transactions.
# It consists of 28 Principle Components Analysis transformed features from V1 to V28.
# Due to confidentiality issue, no additional data of original features is provided; hence pre-analysis of features cannot be done.
# 
# In this kernel I have attempted to get the model which gives best performance while handling imbalanced data.
# Modeling algorithms for balancing data used in the kernel (click on below links to go to respective sections)-
#  - Modeling using Imbalanced data
#  - SMOTE
#  - ADASYN

# <font color=blue size = 4.8>Import necessary libraries</font>

# In[ ]:


#Importing Data manipulation and plotting modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time


# In[ ]:


#Importing libraries for performance measures
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve


# In[ ]:


#Importing libraries For data splitting
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#Import libraries for data balancing
from imblearn.over_sampling import SMOTE, ADASYN


# <font color=blue size = 4.8>Loading Dataset</font>

# In[ ]:


data = pd.read_csv("../input/creditcard.csv")


# In[ ]:


pd.options.display.max_columns = 200


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


sns.countplot(data['Class'])


# <font color=green>As we can see the data is highly imbalanced. Out of 2 lakh cases only 492 are fraud cases.</font>

# In[ ]:


f, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.distplot(data['Amount'].values,ax=axes[0])
axes[0].set_title("Distribution of Transaction Amount")
sns.distplot(data['Time'].values,ax=axes[1])
axes[1].set_title("Distribution of Transaction Time (in seconds)")
plt.show()


# In[ ]:


data.drop(['Time'], inplace = True, axis =1)


# In[ ]:



data['Class'].value_counts()[1]/data.shape[0]


# In[ ]:


y = data.iloc[:,29]
X = data.iloc[:,0:29]
X.shape


# In[ ]:


y.shape


# In[ ]:


X_train, X_test, y_train, y_test =   train_test_split(X,
                                                      y,
                                                      test_size = 0.3,
                                                      stratify = y
                                                      )


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# <font color=blue size = 4.8>Modeling on imbalanced data</font>

# Significance of confusion matrixx that can be seen for this dataset -
#  - True Positive: Number of fraud cases the model predicted as 'fraud'.
#  - False Positive: Number of non-fraud cases that the model predicted as 'fraud'.
#  - True Negative: Number of non-fraud cases that the model predicted as 'non-fraud'.
#  - False Negative: Number of fraud cases that the model predicted as 'non-fraud'.
#  - Precision: Precision is the ratio of correctly predicted fraud cases to total predicted fraud cases.
# 
# Hence we can focus on the model that gives least False Negative values.

# 'scale_pos_weight' is used for imbalanced dataset wherein it balancese the Negative and Positive weights. With the default value of '1', it implies that the positive class has a weight equal to the negative class. 
# 'max_delta_step' specifies maximum delta step to be allowed for each leaf output. It helps in logistic regression when data is highly imbalanced.
# Here is defined a range of scale_pos_weight & max_delta_step. The best combination of these parameters which will give a better accuracy, precision, recall and F score. At the same time keeping FN as minimum as possible.

# <font color=orange size = 3>Modeling using weights</font>

# In[ ]:


max_delta_step= [1,2,3,4,5,6,7,8,9,10]
scale_pos_weight= [1,2,3,4,5,6,7,8,9,10]
num_zeros = (data['Class'] == 0).sum()
num_ones = (data['Class'] == 1).sum()
sp_weight = num_zeros / num_ones
for i in max_delta_step:
    print('--------------------')
    print('Iteration ', i)
    print('--------------------')
    print('scale_pos_weight = {} '.format(i))
    print('max_delta_step = {} '.format(i))
    xgb = XGBClassifier(scale_pos_weight = i,max_delta_step=i)
    xgb.fit(X_train,y_train)
    xgb_predict = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)
    xgb_cm = confusion_matrix(y_test, xgb_predict)
    p_xg,r_xg,f_xg,_ = precision_recall_fscore_support(y_test,xgb_predict)
    print('Accuracy',accuracy_score(y_test, xgb_predict))
    print('Confusion Matrix: \n', xgb_cm)
    print('Precision: ',p_xg)
    print('Recall: ',r_xg)
    print('F score: ',f_xg)


# <font color=green>We can observe that the combination of scale_pos_weight=10 and max_delta_step=10 gives us the best values for Precision, Recall and Fscore. Also the False negatives is also minimum : 29 </font>

# <font color=orange size =3>Modeling using balanced weights in Random Forest</font>

# In[ ]:


rf = RandomForestClassifier(n_estimators=100,random_state=0, n_jobs=-1, class_weight="balanced")


# In[ ]:


rf1 = rf.fit(X_train,y_train)
y_pred_rf = rf1.predict(X_test)
y_pred_rf_prob = rf1.predict_proba(X_test)


# In[ ]:


accuracy_score(y_test,y_pred_rf)


# In[ ]:


confusion_matrix(y_test,y_pred_rf)


# In[ ]:


p_rf,r_rf,f_rf,_  = precision_recall_fscore_support(y_test,y_pred_rf)


# In[ ]:


print('Precision:',p_rf , '\nRecall',r_rf,'\nFscore',f_rf,_ )


# <font color=blue size = 4.8>Synthetic Minority Over sampling Technique (SMOTE)</font>

# In[ ]:


sm = SMOTE(random_state=42)


# In[ ]:


X_smote, y_smote = sm.fit_sample(X_train, y_train)


# In[ ]:


X_smote.shape


# In[ ]:


y_smote.shape


# In[ ]:


np.sum(y_smote)/len(y_smote)

#We can see now the data is balanced


# In[ ]:


y_smote = y_smote.reshape(y_smote.size, 1)
y_smote.shape 


# In[ ]:


xg_smote = XGBClassifier(learning_rate=0.1,
                   reg_alpha= 0,
                   reg_lambda= 1,
                   )
rf_smote = RandomForestClassifier(n_estimators=100,n_jobs=-1)

columns = X_train.columns
X_smote = pd.DataFrame(data = X_smote, columns = columns)

xg_fit = xg_smote.fit(X_smote,y_smote)
rf_fit = rf_smote.fit(X_smote,y_smote)

y_pred_xgb = xg_fit.predict(X_test)
y_pred_rfb = rf_fit.predict(X_test)

y_pred_xgb_prob = xg_fit.predict_proba(X_test)
y_pred_rfb_prob = rf_fit.predict_proba(X_test)

p_rfb,r_rfb,f_rfb,_  = precision_recall_fscore_support(y_test,y_pred_rfb)
p_xgb,r_xgb,f_xgb,_  = precision_recall_fscore_support(y_test,y_pred_xgb)


print('Random Forest:\n')
print('Accuracy - ',accuracy_score(y_test,y_pred_rfb))
print('\nPrecision - ',p_rfb , '\nRecall - ',r_rfb,'\nFscore - ',f_rfb,_ )
print('Confusion Matrix -\n',confusion_matrix(y_test,y_pred_rfb))

print('XGBoost:\n')
print('Accuracy - ',accuracy_score(y_test,y_pred_xgb))
print('\nPrecision - ',p_xgb , '\nRecall - ',r_xgb,'\nFscore - ',f_xgb,_ )
print('Confusion Matrix - \n',confusion_matrix(y_test,y_pred_xgb))


# <font color=green>It is observed that Random Forest gives better results as compared to XGBoost</font>

# <font color=blue size = 4.8>ADAptive SYNthetic (ADASYN)</font>

# In[ ]:


adasyn = ADASYN(random_state=42)
X_ada, y_ada = adasyn.fit_sample(X_train, y_train)


# In[ ]:


X_ada.shape


# In[ ]:


y_ada.shape


# In[ ]:


np.sum(y_ada)/len(y_ada)

y_ada = y_ada.reshape(y_ada.size, 1)
y_ada.shape 

xg_ada = XGBClassifier(learning_rate=0.5,
                   reg_alpha= 0.1,
                   reg_lambda= 1,
                   )


rf_ada = RandomForestClassifier(n_estimators=100,n_jobs=-1)

columns = X_train.columns
X_ada = pd.DataFrame(data = X_ada, columns = columns)

xg_fit = xg_ada.fit(X_ada,y_ada)


rf_fit = rf_ada.fit(X_ada,y_ada)

y_pred_xgb = xg_fit.predict(X_test)


y_pred_rfb = rf_fit.predict(X_test)

y_pred_xgb_prob = xg_fit.predict_proba(X_test)
y_pred_rfb_prob = rf_fit.predict_proba(X_test)

p_rfb,r_rfb,f_rfb,_  = precision_recall_fscore_support(y_test,y_pred_rfb)

p_xgb,r_xgb,f_xgb,_  = precision_recall_fscore_support(y_test,y_pred_xgb)


print('Random Forest:\n')
print('Accuracy - ',accuracy_score(y_test,y_pred_rfb))
print('Precision - \n',p_rfb , 'Recall - \n',r_rfb,'Fscore - \n',f_rfb,_ )
print('Confusion Matrix - \n',confusion_matrix(y_test,y_pred_rfb))

print('XGBoost:\n')
print('Accuracy XGBoost',accuracy_score(y_test,y_pred_xgb))
print('Precision XGBoost:',p_xgb , 'Recall',r_xgb,'Fscore',f_xgb,_ )
print('Confusion Matrix - \n',confusion_matrix(y_test,y_pred_xgb))


# <font color=green>For Adasyn as well it is observed that Random Forest gives better results as compared to XGBoost</font>

# In[ ]:





# In[ ]:




