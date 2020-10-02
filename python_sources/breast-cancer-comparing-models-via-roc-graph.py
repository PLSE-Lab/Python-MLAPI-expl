#!/usr/bin/env python
# coding: utf-8

# ## Basic Imports

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Sklearn Imports

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support


# ## Read data from file

# In[ ]:


import os
os.listdir('./../input')


# In[ ]:


df = pd.read_csv("../input/data.csv")


# ## Analyze data 

# In[ ]:


print(df.shape)


# In[ ]:


print(df.info())


# In[ ]:


df.head()


# In[ ]:


df.isna().any()


# Deleting unneeded columns

# In[ ]:


del df['id']
del df['Unnamed: 32']


# Create our x (input data) and y (output data)

# In[ ]:


x = df.loc[:, 'radius_mean': 'fractal_dimension_worst']
y = df['diagnosis']


# In[ ]:


di = {'M': 1, 'B': 0}
y = y.map(di)
print(y)


# In[ ]:


scale = StandardScaler()
X = scale.fit_transform(x)


# In[ ]:


pca = PCA(n_components = 2)
X = pca.fit_transform(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)


# ### Classifiers
# 
# Create different classifiers and get predictions and prediction probabilities from them.

# In[ ]:


dt = DecisionTreeClassifier()
dt1 = dt.fit(X_train,y_train)
y_pred_dt = dt1.predict(X_test)
y_pred_dt_prob = dt1.predict_proba(X_test)


rf = RandomForestClassifier(n_estimators=100)
rf1 = rf.fit(X_train,y_train)
y_pred_rf = rf1.predict(X_test)
y_pred_rf_prob = rf1.predict_proba(X_test)


etc = ExtraTreesClassifier(n_estimators=100)
etc1 = etc.fit(X_train,y_train)
y_pred_etc= etc1.predict(X_test)
y_pred_etc_prob = etc1.predict_proba(X_test)


knc = KNeighborsClassifier()
knc1 = knc.fit(X_train,y_train)
y_pred_knc= knc1.predict(X_test)
y_pred_knc_prob = knc1.predict_proba(X_test)


xg = XGBClassifier(learning_rate=0.5, reg_alpha= 5, reg_lambda= 0.1)
xg1 = xg.fit(X_train,y_train)
y_pred_xg= xg1.predict(X_test)
y_pred_xg_prob = xg1.predict_proba(X_test)


gbm = GradientBoostingClassifier()
gbm1 = gbm.fit(X_train,y_train)
y_pred_gbm= gbm1.predict(X_test)
y_pred_gbm_prob= gbm1.predict_proba(X_test)


# #### Comparing the output of predictions with actual results

# In[ ]:


print("DecisionTreeClassifier: {0}".format(accuracy_score(y_test,y_pred_dt)))
print("RandomForestClassifier: {0}".format(accuracy_score(y_test,y_pred_rf)))
print("ExtraTreesClassifier: {0}".format(accuracy_score(y_test,y_pred_etc)))
print("KNeighborsClassifier: {0}".format(accuracy_score(y_test,y_pred_knc)))
print("XGBClassifier: {0}".format(accuracy_score(y_test,y_pred_xg)))
print("GradientBoostingClassifier: {0}".format(accuracy_score(y_test,y_pred_gbm)))


# Confusion Matrix

# In[ ]:


print("DecisionTreeClassifier: ", confusion_matrix(y_test,y_pred_dt))
print("RandomForestClassifier: ", confusion_matrix(y_test,y_pred_rf))
print("ExtraTreesClassifier: ", confusion_matrix(y_test,y_pred_etc))
print("GradientBoostingClassifier: ", confusion_matrix(y_test,y_pred_gbm))
print("KNeighborsClassifier: ", confusion_matrix(y_test,y_pred_knc))
print("XGBClassifier: ", confusion_matrix(y_test,y_pred_xg))


# In[ ]:


print("No. of elements in y_test:", len(y_test))


# ROC Graph

# In[ ]:


fpr_dt, tpr_dt, thresholds = roc_curve(y_test, y_pred_dt_prob[: , 1], pos_label= 1)
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)
fpr_etc, tpr_etc, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)
fpr_knc, tpr_knc, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)
fpr_xg, tpr_xg, thresholds = roc_curve(y_test, y_pred_xg_prob[: , 1], pos_label= 1)
fpr_gbm, tpr_gbm, thresholds = roc_curve(y_test, y_pred_gbm_prob[: , 1], pos_label= 1)


# ### AUC Values

# In[ ]:


print("DecisionTreeClassifier: {0}".format(auc(fpr_dt,tpr_dt)))
print("RandomForestClassifier: {0}".format(auc(fpr_rf,tpr_rf)))
print("ExtraTreesClassifier: {0}".format(auc(fpr_etc,tpr_etc)))
print("GradientBoostingClassifier: {0}".format(auc(fpr_gbm,tpr_gbm)))
print("KNeighborsClassifier: {0}".format(auc(fpr_knc,tpr_knc)))
print("XGBClassifier: {0}".format(auc(fpr_xg,tpr_xg)))


# ### Precision/Recall/F-score for each label (0,1)

# In[ ]:


print("DecisionTreeClassifier: ")
print(precision_recall_fscore_support(y_test,y_pred_dt))
print("RandomForestClassifier: ")
print(precision_recall_fscore_support(y_test,y_pred_rf))
print("ExtraTreesClassifier: ")
print(precision_recall_fscore_support(y_test,y_pred_etc))
print("GradientBoostingClassifier: ")
print(precision_recall_fscore_support(y_test,y_pred_gbm))
print("KNeighborsClassifier: ")
print(precision_recall_fscore_support(y_test,y_pred_knc))
print("XGBClassifier: ")
print(precision_recall_fscore_support(y_test,y_pred_xg))


# In[ ]:


# Plot ROC curve now
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)

# Connect diagonals
ax.plot([0, 1], [0, 1], ls="--")

# Labels etc
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve')

# Set graph limits
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])

# Plot each graph now
ax.plot(fpr_dt, tpr_dt, label = "dt")
ax.plot(fpr_rf, tpr_rf, label = "rf")
ax.plot(fpr_etc, tpr_etc, label = "etc")
ax.plot(fpr_knc, tpr_knc, label = "knc")
ax.plot(fpr_xg, tpr_xg, label = "xg")
ax.plot(fpr_gbm, tpr_gbm, label = "gbm")

# Set legend and show plot
ax.legend(loc="lower right")
plt.show()


# ### From graph, we see algorithms xg and knc give us best results.

# In[ ]:




