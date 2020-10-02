#!/usr/bin/env python
# coding: utf-8

# 1.1 Pandas and Numpy

# In[1]:


import numpy as np
import pandas as pd


# 1.2 For plotting

# In[2]:


import matplotlib.pyplot as plt


# 1.3 For modeling

# In[3]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# 1.4 For performance measures

# In[4]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


# 1.5 Import the Data Scaler

# In[5]:


from sklearn.preprocessing import StandardScaler as ss


# 1.6 Import PCA class

# In[6]:


from sklearn.decomposition import PCA


# 1.7 For data splitting

# In[7]:


from sklearn.model_selection import train_test_split


# 1.8 Data Preprocessor for binarization

# In[8]:


from sklearn import preprocessing


# Data importing and viewing the data

# In[19]:


data = pd.read_csv("../input/data.csv")
print("Information of data columns and data type:")
print(data.info())


# In[10]:


print("Quick Glance of the data: ")
print(data.head())


# In[11]:


print("Data Information: ")
print(data.describe())


# In[12]:


print("Shape of the Data: ")
print(data.shape)


# Data Binarization M = 1 , B = 0

# In[20]:


print("Original Values in the diagnosis column and their count: ")
print(data['diagnosis'].value_counts())


# Binarizing now

# In[22]:



lb = preprocessing.LabelBinarizer()
data['diagnosis'] = lb.fit_transform(data['diagnosis'])


# In[26]:


print("Looking for the data categories: ")
print(lb.classes_)


# In[27]:


print("Check the bindarized data: ")
print(data['diagnosis'].value_counts())


# Droppping the unwanted columns

# In[28]:


data = data.drop(["id", "Unnamed: 32"], axis=1)


# Splitting the Features and Target Data into X and y

# In[29]:


X = data.drop("diagnosis", axis=1)
y = data["diagnosis"].values
print(X.shape)
print(y.shape)


# Scale the data

# In[30]:


scale = ss()
X = scale.fit_transform(X)
print(X.shape)
print(X[:5,:])


# Apply PCA

# In[31]:


pca = PCA(n_components = 0.95)
X = pca.fit_transform(X)
print(X.shape)
print(X[:5,:])


# Explained Variance

# In[32]:


print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())


# Split and shuffle data

# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True )


# Create default classifiers

# In[33]:


dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)
etc = ExtraTreesClassifier(n_estimators=100)
knc = KNeighborsClassifier()
xg = XGBClassifier(learning_rate=0.5, reg_alpha= 5, reg_lambda= 0.1)
gbm = GradientBoostingClassifier()


# Train data

# In[37]:


dt1 = dt.fit(X_train,y_train)
rf1 = rf.fit(X_train,y_train)
etc1 = etc.fit(X_train,y_train)
knc1 = knc.fit(X_train,y_train)
xg1 = xg.fit(X_train,y_train)
gbm1 = gbm.fit(X_train,y_train)


# Make predictions

# In[38]:


y_pred_dt = dt1.predict(X_test)
y_pred_rf = rf1.predict(X_test)
y_pred_etc= etc1.predict(X_test)
y_pred_knc= knc1.predict(X_test)
y_pred_xg= xg1.predict(X_test)
y_pred_gbm= gbm1.predict(X_test)


# Get probability values

# In[39]:


y_pred_dt_prob = dt1.predict_proba(X_test)
y_pred_rf_prob = rf1.predict_proba(X_test)
y_pred_etc_prob = etc1.predict_proba(X_test)
y_pred_knc_prob = knc1.predict_proba(X_test)
y_pred_xg_prob = xg1.predict_proba(X_test)
y_pred_gbm_prob= gbm1.predict_proba(X_test)


# Calculate accuracy

# In[40]:


print("DecisionTreeClassifier: {0}".format(accuracy_score(y_test,y_pred_dt)))
print("RandomForestClassifier: {0}".format(accuracy_score(y_test,y_pred_rf)))
print("ExtraTreesClassifier: {0}".format(accuracy_score(y_test,y_pred_etc)))
print("KNeighborsClassifier: {0}".format(accuracy_score(y_test,y_pred_knc)))
print("XGBClassifier: {0}".format(accuracy_score(y_test,y_pred_xg)))
print("GradientBoostingClassifier: {0}".format(accuracy_score(y_test,y_pred_gbm)))


# Calculate Confusion Matrix

# In[41]:


print("DecisionTreeClassifier: ")
print(confusion_matrix(y_test,y_pred_dt))
print("RandomForestClassifier: ")
print(confusion_matrix(y_test,y_pred_rf))
print("ExtraTreesClassifier: ")
print(confusion_matrix(y_test,y_pred_etc))
print("GradientBoostingClassifier: ")
print(confusion_matrix(y_test,y_pred_gbm))
print("KNeighborsClassifier: ")
print(confusion_matrix(y_test,y_pred_knc))
print("XGBClassifier: ")
print(confusion_matrix(y_test,y_pred_xg))


# Calculate ROC graph

# In[42]:


fpr_dt, tpr_dt, thresholds = roc_curve(y_test, y_pred_dt_prob[: , 1], pos_label= 1)
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)
fpr_etc, tpr_etc, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)
fpr_knc, tpr_knc, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)
fpr_xg, tpr_xg, thresholds = roc_curve(y_test, y_pred_xg_prob[: , 1], pos_label= 1)
fpr_gbm, tpr_gbm,thresholds = roc_curve(y_test, y_pred_gbm_prob[: , 1], pos_label= 1)


# Get AUC values

# In[43]:


print("DecisionTreeClassifier: {0}".format(auc(fpr_dt,tpr_dt)))
print("RandomForestClassifier: {0}".format(auc(fpr_rf,tpr_rf)))
print("ExtraTreesClassifier: {0}".format(auc(fpr_etc,tpr_etc)))
print("GradientBoostingClassifier: {0}".format(auc(fpr_gbm,tpr_gbm)))
print("KNeighborsClassifier: {0}".format(auc(fpr_knc,tpr_knc)))
print("XGBClassifier: {0}".format(auc(fpr_xg,tpr_xg)))


# Precision/Recall/F-score for each label (0,1)

# In[44]:


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


# 1.1 Plot ROC curve now

# In[45]:


# Plot ROC curve now
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)

# Connect diagonals
ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line

# Labels etc
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for models')

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

