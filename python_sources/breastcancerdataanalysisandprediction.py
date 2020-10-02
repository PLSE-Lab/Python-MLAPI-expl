#!/usr/bin/env python
# coding: utf-8

# #import all the necessary libraries required for data visualization

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


# In[4]:


data = pd.read_csv("../input/data.csv")


# In[5]:


print(data.head())


# In[6]:


data.describe()


# In[7]:


print(data.describe())


# In[8]:


data.shape


# In[9]:


# feature names as a list
col = data.columns


# In[10]:


print(col)


# In[11]:


# Drop useless variables
data = data.drop(['Unnamed: 32','id'],axis = 1)

# Reassign target
data.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)


# In[12]:


# 2 datasets
M = data[(data['diagnosis'] != 0)]
B = data[(data['diagnosis'] == 0)]


# In[13]:


X = data.drop("diagnosis", axis=1)
y = data["diagnosis"].values


# In[14]:


scale = ss()
X = scale.fit_transform(X)
print(X.shape)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)


# In[16]:


dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=50)
etc = ExtraTreesClassifier(n_estimators=50)
knc = KNeighborsClassifier()
xg = XGBClassifier(learning_rate=0.5, reg_alpha= 5, reg_lambda= 0.1)
gbm = GradientBoostingClassifier()


# In[17]:


#Train the data
dt1 = dt.fit(X_train,y_train)
rf1 = rf.fit(X_train,y_train)
etc1 = etc.fit(X_train,y_train)
knc1 = knc.fit(X_train,y_train)
xg1 = xg.fit(X_train,y_train)
gbm1 = gbm.fit(X_train,y_train)


# In[18]:


#Predict the data
y_pred_dt = dt1.predict(X_test)
y_pred_rf = rf1.predict(X_test)
y_pred_etc= etc1.predict(X_test)
y_pred_knc= knc1.predict(X_test)
y_pred_xg= xg1.predict(X_test)
y_pred_gbm= gbm1.predict(X_test)


# In[19]:


#Fetch probabilities
y_pred_dt_prob = dt1.predict_proba(X_test)
y_pred_rf_prob = rf1.predict_proba(X_test)
y_pred_etc_prob = etc1.predict_proba(X_test)
y_pred_knc_prob = knc1.predict_proba(X_test)
y_pred_xg_prob = xg1.predict_proba(X_test)
y_pred_gbm_prob= gbm1.predict_proba(X_test)


# In[20]:


#Get accuracy scores
accuracy_score(y_test,y_pred_dt)
accuracy_score(y_test,y_pred_rf)
accuracy_score(y_test,y_pred_etc)
accuracy_score(y_test,y_pred_knc)
accuracy_score(y_test,y_pred_xg)
accuracy_score(y_test,y_pred_gbm)


# In[21]:


print(accuracy_score(y_test,y_pred_dt))
print(accuracy_score(y_test,y_pred_rf))
print(accuracy_score(y_test,y_pred_etc))
print(accuracy_score(y_test,y_pred_knc))
print(accuracy_score(y_test,y_pred_xg))
print(accuracy_score(y_test,y_pred_gbm))


# In[22]:


#Confusion matrix
confusion_matrix(y_test,y_pred_dt)


# In[23]:


confusion_matrix(y_test,y_pred_rf)


# In[24]:


#ROC Graph
fpr_dt, tpr_dt, thresholds = roc_curve(y_test, y_pred_dt_prob[: , 1], pos_label= 1)


# In[25]:


fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_rf_prob[: , 1], pos_label= 1)


# In[26]:


#Fetch AUC
auc(fpr_dt,tpr_dt)


# In[27]:


auc(fpr_rf,tpr_rf)


# In[28]:


#Calculate Precision, Recall and F-score
precision_recall_fscore_support(y_test,y_pred_dt)


# In[29]:


precision_recall_fscore_support(y_test,y_pred_rf)


# In[30]:


#Plotting ROC Curve
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.plot(fpr_dt, tpr_dt, label = "dt")
ax.plot(fpr_rf, tpr_rf, label = "rf")
plt.show()


# In[31]:


drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']
data_1 = data.drop(drop_list1,axis = 1 )        
data_1.head()


# In[32]:


ax = sns.countplot(y,label="Count")
y = data.diagnosis
B, M = y.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)


# In[33]:


#correlation map
f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(data_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[34]:


cm = confusion_matrix(y_test,rf.predict(X_test))
sns.heatmap(cm,annot=True,fmt="d")


# In[35]:


# seaborn version : Uncorrelated features
fig = plt.figure(figsize=(12,12))
palette ={0 : 'lightblue', 1 : 'gold'}
edgecolor = 'blue'
plt.subplot(221)
ax1 = sns.scatterplot(x = data['smoothness_mean'], y = data['texture_mean'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('smoothness mean vs texture mean')
plt.subplot(222)
ax2 = sns.scatterplot(x = data['radius_mean'], y = data['fractal_dimension_worst'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('radius mean vs fractal dimension_worst')
plt.subplot(223)
ax3 = sns.scatterplot(x = data['texture_mean'], y = data['symmetry_mean'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('texture mean vs symmetry mean')
plt.subplot(224)
ax4 = sns.scatterplot(x = data['texture_mean'], y = data['symmetry_se'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('texture mean vs symmetry se')

fig.suptitle('Uncorrelated features', fontsize = 20)
plt.savefig('2')
plt.show()


# In[36]:


confusion_matrix(y_test,y_pred_etc)
confusion_matrix(y_test,y_pred_knc)
confusion_matrix(y_test,y_pred_xg)
confusion_matrix(y_test,y_pred_gbm)


# In[37]:


fpr_etc, tpr_etc, thresholds = roc_curve(y_test, y_pred_etc_prob[: , 1], pos_label= 1)


# In[38]:


fpr_knc, tpr_knc, thresholds = roc_curve(y_test, y_pred_knc_prob[: , 1], pos_label= 1)


# In[39]:


fpr_xg, tpr_xg, thresholds = roc_curve(y_test, y_pred_xg_prob[: , 1], pos_label= 1)


# In[40]:


fpr_gbm, tpr_gbm, thresholds = roc_curve(y_test, y_pred_gbm_prob[: , 1], pos_label= 1)


# In[41]:


auc(fpr_etc,tpr_etc)


# In[42]:


auc(fpr_knc,tpr_knc)


# In[43]:


auc(fpr_xg,tpr_xg)


# In[44]:


auc(fpr_gbm,tpr_gbm)


# In[45]:


precision_recall_fscore_support(y_test,y_pred_etc)


# In[46]:


precision_recall_fscore_support(y_test,y_pred_knc)


# In[47]:


precision_recall_fscore_support(y_test,y_pred_xg)


# In[48]:


precision_recall_fscore_support(y_test,y_pred_gbm)


# In[49]:


#Plotting ROC Curve
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


# In[ ]:




