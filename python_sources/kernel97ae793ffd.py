#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('reset', '-f')

import numpy as np
import pandas as pd

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Class for applying multiple data transformation jobs
from sklearn.compose import ColumnTransformer as ct
# Scale numeric data
from sklearn.preprocessing import StandardScaler as ss
#  One hot encode data--Convert to dummy
from sklearn.preprocessing import OneHotEncoder as ohe
#  For clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[4]:


# For modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier


# In[5]:


# For generating dataset
from sklearn.datasets import make_hastie_10_2

# For performance measures
from sklearn.metrics import accuracy_score
# From sklearn.metrics import
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# For data splitting
from sklearn.model_selection import train_test_split


# In[6]:


import os


# In[ ]:





# In[ ]:


#os.chdir("C:\Imp_Docs\Machine Learning\Exercises\Exercise - 4")


# In[7]:


df = pd.read_csv("../input/data.csv")


# In[8]:


df.shape


# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


# Plotting the Countplot graph

sns.countplot(x='diagnosis',data=df)


# In[14]:


# Plotting the Jointplot graph

sns.jointplot(x='radius_mean',y='perimeter_mean',data=df)


# In[15]:


# Checking if there are any Null values
df.isnull().values.any()
df.isnull().sum()


# In[16]:


# Dropping columns "ID" and "Unnamed: 32"
 
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)


# In[17]:


df.columns


# In[18]:


# Assigning values 1 and 0 to "M" and "B"

df.diagnosis[df.diagnosis == 'M'] = 1
df.diagnosis[df.diagnosis == 'B'] = 0


# In[19]:


print(df)


# In[20]:


y = df['diagnosis']
y=y.astype('int')


# In[21]:


y


# In[22]:


# Plotting the Boxplot Graph

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(1,4,1)
ax1.set_xticklabels(labels = 'Radius mean', rotation=90)

sns.boxenplot(x='diagnosis',y='radius_mean',data=df)
ax1 = fig.add_subplot(1,4,2)
sns.boxenplot(x='diagnosis',y='texture_mean',data=df)
ax1 = fig.add_subplot(1,4,3)
sns.boxenplot(x='diagnosis',y='perimeter_mean',data=df)
ax1 = fig.add_subplot(1,4,4)
sns.boxenplot(x='diagnosis',y='area_mean',data=df)


# In[23]:


fig2 = plt.figure(figsize=(12,12))
ax2 = fig2.add_subplot(1,4,1)
sns.boxenplot(x='diagnosis',y='smoothness_mean',data=df)
ax2 = fig2.add_subplot(1,4,2)
sns.boxenplot(x='diagnosis',y='compactness_mean',data=df)
ax2 = fig2.add_subplot(1,4,3)
sns.boxenplot(x='diagnosis',y='concavity_mean',data=df)
ax2 = fig2.add_subplot(1,4,4)
sns.boxenplot(x='diagnosis',y='concave points_mean',data=df)


# In[24]:


# Selecting the Columns 
X = df.loc[:, 'radius_mean' : 'fractal_dimension_worst']


# In[25]:


X.isnull().sum()


# In[26]:


X.head()


# In[27]:


X.shape


# In[28]:


# Scale the Numeric data

scaleit = ss()
s=scaleit.fit_transform(df.loc[:, 'radius_mean' : 'fractal_dimension_worst'])
s=scaleit.fit_transform(X)


# In[29]:


pca = PCA()
principleComp = pca.fit_transform(X)


# In[30]:


principleComp.shape


# In[31]:


pca.explained_variance_ratio_


# In[32]:


X = pca.explained_variance_ratio_.cumsum()


# In[33]:


X


# In[34]:


# Plotting the Distplot graph
sns.distplot(X,bins=5)


# In[35]:


X = principleComp[:,0:11]


# In[36]:


# Splitting and Shuffling the Data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    shuffle = True
                                                    )


# In[37]:


X_train.shape


# In[38]:


X_test.shape


# In[39]:


y_train[:4]


# In[40]:


X_test


# In[41]:


y_train


# In[42]:


y_test


# In[43]:


# Instantiate the Classifiers
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=5)
et = ExtraTreesClassifier(n_estimators=10)
xgb = XGBClassifier(learning_rate=0.5,
                   reg_alpha= 5,
                   reg_lambda= 0.1)
gbm = GradientBoostingClassifier()
kn = KNeighborsClassifier(n_neighbors=10)


# In[44]:


# Train the data
dt1 = dt.fit(X_train,y_train)
rf1 = rf.fit(X_train,y_train)
et1 = et.fit(X_train,y_train)
xgb1 = xgb.fit(X_train,y_train)
gbm1 = gbm.fit(X_train,y_train)
kn1 = kn.fit(X_train,y_train)


# In[45]:


# Data Predictions
y_pred_dt = dt1.predict(X_test)
y_pred_rf = rf1.predict(X_test)
y_pred_et = et1.predict(X_test)
y_pred_xgb= xgb1.predict(X_test)
y_pred_gbm = gbm1.predict(X_test)
y_pred_kn = kn1.predict(X_test)


# Get probability values

# In[46]:


y_pred_dt_prob = dt1.predict_proba(X_test)


# In[47]:


y_pred_rf_prob = rf1.predict_proba(X_test)
y_pred_et_prob = et1.predict_proba(X_test)
y_pred_xgb_prob = xgb1.predict_proba(X_test)
y_pred_gbm_prob= gbm1.predict_proba(X_test)
y_pred_kn_prob = kn1.predict_proba(X_test)


# In[48]:


y_pred_dt_prob


# In[49]:


y_pred_rf_prob


# In[50]:


y_pred_et_prob


# In[51]:


y_pred_xgb_prob


# In[52]:


y_pred_gbm_prob


# In[53]:


y_pred_kn_prob


# In[54]:


# Calculate accuracy
accuracy_score(y_test,y_pred_dt)


# In[55]:


accuracy_score(y_test,y_pred_rf)


# In[56]:


accuracy_score(y_test,y_pred_et)


# In[57]:


accuracy_score(y_test,y_pred_xgb)


# In[58]:


accuracy_score(y_test,y_pred_gbm)


# In[59]:


accuracy_score(y_test,y_pred_kn)


# In[60]:


# Draw Confusion matrix

confusion_matrix(y_test,y_pred_dt)


# In[61]:


confusion_matrix(y_test,y_pred_rf)


# In[62]:


confusion_matrix(y_test,y_pred_et)


# In[63]:


confusion_matrix(y_test,y_pred_xgb)


# In[64]:


confusion_matrix(y_test,y_pred_gbm)


# In[65]:


confusion_matrix(y_test,y_pred_kn)


# In[66]:


tn,fp,fn,tp= confusion_matrix(y_test,y_pred_dt).flatten()


# In[67]:


# ROC graph
fpr_dt, tpr_dt, thresholds = roc_curve(y_test,
                                 y_pred_dt_prob[: , 1],
                                 pos_label= 1
                                 )


# In[68]:


fpr_rf, tpr_rf, thresholds = roc_curve(y_test,
                                 y_pred_rf_prob[: , 1],
                                 pos_label= 1
                                 )


# In[69]:


fpr_et, tpr_et, thresholds = roc_curve(y_test,
                                 y_pred_et_prob[: , 1],
                                 pos_label= 1
                                 )


# In[70]:


fpr_xgb, tpr_xgb, thresholds = roc_curve(y_test,
                                 y_pred_xgb_prob[: , 1],
                                 pos_label= 1
                                 )


# In[71]:


fpr_gbm, tpr_gbm,thresholds = roc_curve(y_test,
                                 y_pred_gbm_prob[: , 1],
                                 pos_label= 1
                                 )


# In[72]:


fpr_kn, tpr_kn, thresholds = roc_curve(y_test,
                                 y_pred_kn_prob[: , 1],
                                 pos_label= 1
                                 )


# In[73]:


fpr_dt


# In[74]:


tpr_dt


# In[75]:


fpr_rf


# In[76]:


tpr_rf


# In[77]:


fpr_et


# In[78]:


tpr_et


# In[79]:


fpr_xgb


# In[80]:


tpr_xgb


# In[81]:


fpr_gbm


# In[82]:


tpr_gbm


# In[83]:


fpr_kn


# In[84]:


tpr_kn


# In[85]:


# Get AUC values

auc(fpr_dt,tpr_dt)


# In[86]:


auc(fpr_rf,tpr_rf)


# In[87]:


auc(fpr_et,tpr_et)


# In[88]:


auc(fpr_gbm,tpr_gbm)


# In[89]:


auc(fpr_xgb,tpr_xgb)


# In[90]:


auc(fpr_kn,tpr_kn)


# In[91]:


# Plot the ROC curve

fig = plt.figure(figsize=(12,10))          # Create window frame
ax = fig.add_subplot(111)   # Create axes

# Also connect diagonals
ax.plot([0, 0], [1, 1], ls="--")   # Dashed diagonal line
# 9.3 Labels etc
ax.set_xlabel('False Positive Rate')  # Final plot decorations
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for models')
# 9.4 Set graph limits
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])

ax.plot(fpr_dt, tpr_dt, label = "Decision Tree")
ax.plot(fpr_rf, tpr_rf, label = "Random Forest")
ax.plot(fpr_et, tpr_et, label = "Extra Trees")
ax.plot(fpr_xgb, tpr_xgb, label = "XGBoost")
ax.plot(fpr_gbm, tpr_gbm, label = "Gradient Boosting")
ax.plot(fpr_kn, tpr_kn, label = "KNeighbors")

# 9.6 Set legend and show plot
ax.legend(loc="lower right")
plt.show()


# In[92]:


# For AUC Graph

models = [(dt, "DecisionTree"), (rf, "RandomForest"), (et, "ExtraTrees"), (gbm, "GradientBoost"),(xgb,"XGBoost"), (kn, "KNeighbors")]
#  Plot the ROC curve
fig = plt.figure(figsize=(12,10))          # Create window frame
ax = fig.add_subplot(111)   # Create axes
# Also connect diagonals
ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line
#  Labels etc
ax.set_xlabel('False Positive Rate')  # Final plot decorations
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for models')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
AUC = []
for clf,name in models:
    clf.fit(X_train,y_train)
    y_pred_prob = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test,
                                     y_pred_prob[: , 1],
                                     pos_label= 1
                                     )
    AUC.append((auc(fpr,tpr)))
    ax.plot(fpr, tpr, label = name)           # Plot on the axes

ax.legend(loc="lower right")
plt.show()


# In[ ]:




