#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# In[5]:


import os


# In[6]:


#libraries for data modelling
from sklearn.ensemble import RandomForestClassifier


# In[7]:


from sklearn.ensemble import GradientBoostingClassifier


# In[8]:


from xgboost.sklearn import XGBClassifier


# In[9]:


from sklearn.ensemble import ExtraTreesClassifier


# In[10]:


from sklearn.neighbors import KNeighborsClassifier


# In[11]:


from sklearn.tree import DecisionTreeClassifier


# In[12]:


#libraries for data preprocessing
from sklearn.preprocessing import StandardScaler


# In[13]:


#model for Dimentionality Reduction
from sklearn.decomposition import PCA


# In[14]:


#libraries for performance measures
from sklearn.metrics import confusion_matrix


# In[15]:


from sklearn.metrics import precision_recall_fscore_support


# In[16]:


from sklearn.metrics import accuracy_score


# In[17]:


from sklearn.metrics import auc, roc_curve


# In[18]:


#libraries For data splitting
from sklearn.model_selection import train_test_split


# In[19]:


os.chdir("../input")


# In[20]:


data_file = pd.read_csv("data.csv")


# In[21]:


pd.options.display.max_columns = 200


# In[22]:


data_file.head()


# In[23]:


data_file.tail()


# In[24]:


data_file.shape


# In[25]:


data_file.info()


# In[26]:


data_file.describe()


# In[27]:


data_file.dtypes


# In[28]:


data_file.isna().sum()


# In[29]:


#dropping the unwanted coloumns
df=data_file.drop(['id','Unnamed: 32'],axis=1)


# In[30]:


df.shape


# In[31]:


df['diagnosis'].unique()


# In[32]:


#Splitting the Features and Target Data into X and y
X =df.iloc[:,1:]
y=df.iloc[:,:1]


# In[33]:


X.shape


# In[34]:


y.shape


# In[35]:


X.head()


# In[36]:


y.head()


# In[37]:


y=y.diagnosis.map({'M':1,'B':0})


# In[38]:


y.head()


# In[39]:


#Scale all numerical features in X  using sklearn's StandardScaler class
from sklearn.preprocessing import StandardScaler


# In[40]:


scaler=StandardScaler()


# In[41]:


X_sc = scaler.fit_transform(X)


# In[42]:


sca_x=pd.DataFrame(X_sc)


# In[43]:


sca_x.head()


# In[44]:


#Perform PCA on numeric features
from sklearn.decomposition import PCA


# In[45]:


pca = PCA(n_components = 2)


# In[46]:


pca=PCA(.95)


# In[47]:


X=pca.fit_transform(sca_x)


# In[48]:


X.shape


# In[49]:


#Split and shuffle data
from sklearn.model_selection import train_test_split


# In[50]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.2,shuffle=True)


# In[51]:


#Create default classifiers
dtc=DecisionTreeClassifier()
knc=KNeighborsClassifier()
xgbc=XGBClassifier()
etc=ExtraTreesClassifier()
gbc=GradientBoostingClassifier()
rfc=RandomForestClassifier()


# In[52]:


# Train data
dtc_train=dtc.fit(X_train,y_train)


# In[53]:


knc_train=knc.fit(X_train,y_train)


# In[54]:


xgbc_train=xgbc.fit(X_train,y_train)


# In[55]:


etc_train=etc.fit(X_train,y_train)


# In[56]:


gbc_train=gbc.fit(X_train,y_train)


# In[57]:


rfc_train=rfc.fit(X_train,y_train)


# In[58]:


# Make predictions
y_pred_dtc=dtc_train.predict(X_test)


# In[59]:


y_pred_etc=etc_train.predict(X_test)


# In[60]:


y_pred_rfc=rfc_train.predict(X_test)


# In[61]:


y_pred_gbc=gbc_train.predict(X_test)


# In[62]:


y_pred_xgbc=xgbc_train.predict(X_test)


# In[63]:


y_pred_knc=knc_train.predict(X_test)


# In[64]:


#Get probability values
y_pred_dtc_prob = dtc_train.predict_proba(X_test)
y_pred_rfc_prob = rfc_train.predict_proba(X_test)
y_pred_etc_prob = etc_train.predict_proba(X_test)
y_pred_knc_prob = knc_train.predict_proba(X_test)
y_pred_xgbc_prob = xgbc_train.predict_proba(X_test)
y_pred_gbc_prob= gbc_train.predict_proba(X_test)


# In[65]:


#xi) Compare the performance of each of these models by calculating metrics as follows:: 
         #a) accuracy,
         #b) Precision & Recall,
         #c) F1 score,
         #d) AUC
        
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


# In[66]:


# Calculate accuracy
accuracy_score(y_test,y_pred_dtc)


# In[67]:


accuracy_score(y_test,y_pred_etc)


# In[68]:


accuracy_score(y_test,y_pred_rfc)


# In[69]:


accuracy_score(y_test,y_pred_gbc)


# In[70]:


accuracy_score(y_test,y_pred_xgbc)


# In[71]:


accuracy_score(y_test,y_pred_knc)


# In[72]:


#Calculate Confusion Matrix
print("DecisionTreeClassifier: ")
confusion_matrix(y_test,y_pred_dtc)


# In[73]:


print("RandomForestClassifier: ")
confusion_matrix(y_test,y_pred_rfc)


# In[74]:


print("ExtraTreesClassifier: ")
confusion_matrix(y_test,y_pred_etc)


# In[75]:


print("GradientBoostingClassifier: ")
confusion_matrix(y_test,y_pred_gbc)


# In[76]:


print("KNeighborsClassifier: ")
confusion_matrix(y_test,y_pred_knc)


# In[77]:


print("XGBClassifier: ")
confusion_matrix(y_test,y_pred_xgbc)


# In[78]:


#Get probability values
y_pred_dtc_prob = dtc_train.predict_proba(X_test)
y_pred_rfc_prob = rfc_train.predict_proba(X_test)
y_pred_etc_prob = etc_train.predict_proba(X_test)
y_pred_knc_prob = knc_train.predict_proba(X_test)
y_pred_xgbc_prob = xgbc_train.predict_proba(X_test)
y_pred_gbc_prob= gbc_train.predict_proba(X_test)


# In[79]:


fpr_dt, tpr_dt, thresholds = roc_curve(y_test, y_pred_dtc_prob[: , 1], pos_label= 1)
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_rfc_prob[: , 1], pos_label= 1)
fpr_etc, tpr_etc, thresholds = roc_curve(y_test, y_pred_rfc_prob[: , 1], pos_label= 1)
fpr_knc, tpr_knc, thresholds = roc_curve(y_test, y_pred_rfc_prob[: , 1], pos_label= 1)
fpr_xg, tpr_xg, thresholds = roc_curve(y_test, y_pred_xgbc_prob[: , 1], pos_label= 1)
fpr_gbm, tpr_gbm,thresholds = roc_curve(y_test, y_pred_gbc_prob[: , 1], pos_label= 1)


# In[80]:


print("DecisionTreeClassifier")
auc(fpr_dt,tpr_dt)


# In[81]:


print("RandomForestClassifier")
auc(fpr_rf,tpr_rf)


# In[82]:


print("ExtraTreesClassifier")
auc(fpr_etc,tpr_etc)


# In[83]:


print("GradientBoostingClassifier")
auc(fpr_gbm,tpr_gbm)


# In[84]:


print("KNeighborsClassifier")
auc(fpr_knc,tpr_knc)


# In[85]:


print("XGBClassifier")
auc(fpr_xg,tpr_xg)


# In[86]:


print("DecisionTreeClassifier: ")
precision_recall_fscore_support(y_test,y_pred_dtc)


# In[87]:


print("RandomForestClassifier: ")
precision_recall_fscore_support(y_test,y_pred_rfc)


# In[88]:


print("ExtraTreesClassifier: ")
precision_recall_fscore_support(y_test,y_pred_etc)


# In[89]:


print("GradientBoostingClassifier: ")
precision_recall_fscore_support(y_test,y_pred_gbc)


# In[90]:


print("KNeighborsClassifier: ")
precision_recall_fscore_support(y_test,y_pred_knc)


# In[91]:


print("XGBClassifier: ")
precision_recall_fscore_support(y_test,y_pred_xgbc)


# In[92]:


#Plot ROC curve now
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

