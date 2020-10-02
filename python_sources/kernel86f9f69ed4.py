#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cancer_df=pd.read_csv('../input/data.csv')


# In[ ]:





# In[3]:


cancer_df.head()


# In[4]:


cancer_df.columns


# In[5]:


# check column types in the dataframe

cancer_df.dtypes


# In[6]:


#check null values on each columns
cancer_df.isna().sum()


# In[7]:


#drop unwanted columns
df=cancer_df.drop(['id','Unnamed: 32'],axis=1)


# In[8]:


df.head()


# In[9]:


df.dtypes


# In[10]:


df['diagnosis'].unique()


# In[11]:


#deviding the dataset into predictor and target sets
y=df.iloc[:,:1]
X=df.iloc[:,1:]


# In[12]:


X.head()


# In[13]:


y.head()


# In[14]:


y.diagnosis=y.diagnosis.map({'M':1,'B':0})


# In[15]:


y.head()


# In[16]:


#Scale all numerical features in X  using sklearn's StandardScaler class
from sklearn.preprocessing import StandardScaler


# In[17]:


scaler=StandardScaler()
scaled_X_array=scaler.fit_transform(X)


# In[18]:


scaled_X=pd.DataFrame(scaled_X_array)


# In[19]:


scaled_X.head()


# In[20]:


#Perform PCA on numeric features, X. Use sklearn's PCA class. Only retain as many principal components (PCs) as explain 95% variance.
from sklearn.decomposition import PCA


# In[21]:


pca=PCA(.95)


# In[22]:


final_X=pca.fit_transform(scaled_X)


# In[23]:


final_X.shape


# In[24]:


#viii) Split X,y into train and test datasets in the ratio of 80:20 using sklearn's train_test_split function. 
#You get: X_train, X_test, y_train, y_test.
from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test=train_test_split(final_X,y,test_size=.2,shuffle=True)


# In[26]:


# ix) Perform modeling on (X_train,y_train) using above listed algorithms (six).

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[27]:


#Create default classifiers

dt=DecisionTreeClassifier()
kn=KNeighborsClassifier()
xgb=XGBClassifier()
et=ExtraTreesClassifier()
gb=GradientBoostingClassifier()
rf=RandomForestClassifier()


# In[28]:


# Train data

dt_train=dt.fit(X_train,y_train)


# In[29]:


kn_train=kn.fit(X_train,y_train)


# In[30]:


xgb_train=xgb.fit(X_train,y_train)


# In[31]:


et_train=et.fit(X_train,y_train)


# In[32]:


gb_train=gb.fit(X_train,y_train)


# In[33]:


rf_train=rf.fit(X_train,y_train)


# In[34]:


# Make predictions
y_pred_dt=dt_train.predict(X_test)


# In[35]:


y_pred_et=et_train.predict(X_test)


# In[36]:


y_pred_rf=rf_train.predict(X_test)


# In[37]:


y_pred_gb=gb_train.predict(X_test)


# In[38]:


y_pred_xgb=xgb_train.predict(X_test)


# In[39]:


y_pred_kn=kn_train.predict(X_test)


# In[40]:


#xi) Compare the performance of each of these models by calculating metrics as follows:: 
         #a) accuracy,
         #b) Precision & Recall,
         #c) F1 score,
         #d) AUC
        
# For performance measures
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


# In[41]:


# Calculate accuracy


# In[42]:


accuracy_score(y_test,y_pred_dt)


# In[43]:


accuracy_score(y_test,y_pred_et)


# In[44]:


accuracy_score(y_test,y_pred_rf)


# In[45]:


accuracy_score(y_test,y_pred_gb)


# In[46]:


accuracy_score(y_test,y_pred_xgb)


# In[47]:


accuracy_score(y_test,y_pred_kn)


# In[48]:


# best accuracy score is for XGBoost model.


# In[49]:


# calculating Precision,Recall and F1 score for each model. 


# In[50]:


from sklearn.metrics import classification_report


# In[51]:


print(classification_report(y_test,y_pred_dt))


# In[52]:


print(classification_report(y_test,y_pred_et))


# In[53]:


print(classification_report(y_test,y_pred_gb))


# In[54]:


print(classification_report(y_test,y_pred_xgb))


# In[55]:


print(classification_report(y_test,y_pred_rf))


# In[56]:


print(classification_report(y_test,y_pred_kn))


# In[57]:


# creating confusion matrics for each model
confusion_matrix(y_test,y_pred_dt)


# In[58]:


confusion_matrix(y_test,y_pred_et)


# In[59]:


confusion_matrix(y_test,y_pred_gb)


# In[60]:


confusion_matrix(y_test,y_pred_xgb)


# In[61]:


confusion_matrix(y_test,y_pred_rf)


# In[62]:


confusion_matrix(y_test,y_pred_kn)


# In[63]:


# Again XGB model has the best results as per the F1 score as well and this has the best confusion matrix too.


# In[64]:


# calculating the AUC values for each model


# In[65]:


# decission tree model


# In[66]:


fpr_dt, tpr_dt, thresholds = roc_curve(y_test,
                                 dt_train.predict_proba(X_test)[: , 1],
                                 pos_label= 1
                                 )


# In[67]:


dt_auc=auc(fpr_dt,tpr_dt)


# In[68]:


# ExtraTreesClassifier model


# In[69]:


fpr_et, tpr_et, thresholds = roc_curve(y_test,
                                 et_train.predict_proba(X_test)[: , 1],
                                 pos_label= 1
                                 )


# In[70]:


et_auc=auc(fpr_et,tpr_et)


# In[71]:


# random forest classifier model


# In[72]:


fpr_rf, tpr_rf, thresholds = roc_curve(y_test,
                                 rf_train.predict_proba(X_test)[: , 1],
                                 pos_label= 1
                                 )
rf_auc=auc(fpr_rf,tpr_rf)


# In[73]:


fpr_gb, tpr_gb, thresholds = roc_curve(y_test,
                                 gb_train.predict_proba(X_test)[: , 1],
                                 pos_label= 1
                                 )
gb_auc=auc(fpr_gb,tpr_gb)


# In[74]:


fpr_xgb, tpr_xgb, thresholds = roc_curve(y_test,
                                 xgb_train.predict_proba(X_test)[: , 1],
                                 pos_label= 1
                                 )
xgb_auc=auc(fpr_xgb,tpr_xgb)


# In[75]:


fpr_kn, tpr_kn, thresholds = roc_curve(y_test,
                                 kn_train.predict_proba(X_test)[: , 1],
                                 pos_label= 1
                                 )
kn_auc=auc(fpr_kn,tpr_kn)


# In[76]:


auc_dict={'dt_auc':dt_auc,'et_auc':et_auc,'rf_auc':rf_auc,'gb_auc':gb_auc,'xgb_auc':xgb_auc,'kn_auc':kn_auc}


# In[77]:


max(auc_dict)


# In[78]:


#again the AUC specifies the XGB is the best model for this prediction.


# In[79]:


#xii) Also draw ROC curve for each


# In[80]:


fig = plt.figure(figsize=(20,10))          
ax = fig.add_subplot(111)   


ax.plot([0, 1], [0, 1], ls="--")  

ax.set_xlabel('False Positive Rate')  
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for models')


ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])


ax.plot(fpr_dt, tpr_dt, label = "dt")
ax.plot(fpr_rf, tpr_rf, label = "rf")
ax.plot(fpr_gb, tpr_gb, label = "gbm")
ax.plot(fpr_xgb, tpr_xgb, label = "xgb")
ax.plot(fpr_et, tpr_et, label = "et")
ax.plot(fpr_kn, tpr_kn, label = "kn")

ax.legend()
plt.show()


# In[81]:


## From all metrices the XGB model gives the best results.


# In[ ]:





# In[ ]:





# In[82]:


# Dtata Explorationa and Visualization:


# In[83]:


cancer_df.head()


# In[84]:


cancer_df.diagnosis.value_counts()


# In[85]:


# Here we will use Seaborn to create a heat map of the correlations between the features.
features_mean= list(cancer_df.columns[1:11])
plt.figure(figsize=(25,15))
sns.heatmap(cancer_df[features_mean].corr(), annot=True, square=True, cmap='coolwarm')
plt.show()


# In[86]:


color_dic = {'M':'red', 'B':'blue'}
colors = cancer_df['diagnosis'].map(lambda x: color_dic.get(x))

sm = pd.scatter_matrix(cancer_df[features_mean], c=colors, alpha=0.4, figsize=((15,15)));

plt.show()


# In[92]:


# plotting the distribution of each type of diagnosis for each of the mean features.

bins = 12
plt.figure(figsize=(15,15))
rows = int(len(features_mean)/2)
features_mean = features_mean[1:]
for i, feature in enumerate(features_mean):
    
    plt.subplot(rows, 2, i+1)
    
    sns.distplot(cancer_df[cancer_df['diagnosis']=='M'][feature], bins=bins, color='r', label='M');
    sns.distplot(cancer_df[cancer_df['diagnosis']=='B'][feature], bins=bins, color='b', label='B');
    

    plt.legend(loc='upper right')
    
plt.tight_layout()
plt.show()


# In[ ]:


bins = 12
plt.figure(figsize=(10,8))


sns.distplot(cancer_df[cancer_df['diagnosis']=='M']['radius_mean'], bins=bins, color='r', label='M');
sns.distplot(cancer_df[cancer_df['diagnosis']=='B']['radius_mean'], bins=bins, color='b', label='B');


# In[ ]:


rows = int(len(features_mean)/2)
rows


# In[94]:


plt.figure(figsize=(15,15))
features_mean = features_mean[1:]
rows = int(len(features_mean)/2)
for i, feature in enumerate(features_mean):
    
    plt.subplot(rows, 2, i+1)
    
    sns.boxplot(x='diagnosis', y=feature, data=cancer_df, palette="Set1")

plt.tight_layout()
plt.show()


# In[89]:


plt.figure(figsize=(10,8))

sns.boxplot(x='diagnosis',y='texture_mean',data=cancer_df,palette="Set1")


# In[88]:


for i, feature in enumerate(features_mean):
    rows = int(len(features_mean)/2)
    print(i,rows,feature)


# In[ ]:




