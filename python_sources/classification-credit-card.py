#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


df=pd.read_csv('../input/Test 1.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# * There are no null values in the data set. It is a clean data set
# * The dataset consists of 10000 observations and 12 variables

# In[ ]:


df.info()


# There are :
# * 3 categorical variables
# * 8 continuous variables
# * 1 boolean output variable

# In[ ]:


cust_id=df.customer_id
type(cust_id)


# * Taking customer id in a series so that we can refer it later
# * cutomer_id is of no use to us in model building. So it can be safely removed

# In[ ]:


df.drop('customer_id',axis=1,inplace=True)
df.head()


# In[ ]:


df['country_reg'].value_counts()


# In[ ]:


df2=pd.get_dummies(df,columns=['demographic_slice','ad_exp','country_reg'])
df2.head()


# Converting all categorical features into encoded features using one hot encoding

# In[ ]:


outcome=np.where(df['card_offer']==True,1,0)
df2['card_offer']=outcome
df2.head()


# #### Splitting data into train and test

# In[ ]:


X=df2.drop('card_offer',axis=1)
Y=df2['card_offer']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)


# ### Logistic Regression

# In[ ]:


log=LogisticRegression(C=0.0007)
log.fit(x_train,y_train)
print('Train score:',log.score(x_train,y_train))
print('Test score:',log.score(x_test,y_test))
#print(log.C_)


# In[ ]:


from sklearn.metrics import confusion_matrix,f1_score,roc_curve,roc_auc_score


# In[ ]:


fpr,tpr,threholds=roc_curve(y_test,log.predict_proba(x_test)[:,1])
optF1=0
optTh=0
for th in threholds:
    preds=np.where(log.predict_proba(x_test)[:,1]>th,1,0)
    f1=f1_score(y_test,preds)
    if(optF1<f1):
        optF1=f1
        optTh=th
print('Optimum F1:',optF1)
print('Optimum Threshold:',optTh)


# In[ ]:


y_pred=np.where(log.predict_proba(x_test)[:,1]>0.15,1,0)
cn=confusion_matrix(y_test,y_pred)
cn=pd.DataFrame(cn,columns=['Predicted 0','Predicted 1'],index=['Actual 0','Actual 1'])
cn


# In[ ]:


f1_score(y_test,y_pred)


# In[ ]:


roc_auc=roc_auc_score(y_test,y_pred)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# * Training accuracy is 84%
# * Testing accuracy is 84%
# * F1 score with optimum threshold is 37%

# ### Decision Tree

# In[ ]:


dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
print('Train score:',dt.score(x_train,y_train))
print('Test score:',dt.score(x_test,y_test))


# In[ ]:


fpr,tpr,threholds=roc_curve(y_test,dt.predict_proba(x_test)[:,1])
optF1=0
optTh=0
for th in threholds:
    preds=np.where(dt.predict_proba(x_test)[:,1]>th,1,0)
    f1=f1_score(y_test,preds)
    if(optF1<f1):
        optF1=f1
        optTh=th
print('Optimum F1:',optF1)
print('Optimum Threshold:',optTh)


# In[ ]:


y_pred=np.where(dt.predict_proba(x_test)[:,1]>0,1,0)
cn=confusion_matrix(y_test,y_pred)
cn=pd.DataFrame(cn,columns=['Predicted 0','Predicted 1'],index=['Actual 0','Actual 1'])
cn


# In[ ]:


f1_score(y_test,y_pred)


# In[ ]:


roc_auc=roc_auc_score(y_test,y_pred)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# * Training accuracy is 100%
# * Testing accuracy is 96%
# * F1 score with optimum threshold is 87%

# ### Random Forest

# In[ ]:


rf=RandomForestClassifier()
rf.fit(x_train,y_train)
print('Train score:',rf.score(x_train,y_train))
print('Test score:',rf.score(x_test,y_test))


# In[ ]:


fpr,tpr,threholds=roc_curve(y_test,rf.predict_proba(x_test)[:,1])
optF1=0
optTh=0
for th in threholds:
    preds=np.where(rf.predict_proba(x_test)[:,1]>th,1,0)
    f1=f1_score(y_test,preds)
    if(optF1<f1):
        optF1=f1
        optTh=th
print('Optimum F1:',optF1)
print('Optimum Threshold:',optTh)


# In[ ]:


y_pred=np.where(rf.predict_proba(x_test)[:,1]>0.3,1,0)
cn=confusion_matrix(y_test,y_pred)
cn=pd.DataFrame(cn,columns=['Predicted 0','Predicted 1'],index=['Actual 0','Actual 1'])
cn


# In[ ]:


f1_score(y_test,y_pred)


# In[ ]:


roc_auc=roc_auc_score(y_test,y_pred)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# * Training accuracy is 99%
# * Testing accuracy is 96%
# * F1 score with optimum threshold is 89%

# ### Boosting

# In[ ]:


gbr=GradientBoostingClassifier()
gbr.fit(x_train,y_train)
print('Train score:',gbr.score(x_train,y_train))
print('Test score:',gbr.score(x_test,y_test))


# In[ ]:


fpr,tpr,threholds=roc_curve(y_test,gbr.predict_proba(x_test)[:,1])
optF1=0
optTh=0
for th in threholds:
    preds=np.where(gbr.predict_proba(x_test)[:,1]>th,1,0)
    f1=f1_score(y_test,preds)
    if(optF1<f1):
        optF1=f1
        optTh=th
print('Optimum F1:',optF1)
print('Optimum Threshold:',optTh)


# In[ ]:


y_pred=np.where(gbr.predict_proba(x_test)[:,1]>optTh,1,0)
cn=confusion_matrix(y_test,y_pred)
cn=pd.DataFrame(cn,columns=['Predicted 0','Predicted 1'],index=['Actual 0','Actual 1'])
cn


# In[ ]:


f1_score(y_test,y_pred)


# In[ ]:


roc_auc=roc_auc_score(y_test,y_pred)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# * Training accuracy is 99%
# * Testing accuracy is 97%
# * F1 score with optimum threshold is 92%

# ### Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


nb=GaussianNB()
nb.fit(x_train,y_train)
print('Train score:',nb.score(x_train,y_train))
print('Test score:',nb.score(x_test,y_test))


# In[ ]:


fpr,tpr,threholds=roc_curve(y_test,nb.predict_proba(x_test)[:,1])
optF1=0
optTh=0
for th in threholds:
    preds=np.where(nb.predict_proba(x_test)[:,1]>th,1,0)
    f1=f1_score(y_test,preds)
    if(optF1<f1):
        optF1=f1
        optTh=th
print('Optimum F1:',optF1)
print('Optimum Threshold:',optTh)


# In[ ]:


y_pred=np.where(nb.predict_proba(x_test)[:,1]>optTh,1,0)
cn=confusion_matrix(y_test,y_pred)
cn=pd.DataFrame(cn,columns=['Predicted 0','Predicted 1'],index=['Actual 0','Actual 1'])
cn


# In[ ]:


f1_score(y_test,y_pred)


# In[ ]:


roc_auc=roc_auc_score(y_test,y_pred)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# * Training accuracy is 84%
# * Testing accuracy is 84%
# * F1 score with optimum threshold is 50%

# ### KNN

# In[ ]:


neighbors = np.arange(1, 20)
train_accuracy_plot = np.empty(len(neighbors))
test_accuracy_plot = np.empty(len(neighbors))
# Loop over different values of k
sc=StandardScaler()
scaledX_train = sc.fit_transform(x_train)
scaledX_test = sc.transform(x_test)
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(scaledX_train,y_train)
    train_accuracy_plot[i] = knn.score(scaledX_train,y_train)
    test_accuracy_plot[i] = knn.score(scaledX_test,y_test)
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy_plot, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy_plot, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train,y_train)
print('Train score:',knn.score(x_train,y_train))
print('Test score:',knn.score(x_test,y_test))


# In[ ]:


fpr,tpr,threholds=roc_curve(y_test,knn.predict_proba(x_test)[:,1])
optF1=0
optTh=0
for th in threholds:
    preds=np.where(knn.predict_proba(x_test)[:,1]>th,1,0)
    f1=f1_score(y_test,preds)
    if(optF1<f1):
        optF1=f1
        optTh=th
print('Optimum F1:',optF1)
print('Optimum Threshold:',optTh)


# In[ ]:


y_pred=np.where(knn.predict_proba(x_test)[:,1]>optTh,1,0)
cn=confusion_matrix(y_test,y_pred)
cn=pd.DataFrame(cn,columns=['Predicted 0','Predicted 1'],index=['Actual 0','Actual 1'])
cn


# In[ ]:


f1_score(y_test,y_pred)


# In[ ]:


roc_auc=roc_auc_score(y_test,y_pred)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# * Training accuracy is 85%
# * Testing accuracy is 83%
# * F1 score with optimum threshold is 33%

# ### Conclusion

# * KNN gives Training accuracy as 85%, Testing accuracy as 83% and F1 score with optimum threshold is 33%
# * Naive Bayes gives Training accuracy as 84%, Testing accuracy as 84% and F1 score with optimum threshold is 50%
# * Logistic gives Training accuracy as 84%, Testing accuracy as 84% and F1 score with optimum threshold is 37%
# * Decision Tree gives Training accuracy as 100%, Testing accuracy as 96% and F1 score with optimum threshold is 87%
# * Random Forest gives Training accuracy as 99%, Testing accuracy as 96% and F1 score with optimum threshold is 89%
# * Gradient Boosting gives Training accuracy as 99%, Testing accuracy as 97% and F1 score with optimum threshold is 92%
# * By comparing all models we found that GradientBoosting gives best accuracy and F1 score for our data set. Thus GradientBoosting will be our final model

# ### Predicting using Gradient Boosting

# In[ ]:


test=pd.read_csv('../input/Test 2.csv')
test.head()


# In[ ]:


test.shape


# In[ ]:


test.isnull().sum()


# In[ ]:


test=pd.get_dummies(test,columns=['demographic_slice','ad_exp','country_reg'])
test.head()


# In[ ]:


X=test.drop(['customer_id','card_offer'],axis=1)
optTh=0.3618855170800892


# In[ ]:


Y_pred=np.where(gbr.predict_proba(X)[:,1]>optTh,1,0)


# In[ ]:


Y_pred.shape


# In[ ]:


test['card_offer']=Y_pred
test['card_offer']=np.where(test['card_offer']==1,True,False)
test.head()


# In[ ]:


test.to_csv('Prediction.csv')


# ### -------------------------------------------------------------------END-------------------------------------------------------------------
