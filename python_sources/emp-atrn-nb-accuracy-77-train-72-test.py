#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


emp_atr=pd.read_csv("../input/HR-Employee-Attrition.csv")
emp_atr.shape


# In[ ]:


emp_atr.info()


# In[ ]:


emp_atr.describe().transpose()


# In[ ]:


emp_atr.Attrition.value_counts()


# In[ ]:


emp_atr.head(10)


# In[ ]:


emp_atr.tail(10)


# In[ ]:


emp_atr.isna().sum()


# In[ ]:


emp_atr[emp_atr.duplicated()]


# In[ ]:


emp_atr.columns


# In[ ]:


cat_col = emp_atr.select_dtypes(exclude=np.number).columns
num_col = emp_atr.select_dtypes(include=np.number).columns
print(cat_col)
print(num_col)


# In[ ]:


for i in cat_col:
    print(emp_atr[i].value_counts())


# In[ ]:


# Get discrete numerical value
num_col_disc=[]
num_col_medium=[]
num_col_cont=[]
print("Attributes with their distinct count")
for i in num_col:
    if emp_atr[i].nunique() <=10:
        print(i,"==",emp_atr[i].nunique(),"== disc")
        num_col_disc.append(i)
    elif (emp_atr[i].nunique() >10 and emp_atr[i].nunique() <100):
        num_col_medium.append(i)    
        print(i,"==",emp_atr[i].nunique(),"== medium")
    else:
        num_col_cont.append(i)
        print(i,"==",emp_atr[i].nunique(),"== cont")
#print(num_col_disc)
#print(num_col_medium)
#print(num_col_cont)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


fig, ax = plt.subplots(3, 3, figsize=(30, 40))
for variable, subplot in zip(cat_col, ax.flatten()):
    cp=sns.countplot(emp_atr[variable], ax=subplot,order = emp_atr[variable].value_counts().index,hue=emp_atr['Attrition'])
    cp.set_title(variable,fontsize=40)
    cp.legend(fontsize=30)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
        label.set_fontsize(36)                
    for label in subplot.get_yticklabels():
        label.set_fontsize(36)        
        cp.set_ylabel('Count',fontsize=40)    
plt.tight_layout()


# In[ ]:


fig, ax = plt.subplots(3, 3, figsize=(30, 40))
for variable, subplot in zip(num_col_disc, ax.flatten()):
    cp=sns.countplot(emp_atr[variable], ax=subplot,order = emp_atr[variable].value_counts().index,hue=emp_atr['Attrition'])
    cp.set_title(variable,fontsize=40)
    cp.legend(fontsize=30)
    for label in subplot.get_xticklabels():
        #label.set_rotation(90)
        label.set_fontsize(36)                
    for label in subplot.get_yticklabels():
        label.set_fontsize(36)        
        cp.set_ylabel('Count',fontsize=40)
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(emp_atr["WorkLifeBalance"],hue=emp_atr["Attrition"])
#emp_atr["WorkLifeBalance"]


# **You can clearly see that attrition value for No is quite high than Yes value. We get this biased data having more data for attrition="No" . We try to see that we can get try to get something out of model **
# 
# **Also with above plot we see that data is not evenly distributed for discrete values**

# In[ ]:


plt.figure(figsize=(20, 10))
sns.boxplot(data=emp_atr[num_col_cont],orient="h")


# In[ ]:


plt.figure(figsize=(20, 10))
sns.barplot(data=emp_atr[num_col_cont],orient="h")


# **From the above boxplot and barplot, we see that outlier exist for monthly income ,Daily rate and Employee number. Also data is distributed wider for monthly income and monthly rate variable **

# In[ ]:


#fill_num_attrition=lambda x: 1 if x=="Yes" else 0
#type(fill_num_attrition)
emp_atr["num_attrition"]=emp_atr["Attrition"].apply(lambda x: 1 if x=="Yes" else 0)
emp_atr["num_attrition"].value_counts()


# In[ ]:


emp_atr_cov=emp_atr.cov()
emp_atr_cov


# In[ ]:


plt.figure(figsize=(40,20))
sns.heatmap(emp_atr_cov,vmin=-1,vmax=1,center=0,annot=True)


# In[ ]:


# Importing necessary package for creating model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[ ]:


# one hot encoding num_attrition
cat_col_rm_tgt=cat_col[1:]
num_col=emp_atr.select_dtypes(include=np.number).columns
one_hot=pd.get_dummies(emp_atr[cat_col_rm_tgt])
emp_atr_df=pd.concat([emp_atr[num_col],one_hot],axis=1)
emp_atr_df.head(10)


# In[ ]:


X=emp_atr_df.drop(columns=['num_attrition'])
y=emp_atr_df[['num_attrition']]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


train_Pred = logreg.predict(X_train)


# In[ ]:


metrics.confusion_matrix(y_train,train_Pred)


# In[ ]:


metrics.accuracy_score(y_train,train_Pred)


# In[ ]:


test_Pred = logreg.predict(X_test)


# In[ ]:


metrics.confusion_matrix(y_test,test_Pred)


# In[ ]:


metrics.accuracy_score(y_test,test_Pred)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, test_Pred))


# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# **We will try to implement KNN for this problem to see accuracy is better than logistic regression**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from math import sqrt


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size = 0.3, random_state = 100)
y_train=np.ravel(y_train)
y_test=np.ravel(y_test)
#y_train = y_train.ravel()
#y_test = y_test.ravel()


# In[ ]:


accuracy_train_dict={}
accuracy_test_dict={}
df_len=round(sqrt(len(emp_atr_df)))
for k in range(3,df_len):
    K_value = k+1
    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
    neigh.fit(X_train, y_train) 
    y_pred_train = neigh.predict(X_train)
    y_pred_test = neigh.predict(X_test)    
    train_accuracy=accuracy_score(y_train,y_pred_train)*100
    test_accuracy=accuracy_score(y_test,y_pred_test)*100
    accuracy_train_dict.update(({k:train_accuracy}))
    accuracy_test_dict.update(({k:test_accuracy}))
    print ("Accuracy for train :",train_accuracy ," and test :",test_accuracy,"% for K-Value:",K_value)


# **From the above iteration we see K=12 had better accuracy**

# In[ ]:


elbow_curve_train = pd.Series(accuracy_train_dict,index=accuracy_train_dict.keys())
elbow_curve_test = pd.Series(accuracy_test_dict,index=accuracy_test_dict.keys())
elbow_curve_train.head(10)


# In[ ]:


ax=elbow_curve_train.plot(title="Accuracy of train VS Value of K ")
ax.set_xlabel("K")
ax.set_ylabel("Accuracy of train")


# In[ ]:


ax=elbow_curve_test.plot(title="Accuracy of test VS Value of K ")
ax.set_xlabel("K")
ax.set_ylabel("Accuracy of test")


# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


NB=GaussianNB()
NB.fit(X_train, y_train)


# In[ ]:


GaussianNB(priors=None,var_smoothing=1e-09)


# In[ ]:


train_pred=NB.predict(X_train)
accuracy_score(train_pred,y_train)


# In[ ]:


test_pred=NB.predict(X_test)
accuracy_score(test_pred,y_test)


# In[ ]:




