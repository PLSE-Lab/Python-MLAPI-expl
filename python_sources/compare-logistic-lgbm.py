#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder  
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
from scipy import sparse  


# In[ ]:


os.listdir("../input/cat-in-the-dat")


# In[ ]:


train=pd.read_csv("../input/cat-in-the-dat/train.csv")
train.head()


# In[ ]:


test=pd.read_csv("../input/cat-in-the-dat/test.csv")
test.head()


# In[ ]:


train=train.drop(index=train[~train.nom_7.isin(test.nom_7)].index)
train=train.drop(index=train[~train.nom_8.isin(test.nom_8)].index)
train=train.drop(index=train[~train.nom_9.isin(test.nom_9)].index)


# In[ ]:


new_test=test.drop(columns=['id'])
new_test.head()


# In[ ]:


plt.figure(figsize=(8,8))
train.target.value_counts().plot(kind='bar',color=['red','plum'])


# In[ ]:


plt.figure(figsize=(8,8))
corr = train.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,annot=True)
plt.title("correlation plot for train data",size=28)


# In[ ]:


plt.figure(figsize=(8,8))
corr = test.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,annot=True)
plt.title("correlation plot for test data",size=28)


# # Logistic Regression

# In[ ]:


new_train=train.drop(columns=['id','target'])
new_train.head()


# In[ ]:


fig,ax=plt.subplots(5,2,figsize=(15,30))
j=0
for i in new_train.columns[:10]:
    sns.barplot(y=new_train[i].value_counts()[:10],x=new_train[i].value_counts()[:10].index,ax=ax[int(j/2),round(j%2)])
    ax[int(j/2),round(j%2)].set_title("bar chart for "+i)
    ax[int(j/2),round(j%2)].set_ylabel("counts")
    j+=1


# In[ ]:


fig,ax=plt.subplots(5,2,figsize=(30,30))
j=0
for i in new_train.columns[10:20]:
    sns.barplot(y=new_train[i].value_counts()[:10],x=new_train[i].value_counts()[:10].index,ax=ax[int(j/2),round(j%2)])
    ax[int(j/2),round(j%2)].set_title("bar chart for "+i)
    ax[int(j/2),round(j%2)].set_ylabel("counts")
    j+=1


# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(15,15))
j=0
for i in new_train.columns[20:23]:
    sns.barplot(y=new_train[i].value_counts()[:10],x=new_train[i].value_counts()[:10].index,ax=ax[int(j/2),round(j%2)])
    ax[int(j/2),round(j%2)].set_title("bar chart for "+i)
    ax[int(j/2),round(j%2)].set_ylabel("counts")
    j+=1


# In[ ]:


new_train=new_train.drop(columns=['bin_0'])
new_test=new_test.drop(columns=['bin_0'])


# In[ ]:


data = pd.concat([new_train, new_test])

dummies = pd.get_dummies(data, columns=data.columns, drop_first=True,sparse=True)
new_train = dummies.iloc[:new_train.shape[0], :]
new_test = dummies.iloc[new_train.shape[0]:, :]
del data
del dummies
new_train = new_train.sparse.to_coo().tocsr()
new_test = new_test.sparse.to_coo().tocsr()


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(new_train,train['target'],test_size=0.001,random_state=0)

lr = LogisticRegression(C=0.095,solver='lbfgs',class_weight='balanced')  
lr.fit(X_train, y_train)  
proba_test = lr.predict_proba(X_test)[:, 1]
LR_result=pd.DataFrame({'pred':proba_test,'real':y_test})
LR_result['pred_0_1']=LR_result.pred.apply(lambda x:1 if x>=0.5 else 0)


# In[ ]:


print('LR_acc: ',sum(LR_result.real==LR_result.pred_0_1)/len(LR_result))


# In[ ]:


lr.predict_proba(new_test)[:, 1]


# # LightGBM

# In[ ]:


import lightgbm as lgb  
import pickle  

X_train=X_train.astype(float)
X_test=X_test.astype(float)
lgb_train = lgb.Dataset(X_train, y_train)  
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train) 
params = {  
    'boosting_type': 'gbdt',  
    'objective': 'binary',  
    'metric': {'binary_logloss', 'auc'},  
    'num_leaves':450,  
    'max_depth': 25,  
    'min_data_in_leaf': 150,  
    'learning_rate': 0.1,  
    'feature_fraction': 0.95,  
    'bagging_fraction': 0.95,  
    'bagging_freq': 10,  
    'lambda_l1': 0,    
    'lambda_l2': 0, 
    'min_gain_to_split': 0.1,  
    'verbose': 0,  
    'is_unbalance': True  
}  


# In[ ]:


gbm = lgb.train(params,  
                lgb_train,  
                num_boost_round=10000,  
                valid_sets=lgb_eval,  
                early_stopping_rounds=500)  


# # testdata

# In[ ]:


lr.fit(new_train, train['target'])  
LR_TEST=lr.predict_proba(new_test)[:, 1]
new_test=new_test.astype(float)
LGBM_TEST= gbm.predict(new_test, num_iteration=gbm.best_iteration) 


# In[ ]:


prediction=pd.DataFrame({'id':test.id,'LR_TEST':LR_TEST,'LGBM_TEST':LGBM_TEST})
submit=pd.DataFrame({'id':test.id,'target':LR_TEST})
prediction.to_csv('prediction.csv',index=False)
submit.to_csv('submission.csv',index=False)


# # Summary 

# Here I want to sum up my work from version 1 to version 44.In this data,I have tried to remove variable, set parameters for model,balance the number of sample for target.Actually,most of scores are 
# concentrated on 0.805 to 0.80.
# 
# Now I get the highest score is 0.80678,I remove the variable 'bin_0' and use logistic regression.
# 
# Actually,I could not get the high score from LGBM,even I tried to set the parameter.
