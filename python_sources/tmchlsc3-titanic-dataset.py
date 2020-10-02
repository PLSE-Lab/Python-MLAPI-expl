#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train=pd.read_csv('/kaggle/input/titanic/train.csv')
train.head(20)


# In[ ]:


test=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.isnull().any()


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(train.isnull(),xticklabels=True,cmap='viridis')


# In[ ]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train)


# In[ ]:


def ageisclass(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 28
        else:
            return 25
    else:
        return Age
    


# In[ ]:


train['Age']=train[['Age','Pclass']].apply(ageisclass,axis=1)


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(train.isnull(),xticklabels=True,cmap='viridis')


# In[ ]:


train.drop(columns='Cabin',axis=1,inplace=True)


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(train.isnull(),xticklabels=True,cmap='viridis')


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


train.isnull().any()


# In[ ]:


train.head(5)


# In[ ]:


train.info()


# In[ ]:


embarked=pd.get_dummies(train['Embarked'],drop_first=True)
sex=pd.get_dummies(train['Sex'],drop_first=True)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train=pd.concat([train,embarked,sex],axis=1)


# In[ ]:


train.head(10)


# In[ ]:


x=train.drop(columns='Survived',axis=1)
y=train['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split as ts
x_train,x_test,y_train,y_test=ts(x,y,test_size=0.47,random_state=42)
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import cross_val_score as cv
e=rf()
i=cv(e,x_train,y_train,cv=5,scoring='accuracy')
i.mean()


# In[ ]:


params={
    'n_estimators':[100,200,300,400,500,800,1000],
    'min_samples_split':[2,4,6,8],
    'max_features':['auto','sqrt','log2'],
    'n_jobs':[1,-1],
    'random_state':[14,24,21,34,45,65,78,101,222,221]
    
}


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV as rcv
b=rf()
z=rcv(b,param_distributions=params,n_iter=10,scoring='accuracy',cv=10)
z.fit(x_train,y_train)


# In[ ]:


z.best_estimator_


# In[ ]:


t=rf(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='log2',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=6,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=45, verbose=0,
                       warm_start=False)
t.fit(x_train,y_train)
y_pred=t.predict(x_test)
submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = y_pred
submission.to_csv('submission2.csv', index=False)


# In[ ]:






# In[ ]:





# In[ ]:




