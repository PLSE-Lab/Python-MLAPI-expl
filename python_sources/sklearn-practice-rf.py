#!/usr/bin/env python
# coding: utf-8

# In this practice notebook, the things I did are below:
# 
# 1.Compare the predict performence between decision tree and random forest model
# 
# 2.Using random forest tree to fill missing value in boston data
# 
# 3.Analyze breast cancer data by using RandomForestClassifier and adjust parameters

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine


# In[ ]:


wine=load_wine()


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(wine.data,wine.target,test_size=0.3)


# In[ ]:


clf=DecisionTreeClassifier(random_state=0)
rfc=RandomForestClassifier(random_state=0)

clf=clf.fit(xtrain,ytrain)
rfc=rfc.fit(xtrain,ytrain)

score_c=clf.score(xtest,ytest)
score_r=rfc.score(xtest,ytest)

print ('single tree {}'.format(score_c)
      ,'random tree {}'.format(score_r))


# In[ ]:


# cross validation
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rfc_s=cross_val_score(rfc,wine.data,wine.target,cv=10)
clf_s=cross_val_score(clf,wine.data,wine.target,cv=10)

plt.plot(range(1,11),rfc_s,color='red',label='randomforest')
plt.plot(range(1,11),clf_s,color='blue',label='decisiontree')
plt.legend()
plt.show


# In[ ]:


clf_l=[]
rfc_l=[]
for i in range(10):
    clf=DecisionTreeClassifier()
    rfc=RandomForestClassifier()

    clf=clf.fit(xtrain,ytrain)
    rfc=rfc.fit(xtrain,ytrain)
    rfc_s=cross_val_score(rfc,wine.data,wine.target,cv=10).mean()
    clf_s=cross_val_score(clf,wine.data,wine.target,cv=10).mean()
    clf_l.append(clf_s)
    rfc_l.append(rfc_s)
plt.plot(range(1,11),rfc_l,color='red',label='randomforest')
plt.plot(range(1,11),clf_l,color='blue',label='decisiontree')
plt.legend()
plt.show    


# In[ ]:


superpa=[]
for i in range(100):
    rfc=RandomForestClassifier(n_estimators=i+1,n_jobs=-1,random_state=0)
    rfc_s=cross_val_score(rfc,wine.data,wine.target,cv=10).mean()
    superpa.append(rfc_s)
print(max(superpa),superpa.index(max(superpa)))  
plt.figure(figsize=[18,5])
plt.plot(range(1,101),superpa)
plt.show()


# In[ ]:


# In the randomforest model, 'oob_score=True' means use the data which are not selected to be the test data. 
# From the figure below, when n_estimators > 10, it's ok for us to use oob data to test our model
j=[]
for i in range(1,51):
    dot=1-(1-1/i)**i
    j.append(dot)
plt.plot(range(1,51),j)
plt.show()


# In[ ]:


rfc=RandomForestClassifier(random_state=0,oob_score=True)
rfc=rfc.fit(wine.data,wine.target)
print(rfc.oob_score_)


# In[ ]:


from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
boston=load_boston()


# In[ ]:


regressor=RandomForestRegressor(n_estimators=100,random_state=0)
xtrain,xtest,ytrain,ytest=train_test_split(boston.data,boston.target,test_size=0.3)
regressor=regressor.fit(xtrain,ytrain)
-cross_val_score(regressor,boston.data,boston.target,cv=10,scoring='neg_mean_squared_error').mean()


# In[ ]:


boston.data.shape


# In[ ]:


x=boston.data
y=boston.target
n_samples=x.shape[0]
n_features=x.shape[1]


# In[ ]:


# assume the missing rate is 50%
rng=np.random.RandomState(0) # set random seed
missing_rate=0.5
n_missing_samples=int(np.floor(n_samples*n_features*missing_rate))


# In[ ]:


# random select the number of n_missing_samples between 0-n_features/n_samples
n_missing_features=rng.randint(0,n_features,n_missing_samples)
n_missing_samples=rng.randint(0,n_samples,n_missing_samples)


# In[ ]:


x_missing=x.copy()
y_missing=y.copy()


# In[ ]:


x_missing[n_missing_samples,n_missing_features]=np.nan
x_missing=pd.DataFrame(x_missing)
x_missing.info()


# In[ ]:


# filled the missing values
from sklearn.impute import SimpleImputer
imp_mean=SimpleImputer(missing_values=np.nan,strategy='mean')
x_missing_mean=imp_mean.fit_transform(x_missing)


# In[ ]:


pd.DataFrame(x_missing_mean).info()


# In[ ]:


imp_0=SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0)
x_missing_0=imp_0.fit_transform(x_missing)


# In[ ]:


#using randomforest model to fill missing value
x_missing_reg=x_missing.copy()
sortindex=np.argsort(x_missing_reg.isnull().sum()).values # sort feature by the number of missing value to find the min


# In[ ]:


for i in sortindex:
    # recreate a new dataset without the min missing feature
    df=x_missing_reg
    fillc=df.iloc[:,i]
    df=pd.concat([df.iloc[:,df.columns!=i],pd.DataFrame(y)],axis=1)
    
    df0=SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0).fit_transform(df)
    ytrain=fillc[fillc.notnull()]
    ytest=fillc[fillc.isnull()]
    xtrain=df0[ytrain.index,:]
    xtest=df0[ytest.index,:]
    
    rfc=RandomForestRegressor(n_estimators=50)
    rfc=rfc.fit(xtrain,ytrain)
    ypredict=rfc.predict(xtest)# the value will be use to fill the missing value
    
    x_missing_reg.loc[x_missing_reg.iloc[:,i].isnull(),i]=ypredict
    


# In[ ]:


x_missing_reg.isnull().sum()


# In[ ]:


# to compare three filling methods accuracy
X=[x,x_missing_mean,x_missing_0,x_missing_reg]
mes=[]
for i in X:
    est=RandomForestRegressor(random_state=0,n_estimators=50)
    scores=cross_val_score(est,i,y,scoring='neg_mean_squared_error',cv=10).mean()
    mes.append(scores*-1)
    


# In[ ]:


[*zip(['x_full','x_missing_mean','x_missing_0','x_missing_reg'],mes)]


# In[ ]:


xlabels=['x_full','x_missing_mean','x_missing_0','x_missing_reg']
colors=['r','g','b','orange']
plt.figure(figsize=[12,6])
ax=plt.subplot(111)
for i in range(len(mes)):
    ax.barh(i,mes[i],color=colors[i],alpha=0.6,align='center')
ax.set_title('Imputation Methods with Boston Data')    
ax.set_xlim(left=min(mes)*0.9,right=max(mes)*1.1)
ax.set_yticks(np.arange(len(mes)))
ax.set_xlabel('MSE')
ax.set_yticklabels(xlabels)
plt.show()


# In[ ]:


from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
data=load_breast_cancer()


# In[ ]:


data.data.shape


# In[ ]:


rfc=RandomForestClassifier(random_state=0,n_estimators=50)
cross_val_score(rfc,data.data,data.target,cv=10).mean()


# In[ ]:


scores=[]
for i in range(0,200,10):
    rfc=RandomForestClassifier(n_estimators=i+1
                              ,random_state=0
                                )
    score=cross_val_score(rfc,data.data,data.target,cv=10).mean()
    scores.append(score)
print (max(scores),(scores.index(max(scores))*10)+1)    
plt.figure(figsize=[18,6])
plt.plot(range(1,201,10),scores)
plt.show()


# In[ ]:


scores=[]
for i in range(40,60):
    rfc=RandomForestClassifier(n_estimators=i+1
                              ,random_state=0
                                )
    score=cross_val_score(rfc,data.data,data.target,cv=10).mean()
    scores.append(score)
print (max(scores),([*range(40,60)][scores.index(max(scores))]))    
plt.figure(figsize=[18,6])
plt.plot(range(40,60),scores)
plt.show()


# In[ ]:


parameters={'max_depth':np.arange(1,20,1)}
rfc=RandomForestClassifier(n_estimators=44
                           ,random_state=0
                                )
gs=GridSearchCV(rfc,parameters,cv=10)
gs.fit(data.data,data.target)
print(gs.best_params_)#best combo
print(gs.best_score_)#best accuracy


# In[ ]:


parameters={'max_features':np.arange(5,30,1)}
rfc=RandomForestClassifier(n_estimators=44
                           ,random_state=0
                                )
gs=GridSearchCV(rfc,parameters,cv=10)
gs.fit(data.data,data.target)
print(gs.best_params_)#best combo
print(gs.best_score_)#best accuracy


# In[ ]:


parameters={'criterion':['entropy','gini']}
rfc=RandomForestClassifier(n_estimators=44
                           ,random_state=0
                                )
gs=GridSearchCV(rfc,parameters,cv=10)
gs.fit(data.data,data.target)
print(gs.best_params_)#best combo
print(gs.best_score_)#best accuracy

