#!/usr/bin/env python
# coding: utf-8

# # PIMA DIABETES DATA
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[ ]:


#READING THE DIABETES DATASET
r=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
r.head()


# In[ ]:


r.isnull().values.any()#check whether the null values are present or not


# In[ ]:


#for data correlation
g=r.corr()
g_index=g.index
plt.figure(figsize=(10,10))
#to plot heatmap
sns.heatmap(r[g_index].corr(),annot=True,cmap='viridis')


# In[ ]:


diabetes={True:1,False:0}


# In[ ]:


r['Outcome']=r['Outcome'].map(diabetes)#handling diabetes feature


# In[ ]:


r.head(10)


# In[ ]:


(r['Outcome']==False).value_counts()#counting values of diabetes


# In[ ]:


#splitting the data into train and test splits
from sklearn.model_selection import train_test_split
x=r.iloc[:,:-1]
y=r.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=15)


# In[ ]:


#counting number of zeros in each feature
print(len(r.loc[r['Pregnancies'] == 0]))
print(len(r.loc[r['Glucose'] == 0]))
print(len(r.loc[r['BloodPressure'] == 0]))
print(len(r.loc[r['SkinThickness'] == 0]))
print(len(r.loc[r['Insulin'] == 0]))
print(len(r.loc[r['BMI'] == 0]))
print(len(r.loc[r['DiabetesPedigreeFunction'] == 0]))
print(len(r.loc[r['Age'] == 0]))
print(len(r.loc[r['Outcome'] == 0]))


# In[ ]:


#replacing all 0s with the feature's values mean
x_train=x_train.replace(0,x_train.values.mean())
x_test=x_test.replace(0,x_test.values.mean())


# In[ ]:


#applying algorithm
from sklearn.ensemble import RandomForestClassifier
t=RandomForestClassifier(random_state=10)
t.fit(x_train,y_train.ravel())


# In[ ]:


import sklearn.metrics as m
y_pred=t.predict(x_test)
m.accuracy_score(y_pred,y_test)


# In[ ]:


#hyperparameter optimisation
params={
  'criterion':['gini','entropy'],
   'min_samples_leaf':[1,2,4],
   'min_samples_split': [2,4,6,8,10],
  'n_estimators':[10,20,30,40,50],
  'max_depth':[100,200,300,400,500,1000,2000],
  'random_state':[1,11,15,20,30,43,57,87,97,68,23,24,21,15]  
}


# In[ ]:


#using randomizedcv to increase the accuracy of randomforestclassifier
from sklearn.model_selection import RandomizedSearchCV as rs
t=RandomForestClassifier()
random_search=rs(t,param_distributions=params,n_iter=5,scoring='accuracy',cv=10)
random_search.fit(x_train,y_train.ravel())


# In[ ]:


random_search.best_estimator_#estimating best parameters


# In[ ]:


t=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=100, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=8,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=11, verbose=0, warm_start=False)


# In[ ]:


t.fit(x_train,y_train)


# In[ ]:


y_pred=t.predict(x_test)


# In[ ]:


cm=m.confusion_matrix(y_test,y_pred)
score=m.accuracy_score(y_test,y_pred)
print(cm)
print(score)


# In[ ]:


#using cross validation 
from sklearn.model_selection import cross_val_score
score=cross_val_score(t,x_train,y_train.ravel(),cv=5)


# In[ ]:


score


# In[ ]:


score.mean()


# In[ ]:




