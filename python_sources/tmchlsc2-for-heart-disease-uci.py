#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing the libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#using three classifiers
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.neighbors import KNeighborsClassifier as kn
from sklearn.ensemble import RandomForestClassifier as rf


# In[ ]:


#reading the dataset
r=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
r.head(10)


# In[ ]:


r.info()


# In[ ]:


r.describe()


# In[ ]:


#feature selection 
e=r.corr()
e_features=e.index
plt.figure(figsize=(20,20))
#plotting the heatmap
sns.heatmap(r[e_features].corr(),annot=True,cmap='viridis')


# In[ ]:


#plotting histogram
r.hist(figsize=(20,20))


# In[ ]:


#checking whether a target dataset is balanced or not
sns.set_style('darkgrid')
sns.countplot(x='target',data=r)


# In[ ]:


r['target'].value_counts()


# In[ ]:


#handling categorical variables
p=pd.get_dummies(r,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])


# In[ ]:


from sklearn.preprocessing import StandardScaler
w=StandardScaler()
sv=['age','trestbps','chol','thalach','oldpeak']
p[sv]=w.fit_transform(p[sv])


# In[ ]:


y=p['target']
x=p.drop(['target'],axis=1)
from sklearn.model_selection import train_test_split
import sklearn.metrics as m
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=34)
b=kn(n_neighbors=5)
b.fit(x_train,y_train)
y_pred=b.predict(x_test)
m.accuracy_score(y_test,y_pred)


# In[ ]:


#using cross validation for finding accuracy corresponding to each neighbours
#using kneighbors
from sklearn.model_selection import cross_val_score as cv
knn_scores=[]
for k in range(1,21):
    knn=kn(n_neighbors=k)
    score=cv(knn,x_train,y_train,cv=10)
    knn_scores.append(score.mean())


# In[ ]:


knn_scores


# In[ ]:


#since 12th neighbor has highest accuracy so therefore re-applying the model 
knn=kn(n_neighbors=12)
knn.fit(x_train,y_train)
score=cv(knn,x_train,y_train,cv=10)


# In[ ]:


score.mean()


# In[ ]:


y_pred=knn.predict(x_test)
print('Accuracy of Knn is =',m.accuracy_score(y_test,y_pred))


# In[ ]:


#hyperparameter tuning for random forest
params={
    'n_estimators':[100,150,200,250,300,400,500,1000,2000],
    'max_depth':[2,4,7,9,12],
    'min_samples_split':[1,2,4,8],
    'min_samples_leaf':[1,3,5,7,9,12],
    'max_features':['sqrt','log2'],
    'n_jobs':[1,-1],
    'random_state':[11,21,32,44,55,67]
    
}


# In[ ]:


#using RandomizedCV to predict best parameters for Random Forest Classifier
from sklearn.model_selection import RandomizedSearchCV as rcv
t=rf()
f=rcv(t,param_distributions=params,n_iter=10,cv=10,n_jobs=-1,scoring='accuracy')
f.fit(x_train,y_train)


# In[ ]:


f.best_estimator_


# In[ ]:


u=rf(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=12, max_features='log2',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=12, min_samples_split=8,
                       min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=1,
                       oob_score=False, random_state=44, verbose=0,
                       warm_start=False)

u.fit(x_train,y_train)
y_pred=u.predict(x_test)
print('Accuracy of Random Forest is=',m.accuracy_score(y_test,y_pred))


# In[ ]:


#Hyperparamter tuning using Decision Tree
param={
    'criterion':['gini','entropy'],
    'max_depth':[2,4,7,9,12],
    'min_samples_split':[1,2,4,8],
    'min_samples_leaf':[1,3,5,7,9,12],
    'max_features':['sqrt','log2'],
    'random_state':[11,21,32,44,55,67]
    
    
}


# In[ ]:


#using RandomizedCV to predict best parameters for Desicion Tree Classifier
t=dt()
f=rcv(t,param_distributions=param,n_iter=10,cv=10,n_jobs=-1,scoring='accuracy')
f.fit(x_train,y_train)


# In[ ]:


f.best_estimator_


# In[ ]:


o=dt(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=7, max_features='log2', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=8,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                     random_state=32, splitter='best')
o.fit(x_train,y_train)
y_pred=o.predict(x_test)
print('Accuracy of Decision Tree is=',m.accuracy_score(y_test,y_pred))

