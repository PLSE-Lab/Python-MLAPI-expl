#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore',message='DeprecationWarning')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:





# ## Loading datasets

# In[ ]:


train=pd.read_csv('../input/train.csv').copy()
test=pd.read_csv('../input/test.csv').copy()


# In[ ]:





# In[ ]:


train.describe()


# In[ ]:


train.info()


# There is no missing values in the dataset.

# In[ ]:


train['type'].unique()


# There are 3 types of monsters in the dataset

# In[ ]:


train=train.drop(['id'],axis=1)
test_df=test.drop(['id'],axis=1)


# In[ ]:


train.head()


# In[ ]:


np.sort(train['color'].unique())==np.sort(test['color'].unique())


# ### Encoding categorical variable

# In[ ]:


color_le=preprocessing.LabelEncoder()
color_le.fit(train['color'])
train['color_le']=color_le.transform(train['color'])


# In[ ]:


sns.pairplot(train.drop(['color'],axis=1),palette='muted',diag_kind='kde',hue='type')


# In[ ]:


train.drop(['color_le'],axis=1,inplace=True)


# correlation

# In[ ]:


sns.heatmap(train.corr(),annot=True)


# In[ ]:





# on hot encoding

# In[ ]:


df=pd.get_dummies(train.drop(['type'],axis=1))
X_train,X_test,y_train,y_test=train_test_split(df,train['type'],random_state=0)


# 
# ## Baseline modelling

# In[ ]:


tr=DecisionTreeClassifier(random_state=0)
tr.fit(X_train,y_train)
y_pre=tr.predict(X_test)

print("accuracy score is ",metrics.accuracy_score(y_test,y_pre))
print('\n',metrics.classification_report(y_test,y_pre))


# In[ ]:


sns.barplot(y=X_test.columns,x=tr.feature_importances_)


# In[ ]:


params={'max_depth':np.linspace(1, 16, 16, endpoint=True),'min_samples_split':np.linspace(.1, 1,10, endpoint=True),"max_features":[1,4,6]}


# In[ ]:


accuracy=metrics.make_scorer(metrics.accuracy_score)


# In[ ]:


tr=DecisionTreeClassifier()
clf=GridSearchCV(tr,param_grid=params,scoring=accuracy,cv=5,n_jobs=-1)
clf.fit(X_train,y_train)
print('best score',clf.best_score_)
print('param',clf.best_params_)


# Random forrests

# In[ ]:


rf=RandomForestClassifier(random_state=0)
rf.fit(X_train,y_train)
y_pre=rf.predict(X_test)

print('rf basline score',metrics.accuracy_score(y_test,y_pre))
print('\n',metrics.classification_report(y_test,y_pre))


# Gradient Boosting

# In[ ]:


gb=GradientBoostingClassifier()
gb.fit(X_train,y_train)
y_pre=gb.predict(X_test)

print('score',metrics.accuracy_score(y_test,y_pre))


#  Tuning random forrest classifier

# In[ ]:


rf.get_params()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)


# In[ ]:


rf1=RandomForestClassifier(random_state=0)
rf1_rand=RandomizedSearchCV(rf1,param_distributions=random_grid,n_iter=100,cv=3,n_jobs=-1)
rf1_rand.fit(X_train,y_train)
print(rf1_rand.best_params_)


# In[ ]:


best_random=rf1_rand.best_estimator_
best_random.fit(X_train,y_train)
y_pre=best_random.predict(X_test)
print(metrics.accuracy_score(y_test,y_pre))


# In[ ]:


params={'max_features':['auto',],'bootstrap':[False],'max_depth':[50,60,70,56],'min_samples_leaf':[1,2],'n_estimators':[100,120,130,140],'min_samples_split':[5,10,15,20]}
rf=RandomForestClassifier()
gcv=GridSearchCV(rf,param_grid=params,cv=5,n_jobs=-1,scoring=accuracy)
gcv.fit(X_train,y_train)
print('score',gcv.best_params_)


# In[ ]:


y_pre=gcv.predict(X_test)
print('score',metrics.accuracy_score(y_test,y_pre))


# In[ ]:


gcv.param_grid


# Making submission

# In[ ]:


test_=pd.get_dummies(test_df)


# In[ ]:


pre=gcv.predict(test_)


# In[ ]:


df=pd.DataFrame({'id':test['id'],'type':pre},columns=['id','type'])
csv=df[['id','type']].to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:




