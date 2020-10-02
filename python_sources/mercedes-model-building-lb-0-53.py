#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

from sklearn.ensemble import GradientBoostingRegressor
# feature selection (from supportive model)
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV


import xgboost as xgb


# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# 

# In[ ]:


for col in train.select_dtypes(['object']).columns:
    lb=LabelEncoder()
    lb.fit(list(train[col].values.astype(str))+list(test[col].values.astype(str)))
    train[col]=lb.transform(list(train[col].astype(str)))
    test[col]=lb.transform(list(test[col].astype(str)))
    print(col, 'Done')


# 

# In[ ]:


x_train=train.drop(['y','ID'],1)
y_train=train['y']

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=100)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,15))
xgb.plot_importance(model, height=0.8, ax=ax, max_num_features=30)
plt.show()


# 

# In[ ]:


for index in train.index:
    train.ix[index, 'ones_count']=len(train.ix[index][train.ix[index]==1])
    train.ix[index, 'zeros count']=len(train.ix[index][train.ix[index]==0])
    
    test.ix[index, 'ones_count']=len(test.ix[index][test.ix[index]==1])
    test.ix[index, 'zeros count']=len(test.ix[index][test.ix[index]==0])
    
train.head()


# 

# In[ ]:


training, validation = train_test_split(train, test_size=0.2, random_state=4242)

test_ids=test['ID']

#We first select all the features
x_train1, y_train1 = training.drop(['y'],1), training['y']
x_val1, y_val1 = validation.drop(['y'],1), validation['y']
x_test=test
training.shape, validation.shape


# 

# 

# In[ ]:


lr1=LinearRegression(normalize=False, fit_intercept=True)
lr1.fit(x_train1, y_train1)
pred1=lr1.predict(x_val1)
print(r2_score(pred1, y_val1))


# 

# In[ ]:


lr1.fit(train.drop(['ID','y'],1), train['y'])
sub1=lr1.predict(test.drop('ID',1))
#output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': sub1})
#output.to_csv('/Users/adityavyas/Desktop/mercedes.csv', index=False)


# 

# In[ ]:


imp_features=['X0','X5','X8','X6','X1','X2','X3','y']

x_train2, y_train2 = training[imp_features].drop('y',1), training[imp_features]['y']
x_val2, y_val2 = validation[imp_features].drop('y',1), validation[imp_features]['y']

lr2=LinearRegression(normalize=True, fit_intercept=True)
lr2.fit(x_train2, y_train2)
pred2=lr2.predict(x_val2)
print(r2_score(y_val2, pred2))


# 

# 

# In[ ]:


from sklearn.decomposition import PCA

full_data=pd.concat((train.drop(['y'],1), test), keys=['train', 'test'])

pca=PCA()
pca.fit(full_data)


plt.figure(1,figsize=(15,10))
list_k=range(1, (len(full_data.columns))+1)
# len(pca.explained_variance_ratio_.cumsum())
plt.scatter(list_k,pca.explained_variance_ratio_.cumsum())
plt.xlim([0,100])

plt.figure(2,figsize=(15,10))
plt.scatter(list_k,pca.explained_variance_ratio_.cumsum())
plt.xlim([0,20])
plt.show()


# We now use a GridSearch to select the best number of components and create a pipeline to make our predictions.

# 

# In[ ]:


from sklearn.pipeline import Pipeline

X_TRAIN, Y_TRAIN = train.drop(['y'],1), train['y']
X_TEST = test
n_components = [50, 100, 200]

lr_pca = LinearRegression()
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('linear', lr_pca)])

estimator = GridSearchCV(pipe,dict(pca__n_components=n_components))

estimator.fit(X_TRAIN, Y_TRAIN)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()


# In[ ]:


sub4=estimator.predict(test)
output_ = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': sub4})
#output_.to_csv('/Users/adityavyas/Desktop/mercedes4.csv', index=False)


# This got me an LB score of **0.535**.

# 

# In[ ]:


n_components = [50, 100, 200]

lr_ridge = Ridge()
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('linear', lr_ridge)])

estimator_ridge = GridSearchCV(pipe,dict(pca__n_components=n_components))

estimator.fit(X_TRAIN, Y_TRAIN)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()


# In[ ]:


estimator_ridge.fit(x_train1, y_train1)
y_pred_ridge=estimator_ridge.predict(x_val1)
print(r2_score(y_val1, y_pred_ridge))


# 

# 

# In[ ]:



X_TRAIN, Y_TRAIN = train.drop(['y'],1), train['y']
X_TEST = test
X_train, X_val, Y_train, Y_val = train_test_split(X_TRAIN,Y_TRAIN, test_size=0.2, random_state=4242)


xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

d_train = xgb.DMatrix(X_train, label=Y_train)
d_val = xgb.DMatrix(X_val, label=Y_val)

watchlist = [(d_train, 'train'), (d_val, 'validation')]

bst1 = xgb.train(xgb_params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)


# This gives a public LB of **0.5369**

# 

# In[ ]:


gbooster = GradientBoostingRegressor(
                max_depth=4, 
                learning_rate=0.005, 
                random_state=42, 
                subsample=0.95, 
                n_estimators=100)

gbooster.fit(train.drop(['y'],1), train['y'])


# In[ ]:


features = pd.DataFrame()
feat_names = train.drop(['y'],1).columns

importances = gbooster.feature_importances_
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()


# Let us take these important features and make a model with just these features

# In[ ]:




