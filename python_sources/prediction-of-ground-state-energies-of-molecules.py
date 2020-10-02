#!/usr/bin/env python
# coding: utf-8

# This kernel follows the [paper](http://arxiv.org/abs/1609.07124) and the [blog](http://burakhimmetoglu.com/machine-learning-meets-quantum-mechanics/) both by Burak Himmetoglu.
# 

# Work in progress :)

# In[24]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import time
start_time = time.time()


# > **1. Data understanding**

# In[25]:


df = pd.read_csv('../input/energy-molecule/roboBohr.csv')


# Size of dataset

# In[26]:


print(df.shape)


# Columns in the dataset
# 
# * **Unnamed: 0** - Index for each molecule.
# * **0-1274** - 1275 entries in the Coulomb matrix that act as molecular features.
# * **pubchem_id** - Pubchem Id where the molecular structures are obtained. A unique identifier for each molecule
# * **Eat** - atomization energy calculated by simulations using the Quantum Espresso package.

# In[27]:


df.columns


# Checking for missing values

# In[28]:


df.isnull().sum().sum()


# In[29]:


df.head(5)


# Unique columns that can be dropped

# In[30]:


df = df.drop(['Unnamed: 0', 'pubchem_id'], axis = 1)


# Target Variable: **Eat**

# In[31]:


df.Eat.describe()


# In[32]:


sns.distplot(df['Eat'], kde=True, color="g")
plt.xlabel('Atomization Energy')
plt.ylabel('Frequency')
plt.title('Atomization Energy Distribution');


# > **2. Feature Extraction**

# PCA

# In[33]:


# data taken from https://github.com/bhimmetoglu/RoboBohr/tree/master/data
columbl = pd.read_csv('../input/columbl/coulombL.csv', header=None, index_col=0)


# In[34]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
z = pca.fit_transform(columbl)
z = pd.DataFrame(z)


# In[35]:


cmap = sns.cubehelix_palette(light=1, as_cmap=True)
ax = sns.scatterplot(x=-z[0], y=z[1],
                     size=df.Eat,hue=df.Eat,
                     palette=cmap, sizes=(5, 60))
plt.xlabel('z1')
plt.ylabel('z2')
plt.title('Atomization Energy as function of two pca components');


# Adding the PCA components to the dataset for modeling

# In[36]:


#z.rename(columns={0:'pca_0', 1:'pca_1'}, inplace=True)
#df = pd.concat([df, z], axis=1, sort=False)


# > **3. Modeling**

# Here I decided to check different methods linear models, nearest neighbors and boosted trees. The [paper](https://arxiv.org/abs/1609.07124) stated that the best method is booosted trees used with xgboost. Let's check this out here. In addition to that I will present the results I got from stacking and averaging some of the models to get better outcomes. 

# In[37]:


Y = df['Eat']
df = df.drop(['Eat'], axis = 1)


# In[38]:


from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
x_train, x_test, y_train, y_test = train_test_split(df, Y, test_size=0.3)


# Inspired from [this kernel](http://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)

# In[67]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error

lgbm_model = LGBMRegressor(num_leaves = 25, n_estimators = 250, min_child_weight = 5, max_depth = 4, learning_rate = 0.08,
                           colsample_bytree = 0.3)

xgb_model = XGBRegressor(objective='reg:linear', eval_metric = 'rmse', learning_rate = 0.0625, reg_lambda = 0,
                         max_depth = 6, colsample_bytree = 0.2, min_child_weight = 10, n_estimators = 400)
rf_model = RandomForestRegressor(n_estimators = 100, min_samples_split = 3, max_features = 'auto', max_depth = 8)
knn_model = KNeighborsRegressor(weights = 'distance', n_neighbors = 3, leaf_size = 90)
ridge_model = Ridge(alpha = 1000)
lasso_model = Lasso(alpha = 0.01, max_iter=10000)
enet_model = ElasticNet(alpha = 0.01, l1_ratio = 0.1, max_iter=10000)


# In[40]:


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin


class AveragingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors):
        self.regressors = regressors
        self.predictions = None

    def fit(self, X, y):
        for regr in self.regressors:
            regr.fit(X, y)
        return self

    def predict(self, X):
        self.predictions = np.column_stack([regr.predict(X) for regr in self.regressors])
        return np.mean(self.predictions, axis=1)
    
    
averaged_model = AveragingRegressor([xgb_model, lgbm_model])


# In[41]:


from mlxtend.regressor import StackingCVRegressor

stacked_model = StackingCVRegressor(
    regressors=[xgb_model, lgbm_model],
    meta_regressor=Ridge()
)


# In[68]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

def rmse_fun(predicted, actual):
    return np.sqrt(np.mean(np.square(predicted - actual)))

rmse = make_scorer(rmse_fun, greater_is_better=False)

models = [
     ('XGBoost', xgb_model),
     ('LightGBM', lgbm_model),
     ('RandomForest', rf_model),
     ('Ridge', ridge_model),
     ('Lasso', lasso_model),
     ('ElasticNet', enet_model),
     ('KNN', knn_model),
     ('Averaged', averaged_model),
     ('Stacked', stacked_model),
]

scores = [
    -1.0 * cross_val_score(model, x_train.values, y_train.values, scoring=rmse, cv=5).mean()
    for _,model in models
]

dataz = pd.DataFrame({ 'Model': [name for name, _ in models], 'Error (RMSE)': scores })
dataz.plot(x='Model', kind='bar')


# In[43]:


dataz


# In[74]:


stacked_model.fit(x_train, y_train)
y_pred_stack = stacked_model.predict(x_test)
print("The score for stacked models (xgb + lgb) is "+str(np.sqrt(mean_squared_error(y_test, y_pred_stack))))


# In[75]:


averaged_model.fit(x_train, y_train)
y_pred_avg = averaged_model.predict(x_test)
print("The score for averaged models (xgb + lgb) is "+str(np.sqrt(mean_squared_error(y_test, y_pred_avg))))


# In[76]:


xgb_model.fit(x_train, y_train)
y_pred_xgb = xgb_model.predict(x_test)
print("The score for xgb model is "+str(np.sqrt(mean_squared_error(y_test, y_pred_xgb))))


# In[58]:


print('Valid stacked mean: %.3f' % y_pred_stack.mean())
print('Valid avg mean: %.3f' % y_pred_avg.mean())
print('Valid xgb mean: %.3f' % y_pred_xgb.mean())

print('Test mean: %.3f' % y_test.mean())


# In[73]:


fig, ax = plt.subplots(nrows=4, sharex=True, sharey=True, figsize=(10,10))
sns.distplot(y_pred_stack, ax=ax[0], color='blue', label='validation stack')
sns.distplot(y_pred_avg, ax=ax[1], color='red', label='validation avg')
sns.distplot(y_pred_xgb, ax=ax[2], color='orange', label='validation xgb')
sns.distplot(y_test, ax=ax[3], color='green', label='test')
ax[0].legend(loc=0)
ax[1].legend(loc=0)
ax[2].legend(loc=0)
ax[3].legend(loc=0)
plt.show()


# In[61]:


print(" Seconds %0.3f" % (time.time() - start_time))

