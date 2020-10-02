#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('precision', 2)
pd.set_option('display.max_columns', 100)

sns.set(style="white", color_codes=True)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


prefix = '/kaggle/input/house-prices-advanced-regression-techniques/'
train_data = pd.read_csv(prefix + 'train.csv')
test_data = pd.read_csv(prefix + 'test.csv')

combine = [train_data, test_data]
data = train_data

data.head()


# In[ ]:


import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
#from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import skew


# In[ ]:


def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5

RMSE = make_scorer(mean_squared_error_, greater_is_better=False)


# In[ ]:


outlier_idx = [4, 11, 13, 20, 46, 66, 70, 167, 178, 185, 199, 224, 261, 309, 313, 318, 349, 412, 423, 440, 454, 477, 478, 523, 540, 581, 588, 595, 654, 688, 691, 774, 798, 875, 898, 926, 970, 987, 1027, 1109, 1169, 1182, 1239, 1256, 1298, 1324, 1353, 1359, 1405, 1442, 1447]
print("%d outliers to be dropped" % len(outlier_idx))

print("before:", train_data.shape)
train_data.drop(train_data.index[outlier_idx], inplace=True)
print("after:", train_data.shape)


# In[ ]:


all_data = pd.concat((train_data.loc[:,'MSSubClass':'SaleCondition'],
                      test_data.loc[:,'MSSubClass':'SaleCondition']))
print(all_data.shape)


# In[ ]:


print("before:", len(all_data.columns))
to_delete = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
all_data.drop(to_delete, axis=1, inplace=True)
print("after:", len(all_data.columns))


# In[ ]:


train_data["SalePrice"].head()


# In[ ]:


train_data["SalePrice"] = np.log1p(train_data["SalePrice"])
train_data["SalePrice"].head()


# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
print("Numeric features:", numeric_feats)

skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
print("Skewed features:", skewed_feats)


# In[ ]:


all_data[skewed_feats].head()


# In[ ]:


all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data[skewed_feats].head()


# In[ ]:


all_data.head()


# In[ ]:


all_data = pd.get_dummies(all_data)
all_data.head()


# In[ ]:


all_data = all_data.fillna(all_data.mean())
all_data.head()


# In[ ]:


X_train = all_data[:train_data.shape[0]]
y_train = train_data.SalePrice
X_test = all_data[train_data.shape[0]:]

print(type(X_train), type(X_test), type(y_train))
print(X_train.shape, X_test.shape, y_train.shape)


# In[ ]:


def create_submission(algorithm, prediction, score):
    now = datetime.datetime.now()
    sub_file = 'submission_' + algorithm + '_' +         str(round(score, 4)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Creating submission: ', sub_file)
    pd.DataFrame({'Id': test_data['Id'].values, 'SalePrice': prediction}).to_csv(sub_file, index=False)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
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
pprint(random_grid)


# In[ ]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf,     param_distributions = random_grid, n_iter = 10, cv = 3,     verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, y_train)

# view the best parameters from fitting the random search
pprint(rf_random.best_params_)


# In[ ]:


rfr = RandomForestRegressor(n_jobs=1, random_state=0)
param_grid = {}#'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]}

param_grid = {
    'bootstrap': [True],
    'max_depth': [50, 100],
    'max_features': ['auto'],
    'min_samples_leaf': [2],
    'min_samples_split': [5],
    'n_estimators': [200, 2000]}

#{'bootstrap': True, 'max_depth': 50, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 2000}

model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=-1, cv=10, scoring=RMSE, verbose=2)
model.fit(X_train, y_train)
score = -model.best_score_

print('Random forecast regression...')
print(rfr)
print('Best Params:\n', model.best_params_)
print('Best CV Score:', score)

y_pred = model.predict(X_test)

create_submission('rfr', np.expm1(y_pred), score)


# In[ ]:


gbr = GradientBoostingRegressor(random_state=0)
param_grid = {
    'n_estimators': [100, 250, 500, 750, 1000],
    'max_features': [5, 10, 15, 20],
    'max_depth': [4, 6, 8, 10, 12, 14],
    'learning_rate': [0.05, 0.1, 0.15, 0.2],
    'subsample': [0.4, 0.6, 0.8]
}

#Best Params: {'learning_rate': 0.05, 'max_depth': 4, 'max_features': 15, 'n_estimators': 1000, 'subsample': 0.6}
#{'max_depth': 6, 'max_features': 15, 'n_estimators': 500}

model = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=-1, cv=10, scoring=RMSE, verbose=1)
model.fit(X_train, y_train)
score = -model.best_score_

print('Gradient boosted tree regression...')
print(gbr)
print('Best Params:\n', model.best_params_)
print('Best CV Score:', score)

y_pred = model.predict(X_test)

create_submission('gbr', np.expm1(y_pred), score)


# In[ ]:


etr = ExtraTreesRegressor(n_jobs=1, random_state=0)
param_grid = {}#'n_estimators': [500], 'max_features': [10,15,20]}

model = GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=-1, cv=10, scoring=RMSE)
model.fit(X_train, y_train)
score = -model.best_score_

print('Extra trees regression...')
print(etr)
print('Best Params:\n', model.best_params_)
print('Best CV Score:', score)

y_pred = model.predict(X_test)

create_submission('etr', np.expm1(y_pred), score)


# In[ ]:


xgbr = xgb.XGBRegressor(seed=0)
param_grid = {
#        'n_estimators': [500],
#        'learning_rate': [ 0.05],
#        'max_depth': [ 7, 9, 11],
#        'subsample': [ 0.8],
#        'colsample_bytree': [0.75,0.8,0.85],
}

model = GridSearchCV(estimator=xgbr, param_grid=param_grid, n_jobs=-1, cv=10, scoring=RMSE)
model.fit(X_train, y_train)
score = -model.best_score_

print('eXtreme Gradient Boosting regression...')
print(xgbr)
print('Best Params:\n', model.best_params_)
print('Best CV Score:', score)

y_pred = model.predict(X_test)

create_submission('xgbr', np.expm1(y_pred), score)


# In[ ]:




