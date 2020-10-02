#!/usr/bin/env python
# coding: utf-8

# # This Notebook is to pull ideas from my previous projects as well as some other kernels. I do not work on boosting or stacking very often, so this is great practice

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import boxcox1p

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder,  RobustScaler
from sklearn.base import clone
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge

import lightgbm as lgb


# In[ ]:


train = pd.read_csv('../input/train.csv').drop('Id', axis=1)
test = pd.read_csv('../input/test.csv')


# In[ ]:


# remove outliers as shown by other notebooks
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


y = train['SalePrice'].values
y_1p = np.log1p(y)

all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice', 'Id'], axis=1, inplace=True)


# In[ ]:


# take a peek at the data
train.head()


# In[ ]:


# my performance benchmark (Root Mean Squared Logrithmic Error)
def rmsle(actual, predicted):
    """
    Computes the root mean squared log error.
    This function computes the root mean squared log error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The root mean squared log error between actual and predicted
    """
    
    def sle(actual, predicted):

        return np.power(np.log(np.array(actual)+1) - np.log(np.array(predicted)+1), 2)
    
    return np.sqrt(np.mean(sle(actual, predicted)))


# In[ ]:


# seperate into data type
numerical_cols = all_data.select_dtypes(include=[int, float]).columns.tolist()
catigorical_cols = all_data.select_dtypes(exclude=[int, float]).columns.tolist()


# In[ ]:


print('catigorical variables: {}'.format(len(catigorical_cols)))
print('numerical variables:   {}'.format(len(numerical_cols)))


# # Dealing with Missing Catigorical Data

# In[ ]:


# find columns with missing data
nan = pd.DataFrame()
nan['total'] = all_data.isnull().sum(axis=0).sort_values(ascending=False)
nan['percent'] = nan['total'] / all_data.shape[0]
nan.head()


# In[ ]:


# visually inspect columns with missing data for any trend
cols = nan.head(4).index.tolist()
for col in cols:
    train[[col, 'SalePrice']].dropna().plot(x=col, y='SalePrice')


# In[ ]:


# drop out top 4 columns that have a ton of nans
catigorical_cols = [x for x in catigorical_cols if x not in nan.head(4).index.tolist()]


# In[ ]:


# take y out of numerical cols
numerical_cols = [x for x in numerical_cols if x not in ['SalePrice']]


# # Numerical Columns

# In[ ]:


# take y out of numerical cols
numerical_cols = [x for x in numerical_cols if x not in ['SalePrice']]


# In[ ]:


# numerical columns
all_data["LotFrontage"] = boxcox1p(all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median())), 0.15)

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
            'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', "MasVnrArea"):
    all_data[col] = boxcox1p(all_data[col].fillna(0.), 0.15)


# # Catigorical Columns

# In[ ]:


# catigorical columns
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'MasVnrType', 'MSSubClass'):
    all_data[col] = all_data[col].fillna('None')
    
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


# I chose to use an imputer to impute the mode of each column. There are few enough numerical columns that you could go through and manually check, but that doesn't apply to many problems in high dimensions.

# # Finish Processing the Data

# In[ ]:


# fix and label the catigorical data
cat = pd.get_dummies(all_data[catigorical_cols].fillna('None'))

imputer = Imputer(strategy='most_frequent')
num = imputer.fit_transform(all_data[numerical_cols])

processed = np.concatenate([cat, num], axis=1)

X_tr = processed[:train.shape[0]]
X_te = processed[train.shape[0]:]

print(X_tr.shape, X_te.shape, y.shape)


# # Modeling

# In[ ]:


# can cross validate the parameters if you want
min_samples_split = 7
max_depth = 8
max_features='sqrt'
subsample = 0.8 
loss='ls'

param_test = {
#     'n_estimators': np.arange(45, 60, 1),
#           'max_depth': np.arange(10, 30, 2),
#     'max_features': np.arange(1, 40, 3)
#           'min_samples_split': np.arange(2, 10, 1)
#     'alpha': np.arange(0.01, .1, .01),
    'l1_ratio': np.arange(0, 1, 0.07)
#     'min_samples_leaf': np.arange(1, 10, 1),
#     'min_impurity_decrease': np.arange(5, 9, .1) 
}

gboost = GradientBoostingRegressor(loss='huber',
                                      n_estimators=3000,
                                      learning_rate=0.03,
                                      max_depth=4,
                                      min_samples_split=10,
                                      max_features=13)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

lregress = ExtraTreesRegressor(n_estimators=6000,
                               criterion='mse',
                               max_features=30,
                               max_depth=25,
                               min_samples_split=4,
                               min_samples_leaf=2)
elastic = ElasticNet(2.052, l1_ratio=0.5)



# gsearch1 = GridSearchCV(estimator=elastic, param_grid=param_test,
#                          n_jobs=-1, cv=10)

# gsearch1.fit(X_tr, y_1p)
# gsearch1.best_params_, 


# In[ ]:


reg = gboost

# set up training set
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_1p, test_size=0.20)

reg.fit(X_train, y_train)
# score.append(rmsle(y_val, reg.predict(X_val)))
    
print('RMSLE:     {}'.format(rmsle(np.expm1(y_val), np.expm1(reg.predict(X_val)))))
print('R_squared: {}'.format(reg.score(X_val, y_val)))

ax = sns.distplot((np.expm1(reg.predict(X_val)) - np.expm1(y_val)))
plt.show()


# In[ ]:


reg = lasso

# set up training set
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_1p, test_size=0.20)

reg.fit(X_train, y_train)
# score.append(rmsle(y_val, reg.predict(X_val)))
    
print('RMSLE:     {}'.format(rmsle(np.expm1(y_val), np.expm1(reg.predict(X_val)))))
print('R_squared: {}'.format(reg.score(X_val, y_val)))

ax = sns.distplot((np.expm1(reg.predict(X_val)) - np.expm1(y_val)))
plt.show()


# In[ ]:


# I stole this from another notebook as it was written much better then my version.

# Takes models and using k-folding, predicts around the fold, and then uses another model to predict previous model output

class StackingAveragedModels():
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# In[ ]:


gboost was doing the best by far on the data
stack_model = StackingAveragedModels(base_models=[gboost, gboost, lasso, lasso],
                                  meta_model= make_pipeline(RobustScaler(), LinearRegression()))
reg = stack_model.fit(X_train, y_train)
    
print('RMSLE:     {}'.format(rmsle(np.expm1(y_val), np.expm1(reg.predict(X_val)))))

ax = sns.distplot((reg.predict(X_val) - y_val))
plt.show()


# In[ ]:


stack_model = StackingAveragedModels(base_models=[gboost, lasso],
                                  meta_model= lasso)
stack = stack_model.fit(X_tr, y_1p)
gboost = gboost.fit(X_tr, y_1p)
lasso = lasso.fit(X_tr, y_1p)


# In[ ]:


out = pd.DataFrame()
out['Id'] = test['Id']
out['SalePrice'] = np.expm1(stack.predict(X_te)) * 0.7 + np.expm1(gboost.predict(X_te)) * 0.15 + np.expm1(lasso.predict(X_te)) * 0.15
out.to_csv('submission.csv',index=False)


# In[ ]:




