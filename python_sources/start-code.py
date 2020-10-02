#!/usr/bin/env python
# coding: utf-8

# # Loading the Data

# In[ ]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Read the train data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

# Read the test data
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
target = 'SalePrice'


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


# # Data Clean-up

# ## Outliers

# Let's explore these outliers

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[ ]:


#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# Note :
# Outliers removal is note always safe. We decided to delete these two as they are very huge and really bad ( extremely large areas for very low prices).
# 
# There are probably others outliers in the training data. However, removing all them may affect badly our models if ever there were also outliers in the test data. That's why , instead of removing them all, we will just manage to make some of our models robust on them. You can refer to the modelling part of this notebook for that.

# ## Features engineering

# let's first concatenate the train and test data in the same dataframe

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
Y_train = train.SalePrice.values
all_data = pd.concat((train, test), sort=False).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# ### Missing Data

# In[ ]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# ### Data Correlation

# In[ ]:


#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True) #annot=True


# ### Imputing missing values

# We impute them by proceeding sequentially through features with missing values

# * **PoolQC** : data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%)  
# and majority of houses have no Pool at all in general.

# In[ ]:


all_data["PoolQC"] = all_data["PoolQC"].fillna("None")


# * **LotFrontage **: Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.

# In[ ]:


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# In[ ]:


# Split train and test agian
train = all_data[:ntrain]
test = all_data[ntrain:]


# ## Self defined mixed Imputation Function for  features with not enough info

# In[ ]:


#Self Defined Imputation for both categorical and numerical datas
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# ### Imputation along with track of what was imputed

# In[ ]:


imputed_train = train.copy()
imputed_test = test.copy()

cols_with_missing = (col for col in test.columns 
                                 if train[col].isnull().any() or test[col].isnull().any() )
for col in cols_with_missing:
    imputed_train[col + '_was_missing'] = imputed_train[col].isnull()
    imputed_test[col + '_was_missing'] = imputed_test[col].isnull()
imputed_train_cols = imputed_train.columns
imputed_test_cols = imputed_test.columns

# Imputation
my_imputer = DataFrameImputer()
imputed_train_array = my_imputer.fit_transform(imputed_train)
imputed_test_array = my_imputer.transform(imputed_test)

imputed_train =  pd.DataFrame(imputed_train_array, columns=imputed_train_cols)
imputed_test =  pd.DataFrame(imputed_test_array, columns=imputed_test_cols)


# In[ ]:


imputed_train.info()


# In[ ]:


imputed_train.head()


# In[ ]:


train = imputed_train.copy()
test = imputed_test.copy()


# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
all_data = pd.concat((train, test)).reset_index(drop=True)
print("all_data size is : {}".format(all_data.shape))


# In[ ]:


#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# In[ ]:


# Exclude data that lies outside 2 standard deviations from the mean
# m = np.mean(train[''])
# s = np.std(train[''])
# train = train[train[''] <= m + 2*s]
# train = train[train[''] >= m - 2*s]


# ## More features engeneering

# **Transforming some numerical variables that are really categorical**

# In[ ]:


#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# **Adding  more important feature**

# In[ ]:


# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# In[ ]:


# Split train and test agian
train = all_data[:ntrain]
test = all_data[ntrain:]


# # Date-Time conversion

# In[ ]:


from datetime import datetime
# train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
# test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
# train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
# test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date


# # Data Visualisation

# ## Initial Analysis

# In[ ]:


plt.rcParams['figure.figsize'] = [16, 10]


# In[ ]:


plt.hist(Y_train, bins=100)
plt.xlabel(target)
plt.ylabel('number of train records')
plt.show()


# In[ ]:


Y_train_log = np.log(Y_train + 1)
plt.hist(Y_train_log, bins=100)
plt.xlabel('log_'+target)
plt.ylabel('number of train records')
plt.show()
sns.distplot(Y_train_log, bins =100)


# # Data Enrichment
# Adding data from external sources

# ## Split train data to X and Y

# In[ ]:


# pull data into target (y) and predictors (X)
X_train = train
X_test = test


# # Encoding categorical variables

# In[ ]:


# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. This is convenient, though
# a little arbitrary.
encoded_X_train = X_train.copy()
encoded_X_test = X_test.copy()
low_cardinality_cols = [cname for cname in encoded_X_train.columns if 
                                encoded_X_train[cname].nunique() < 30 and
                                encoded_X_train[cname].dtype == "object"]
high_cardinality_cols = [cname for cname in encoded_X_train.columns if 
                                encoded_X_train[cname].nunique() >= 30 and
                                encoded_X_train[cname].dtype == "object"]
numeric_cols = [cname for cname in encoded_X_train.columns if 
                                encoded_X_train[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
encoded_X_train = encoded_X_train[my_cols]
encoded_X_test  = encoded_X_test[my_cols]
#final encoding: in only encodes string categorical values unless columns are specified
train_objs_num = len(encoded_X_train)
#Combine tran test data
dataset = pd.concat(objs=[encoded_X_train, encoded_X_test], axis=0)
dataset_preprocessed = pd.get_dummies(dataset)
#Splitting again
encoded_X_train = dataset_preprocessed[:train_objs_num]
encoded_X_test = dataset_preprocessed[train_objs_num:]


# In[ ]:


#Columns still not used for predictions
for feat in high_cardinality_cols:
    print(feat+": "+str(train[feat].nunique()))


# In[ ]:


encoded_X_train.columns


# In[ ]:


encoded_X_train.info()


# In[ ]:


encoded_X_train.head()


# # Modelling

# In[ ]:


final_X_train = encoded_X_train.copy()
final_Y_train = Y_train.copy()
final_X_test = encoded_X_test.copy()


# ### Import librairies

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb


# ### Define a cross validation strategy

# In[ ]:


#Validation function
n_folds = 5

def score_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    error= np.sqrt(-cross_val_score(model, final_X_train, final_Y_train, scoring="neg_mean_squared_error", cv = kf))
    return(error)


# In[ ]:


def accu_score(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# ## Base models

# * LASSO Regression :

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =1,max_iter=100000, random_state=1))


# * Elastic Net Regression :

# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=1, l1_ratio=.9, random_state=3))


# * Kernel Ridge Regression :

# In[ ]:


KRR = KernelRidge(alpha=1, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None)


# * Gradient Boosting Regression :

# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# * XGBoost :

# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# * LightGBM :

# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# ### Base models scores

# In[ ]:


score = score_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = score_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = score_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = score_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = score_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = score_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# ## Stacking averaged Models Class

# In[ ]:


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            instance = clone(model)
            self.base_models_[i].append(instance)
            out_of_fold_predictions[:,i] = cross_val_predict(instance, X, y, cv=kfold)
            instance.fit(X,y)
        
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models])
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# ### Stacking Averaged models Score

# In[ ]:


stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

#score = score_cv(stacked_averaged_models)
#print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# ### Ensembling StackedRegressor, XGBoost and LightGBM

# * StackedRegressor:

# In[ ]:


stacked_averaged_models.fit(final_X_train, final_Y_train)
stacked_train_pred = stacked_averaged_models.predict(final_X_train)
stacked_pred = stacked_averaged_models.predict(final_X_test)
print(accu_score(final_Y_train, stacked_train_pred))


# * XGBoost:

# Tuning XGB Parameters

# In[ ]:


X_train_cv, X_test_cv, Y_train_cv, Y_test_cv = train_test_split(final_X_train, final_Y_train, test_size=0.25)


# In[ ]:


from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

def objective(space):

    clf = xgb.XGBRegressor(n_estimators = 1000,
                            max_depth = int(space['max_depth']),
                            min_child_weight = int(space['min_child_weight']),
                            subsample = space['subsample'])

    eval_set  = [( X_train_cv, Y_train_cv), ( X_test_cv, Y_test_cv)]

    clf.fit(X_train_cv, Y_train_cv,
            eval_set=eval_set,
            early_stopping_rounds=30, verbose = False)

    pred = clf.predict(X_test_cv)
    score = mean_absolute_error(pred, Y_test_cv)
    print("SCORE: "+ str(score))

    return{'loss':score, 'status': STATUS_OK }


space ={
        'max_depth': hp.quniform("x_max_depth", 5, 30, 1),
        'min_child_weight': hp.quniform ('x_min_child', 1, 10, 1),
        'subsample': hp.uniform ('x_subsample', 0.8, 1)
    }

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print (best)


# In[ ]:


my_model = xgb.XGBRegressor(n_estimators = 1000,
                            max_depth = int(best['x_max_depth']),
                            min_child_weight = int(best['x_min_child']),
                            subsample = best['x_subsample'])


# In[ ]:


my_model.fit(final_X_train, final_Y_train, verbose=False)
xgb_train_pred = my_model.predict(final_X_train)
xgb_pred = my_model.predict(final_X_test) #np.expm1 if logY is used for training
print(accu_score(final_Y_train, xgb_train_pred))


# * LightGBM:

# In[ ]:


model_lgb.fit(final_X_train, final_Y_train)
lgb_train_pred = model_lgb.predict(final_X_train)
lgb_pred = model_lgb.predict(final_X_test)
print(accu_score(final_Y_train, lgb_train_pred))


# In[ ]:


'''Score on the entire Train data when averaging'''

print('Score score on train data:')
print(accu_score(final_Y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))


# ### Ensemble prediction:

# In[ ]:


ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15


# ### Final submission

# In[ ]:


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)


# # Interpreting XGBoost

# In[ ]:


#XGB feature importances
import shap

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(final_X_train)

# create a SHAP dependence plot to show the effect of a single feature across the whole dataset
shap.dependence_plot('OverallQual', shap_values, final_X_train)


# In[ ]:


# summarize the effects of all the features
shap.summary_plot(shap_values, final_X_train)


# In[ ]:


shap.summary_plot(shap_values, final_X_train, plot_type="bar")


# # Plots after fitting -- Partial Dependece

# In[ ]:


from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor

my_model_plot = GradientBoostingRegressor()
# fit the model as usual
my_model_plot.fit(final_X_train, final_Y_train)
# Here we make the plot
my_plots = plot_partial_dependence(my_model_plot,       
                                   features=[0,1, 2], # column numbers of plots we want to show
                                   X=final_X_train,            # raw predictors data.
                                   feature_names=['MSSubClass', 'LotArea', 'YearBuilt'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis


# In[ ]:


print("Kernel Completed")


# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
