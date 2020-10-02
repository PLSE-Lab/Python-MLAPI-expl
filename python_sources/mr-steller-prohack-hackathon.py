#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from statistics import mean
import re
from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn 


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/prohack12/train.csv')
test = pd.read_csv('/kaggle/input/prohack12/test.csv')
submission = pd.read_csv('/kaggle/input/prohack/sample_submit.csv')


# In[ ]:


train.head(3)


# In[ ]:


train_features = train.drop(['y'], axis = 1)
train_dependent = train['y']


# In[ ]:


#categorical and numerical variables
categorical_cols = []
numerical_cols = []
for col in train_features.columns:
    if train_features[col].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        numerical_cols.append(col)
    elif train_features[col].dtype == object:
        categorical_cols.append(col)

        
features = numerical_cols + categorical_cols
train_features = train[features]
test = test[features]

print('test', test.shape)
print('train_features', train_features.shape)
print('train_dependent', train_dependent.shape)


# In[ ]:


features = pd.concat([train_features, test], axis = 0)
features.shape


# In[ ]:


#Null numerical values (percentage)
null = features[numerical_cols].isna().sum().sort_values(ascending = False)
null_per = (null/4755) * 100
null_perc = pd.DataFrame(null_per)
null_perc.head(50)


# In[ ]:


#percentages plot
f, ax = plt.subplots(figsize=(25, 22))
plt.xticks(rotation='90')
sns.barplot(x=null_per.index, y=null_per)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# This dataset is just messed up maaaan. so sad

# In[ ]:


#columns with null values < 60%
for entry, column in zip(null_perc.iloc[:, 0], null_perc.index):
    if entry <= 60:
        print(column)


# ### The latter

# * GGP - Simple Imputer based on mean
# * Population using at least basic sanitation services (%) - Simple Imputer based on mean
# * Population using at least basic drinking-water services (%) - SimpleImputer based on mean
# * Intergalactic Development Index (IDI), Rank - SimpleImputer based on most_frequent
# * Intergalactic Development Index (IDI) - SimpleImputer based on mean
# * Education Index -  SimpleImputer based on mean
# * Mean years of education (galactic years) -  SimpleImputer based on mean
# * Expected years of education (galactic years) -  SimpleImputer based on mean
# * Income Index -  SimpleImputer based on mean
# * Gross income per capita -  SimpleImputer based on mean
# * existence expectancy at birth - SimpleImputer based on mean
# * existence expectancy index-  SimpleImputer based on mean

# In[ ]:


features.columns


# In[ ]:


#imputing with column means.
columns = [1,2,3,4,5,6,7,8,10,11,12]
for col in columns:
    x = features.iloc[:, col].values
    x = x.reshape(-1,1)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(x)
    x = imputer.transform(x)
    features.iloc[:, col] = x    


# In[ ]:


#imputing 'Intergalactic Development Index (IDI), Rank' with most_frequent.
x = features.iloc[:, 9].values
x = x.reshape(-1,1)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x)
x = imputer.transform(x)
features.iloc[:, 9] = x    


# In[ ]:


x = features.iloc[:, 3].values
x = x.reshape(-1,1)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x)
x = imputer.transform(x)
features.iloc[:, 3] = x    


# In[ ]:


#Null numerical values (percentage)
null = features[numerical_cols].isna().sum().sort_values(ascending = False)
null_perc = (null/4755) * 100
null_perc = pd.DataFrame(null_perc)
null_perc.tail(15)         


# ### The former (>60%)

# In[ ]:


features_to_be_dropped = []
for feature in features.columns:
    null_perc = null_perc
    for entry, column in zip(null_perc.iloc[:, 0], null_perc.index):
        if entry > 60:
            over_60 = column
            features_to_be_dropped.append(over_60)
            
print('\nFeatures with null values over 60%:\n')
print(features_to_be_dropped)

features = features.drop(features_to_be_dropped, axis=1).copy()
features.shape


# In[ ]:


features.head()


# In[ ]:


features.isna().any()


# ### Categorical Features
# 
# There's only one categorical features, **galaxy**, which is more like **ID**. I'm gonna drop it then work with the remaining 14 numerical columns.

# In[ ]:


features = features.drop(['galaxy'], axis = 1)
features.columns


# ### Feature Generation
# * existence_expectancy_trend - based on 'existence expectancy index', 'existence expectancy at birth'
# * pop using at least basic needs - based on drinking water services and sanitation services
# * capital formation : per capita - based on capital formation and gross income per capita
# 

# In[ ]:


features['existence_expectancy_trend '] = features['existence expectancy at birth'] / features['existence expectancy index']
features['access to basic needs'] = (features['Population using at least basic drinking-water services (%)'] + features['Population using at least basic sanitation services (%)']) / 2
features['capital formation to per capita'] = (features['Gross capital formation (% of GGP)'] * 100) / features['Gross income per capita']


# ### Distribution test and Skewness

# In[ ]:


# Target variable
sns.distplot(train_dependent.index , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_dependent)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('y distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_dependent.index, plot=plt)
plt.show()


# Linear models love normally distributed data but the plot shows that the variable is more **uniformly** distributed than **normally**. I tried applying boxcox and log transformations to normalize it but both miu and sigma were reducing when the latter should approach 1. So I'm gonna leave it as it is and comment out the code for the transformations.

# In[ ]:


# #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

#log transformation
# train_dependent = np.log1p(train_dependent)

#boxcox transformation.
# train_dependent = boxcox1p(train_dependent, boxcox_normmax(train_dependent + 1))

# #Check the new distribution 
# sns.distplot(train_dependent.index , fit=norm);

# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(train_dependent)
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')

# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(train_dependent.index, plot=plt)
# plt.show()


# In[ ]:


features.columns


# In[ ]:


f, axes = plt.subplots(2, 4, figsize=(67, 40), sharex=True)
sns.distplot( features["galactic year"] , color="skyblue", ax=axes[0, 0])
sns.distplot( features["existence expectancy index"] , color="olive", ax=axes[0, 1])
sns.distplot( features["existence expectancy at birth"] , color="gold", ax=axes[0, 2])
sns.distplot( features["Gross income per capita"] , color="teal", ax=axes[0, 3])
sns.distplot( features["Income Index"] , color="skyblue", ax=axes[1, 0])
sns.distplot( features["Expected years of education (galactic years)"] , color="olive", ax=axes[1, 1])
sns.distplot( features["Mean years of education (galactic years)"] , color="gold", ax=axes[1, 2])
sns.distplot( features["Intergalactic Development Index (IDI)"] , color="teal", ax=axes[1, 3])


# In[ ]:


f, axes = plt.subplots(2, 4, figsize=(57, 40), sharex=True)
sns.distplot( features["Education Index"] , color="skyblue", ax=axes[0, 0])
sns.distplot( features["Intergalactic Development Index (IDI), Rank"] , color="olive", ax=axes[0, 1])
sns.distplot( features["Population using at least basic drinking-water services (%)"] , color="gold", ax=axes[0, 2])
sns.distplot( features["Population using at least basic sanitation services (%)"] , color="teal", ax=axes[0, 3])
sns.distplot( features["Gross capital formation (% of GGP)"] , color="skyblue", ax=axes[1, 0])
sns.distplot( features["existence_expectancy_trend "] , color="olive", ax=axes[1, 1])
sns.distplot( features["access to basic needs"] , color="gold", ax=axes[1, 2])
sns.distplot( features["capital formation to per capita"] , color="teal", ax=axes[1, 3])


# In[ ]:


# numerical variables
norm_features = ['galactic year','Gross income per capita',
       'Intergalactic Development Index (IDI), Rank',
       'Population using at least basic drinking-water services (%)',
       'Population using at least basic sanitation services (%)',
       'Gross capital formation (% of GGP)', 'existence_expectancy_trend ',
       'access to basic needs', 'capital formation to per capita']
skew_features = features[norm_features].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

low_skew = skew_features[skew_features < -0.5]
low_skew_index = low_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})

print("There are {} numerical features with Skew < -0.5 :".format(low_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :low_skew})
skew_features


# In[ ]:


print(skew_index)


# In[ ]:


# # Normalize skewed features with boxcox transformation
# for i in skew_index:
#     features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
# for j in low_skew_index:
#     features[j] = boxcox1p(features[j], boxcox_normmax(features[j] + 1))
for i in skew_index:
    log_max = np.log(features[i].max())
    features[i] = features[i]**(1/log_max)


# In[ ]:


for i in low_skew_index:
    log_max = np.log(features[i].max())
    features[i] = features[i]**(1/log_max)


# In[ ]:


f, axes = plt.subplots(2, 4, figsize=(57, 40), sharex=True)
sns.distplot( features["existence_expectancy_trend "] , color="skyblue", ax=axes[0, 0])
sns.distplot( features["Gross income per capita"] , color="olive", ax=axes[0, 1])
sns.distplot( features["Population using at least basic drinking-water services (%)"] , color="gold", ax=axes[0, 2])
sns.distplot( features["Population using at least basic sanitation services (%)"] , color="teal", ax=axes[0, 3])
sns.distplot( features["Gross capital formation (% of GGP)"] , color="skyblue", ax=axes[1, 0])
sns.distplot( features["existence_expectancy_trend "] , color="olive", ax=axes[1, 1])
sns.distplot( features["access to basic needs"] , color="gold", ax=axes[1, 2])
sns.distplot( features["capital formation to per capita"] , color="teal", ax=axes[1, 3])


# In[ ]:


# skewness after
norm_features = ['galactic year','Gross income per capita',
       'Intergalactic Development Index (IDI), Rank',
       'Population using at least basic drinking-water services (%)',
       'Population using at least basic sanitation services (%)',
       'Gross capital formation (% of GGP)', 'existence_expectancy_trend ',
       'access to basic needs', 'capital formation to per capita']
skews = features[norm_features].apply(lambda x: skew(x)).sort_values(ascending=False)
skews


# good enough. 

# ### Multicolinearity

# In[ ]:


#defining numerical features again to include the added features for the correlation plot to be plotted.
numerical_cols= []
for column in train_features.columns:
    if train_features[column].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        numerical_cols.append(column)

new_train_set = pd.concat([features.iloc[:len(train_dependent), :], train_dependent], axis=1)


# In[ ]:


def correlation_map(f_data, f_feature, f_number):
    f_most_correlated = f_data.corr().nlargest(f_number,f_feature)[f_feature].index
    f_correlation = f_data[f_most_correlated].corr()
    
    f_mask = np.zeros_like(f_correlation)
    f_mask[np.triu_indices_from(f_mask)] = True
    with sns.axes_style("white"):
        f_fig, f_ax = plt.subplots(figsize=(12, 10))
        f_ax = sns.heatmap(f_correlation, mask=f_mask, vmin=0, vmax=1, square=True,
                           annot=True, annot_kws={"size": 10}, cmap="BuPu")

    plt.show()

correlation_map(new_train_set, 'y', 20)


# basic needs, sanitation and water: high corrrelation.
# basic needs has a higher in relation to y, so I'm gonna drop sanitation and water.

# In[ ]:


features = features.drop(['Population using at least basic drinking-water services (%)', 'Population using at least basic sanitation services (%)'], axis = 1)


# ### Outliers

# credits to [this kernel on outlier detection using chauvenets creterion](http://kaggle.com/nroman/detecting-outliers-with-chauvenet-s-criterion)

# In[ ]:


def chauvenet(array):
    mean = array.mean()           # Mean of incoming array
    stdv = array.std()            # Standard deviation
    N = len(array)                # Lenght of incoming array
    criterion = 1.0/(2*N)         # Chauvenet's criterion
    d = abs(array-mean)/stdv      # Distance of a value to mean in stdv's
    prob = erfc(d)                # Area normal dist.    
    return prob < criterion       # Use boolean array outside this function


# In[ ]:


features_outliers = dict()
for col in [col for col in features.columns if 'var_' in col]:
    features_outliers[col] = features[chauvenet(train[col].values)].shape[0]
features_outliers = pd.Series(features_outliers)
features_outliers


# It's an empty series, meaning we're safe on the outliers side.
# We can check that visually using scatterplots.

# In[ ]:


f, axes = plt.subplots(3, 2, figsize=(37, 30), sharex=True)
sns.scatterplot(x = new_train_set["existence_expectancy_trend "] ,y = new_train_set['y'], color="skyblue", ax=axes[0, 0])
sns.scatterplot(x = new_train_set["Gross income per capita"] ,y = new_train_set['y'], color="olive", ax=axes[0, 1])
sns.scatterplot(x = new_train_set["Gross capital formation (% of GGP)"] ,y = new_train_set['y'], color="skyblue", ax=axes[1, 0])
sns.scatterplot(x = new_train_set["existence_expectancy_trend "] ,y = new_train_set['y'], color="olive", ax=axes[1, 1])
sns.scatterplot(x = new_train_set["access to basic needs"] ,y = new_train_set['y'], color="gold", ax=axes[2, 0])
sns.scatterplot(x = new_train_set["capital formation to per capita"] ,y = new_train_set['y'], color="teal", ax=axes[2, 1])


# In[ ]:


f, axes = plt.subplots(4, 2, figsize=(37, 30), sharex=True)
sns.scatterplot(x = new_train_set["galactic year"] ,y = new_train_set['y'], color="skyblue", ax=axes[0, 0])
sns.scatterplot(x = new_train_set["existence expectancy index"] ,y = new_train_set['y'], color="olive", ax=axes[0, 1])
sns.scatterplot(x = new_train_set["existence expectancy at birth"] ,y = new_train_set['y'], color="skyblue", ax=axes[1, 0])
sns.scatterplot(x = new_train_set["Gross income per capita"] ,y = new_train_set['y'], color="olive", ax=axes[1, 1])
sns.scatterplot(x = new_train_set["Income Index"] ,y = new_train_set['y'], color="gold", ax=axes[2, 0])
sns.scatterplot(x = new_train_set["Expected years of education (galactic years)"] ,y = new_train_set['y'], color="teal", ax=axes[2, 1])
sns.scatterplot(x = new_train_set["Mean years of education (galactic years)"] ,y = new_train_set['y'], color="teal", ax=axes[3, 0])
sns.scatterplot(x = new_train_set["Intergalactic Development Index (IDI)"] ,y = new_train_set['y'], color="teal", ax=axes[3, 1])


# Capital formation doesn't look good. Solving that using percentiles but applying to all columns.

# In[ ]:


#Capping the outlier rows with Percentiles
cols = features.columns
for col in cols:
    upper_lim = features[col].quantile(.95)
    lower_lim = features[col].quantile(.05)
    features.loc[(features[col] > upper_lim),col] = upper_lim
    features.loc[(features[col] < lower_lim),col] = lower_lim
    
print(features.shape)


# In[ ]:


features.dtypes


# There's no categorical features to encode, thus we skip that.

# ### Reconstruct train and test sets

# In[ ]:


x_train = features.iloc[:len(train_dependent), :]
x_test = features.iloc[len(train_dependent):, :]
y_train = train_dependent
train_set = pd.concat([x_train, y_train], axis=1)

print('x train', x_train.shape)
print('y train', y_train.shape)
print('train set', train_set.shape)
print('x test', x_test.shape)


# Imputing forgotten nans. Not done in the most efficient order but gets the job done.

# In[ ]:


x = x_train.iloc[:, 3].values
x = x.reshape(-1,1)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x)
x = imputer.transform(x)
x_train.iloc[:, 3] = x    


# In[ ]:


x = x_train.iloc[:, -1].values
x = x.reshape(-1,1)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x)
x = imputer.transform(x)
x_train.iloc[:, -1] = x    


# ## Modelling.

# Credits to this awesome [kernel](http://kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook#Modelling)

# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)
    rmse= np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[ ]:


#lasso regression
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=0))

#elastic net regression
enet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=1))

#lightgbm
lgb = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

#gradboost
gdb = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

#kernelridge
kr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

#xgboost
xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# ### Model scores

# In[ ]:


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(enet)
print("\nElastic net score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(lgb)
print("\nlightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(gdb)
print("\ngradboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(kr)
print("\nkernel ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(xgb)
print("\nxgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# Viewing the performances using a barplot

# In[ ]:


l = [0.0432, 0.0431, 0.0310, 0.0310, 0.0430, 0.0337]
c = ['lasso', 'elasticnet', 'lgb', 'gradboost', 'kernel ridge', 'xgboost']
# data = {'lasso': [0.0432], 'elasticnet': [0.0431], 'lgb': [0.0310], 'gradboost': [0.0310], 'kernel ridge': [0.0430], 'xgboost':  [0.0337]}
df = pd.DataFrame(l, index = c)
sns.barplot(x = df.index, y = df[0])


# ### Stacking

# In[ ]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, x_train, y_train):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(x_train, y_train)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, x_train):
        predictions = np.column_stack([
            model.predict(x_train) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# In[ ]:


#averaging models and observing score change.
averaged_models = AveragingModels(models = (lasso, enet, lgb, xgb, gdb))


# In[ ]:


score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# score better than 3 of the models

# ### Averaging w/a meta model trained with predictions of base models under kfold

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


# performance

# In[ ]:


stacked_averaged_models = StackingAveragedModels(base_models = (xgb, gdb, averaged_models),
                                                 meta_model = lgb)

# score = rmsle_cv(stacked_averaged_models)
# print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# ### Final fitting and prediction

# In[ ]:


# evaluation
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[ ]:


# stacked model
stacked_averaged_models.fit(x_train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(x_train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(x_test.values))
print(rmsle(y_train, stacked_train_pred))


# scored better than the models and the averaged model too.

# In[ ]:


#gradboost
gdb.fit(x_train, y_train)
gdb_train_pred = gdb.predict(x_train)
gdb_pred = np.expm1(gdb.predict(x_test))
print(rmsle(y_train, gdb_train_pred))


# In[ ]:


# xgb
xgb.fit(x_train, y_train)
xgb_train_pred = xgb.predict(x_train)
xgb_pred = np.expm1(xgb.predict(x_test))
print(rmsle(y_train, xgb_train_pred))


# In[ ]:


# lgb
lgb.fit(x_train.values, y_train)
lgb_train_pred = lgb.predict(x_train)
lgb_pred = np.expm1(lgb.predict(x_test.values))
print(rmsle(y_train, lgb_train_pred))


# In[ ]:


l = [0.021496434619159566, 0.015215194899359909, 0.030173101327980283, 0.021534161176007767
]
c = ['stacked', 'gradboost', 'xgb', 'lgb']
df = pd.DataFrame(l, index = c)
sns.barplot(x = df.index, y = df[0])


# ditch xgboost

# In[ ]:


'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.05 +
               gdb_train_pred*0.9 + lgb_train_pred*0.05))


# In[ ]:


ensemble = stacked_pred*0.05 + gdb_pred*0.90 + lgb_pred*0.05


# In[ ]:


ensemble.shape


# ### Submission

# In[ ]:


submission.shape


# In[ ]:


submission = submission
index = list(submission.index)
pred = list(ensemble)
file = pd.DataFrame({'index':index, 'pred': pred})
submission = file.set_index('index')
submission.to_csv('sub.csv')


# In[ ]:


submission.head(3)


# ## Energy allocation headache

# In[ ]:


get_ipython().system('pip install pulp')


# ##### First let B  = Budget = 50,000
# Then let Ei = Energy allocation i
# Then let c = (1/1000) * (Index increase potential) ^ 2
# 
# So, now you want to maximize SUM( c x E )
# 
# By using these variable names I'm trying to clarify and separate the constants from the variable. E in this case is the optimization variable, while c is a constant basically. This is one of the reasons why it is a Linear programming problem. Otherwise, it could be convex, convex quadratic, non-convex problems, and so on.
# 
# The optimizer would assign values to E such that the maximum possible value of F = SUM(c x E) would be attained. F is known as the objective function, in other parlance; cost function or loss.
# 
# 
# I am not yet at the constraint, there are more or less 3 steps to an optimization process. Translate problem, define objective function, define constraints, then run solver.
# 
# The optimizer would assign values to E such that the maximum possible value of F = SUM(c x E) would be attained. F is known as the objective function, in other parlance; cost function or loss.
# 
# 
# For the constraints, the first is that B = 50,000. So all the energies assigned to the galaxies (that means SUM( Ei )) has to be less than 50,000.
# 
# 
# For the constraints, the first is that B = 50,000. So all the energies assigned to the galaxies (that means SUM( Ei )) has to be at most 50,000.  
# 
# The next constrain is that energy for each galaxy (that means Ei ) has to be between 0 and 100.
# 
# So constraint 2 is 0 <= Ei <= 100
# 
# The third constraint requires more nuance. One way to do it is to inject a binary variable (b). So b is where eei of galaxy i is less than 0.7. You can do it with python. It will return True and False values. Then simply add 0 (+0) to the outcome, and an automatic type coercion will change the T/F to 1/0. You then multiply this with the galaxies, to get the ones with eei < 0.7. Once this is done, the constraint becomes SUM(Ei (of b)) <= 10% x B.
# 
# 
# Alternatively, you could split it into two optimization problems since it is LP, One with the galaxies where eei > 0.7, and B = 45,000 and another one with galaxies eei < 0.7 and B = 5000. Then combine your predictions.
# 
# You can have a look at PuLP or Scipy.optimize
# 
# The challenge here is that your predictions y determine your objective function F. So if those are wrong, your optimization would be wrong too. It'll be quite wrong because LP will assign that maximum possible ie 100 to the galaxies and 0 to others. This is a case of uncertain data, there are many research articles on this subject, you might want to look at stochastic optimization or robust LP. Nonetheless, I don't think the challenge requires something that advanced. You can think of a way to overcome this problem. This is my trick; use different models to predict y, then for each model's prediction of y, perform the optimization. Then you take the average. This means that, you are less likely to have just 0s and 100s in your opt energies (edited) 

# ### Using ensemble predictions

# In[ ]:


x_test['y_pred'] = pred
x_test['potential for increase'] = -np.log(x_test['y_pred'] + .01) + 3
y = x_test['potential for increase']
y.head()


# In[ ]:


#creating the new column indicating <0.7 and >= in the eei
x_test['condition_1_for_true'] = np.where(x_test['existence expectancy index'] < 0.7, 1, 0)


# * E is the optimisation variable.(Energy allocation)
# * Ei is the energy allocation per galaxy.
# * c is a constant.
# * b is the eei condition column
# * B is the total energy available

# In[ ]:


# b = x_test[x_test['condition_1_for_true'] == 1]
b = x_test['condition_1_for_true']
c = (1/1000) * (y ** 2)
c.head(3)


# In[ ]:


sns.countplot(b)
plt.ylabel('freq')
plt.xlabel('1 for galaxies with eei < 0.7')


# In[ ]:


b.values


# In[ ]:


import pulp as p 
  
# Create a LP Minimization problem 
prob = p.LpProblem('Problem', p.LpMaximize)  

# Create problem Variables 
variables = []
n = range(0,np.size(b),1)
for i in n:
    variables.append(i)

# B = p.LpVariable("B", lowBound = 0)  
E = p.LpVariable.dicts("E", variables, lowBound = 0, upBound = 100)

# Objective Function 
prob += p.lpSum(p.lpDot(E.values(), c))
# prob += np.sum(c * E.values())  #objective is to maximise the sum of the column.

# Constraints:   
# prob += B == 50000
prob += p.lpSum(E.values()) <= 50000
prob += p.lpSum(p.lpDot(E.values(), b.values)) == 5000

# # Display the problem 
print(prob) 

status = prob.solve()   # Solver 
print(p.LpStatus[status])   # The solution status 

# Printing the final solution 
print(p.value(B), p.value(prob.objective))  


# In[ ]:


from pulp import value

def print_result(problem):
    print('Optimization status:', problem.status)
    print('Final value of the objective:', value(problem.objective))
    print('Final values of the variables:')
    for var in problem.variables():
        global x
        x = var, '=', value(var)
        print(x)
        
print_result(prob)


# ### Using stacked pred

# In[ ]:


x_test['stacked_pred'] = list(stacked_pred)
x_test['potential for increase stacked_pred'] = -np.log(x_test['stacked_pred'] + .01) + 3
y = x_test['potential for increase stacked_pred']
c = (1/1000) * (y ** 2)

import pulp as p 
from pulp import value

  
# Create a LP Minimization problem 
prob = p.LpProblem('Problem', p.LpMaximize)  

# Create problem Variables 
variables = []
n = range(0,np.size(b),1)
for i in n:
    variables.append(i)

# B = p.LpVariable("B", lowBound = 0)  
E = p.LpVariable.dicts("E", variables, lowBound = 0, upBound = 100)

# Objective Function 
prob += p.lpSum(p.lpDot(E.values(), c))
# prob += np.sum(c * E.values())  #objective is to maximise the sum of the column.

# Constraints: 
l = [100] * 890
    
# prob += B == 50000
prob += p.lpSum(E.values()) <= 50000
prob += p.lpSum(p.lpDot(E.values(), b.values)) == 5000

# # Display the problem 
print(prob) 

status = prob.solve()   # Solver 

def print_result(problem):
    print('Optimization status:', problem.status)
    print('Final value of the objective:', value(problem.objective))
    print('Final values of the variables:')
    for var in problem.variables():
        global y
        y = var, '=', value(var)
        print(y)
        
print_result(prob)


# ### Using gdb

# In[ ]:


x_test['gdb_pred'] = list(gdb_pred)
x_test['potential for increase gdb'] = -np.log(x_test['gdb_pred'] + .01) + 3
y = x_test['potential for increase gdb']
c = (1/1000) * (y ** 2)

import pulp as p 
from pulp import value

  
# Create a LP Minimization problem 
prob = p.LpProblem('Problem', p.LpMaximize)  

# Create problem Variables 
variables = []
n = range(0,np.size(b),1)
for i in n:
    variables.append(i)

# B = p.LpVariable("B", lowBound = 0)  
E = p.LpVariable.dicts("E", variables, lowBound = 0, upBound = 100)

# Objective Function 
prob += p.lpSum(p.lpDot(E.values(), c))
# prob += np.sum(c * E.values())  #objective is to maximise the sum of the column.

# Constraints: 
l = [100] * 890
    
# prob += B == 50000
prob += p.lpSum(E.values()) <= 50000
prob += p.lpSum(p.lpDot(E.values(), b.values)) == 5000

# # Display the problem 
print(prob) 

status = prob.solve()   # Solver 

def print_result(problem):
    print('Optimization status:', problem.status)
    print('Final value of the objective:', value(problem.objective))
    print('Final values of the variables:')
    for var in problem.variables():
        global z
        z = var, '=', value(var)
        print(z)
        
print_result(prob)


# ### Using lgb

# In[ ]:


x_test['lgb_pred'] = list(lgb_pred)
x_test['potential for increase lgb'] = -np.log(x_test['lgb_pred'] + .01) + 3
y = x_test['potential for increase lgb']
c = (1/1000) * (y ** 2)

import pulp as p 
from pulp import value

  
# Create a LP Minimization problem 
prob = p.LpProblem('Problem', p.LpMaximize)  

# Create problem Variables 
variables = []
n = range(0,np.size(b),1)
for i in n:
    variables.append(i)

# B = p.LpVariable("B", lowBound = 0)  
E = p.LpVariable.dicts("E", variables, lowBound = 0, upBound = 100)

# Objective Function 
prob += p.lpSum(p.lpDot(E.values(), c))
# prob += np.sum(c * E.values())  #objective is to maximise the sum of the column.

# Constraints: 
l = [100] * 890
    
# prob += B == 50000
prob += p.lpSum(E.values()) <= 50000
prob += p.lpSum(p.lpDot(E.values(), b.values)) == 5000

# # Display the problem 
print(prob) 

status = prob.solve()   # Solver 

def print_result(problem):
    print('Optimization status:', problem.status)
    print('Final value of the objective:', value(problem.objective))
    print('Final values of the variables:')
    for var in problem.variables():
        global w
        w = var, '=', value(var)
        print(w)
        
print_result(prob)


# 1. ## Energy allocation option 2.

# In[ ]:


x_test.head()


# In[ ]:


##two
ss=pd.read_csv('sub.csv')

#The index represent the y_pred 
index = ss['pred']
pot_inc = -np.log(index+0.01)+3
p2= pot_inc**2
ss["p2"] = p2
ss['opt_pred'] = 0
ss['eei'] = x_test['existence expectancy index']

#Sorting using Likelyincreasing index
ss=ss.sort_values('p2',ascending=False)
#Droping The old index 
ss=ss.reset_index(drop=True)


# In[ ]:


ss.head()


# In[ ]:


n = 340
#Giving the max of Energy to the 340 first element (ordered using the likely Increasing Index)
ss.opt_pred[:n]=100
ss.opt_pred[n:] = 0
c=100
alpha = 0.62685
for i in range(n,374):
  if c>=alpha: 
    c=c-alpha
    ss.loc[i,'opt_pred'] =c
alpha=0.067345
for i in range(374,455):
  if c>=alpha: 
    c=c-alpha
    ss.loc[i,'opt_pred'] =c
  else:
    ss.loc[i,'opt_pred'] = 0
alpha = 0.03
for i in range(455,465):
  if c>=alpha: 
    c=c-alpha
    ss.loc[i,'opt_pred'] =c
alpha=0.4339465
for i in range(465,890):
  if c>=alpha: 
    c=c-alpha
    ss.loc[i,'opt_pred'] =c
  else:
    ss.loc[i,'opt_pred'] = 0

print(ss.opt_pred.sum())


# In[ ]:


#Checking if the sum of opt_pred in rows having eei<0.7 is >5000
print("sum",ss.opt_pred.sum())
print("left", (50000-ss.opt_pred.sum())) 
print("eei Sum",ss[ss.eei<0.7]['opt_pred'].sum())


# In[ ]:


#We aren't so sure that allocating 100 Zillion DSML to the 340 first rows is optimal  
ss.opt_pred=0
n = 340
ss.opt_pred[:n]=100
ss.opt_pred[n:] = 0
c=100
#This is a simple test we could add more steps or maybe changing these one
for i in range(n,890):
  if ss.pred[i]-ss.pred[i-1]<4*10**(-6):
    alpha=0.13
  if ss.pred[i]-ss.pred[i-1]>=4*10**(-6) and ss.pred[i]-ss.pred[i-1]<4*10**(-5):
    alpha=0.21222895
  if ss.pred[i]-ss.pred[i-1]>=4*10**(-5) and ss.pred[i]-ss.pred[i-1]<10**(-4):
    alpha=0.33
  if ss.pred[i]-ss.pred[i-1]>=10**(-4) and ss.pred[i]-ss.pred[i-1]<5*10**(-3) :
    alpha=0.48
  if ss.pred[i]-ss.pred[i-1]>=5*10**(-3) and ss.pred[i]-ss.pred[i-1]<10**(-3):
    alpha=0.58
  if ss.pred[i]-ss.pred[i-1]>=10**(-3):
    alpha=0.73
  c-=alpha
  if c-alpha>0:
    ss.loc[i,'opt_pred'] =c
  else:
    ss.loc[i,'opt_pred'] =0

print(ss.opt_pred.sum())


# In[ ]:


#Checking if the sum of opt_pred in rows having eei<0.7 is >5000
print("sum",ss.opt_pred.sum())
print("left", (50000-ss.opt_pred.sum()))
print("eei",ss[ss.eei<0.7]['opt_pred'].sum())


# In[ ]:


#Reordering the list using the real index
ss=ss.sort_values('index',ascending=True)
ss=ss.reset_index(drop=True)


# In[ ]:


ss[['index', 'pred', 'opt_pred']].to_csv('f.csv', index=False)

