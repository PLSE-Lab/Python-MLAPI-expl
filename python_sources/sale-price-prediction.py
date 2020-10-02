#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pds.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from statistics import mode
from scipy.special import boxcox1p
from sklearn.preprocessing import LabelEncoder,RobustScaler
from sklearn.linear_model import Ridge, Lasso,ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from catboost import Pool, CatBoostRegressor, cv
import xgboost as xgb
import lightgbm as lgb
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# In[ ]:


train.shape,test.shape


# In[ ]:


train.describe()


# now looking at the target variable, we donnot need the ID as it is not usefull for the model prediction we can drop it.

# In[ ]:


train_ID = train['Id']
test_ID = test['Id']

# Now drop the 'Id' colum since it's unnecessary for the prediction process
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# In[ ]:


sns.regplot(train["GrLivArea"],y=train["SalePrice"],fit_reg=True)
plt.show()


# we can see there are few outliers in the dataset we can go ahead and remove them as these would affect over model prediction.

# In[ ]:


# Removing two very extreme outliers in the bottom right hand corner
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# Re-check graph
sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=True)
plt.show()


# now taking a look at the target values for any kind of skewness  present in them.

# In[ ]:


(mu,sigma)=norm.fit(train.SalePrice)
sns.distplot(train.SalePrice,fit=norm)
plt.legend(["$\mu=$ {:.2f} and $\sigma=$ {:.2f}".format(mu,sigma)],loc="best")


# From the above plot we can see that the data is bit right skewed. we can make it to normal by applying the log for the target values.

# In[ ]:


train.SalePrice = np.log1p(train.SalePrice)


# In[ ]:


(mu,sigma)=norm.fit(train.SalePrice)
sns.distplot(train.SalePrice,fit=norm)
plt.legend(["$\mu=$ {:.2f} and $\sigma=$ {:.2f}".format(mu,sigma)],loc="best")


# First we will make the target values seperate from the Training values and combine th train and test sets to make all the work together rather doing seperate for each once.

# In[ ]:


train_nS=train.shape[0]
test_nS=test.shape[0] # shpaes of train and tests for sperating them back

train_y=train.SalePrice.values
full_data=pd.concat((train,test)).reset_index(drop=True) #concating the train and test sets

full_data.drop(["SalePrice"],axis=1,inplace=True) #dropping the target values

full_data.shape


# Now our target values looks well distributed.

# Now looking at the missing values and imputing with the appropriate values. Now will  find the percenatge of missing values in each features.

# In[ ]:


missing_data_rank=(full_data.isnull().sum()/len(full_data))*100
print("total number of columns with values misiing : {}".format(missing_data_rank[missing_data_rank>0].count()))
missed =pd.DataFrame({"Missing Percentage": missing_data_rank[missing_data_rank>0].sort_values(ascending =False)})


# We can see that only few features have high rank of missing values in them and rest are prety good enough with less than 10%.
# Now lets dive into data and look how can we fill up this missed values: there are two ways in deeling the missing values:
# 1. We can drop the row for the values misisng in them, tough this is not the ideal choice if we havel less traning data and droping data can lead the model to ineffecinet.
# 2. Than droping the row and loosing the data, we can fill up or impute the mising values with the appropriate values like using mean, median and mode values.
# So am going to keep the missing rows and fill them with appropriate values.

# In[ ]:


missed_features=list(missed.index)


# In[ ]:


full_data.head(30)


# In[ ]:


full_data.GarageQual.unique()


# In[ ]:


# All columns where missing values can be replaced with 'None'
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
    full_data[col] = full_data[col].fillna('None')


# In[ ]:



# All columns where missing values can be replaced with 0
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    full_data[col] = full_data[col].fillna(0)


# In[ ]:




# All columns where missing values can be replaced with the mode (most frequently occurring value)
for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 'Utilities'):
    full_data[col] = full_data[col].fillna(full_data[col].mode()[0])

# Imputing LotFrontage with the median (middle) value
full_data['LotFrontage'] = full_data.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))


# In[ ]:


full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']


# In[ ]:


missing_data=(full_data.isnull().sum()/len(full_data))*100
print("total number of columns with values misiing : {}".format(missing_data[missing_data>0].count()))
missed =pd.DataFrame({"Missing Percentage": missing_data[missing_data>0].sort_values(ascending =False)})


# Now we cleaned up our missing values and its time to move to the categorical values as our models can work with only numeric data, now we change our categorical values to numeric by labeling them. 
# 

# In[ ]:


full_data.info()


# I thing few of the features should be categorical but are as numeric in data, so I am going to change them to categorical values.

# In[ ]:


# Converting those variables which should be categorical, rather than numeric
for col in ('MSSubClass', 'OverallCond', 'YrSold', 'MoSold'):
    full_data[col] = full_data[col].astype(str)
    
full_data.info()


# As we alredy change the right skewed target values to normal distribution, now i am going to cange alll the numeric data to normal distribution if any data is skewd internally.

# In[ ]:


# Applying a log(1+x) transformation to all skewed numeric features
numeric_feats = full_data.dtypes[full_data.dtypes != "object"].index

# Compute skewness
skewed_feats = full_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(15)


# 
# 
# Box Cox Transformation of (highly) skewed features
# 
# Skewed features are a formality when dealing with real-world data. Transformation techniques can help to stabilize variance, make data more normal distribution-like and improve the validity of measures of association.
# 
# The problem with the Box-Cox Transformation is estimating lambda. This value will depend on the existing data, and as such should be considered when performing cross validation on out of sample datasets.
# 

# In[ ]:


# Check on number of skewed features above 75% threshold
skewness = skewness[abs(skewness) > 0.75]
print("Total number of features requiring a fix for skewness is: {}".format(skewness.shape[0]))


# In[ ]:


# Now let's apply the box-cox transformation to correct for skewness
skewed_features = skewness.index
lam = 0.15
for feature in skewed_features:
    full_data[feature] = boxcox1p(full_data[feature], lam)


# Now find any features which is highly represented i.e the values in the features are same to the  extend of 97%, this values donot play any role in the model prediction so we will drop those features. 

# In[ ]:


full_data = full_data.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC'], axis=1)


# In[ ]:


#highlyrepeated_values= [col for col in full_data.select_dtypes(exclude=['number']) if 1 - sum(full_data[col] == mode(full_data[col]))/len(full_data) < 0.03]
# Dropping these columns from both datasets
#full_data = full_data.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC'], axis=1)


# 
# Label encoding
# 
# This step build on the previous step whereby all text data will become numeric. This is a requirement for Machine Learning, that is, only numerical data can be fed into a predictive model. There are many other encoding techniques available, some of which more powerful than Label Encoding which does incur the risk of falsely ranking variables, e.g. coding three locations into 0, 1 and 2 might imply that 2 is a higher value than 0, which is incorrect as the numbers just represent different categories (locations). This is a simple approach, however, and therefore I'm going to stick with it for the current kernel.

# In[ ]:


full_data.info()


# In[ ]:


obj_features=list(full_data.select_dtypes(include="object").columns)
len(obj_features)


# In[ ]:


# List of columns to Label Encode
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# Process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(full_data[c].values)) 
    full_data[c] = lbl.transform(list(full_data[c].values))

# Check on data shape        
print('Shape all_data: {}'.format(full_data.shape))


# In[ ]:


full_data.info()


# In[ ]:


full_data=pd.get_dummies(full_data)


# In[ ]:


full_data.shape


# In[ ]:


full_data.i


# In[ ]:



# Now to return to separate train/test sets for Machine Learning
train_x = full_data[:train_nS]
test_x= full_data[train_nS:]


# In[ ]:


# Defining two rmse_cv functions

def rmse_cv(model):
    
    rmse=np.sqrt(-cross_val_score(model, train_x,train_y,scoring="neg_mean_squared_error",cv=10))
    return (rmse)


# Ridge Regression:

# In[ ]:



alphas = [0.05, 0.1, 0.3, 1, 3, 5,7, 10, 15, 30]
#alphas=np.arange(0.05,30,0.05)
# Iterate over alpha's
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]


# In[ ]:


print(cv_ridge)
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")


# In[ ]:


# 5 looks like the optimal alpha level, so let's fit the Ridge model with this value
#model_ridge = Ridge(alpha = 10)
model_ridge = Ridge(alpha = 7)


# In[ ]:


alphas = [0.01, 0.005, 0.001, 0.0002,0.0003,0.0004,0.0005,0.0001]
#alphas=np.arange(0.0001,0.01,0.0005)
# Iterate over alpha's
cv_lasso = [rmse_cv(Lasso(alpha = alpha,random_state=1)).mean() for alpha in alphas]

# Plot findings
cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")
print(cv_lasso)


# In[ ]:


# Initiating Lasso model
model_lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0004))


# ElasticNet:

# In[ ]:


# Setting up list of alpha's
alphas = [0.01, 0.005, 0.001, 0.00055,0.0006, 0.0001]
#alphas=np.arange(0.0001,1,0.0004)
# Iterate over alpha's
cv_elastic = [rmse_cv(ElasticNet(alpha = alpha)).mean() for alpha in alphas]

# Plot findings
cv_elastic = pd.Series(cv_elastic, index = alphas)
cv_elastic.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")
print(cv_elastic)


# In[ ]:


# Initiating ElasticNet model
model_elasticnet = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.0006))


# 
#  4. Kernel ridge regression
# 
# OK, this is not strictly a generalized linear model. Kernel ridge regression (KRR) combines Ridge Regression (linear least squares with l2-norm regularization) with the 'kernel trick'. It thus learns a linear function in the space induced by the respective kernel and the data. For non-linear kernels, this corresponds to a non-linear function in the original space.
# 

# In[ ]:


# Setting up list of alpha's
alphas = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

# Iterate over alpha's
cv_krr = [rmse_cv(KernelRidge(alpha = alpha)).mean() for alpha in alphas]

# Plot findings
cv_krr = pd.Series(cv_krr, index = alphas)
cv_krr.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")
print(cv_krr)


# In[ ]:


# Initiatiing KernelRidge model
model_krr = make_pipeline(RobustScaler(), KernelRidge(alpha=7, kernel='polynomial', degree=2.65, coef0=6.9))


# 
# B. Ensemble methods (Gradient tree boosting)
# 
# Boosting is an ensemble technique in which the predictors are not made independently, but sequentially.
# 
# This technique employs the logic in which the subsequent predictors learn from the mistakes of the previous predictors. Therefore, the observations have an unequal probability of appearing in subsequent models and ones with the highest error appear most. The predictors can be chosen from a range of models like decision trees, regressors, classifiers etc. Because new predictors are learning from mistakes committed by previous predictors, it takes less time/iterations to reach close to actual predictions. But we have to choose the stopping criteria carefully or it could lead to overfitting on training data. Gradient Boosting is an example of a boosting algorithm, and these are what i'll be applying to the current data next.
#  5. Gradient Boosting
# 
# For the Gradient Boosting algorithm I will use 'huber' as the loss function as this is robust to outliers. The other parameters on display originate from other kernels tackling this challenge, followed by trial and error to refine them to this specific dataset. Again, applying GridSearchCV will help to define a better set of parameters than those currently on display.
# 

# In[ ]:


# Initiating Gradient Boosting Regressor
model_gbr = GradientBoostingRegressor(n_estimators=1200, 
                                      learning_rate=0.05,
                                      max_depth=4, 
                                      max_features='sqrt',
                                      min_samples_leaf=15, 
                                      min_samples_split=10, 
                                      loss='huber',
                                      random_state=5)
cv_gbr=rmse_cv(model_gbr).mean()
cv_gbr


# In[ ]:




# Initiating XGBRegressor
model_xgb = xgb.XGBRegressor(colsample_bytree=0.2,
                             learning_rate=0.025,
                             max_depth=3,
                             n_estimators=1550)
cv_xgb = rmse_cv(model_xgb).mean()
cv_xgb


# In[ ]:


# Initiating LGBMRegressor model
model_lgb = lgb.LGBMRegressor(objective='regression',
                              num_leaves=4,
                              learning_rate=0.05, 
                              n_estimators=1080,
                              max_bin=75, 
                              bagging_fraction=0.80,
                              bagging_freq=5, 
                              feature_fraction=0.232,
                              feature_fraction_seed=9, 
                              bagging_seed=9,
                              min_data_in_leaf=6, 
                              min_sum_hessian_in_leaf=11)
cv_lgb = rmse_cv(model_lgb).mean()
cv_lgb


# In[ ]:


# Fitting all models with rmse_cv function, apart from CatBoost
cv_ridge = rmse_cv(model_ridge).mean()
cv_lasso = rmse_cv(model_lasso).mean()
cv_elastic = rmse_cv(model_elasticnet).mean()
cv_krr = rmse_cv(model_krr).mean()
cv_gbr = rmse_cv(model_gbr).mean()
cv_xgb = rmse_cv(model_xgb).mean()
cv_lgb = rmse_cv(model_lgb).mean()


# In[ ]:




# Creating a table of results, ranked highest to lowest
results = pd.DataFrame({
    'Model': ['Ridge',
              'Lasso',
              'ElasticNet',
              'Kernel Ridge',
              'Gradient Boosting Regressor',
              'XGBoost Regressor',
              'Light Gradient Boosting Regressor',
              ],
    'Score': [cv_ridge,
              cv_lasso,
              cv_elastic,
              cv_krr,
              cv_gbr,
              cv_xgb,
              cv_lgb]})

# Build dataframe of values
result_df = results.sort_values(by='Score', ascending=True).reset_index(drop=True)
result_df.head(8)


# 
# 3. Stacking algorithms
# 
# I've ran eight models thus far, and they've all performed pretty well. I'm now quite keen to explore stacking as a means of achieving an even higher score. In a nutshell, stacking uses as a first-level (base) the predictions of a few basic classifiers and then uses another model at the second-level to predict the output from the earlier first-level predictions. Stacking can be beneficial as combining models allows the best elements of their predictive power on the given challenged to be pooled, thus smoothing over any gaps left from an individual model and increasing the likelihood of stronger overall model performance.
# 
# Ok, let's get model predictions and then stack the results!
# 

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


# In[ ]:



#Averaged base models score
averaged_models = AveragingModels(models = (model_elasticnet, model_gbr, model_krr, model_lasso))
score = rmse_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


n_folds = 5
def rmsle_cv(model):
    #kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train_x.values, train_y, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)


# In[ ]:


#Stacking averaged Models Class
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    
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
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)




# In[ ]:


stacked_averaged_models = StackingAveragedModels(base_models = (model_elasticnet, model_gbr, model_krr),meta_model = model_lasso)
score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# In[ ]:


#define a rmsle evaluation function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[ ]:


#Final Training and Prediction
stacked_averaged_models.fit(train_x.values, train_y)
stacked_train_pred = stacked_averaged_models.predict(train_x.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test_x.values))
print(rmsle(train_y, stacked_train_pred))


# **XGB**

# In[ ]:


model_xgb.fit(train_x, train_y)
xgb_train_pred = model_xgb.predict(train_x)
xgb_pred = np.expm1(model_xgb.predict(test_x))
print(rmsle(train_y, xgb_train_pred))


# **LGB**

# In[ ]:



model_lgb.fit(train_x, train_y)
lgb_train_pred = model_lgb.predict(train_x)
lgb_pred = np.expm1(model_lgb.predict(test_x.values))
print(rmsle(train_y, lgb_train_pred))


# In[ ]:




print('RMSLE score on train data:')
print(rmsle(train_y,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))


# In[ ]:


ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
#ensemble=xgb_pred


# In[ ]:


# Create stacked model
#stacked = (lasso_pred + elastic_pred + ridge_pred + xgb_pred + lgb_pred + krr_pred + gbr_pred) / 7
# Setting up competition submission
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble #stacked
sub.to_csv('house_price_predictions.csv',index=False)


# In[ ]:




