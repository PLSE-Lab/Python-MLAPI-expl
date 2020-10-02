#!/usr/bin/env python
# coding: utf-8

# **HOUSE PRICES - Advanced Linear Regression**
# 
# * In this notebook I've tried to keep ALL the imputing, transformations, scaling, etc - to be done via PIPELINE (using classes).
# * This allowed me to experiment quickly with different options, parameters, etc. 
# * Found for example that manually eliminating features, using the feature importance tool with a class that drops columns - is better than PCA.
# * I cannot overestimate the importance of the Learning / Validation curves - it is the ONLY way to identify UNDER or OVERfitting.
# * With some tweaking you can achieve 12.24 on Kaggle with this notebook
# * But hey, play with it and let me know your thoughts and ideas on this notebook.
# ______________________________________________________________________________________________________________________________________________________________________________
# * Many thanks to:
# * Serigne https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# * MassQuantity https://www.kaggle.com/massquantity/all-you-need-is-pca-lb-0-11421-top-4
# * Their notebooks helped me a LOT ! Hopefully this notebook will help others as well.

# **Load modules**
# 
# 

# In[ ]:


from operator import itemgetter    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

from pandas.tools.plotting import scatter_matrix
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, cross_val_predict, StratifiedKFold, train_test_split, learning_curve, ShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
import lightgbm as lgb

print("Modules imported \n")

print("Files in current directory:")
import os
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
# Any results you write to the current directory are saved as output.


# **Load raw data**

# In[ ]:


# Load raw data
train = pd.read_csv('../input/train.csv') 
test = pd.read_csv('../input/test.csv') 

# Locally 
#train = pd.read_csv('/Users/Alex/Desktop/HousingLinReg/train.csv') 
#test = pd.read_csv('/Users/Alex/Desktop/HousingLinReg/test.csv') 
print("train ", train.shape)
print("test ", test.shape)


# **Visualization**

# In[ ]:


# Histograms
train.hist(bins=50, figsize=(20,15))
plt.show()


# In[ ]:


# Check contents of ONE column
print(train["MSSubClass"].value_counts())


# In[ ]:


#Deleting outliers
#plt.figure(figsize=(12,6))
#plt.scatter(x=train.GrLivArea, y=train.SalePrice)
#plt.xlabel("GrLivArea", fontsize=13)
#plt.ylabel("SalePrice", fontsize=13)
#plt.ylim(0,800000)

#train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#plt.figure(figsize=(12,6))
#plt.scatter(x=train.GrLivArea, y=train.SalePrice)
#plt.xlabel("GrLivArea", fontsize=13)
#plt.ylabel("SalePrice", fontsize=13)
#plt.ylim(0,800000)

#print("train without outliers ", train.shape)


# In[ ]:


# Pearson Correlation Coefficient
corr_matrix = train.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)


# In[ ]:


# Scatter_matrix
attributes = ["OverallQual", "YrSold", "YearBuilt","SalePrice"]
scatter_matrix(train[attributes], figsize=(12, 8))


# In[ ]:


# Zoom in on one plot
train.plot(kind="scatter", x="OverallQual", y="SalePrice",alpha=0.2)


# **Check for missing data**

# In[ ]:


# Some transformation should be done on train + test
# If not - there's a difference between the columns in train and test after get_dummies as there are different options in train vs test
# Scaler - is important to be done separately as not to influence the mean and std of train with those of test, this leads to snooping on the test and overfitting

trainWprice = pd.DataFrame(train)
trainNoPrice = trainWprice.drop("SalePrice", axis=1)

full=pd.concat([trainNoPrice,test], ignore_index=True)
full.drop(['Id'],axis=1, inplace=True)

print("train ", train.shape)
print("test ", test.shape)
print("full without Id and no SalePrice ", full.shape)


# In[ ]:


# ### Missing Data
ColsMissingValues = full.isnull().sum()
print("There are ", len(ColsMissingValues[ColsMissingValues>0]), " features with missing values")
#print("_"*80)
all_data_na = (full.isnull().sum() / len(full)) * 100
all_data_na = all_data_na.sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head(len(ColsMissingValues[ColsMissingValues>0])))


# **Classes to be used by the pipeline**  with feature engineering: Imputing missing data, Adding / Removing features, Skew, PCA, Scaling

# In[ ]:


class feat_eng(BaseEstimator, TransformerMixin):
    def __init__(self, fill_missvals = True):
        self.fill_missvals = fill_missvals
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self.fill_missvals:
            X["PoolQC"] = X["PoolQC"].fillna("None")
            X["MiscFeature"] = X["MiscFeature"].fillna("None")
            X["Alley"] = X["Alley"].fillna("None")
            X["Fence"] = X["Fence"].fillna("None")
            X["FireplaceQu"] = X["FireplaceQu"].fillna("None")
            X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
            X['MSZoning'] = X['MSZoning'].fillna(X['MSZoning'].mode()[0])            
            X["Functional"] = X["Functional"].fillna("Typ")
            X['Electrical'] = X['Electrical'].fillna(X['Electrical'].mode()[0])
            X['KitchenQual'] = X['KitchenQual'].fillna(X['KitchenQual'].mode()[0])
            X['Exterior1st'] = X['Exterior1st'].fillna(X['Exterior1st'].mode()[0])
            X['Exterior2nd'] = X['Exterior2nd'].fillna(X['Exterior2nd'].mode()[0])
            X['SaleType'] = X['SaleType'].fillna(X['SaleType'].mode()[0])
            X['MSSubClass'] = X['MSSubClass'].fillna("None")

            for col in ('GarageType', 'GarageFinish', 'GarageQual', 
                        'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                        'BsmtFinType1', 'BsmtFinType2','MasVnrType'):
                X[col] = X[col].fillna('None')
                
            for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea'):
                X[col] = X[col].fillna(0)
                
            for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
                X[col] = X[col].fillna(0)
                
            X['MSSubClass'] = X['MSSubClass'].apply(str) 
            X['OverallCond'] = X['OverallCond'].astype(str)
            X['YrSold'] = X['YrSold'].astype(str)
            X['MoSold'] = X['MoSold'].astype(str)
            
            X = X.drop(['Utilities'], axis=1)
            
        return X


# In[ ]:


class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self,additional=1):
        self.additional = additional
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self.additional==1:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
            #X["Age"] = X["YrSold"] - X["YearBuilt"]
        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
            #X["Age"] = X["YrSold"] - X["YearBuilt"]

        return X


# In[ ]:


class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self,skew=0.75):
        self.skew = skew
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X_numeric=X.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        
        return X


# In[ ]:


class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):      
        lab = LabelEncoder()
        
        cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

        for c in cols:
            X[c] = lab.fit_transform(X[c])
        
        return X


# In[ ]:


class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])


# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = 1-np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = 1-np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[ ]:


class drop_cols(BaseEstimator, TransformerMixin):
    def __init__(self, remove_cols = True):
        self.remove_cols = remove_cols
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self.remove_cols:
            del X['PoolQC']
            #del X['YrSold']
            #del X['YearBuilt']
            del X['BsmtFinType1']
            del X['LowQualFinSF']
            del X['MoSold']
            del X['Electrical']
            del X['BldgType']
            #del X['HouseStyle']
            #del X['BsmtUnfSF']
            #del X['BsmtFinType2']
            #del X['BsmtFullBath']
            #del X['BsmtExposure']
            #del X['BsmtQual']
            #del X['BsmtCond']
            del X['SaleType']
            del X['BsmtFinSF2']
            #del X['HeatingQC']
            #del X['Heating']
            del X['Exterior2nd']
            #del X['Foundation']
            del X['ExterCond']
            #del X['BedroomAbvGr']
            del X['2ndFlrSF']
            #del X['KitchenAbvGr']
            #del X['RoofMatl']
            del X['3SsnPorch']
            del X['Exterior1st']
            del X['MasVnrType']
            #del X['ExterQual']
            #del X['PavedDrive']
            #del X['FireplaceQu']
            #del X['Functional']
            #del X['GarageType']
            del X['GarageFinish']
            #del X['MSSubClass']
            del X['Alley']
            #del X['LotShape']
            del X['PoolArea']
            #del X['LandContour']
            #del X['Condition1']         
            #del X['SaleCondition']
            #del X['Condition2']
            del X['RoofStyle']
            del X['MiscFeature']
            del X['Fence']
            #del X['GarageCond']
            #del X['GarageQual']
            #del X['MSZoning']
            #del X['Neighborhood']
            del X['BsmtHalfBath']
            del X['Street']
            #del X['KitchenQual']
            del X['LotConfig']
            #del X['OverallQual']
            #del X['FullBath']
            #del X['TotRmsAbvGrd']
            #del X['GarageArea']
            #del X['LandSlope']
            del X['TotalBsmtSF']
            del X['GarageYrBlt']
            #del X['LotFrontage']
          
        return X


# In[ ]:


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


# **PIPELINE**

# In[ ]:


# Prepare the data, pipeline for models evaluation

# PIPELINE
pipe = Pipeline([
    ('feat_eng', feat_eng()),
    ('add_feature', add_feature(additional=2)),
    ('lab_enc', labelenc()), 
    ('drop_cols', drop_cols()), 
    ('skew_dummies', skew_dummies(skew=1)), 
    
    ])
###                   ** skew_dummies is taking care of the labelencoder AND onehotencoder with get_dummies() !  **

train = pd.read_csv('../input/train.csv') 
test = pd.read_csv('../input/test.csv') 

trainWprice = pd.DataFrame(train)
#trainWprice = trainWprice.drop(trainWprice[(trainWprice['GrLivArea']>4000) & (trainWprice['SalePrice']<300000)].index)
trainNoPrice = trainWprice.drop("SalePrice", axis=1)

full=pd.concat([trainNoPrice,test], ignore_index=True)
full.drop(['Id'],axis=1, inplace=True)
print("full without Id and no price ", full.shape)

FullDataPipe = pipe.fit_transform(full)
print("FullDataPipe ", FullDataPipe.shape)

n_train=train.shape[0]
trainFinal = pd.DataFrame(FullDataPipe[:n_train])
testFinal = pd.DataFrame(FullDataPipe[n_train:])
y= train.SalePrice
yFinal = np.log(train.SalePrice)

# Scaler should be run separately on train and test to prevent overfitting
scaler = RobustScaler()
trainFinal = scaler.fit_transform(trainFinal)
testFinal = scaler.fit_transform(testFinal)

# PCA should be run separately on train and test 
#pca = PCA(n_components = 0.999) 
# Check the number of feats that will keep 99.9% variance. Note the number may be different for train vs test
#pca = PCA(n_components = 200) 
#trainFinal = pca.fit_transform(trainFinal)
#testFinal = pca.fit_transform(testFinal)

print("trainFinal", trainFinal.shape)
print("testFinal", testFinal.shape)
print("yFinal", yFinal.shape)


# In[ ]:


# FEATURE IMPORTANCE - Needs its own SEPARATE pipeline without Scaler or PCA in order to see the features' NAMES and not their numbers
# Useful even AFTER PCA - check the relevance of features for prediction

trainFinalFI = pd.DataFrame(trainFinal)
yFinalFI = yFinal

lasso=Lasso(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
lasso.fit(trainFinalFI,yFinalFI)

FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=trainFinalFI.columns)

# Focus on those with 0 importance
#print(FI_lasso.sort_values("Feature Importance",ascending=False).to_string())
#print("_"*80)
FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# CHECK if any Missing Data
trainFinal = pd.DataFrame(trainFinal)

ColsMissingValues = trainFinal.isnull().sum()
print("There are ", len(ColsMissingValues[ColsMissingValues>0]), " features with missing values")
#print("_"*80)
all_data_na = (trainFinal.isnull().sum() / len(trainFinal)) * 100
all_data_na = all_data_na.sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head(len(ColsMissingValues[ColsMissingValues>0])))
print("_"*80)
print("trainFinal ", trainFinal.shape)
print("yFinal ", yFinal.shape)


# **Cross Validation**

# In[ ]:


# define cross validation 
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse


# **MODELS**

# In[ ]:


# Lin reg ALL 14 models HYPERPARAMS NOT optimized
#models = [LinearRegression(),Ridge(),Lasso(),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),
#          ElasticNet(),SGDRegressor(),BayesianRidge(),KernelRidge(),ExtraTreesRegressor(),XGBRegressor(),lgb.LGBMRegressor()]
#names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb", "LightGBM"]


# In[ ]:


# Some initial models with Hyper params optimized
models = [
    Lasso(alpha =0.0005, random_state=1),
    ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3),
    KernelRidge(alpha=2.2, coef0=2.5, degree=3, gamma=None, kernel='polynomial',kernel_params=None),
    SVR(C=2, cache_size=200, coef0=0.1, degree=3, epsilon=0.005, gamma=0.005,
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.01, verbose=False),
    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.05, loss='huber', max_depth=5,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=10, min_samples_split=10,
             min_weight_fraction_leaf=0.0, n_estimators=2000,
             n_iter_no_change=None, presort='auto', random_state=None,
             subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0,
             warm_start=False),
    #XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
    #                         learning_rate=0.05, max_depth=3, 
    #                         min_child_weight=1.7817, n_estimators=2200,
    #                         reg_alpha=0.4640, reg_lambda=0.8571,
    #                         subsample=0.5213, silent=1,
    #                         random_state =7, nthread = -1),
    #lgb.LGBMRegressor(objective='regression',num_leaves=5,
    #                          learning_rate=0.05, n_estimators=720,
    #                          max_bin = 55, bagging_fraction = 0.8,
    #                          bagging_freq = 5, feature_fraction = 0.2319,
    #                          feature_fraction_seed=9, bagging_seed=9,
    #                          min_data_in_leaf =6, min_sum_hessian_in_leaf = 11),
    
         ]

names = ["LASSO", "ELA","KER", "SVR ", "GBR" ]


# In[ ]:


# Run the models and compare
ModScores = {}

for name, model in zip(names, models):
    score = rmse_cv(model, trainFinal, yFinal)
    ModScores[name] = score.mean()
    print("{}: {:.6f}".format(name,score.mean()))

print("trainFinal ", trainFinal.shape)
print("_"*80)
for key, value in sorted(ModScores.items(), key = itemgetter(1), reverse = False):
    print(key, value)


# **Hyper params optimization of the  models with Grid Search**

# In[ ]:


# SEARCH GRID FOR HYPERPARAMS OPTIMIZATION

# SVR
param_grid = [{'kernel': ["rbf"], 'degree': [3]},
              {'gamma': [0.005],'coef0': [0.1,0.05],'tol': [0.01],
              'C': [2.5,2.2],'epsilon': [0.005],},]


# In[ ]:


# Grid Search Optimization for models
model4cv = SVR()

grid_search = GridSearchCV(model4cv, param_grid)
grid_search.fit(trainFinal, yFinal)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
print("_"*80)
print(grid(grid_search.best_estimator_).grid_get(trainFinal, yFinal,{}))


# In[ ]:


# Optimized hyper params for models
lasso = Lasso(alpha =0.0005, random_state=1)

ker = KernelRidge(alpha=2.2, coef0=2.5, degree=3, gamma=None, kernel='polynomial',
      kernel_params=None)

svr = SVR(C=2, cache_size=200, coef0=0.1, degree=3, epsilon=0.005, gamma=0.005,
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.01, verbose=False)

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

ela = ElasticNet(alpha=0.01, copy_X=True, fit_intercept=True, l1_ratio=0.05,
      max_iter=5000, normalize=False, positive=False, precompute=False,
      random_state=None, selection='cyclic', tol=0.0001, warm_start=False)


# **Final model fit, evaluation & prediction**

# In[ ]:


# Final model fit, evaluation & prediction

model = SVR(C=2, cache_size=200, coef0=0.1, degree=3, epsilon=0.005, gamma=0.005,
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.01, verbose=False)
#model = AveragingModels(models = (svr, ker))
#model = AveragingModels(models = (ela, svr, ker, gbr, lasso, xgb, lgb))

model.fit(trainFinal, yFinal)
score = rmse_cv(model, trainFinal, yFinal)
print(" model score: {:.5f} ({:.4f})\n".format(score.mean(), score.std()))

pred = np.exp(model.predict(testFinal))
pred = np.around(pred, decimals=4, out=None)
print(pred)


# **Learning / Validation Curves - Jtrain vs Jcv to help identify under or overfitting**

# In[ ]:


# LEARNING CURVE
X, y = trainFinal, yFinal
estimator = model

title = "Learning Curves (SVR)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(estimator, title, X, y, ylim=(0.01, 0.17), cv=cv, n_jobs=4)


# **SUBMISSION**

# In[ ]:


# SUBMISSION
result=pd.DataFrame({'Id':test.Id, 'SalePrice':pred})
result.to_csv("submission.csv",index=False)
print(result)

