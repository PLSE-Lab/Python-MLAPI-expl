#!/usr/bin/env python
# coding: utf-8

# # Kernel (V.02) Summary
# 
# 
#    In this Kernel we wil predict the saleprice based on a quick EDA, trying as much as possible to deal with the features as unknown variables, like we usually encounter in  KAGGLE competitions, and some intermediate regressors manipulation.
# 
#   The missing values will be predicted with a regressor for numerical features and classifier for the categorical ones using the Saleprice as training input for the general train data and the most correlated feature (GrivLiv) for the remaining test data.
#   
#   The outliers will be spotted with the sklearn's LOF function that gives an outlying factor for all 2-D datapoints then ploted for visual confirmation.
#   
#   The skewness of numerical columns will be delt with using the BoxCox transformation trying to reduce it as much as possible for better predictions. (The Log transformation is a specific case of the BoxCox : lambda=0).
#   
#   Basic one-hot encoding for categoricals, Standard Scaler and PCA to feed the data to our models.
#   
#   The modeling part will rely on basic sklearn regressors, we will staring by performing grid_search on 5-fold cross validation, do some blending  and finally stack the blended models.
#   
#   That's all folks !  :)
#   
#   
# ### Forthcoming improvement ( hoping it will improve though...)
# 
# 
# 
# *   I'm trying to use neural nets , wrap them in sklearn, perform gridcv and blend/stack, the problem is I achevied once a good rmse score but when i started tweaking the model, I lost it ( going 5-10 times higher) even if I went back to the prior configuration. **WIP** . It will be very helpfull if somebody can help with this issue.
# *   I've tried an Auto-encoder for dimensiality reduction, then concatenate with original data but no improvement.
# *   Hoping the denoising auto-encoder will do better (using gaussian and swap noise). **WIP**
# *  As I skipped a lot of the feature engeneering part, I want to find a way to use a variational aito-encoder to generate sampled features  from the latent space. I steel need to develop that
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# # Content :
# 1- Load the data and target analysis
# 
# 2- Fill missing values with regression
# 
# 3- Automatic detection of outliers
# 
# 4- Feature selection
# 
# 5- Modeling (PCA, Gridsearch, Blending and stacking)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


# # 1 - Load the data and target analysis 

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_ID = train.Id
test_ID = test.Id

n_target = train.SalePrice

_=train.pop('Id')
_=test.pop('Id')


# In[ ]:


def show_dist(x):
    sns.distplot(x, fit=norm)
    (mu, sigma) = norm.fit(x)

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu,sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')

    fig = plt.figure()
    res = stats.probplot(x, plot=plt)
    plt.show()
    print("Skewness : %.2f" % x.skew())
    print("Kurtosis : %.2f" % x.kurt())
    return


# In[ ]:


show_dist(n_target)


# In[ ]:


target = np.log(n_target)
show_dist(target)


# # 2 - Missing data

# In[ ]:


def na_count(df):
    total = df.isna().sum().sort_values(ascending=False)
    percent = 100*(total/df.shape[0])
    return pd.concat([total, percent], axis=1, keys=['Total NA', '%'])


# In[ ]:


na_count(train).head(10)


# In[ ]:


def reg_on_na(ser,target, p=0.25):
    n_missing = ser.isna().sum()
    percent = n_missing/ser.shape[0]
    if n_missing == 0:
        print("no missing values in :"+str(ser.name))
    else:    
        if percent < p :
            # missing value index
            m_id = ser[ser.isna()].index
            # labels : non-NA values of our missing series
            Y = np.array(ser.drop(m_id)).reshape(-1,1)
            # target values (salerice) that we will train to predict missing feature
            # Single feature data must be reshaped before training
            X = np.array(target.drop(m_id)).reshape(-1,1)
            # Missing saleprices upon which we will make prediction
            Xm = np.array(target[m_id]).reshape(-1,1)
            reg = LassoCV(cv=5, random_state=0).fit(X,Y)
            ser[m_id] = reg.predict(Xm)
        else :
            print("You should drop :"+str(ser.name))   
    return ser

def class_on_na(ser,target, p=0.25):
    n_missing = ser.isna().sum()
    percent = n_missing/ser.shape[0]
    if n_missing == 0:
        print("no missing values in :"+str(ser.name))
    else:    
        if percent < p :
            # missing value index
            m_id = ser[ser.isna()].index
            # labels : non-NA values of our missing series
            Y = np.array(ser.drop(m_id)).reshape(-1,1)
            # target values (salerice) that we will train to predict missing feature
            # Single feature data must be reshaped before training
            X = np.array(target.drop(m_id)).reshape(-1,1)
            # Missing saleprices upon which we will make prediction
            Xm = np.array(target[m_id]).reshape(-1,1)
            clas = SVC(gamma=2, C=1).fit(X,Y)
            ser[m_id] = clas.predict(Xm)
        else :
            print("You should drop :"+str(ser.name))   
    return ser


# In[ ]:


def fill_missing(df, target=target):
    num_cols = df.select_dtypes([np.number]).columns
    cat_cols = df.select_dtypes([np.object]).columns
    dfnum = df[num_cols].copy()
    dfcat = df[cat_cols].copy()
    num_cols_miss = dfnum.isna().sum()[dfnum.isna().sum()>0].index
    cat_cols_miss = dfcat.isna().sum()[dfcat.isna().sum()>0].index
    df[num_cols_miss] = df[num_cols_miss].apply(lambda x: reg_on_na(x, target))
    df[cat_cols_miss] = df[cat_cols_miss].apply(lambda x: class_on_na(x, target))
    return df


# In[ ]:


train = fill_missing(train,target)

test.Utilities.fillna('AllPub', inplace=True)
test = fill_missing(test, test.GrLivArea)


# # 3 - Removing outliers

# In[ ]:


def outliers(x, y=n_target, top=5, plot=True):
    lof = LocalOutlierFactor(n_neighbors=40, contamination=0.1)
    x_ =np.array(x).reshape(-1,1)
    preds = lof.fit_predict(x_)
    lof_scr = lof.negative_outlier_factor_
    out_idx = pd.Series(lof_scr).sort_values()[:top].index
    if plot:
        f, ax = plt.subplots(figsize=(9, 6))
        plt.scatter(x=x, y=y, c=np.exp(lof_scr), cmap='RdBu')
    return out_idx
    


# In[ ]:


outs = outliers(train['GrLivArea'], top=5)
train = train.drop(outs)
target = target.drop(outs)
n_target = n_target.drop(outs)
ntrain = train.shape[0]
ntest = test.shape[0]
alldata = pd.concat([train, test]).reset_index(drop=True)

alldata.drop(['SalePrice','Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
print(alldata.shape)


# In[ ]:


show_dist(target)


# # 4 - Feature engeneering

# In[ ]:


# All the feature engeneering i've got
alldata['TotalSF'] = alldata['TotalBsmtSF'] + alldata['1stFlrSF'] + alldata['2ndFlrSF']
alldata['Area_Qual'] = alldata['TotalSF']*alldata['OverallQual']


# In[ ]:


numeric_feats = alldata.select_dtypes([np.number]).columns

# Check the skew of all numerical features
skewed_feats = alldata[numeric_feats].apply(skew).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[ ]:


skewness = skewness[abs(skewness) > 0.75]

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.10
_skewed = alldata[numeric_feats].apply(lambda x: boxcox1p(x,lam)).apply(skew).sort_values(ascending=False)
skewness['boxed'] = pd.Series(_skewed)
alldata[numeric_feats] = alldata[numeric_feats].apply(lambda x: boxcox1p(x,lam))
skewness.head(10)


# # 5 - Modeling

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from mlxtend.regressor import StackingCVRegressor

import time


# In[ ]:


alldata = pd.get_dummies(alldata)
alldata = RobustScaler().fit_transform(alldata) 
alldata = PCA(n_components=0.999).fit_transform(alldata) 
train = alldata[:ntrain]
test = alldata[ntrain:]
y_train = n_target.values
print(train.shape)
print(test.shape)


# In[ ]:


n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
    rmse= np.sqrt(-cross_val_score(model, train, target.values, scoring="neg_mean_squared_error", cv = kf))
    return rmse


# In[ ]:


class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
        grid_search = GridSearchCV(self.model,param_grid,cv=kf, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])


# In[ ]:


pd.options.display.float_format = '{:.4f}'.format
grid( Lasso(random_state=1)).grid_get(train,target.values,{'alpha': [0.0004,0.0005,0.0007,0.0009],'max_iter':[10000, 15000]})


# In[ ]:


# grid(lgb.LGBMRegressor(objective='regression',
#                               max_bin = 55, bagging_fraction = 0.8,
#                               bagging_freq = 5, feature_fraction = 0.2319,
#                               feature_fraction_seed=9, bagging_seed=9,
#                               min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)).grid_get(train,target.values,{'num_leaves': [5, 10 ,15],
#                                                                                                                 'learning_rate': [0.05, 0.01, 0.001],
#                                                                                                                'n_estimators': [500, 720, 900]})


# In[ ]:


grid(ElasticNet(random_state=3)).grid_get(train,target.values,{'alpha': [0.0004,0.0005,0.0007,0.001], 'l1_ratio': [0.3, 0.6, 0.9]})


# In[ ]:


grid(KernelRidge()).grid_get(train,target.values,{'alpha': [0.1,0.3,0.6,0.9], 'kernel':['linear','polynomial'], 'degree':[2,3,4], 'coef0':[0.01,2.5,5]})


# In[ ]:


grid(Ridge(random_state=5)).grid_get(train,target.values,{'alpha': [5,10,20,30]})


# In[ ]:


grid(SVR()).grid_get(train,target.values,{'gamma': [0.00001,0.00005,0.0009], 'epsilon': [0.001,0.005,0.009]})


# In[ ]:


models = [ Lasso(alpha =0.0005, random_state=1), 
         ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3), 
         KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5), 
         Ridge(alpha=20), 
         SVR(gamma= 0.0009,kernel='rbf',C=13,epsilon=0.005),
         BayesianRidge(),
         LinearSVR(),
         SGDRegressor(max_iter=1000,tol=1e-3)
         ]
names = ['Lasso', 'ElasticNet', 'KernelRidge', 'Ridge', 'SVR', 'BayesianRidge', 'LinearSVR', 'SGDRegressor']


# In[ ]:


for name, model in zip(names, models):
    start = time.time()
    score = rmsle_cv(model)
    end = time.time()
    print("{}: {:.6f}, {:.4f} in {:.3f} s".format(name,score.mean(),score.std(),end-start))


# In[ ]:


class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self,mod,weight):
        self.mod = mod
        self.weight = weight
        
    def fit(self,X,y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X,y)
        return self
    
    def predict(self,X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]
            w.append(np.sum(single))
        return w


# In[ ]:


sel_models = [Lasso(alpha =0.0005, random_state=1, max_iter=10000), 
             ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3), 
             KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5), 
             Ridge(alpha=20), 
             SVR(gamma= 0.0009,kernel='rbf',C=13,epsilon=0.005),
             BayesianRidge(),
             LinearSVR(),
              ]

 


# In[ ]:


np.random.seed(42)
stack = StackingCVRegressor(regressors=sel_models,
                            meta_regressor=SVR(gamma= 0.0009,kernel='rbf',C=13,epsilon=0.005),
                            use_features_in_secondary=True)
start = time.time()
score = rmsle_cv(stack)
end = time.time()
print("Stacked : {:.6f}, (+/-) {:.4f} in {:.3f} s".format(score.mean(),score.std(),end-start))


# In[ ]:


sel_models = [
             KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5), 
             Ridge(alpha=20), 
             SVR(gamma= 0.0009,kernel='rbf',C=13,epsilon=0.005),
             BayesianRidge(),
             stack
              ]

weights = [0.2,0.15,0.2,0.15,0.3]

start = time.time()
blended = AverageWeight(mod = sel_models ,weight=weights)
score = rmsle_cv(blended)
end = time.time()
print("Weighted avg: {:.6f}, {:.4f} in {:.3f} s".format(score.mean(),score.std(),end-start))


# In[ ]:


blended.fit(train, target.values)


# In[ ]:


pred = np.exp(blended.predict(test))
results = pd.DataFrame({'Id':test_ID, 'SalePrice':pred})
q1 = results['SalePrice'].quantile(0.0042)
q2 = results['SalePrice'].quantile(0.99)

results['SalePrice'] = results['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
results['SalePrice'] = results['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
results.to_csv("submission.csv",index=False)

