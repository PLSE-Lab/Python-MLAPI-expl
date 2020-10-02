#!/usr/bin/env python
# coding: utf-8

#  # A Complete Guide for Regression Problem
#  ## from Feature Engineering to Baseline Modeling
#  
# My goal is to demonstrate several practical strategies and useful visualization graphs to build up regression models including the linear, distance-based, and tree-based ones for the beginners(like me). 
# I'll only do the baseline models, and leave the advanced model or training strategies like early stopping or cross validation for the readers.
# Or I'll elaborate on it when I have free time...
#  
#  ## implementation outlines
#  
#  ### data visualization
# - heatmap
# - barplot
# - distplot
# - probplot
# - regplot
# - pairplot
#  
# ###  feature engineering
# - feature crossing
# - one-hot encoding
# - binning
# - categorizing
# - log transformation
# - outliers removal
# - manipuluate both categoized and numerical data
# - dealing with null value
# 
# ### modeling
# - linear regression
# - distance-based method
#     - using NN in Keras with Tensorflow Backend
# - tree-based method
#     - using XGBoost
# 
# ### training strategies
# - train, test and validation set
# - ~~early stopping~~
# - ~~cross validation~~
# 
# For problems or you may consider I had snippets of your code, please feel to contact me!

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import binned_statistic
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Feature Selection
# - we'll first have a glance at our data
# - we'll pick several features with positive correlation with our target, and do some feature engineering if needed

# In[ ]:


train_csv = pd.read_csv('../input/train.csv')
final_csv = pd.read_csv('../input/test.csv')
# I won't call it test set here, since we're going to create our test set and validation set
# and pick up the strongest model to make prediction in the "test.csv"


# In[ ]:


# first we wanna describe the target columns to make sure we're dealing with regression problem (predict value)
# and we'll look into more detail when we have to do feature engineering
train_csv['SalePrice'].describe()


# In[ ]:


print("How many feature candidates do we have? %d" % (len(train_csv.columns) - 1))


# we'll first narrow down the number of features with 2 criteria:
# 1. not too many null values
# 2. positive correlation with our target, SalcePrice

# In[ ]:


# first we'll visualize null count
null_in_train_csv = train_csv.isnull().sum()
null_in_train_csv = null_in_train_csv[null_in_train_csv > 0]
null_in_train_csv.sort_values(inplace=True)
null_in_train_csv.plot.bar()


# In[ ]:


# visualize correlation map
sns.heatmap(train_csv.corr(), vmax=.8, square=True);


# In[ ]:


arr_train_cor = train_csv.corr()['SalePrice']
idx_train_cor_gt0 = arr_train_cor[arr_train_cor > 0].sort_values(ascending=False).index.tolist()
print("How many feature candidates have positive correlation with SalePrice(including itself)? %d" % len(idx_train_cor_gt0))


# In[ ]:


# we shall list them all, and pick up those we're interested
arr_train_cor[idx_train_cor_gt0]


# I'll only pick a few to further demonstrate feature engineering techniques, at most 10 candidates. 
# After look it up in the [data description](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), here's how I pick
# - *OverallQual* for sure, since its high correlation, and the delegate of categorized data [1-10]
# - *GrLivArea*, also because its high correlation and delegate of numerical data
# - I'll pick *GarageCars* rather than *GarageArea*, it's somehow like binned
# - I've noticed that there are *TotalBsmtSF*, *1stFlrSF*, and *2ndFlrSF*, they seem like related to *GrLivArea*, they're all about the area in the house, I'll do the feature cross techniques with it
# - I'll pick *MasVnrArea* although there're a few(8) null values in it, we can fill with mean or something else reasonable
# - I'll pick *Fireplaces* for another categorized example, since it sounds like a luxuriness indicator
# 
# Here're why I don't pick
# - I'll drop *TotRmsAbvGrd* since it looks like highly related to *GrLivArea*, same applies on *GarageCars* and *GarageArea* cases
# - I'd rather not pick those time series related for simplicity
# 
# So, we have 8 feature candidates now.

# In[ ]:


idx_meta = ['SalePrice','GrLivArea', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'OverallQual', 'Fireplaces', 'GarageCars']
train_meta = train_csv[idx_meta].copy()
train_meta.head(n=5)


# ## Feature Engineering

# In[ ]:


null_in_masvnrarea = train_meta[train_meta['MasVnrArea'].isnull()].index.tolist()
zero_in_masvnrarea = train_meta['MasVnrArea'][train_meta['MasVnrArea'] == 0].index.tolist()
print("How many null value in MasVnrArea? %d / 1460" % len(null_in_masvnrarea))
print("How many zero value in MasVnrArea? %d / 1460" % len(zero_in_masvnrarea))


# In[ ]:


# we'll fill in the null value with 0 from the analysis above
train_meta['MasVnrArea'][null_in_masvnrarea] = 0
print("How many null value in MasVnrArea after filling in null value? %d / 1460" % train_meta['MasVnrArea'].isnull().sum())


# In[ ]:


# overview
sns.pairplot(train_meta)


# we can observe something in the pairplot
# - there're a few outliers
# - we may need to use binning to deal with those zero values in the numerical data
# - *GrLivArea* and *1stFlrSF* relation to *SalePrice* look alike, and that's reasonable
# - we may use log transformation to *SalePrice* and *GrLivArea*, however, it can't be applied to those numerical data with lots of zero

# 

# ### Outlier Removal

# In[ ]:


# GrLivArea
train_meta[(train_meta['GrLivArea'] > 4000) & (train_meta['SalePrice'] < 200000)].index.tolist()


# In[ ]:


# TotalBsmtSF
train_meta[(train_meta['TotalBsmtSF'] > 4000) & (train_meta['SalePrice'] < 200000)].index.tolist()


# In[ ]:


train_meta[(train_meta['1stFlrSF'] > 4000) & (train_meta['SalePrice'] < 200000)].index.tolist()


# In[ ]:


# Thus, we'll remove [523, 1298]
train_clean = train_meta.drop([523,1298])


# ### Categorize

# In[ ]:


nonzero_in_masvnrarea = train_clean['MasVnrArea'][train_clean['MasVnrArea'] != 0].index.tolist()
print("How many non-zero value in MasVnrArea now? %d / 1458" % len(nonzero_in_masvnrarea))


# In[ ]:


# I'll categorize into zero and non-zero
train_clean['has_MasVnrArea'] = 0
train_clean['has_MasVnrArea'][nonzero_in_masvnrarea] = 1


# ### Binning and feature cross
# - *TotalBsmtSF* -> 3
# - *1stFlrSF* -> 4
# - *2ndFlrSF* -> 5

# In[ ]:


train_clean['TotalBsmtSF'][train_clean['TotalBsmtSF'] > 0].describe()


# In[ ]:


bins_totalbsmtsf = [-1, 1, 1004, 4000]
train_clean['binned_TotalBsmtSF'] = np.digitize(train_clean['TotalBsmtSF'], bins_totalbsmtsf)


# In[ ]:


train_clean['1stFlrSF'].describe()


# In[ ]:


bins_1stflrsf = [0, 882, 1086, 1390, 4000]
train_clean['binned_1stFlrSF'] = np.digitize(train_clean['1stFlrSF'], bins_1stflrsf)


# In[ ]:


train_clean['2ndFlrSF'][train_clean['2ndFlrSF'] > 0].describe()


# In[ ]:


bins_2ndflrsf = [-1, 1, 625, 772, 924, 4000]
train_clean['binned_2ndFlrSF'] = np.digitize(train_clean['2ndFlrSF'], bins_2ndflrsf)


# In[ ]:


train_clean['SFcross'] = (train_clean['binned_TotalBsmtSF'] - 1) * (5 * 4) + (train_clean['binned_1stFlrSF'] - 1) * 5 + train_clean['binned_2ndFlrSF']


# ### Log Transformation

# In[ ]:


def draw2by2log(arr):
    fig = plt.figure();
    plt.subplot(2,2,1)
    sns.distplot(arr, fit=norm);
    plt.subplot(2,2,3)
    stats.probplot(arr, plot=plt);
    plt.subplot(2,2,2)
    sns.distplot(np.log(arr), fit=norm);
    plt.subplot(2,2,4)
    stats.probplot(np.log(arr), plot=plt);


# In[ ]:


draw2by2log(train_clean['SalePrice'])


# In[ ]:


draw2by2log(train_clean['GrLivArea'])


# Although it's more "normal" after the log transformation, I didn't get a better result when using it as a feature. I'll still use un-transformed numerical data in data sets.

# ### Training Data Ready for Tree-based Algorithm

# In[ ]:


train_clean.head(n=5)


# In[ ]:


idx_tree = ['SalePrice', 'GrLivArea', 'OverallQual', 'Fireplaces', 'GarageCars', 'has_MasVnrArea', 'SFcross']
train_tree = train_clean[idx_tree]
train_tree.head(n=5)


# In[ ]:


sns.pairplot(train_tree)


# - we can compare with the previous pairplot before feature engineering, we have less but more sophisticated features
# - then, we also need to construct the training data for distance-based algorithm

# ### one-hot encoding (dummy variables)

# In[ ]:


print("Max Fireplaces value in train.csv: %d, in test.csv: %d" % (train_csv['Fireplaces'].max(), final_csv['Fireplaces'].max()) )
print("Min Fireplaces value in train.csv: %d, in test.csv: %d" % (train_csv['Fireplaces'].min(), final_csv['Fireplaces'].min()) )


# In[ ]:


print("Max GarageCars value in train.csv: %d, in test.csv: %d" % (train_csv['GarageCars'].max(), final_csv['GarageCars'].max()) )
print("Min GarageCars value in train.csv: %d, in test.csv: %d" % (train_csv['GarageCars'].min(), final_csv['GarageCars'].min()) )


# In[ ]:


dummy_fields = ['OverallQual', 'Fireplaces', 'GarageCars', 'has_MasVnrArea', 'SFcross']
train_dist = train_tree[['SalePrice', 'OverallQual', 'GrLivArea']].copy()
for field in dummy_fields:
    dummies = pd.get_dummies(train_tree.loc[:, field], prefix=field)
    train_dist = pd.concat([train_dist, dummies], axis = 1)
train_dist['GarageCars_5'] = 0
train_dist['Fireplaces_4'] = 0
train_dist.head(n=5)


# In[ ]:


print("The dimension for the input of distance-based model is %d x %d" % (train_dist.shape[0], train_dist.shape[1] - 1))
# SalePrice is not input, so minus one


# ## Modeling

# ### training, test and validation sets (ignore k-fold cross validation)

# In[ ]:


from sklearn.model_selection import train_test_split
random_state = 7


# In[ ]:


xt_train_test, xt_valid, yt_train_test, yt_valid = train_test_split(train_tree['SalePrice'], train_tree.drop('SalePrice', axis=1), test_size=.2, random_state=random_state)
xd_train_test, xd_valid, yd_train_test, yd_valid = train_test_split(train_dist['SalePrice'], train_dist.drop('SalePrice', axis=1), test_size=.2, random_state=random_state)


# In[ ]:


xt_train, xt_test, yt_train, yt_test = train_test_split(yt_train_test, xt_train_test, test_size=.2, random_state=random_state)
xd_train, xd_test, yd_train, yd_test = train_test_split(yd_train_test, xd_train_test, test_size=.2, random_state=random_state)


# In[ ]:


print("number of training set: %d\nnumber of testing set: %d\nnumber of validation set: %d\ntotal: %d" % (len(xt_train), len(xt_test), len(xt_valid), (len(xt_train)+len(xt_test)+len(xt_valid))))


# In[ ]:


def rmse(arr1, arr2):
    return np.sqrt(np.mean((arr1-arr2)**2))


# ### linear model

# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(xd_train, yd_train)
yd_lm = lm.predict(xd_test)
rmse_linear = rmse(yd_test, yd_lm)
sns.regplot(yd_test, yd_lm)
print("RMSE for Linear Regression Model in sklearn: %.2f" % rmse_linear)


# ### Neural Network in Keras using Tensorflow backend

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


def baseline_nn_model(dims):
    model = Sequential()
    model.add(Dense(dims, input_dim=dims,kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[ ]:


def larger_nn_model(dims):
    model = Sequential()
    model.add(Dense(dims, input_dim=dims,kernel_initializer='normal', activation='relu'))
    model.add(Dense(35, kernel_initializer='normal', activation='relu'))
    model.add(Dense(15, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[ ]:


def use_keras_nn_model(nn_model, x, y, xx, yy, epoch):
    print("start training")
    for step in range(epoch + 1):
        cost = nn_model.train_on_batch(x.as_matrix(), y.as_matrix())
        if step % 100 == 0:
            print("train cost: %.2f" % cost)
    print("start testing")
    yy_predict = nn_model.predict(xx.as_matrix()).reshape(len(yy),)
    res = rmse(yy, yy_predict)
    sns.regplot(yy, yy_predict)
    print("RMSE for NN Model in Keras(Tensorflow): %.2f" % res)
    return res


# In[ ]:


rmse_baselinenn = use_keras_nn_model(baseline_nn_model(xd_train.shape[1]), xd_train, yd_train, xd_test, yd_test, 700)


# In[ ]:


rmse_largernn = use_keras_nn_model(larger_nn_model(xd_train.shape[1]), xd_train, yd_train, xd_test, yd_test, 500)


# In[ ]:


rmse_nn = min(rmse_baselinenn, rmse_largernn)


# ### xgboost

# In[ ]:


import xgboost as xgb
from xgboost import plot_importance


# In[ ]:


params = {
    'booster': 'gbtree',
    'objective': 'reg:gamma',
    'gamma': 0.1,
    'max_depth': 5,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

plst = params.items()
dtrain = xgb.DMatrix(xt_train, yt_train)
dtest = xgb.DMatrix(xt_test)
num_rounds = 500
xgb_model = xgb.train(plst, dtrain, num_rounds)
yt_xgb = xgb_model.predict(dtest)
rmse_xgb = rmse(yt_test, yt_xgb)
sns.regplot(yt_test, yt_xgb)
print("RMSE for xgboost: %.2f" % rmse_xgb)


# In[ ]:


plot_importance(xgb_model)
# it shows that the feature crossing is actually working


# In[ ]:


print("The minimum RMSE goes to: %.2f" % min([rmse_linear, rmse_nn, rmse_xgb]))
# xgboost turns out to be a better model here


# ## Use on final

# ### feature engineering on final_csv

# In[ ]:


idx_clean_final = idx_meta.copy()
idx_clean_final.remove('SalePrice')
final_clean = final_csv[idx_clean_final]
final_clean.head(n=5)


# In[ ]:


final_clean['binned_TotalBsmtSF'] = np.digitize(final_clean['TotalBsmtSF'], bins_totalbsmtsf)
final_clean['binned_1stFlrSF'] = np.digitize(final_clean['1stFlrSF'], bins_1stflrsf)
final_clean['binned_2ndFlrSF'] = np.digitize(final_clean['2ndFlrSF'], bins_2ndflrsf)
final_clean['SFcross'] = (final_clean['binned_TotalBsmtSF'] - 1) * (5 * 4) + (final_clean['binned_1stFlrSF'] - 1) * 5 + final_clean['binned_2ndFlrSF']
final_clean['has_MasVnrArea'] = (final_clean['MasVnrArea'] > 0).astype(float)
final_clean.head(n=5)


# In[ ]:


idx_tree_final = idx_tree.copy()
idx_tree_final.remove('SalePrice')
final_tree = final_clean[idx_tree_final]
final_tree.head(n=5)


# ### test on final data

# In[ ]:


dtest_final = xgb.DMatrix(final_tree)
yt_final = xgb_model.predict(dtest_final)
summission = pd.concat([final_csv['Id'], pd.DataFrame(yt_final)], axis=1)
summission.columns = ['Id', 'SalePrice']


# In[ ]:


sns.distplot(summission['SalePrice'])


# In[ ]:


summission.to_csv('summission.csv', encoding='utf-8', index = False)


# ## Reference

# 1. https://www.kaggle.com/yiidtw/house-price/
# 2. https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# 3. https://www.kaggle.com/dgawlik/house-prices-eda
# 4. http://violin-tao.blogspot.com/2017/05/keras.html
# 5. https://zhuanlan.zhihu.com/p/31182879 
# 6. https://developers.google.com/machine-learning/crash-course/
# 7. https://yiidtw.github.io/blog/2018-06-14-class-note-of-machine-learning-crash-course/
# 8. https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
