#!/usr/bin/env python
# coding: utf-8

# Here is more or less what our best single model was. \n (our best submission was a blend of different runs.)
# 
# 
# We fight a lot to get a correct Cross Validation scheme and finally Oleg got something while removing almost all features and using a GroupKfold by description which actually acts as proxy to the sku_hash.
# 
# 
# We forgot to include back the TFIDF and PCA pictures features in our submission this is what cause the difference in score between this kernel and our LB, too bad for the 0.00024 difference between the 5th rank...
# 
# 
# Usefull features were the quantity, price, real month and result of linear least-squares regression of quantity for the 7 days available . 
# 
# 
# Thanks again to Lu and Oleg and also to the organisers and mentors .
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import skew, kurtosis,linregress
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import   GroupKFold
import lightgbm as lgb
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Calculate a linear least-squares regression for the values of the time series  
def sales_features(x):
    x=np.array(x)
    linReg = linregress(range(len(x)), x)
    return  getattr(linReg, 'rvalue') , getattr(linReg, 'intercept') , getattr(linReg, 'slope') , getattr(linReg, 'stderr') , kurtosis(x) , skew(x)

#just to expand the series to df      
def expand_series_to_df(df):
    extra_features = [col for col in df.columns if df[col].dtype == 'object' and 'sku' not in col]
    for col in extra_features :
        tags = df[col].apply(pd.Series)
        tags = tags.rename(columns = lambda x : 'extra_' + col + '_'+ str(x))
        df = pd.concat([df[:], tags[:]], axis=1)
        del df[col]
    return df

#to plot the feature importange of LGBM by gain
def plot_importances(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    plt.figure(figsize=(8, 18))
    data_imp = importances_.sort_values('mean_gain', ascending=False)
    sns.barplot(x='gain', y='feature', data=data_imp[:2000])
    plt.tight_layout()
    plt.savefig('importances.png')
    plt.show()
    
#train LGBM with GroupKF
def train_classifiers(df=None, y=None, feature_to_keep=None):
    folds = GroupKFold(n_splits=5)
    clfs = []
    importances = pd.DataFrame()
    oof_preds =np.zeros(df.shape[0])

    for fold_, (trn_, val_) in enumerate(folds.split(df, y, df['en_US_description'].values)):
        full_train = df[feature_to_keep].copy()
        print(full_train.shape)
        trn_x, trn_y = full_train.loc[trn_], y[trn_]
        val_x, val_y = full_train.loc[val_], y[val_]

        clf = lgb.LGBMRegressor(boosting_type='gbdt', 
                                class_weight=None, 
                                colsample_bytree=0.9, 
                                learning_rate=0.01, 
                                max_depth=5, 
                                metric='rmse',
                                min_child_samples=20, 
                                min_child_weight=10, 
                                min_split_gain=0.01,
                                n_estimators=10000, 
                                n_jobs=-1, num_leaves=31,
                                objective='regression', 
                                random_state=None, 
                                reg_alpha=0.01,
                                reg_lambda=0.01, silent=-1, 
                                subsample=0.9, 
                                subsample_for_bin=200000,
                                subsample_freq=1, 
                                verbose=-1)
        print(clf)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric='rmse',
            verbose=100,
            early_stopping_rounds=100
        )
        oof_preds[val_] = clf.predict(val_x, num_iteration=clf.best_iteration_)
        
        imp_df = pd.DataFrame()
        imp_df['feature'] = full_train.columns
        imp_df['gain'] = clf.feature_importances_
        imp_df['fold'] = fold_ + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)
        clfs.append(clf)
    
    err = sqrt(mean_squared_error(y, oof_preds))
    print('{} LGBM RMSE: {}'.format(cnt + 1, err))
    
    return clfs, importances, err, oof_preds


# In[ ]:


data_path = '../input/kaggledays-paris/'
train = pd.read_csv(data_path + 'train.csv',encoding='UTF-8')
test = pd.read_csv(data_path + 'test.csv',encoding='UTF-8')

colorfile = pd.read_csv('../input/color-feature/color_transcode2.csv',encoding='UTF-8')
train = train.merge(colorfile, on='color', how='left')
test  = test.merge(colorfile,  on='color', how='left')

train['color'] = train['new_color']
test['color']  = test['new_color']

print(train.shape, test.shape)
print(train.head())

train['is_test'] = 0
test['is_test'] = 1
df_all = pd.concat([train, test], axis=0)
print('train ',train.shape)
print('test ',test.shape)


# In[ ]:


train.head()


# In[ ]:


#we Preprocess all the en_US_description with tfidf
print('Preprocessing text...')
tfidf = TfidfVectorizer(max_features=5,norm='l2',)
tfidf.fit(df_all[ 'en_US_description'].astype(str).values)
tfidf_all = np.array(tfidf.transform(df_all[ 'en_US_description'].astype(str).values).toarray(), dtype=np.float16)

for i in tqdm(range(5)):
        df_all['en_US_description_tfidf_' + str(i)] = tfidf_all[:, i]
        
del tfidf, tfidf_all
gc.collect()

print('Done.')
print('train ',train.shape)
print('test ',test.shape)


# In[ ]:


#we label encode all the categorial feature but at the end we didnt use all of them
print('Label Encoder...')
cols = [f_ for f_ in df_all.columns if df_all[f_].dtype == 'object' and 'sku' not in f_ and 'ID' not in f_]
print(cols)
cnt = 0
for c in tqdm(cols):
        le = LabelEncoder()
        df_all[c] = le.fit_transform(df_all[c].astype(str))
        cnt += 1

        del le
print('len(cols) = {}'.format(cnt))

train = df_all.loc[df_all['is_test'] == 0].drop(['is_test'], axis=1)
test = df_all.loc[df_all['is_test'] == 1].drop(['is_test'], axis=1)

print('train ',train.shape)
print('test ',test.shape)


# In[ ]:


# the images features = PCA  on the resnet embedings
vimages = pd.read_csv(data_path + 'vimages.csv')

pca = PCA(n_components=10)
vpca = pca.fit_transform(vimages.drop('sku_hash', axis=1).values)
vimages_pca = vimages[['sku_hash']]
for i in tqdm(range(vpca.shape[1])):
    vimages_pca['dim_pca_{}'.format(i)] = vpca[:, i]
    
train = train.merge(vimages_pca, on='sku_hash', how='left')
test = test.merge(vimages_pca, on='sku_hash', how='left')
print('train ',train.shape)
print('test ',test.shape)


# Sales features are based on the sum of quantity by product by date , and also on the sum of quantity by model  by date

# In[ ]:


sales = pd.read_csv(data_path + 'sales.csv')
sales = sales.merge(df_all[['sku_hash','model','function']], on='sku_hash', how='left')
sales = sales.sort_values('Date', ascending=True)
salesg = sales.groupby('sku_hash').agg({
    'sales_quantity': ['sum', 'mean'],
    'Month_transaction': ['last'],
     }).reset_index()
salesg.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in salesg.columns]
print(salesg.head())

train = train.merge(salesg, on='sku_hash', how='left')
test  = test.merge(salesg, on='sku_hash', how='left')

aggs={}
aggs['sales_quantity']= {'sum':'sum'}

## by item 
agg_tmp = sales.groupby(['sku_hash','Date']).agg(aggs)
agg_tmp.columns = [ '{}_{}'.format(k, agg) for k in aggs.keys() for agg in aggs[k]]
agg_tmp=agg_tmp.reset_index()

agg_tmp = agg_tmp.sort_values('Date', ascending=True)

XX_sales =  agg_tmp.groupby(['sku_hash'])['sales_quantity_sum'].apply(lambda x: sales_features(x))
XX_sales  = expand_series_to_df(XX_sales.reset_index())
XX_sales.columns = ['sku_hash', 
                    'sales_quantity_sum_linear_trend_rvalue',
                    'sales_quantity_sum_linear_trend_intercept',
                    'sales_quantity_sum_linear_trend_attr_slope',
                    'sales_quantity_sum_linear_trend_attr_stderr',
                    'sales_quantity_sum_kurtosis',
                    'sales_quantity_sum_skewness'
                    ]

train = train.merge(XX_sales, on='sku_hash', how='left')
test = test.merge(XX_sales,   on='sku_hash', how='left')
print('train ',train.shape)
print('test ',test.shape)


# In[ ]:


# by model
agg_tmp = sales.groupby(['model','Date']).agg(aggs)
agg_tmp.columns = [ '{}_{}'.format(k, agg) for k in aggs.keys() for agg in aggs[k]]
agg_tmp=agg_tmp.reset_index()
agg_tmp = agg_tmp.sort_values('Date', ascending=True)
XX_sales_BYMODEL =  agg_tmp.groupby(['model'])['sales_quantity_sum'].apply(lambda x: sales_features(x))
XX_sales_BYMODEL  = expand_series_to_df(XX_sales_BYMODEL.reset_index())
XX_sales_BYMODEL.columns = ['model', 
                    'BYMODELsales_quantity_sum_linear_trend_rvalue',
                    'BYMODELsales_quantity_sum_linear_trend_intercept',
                    'BYMODELsales_quantity_sum_linear_trend_attr_slope',
                    'BYMODELsales_quantity_sum_linear_trend_attr_stderr',
                    'BYMODELsales_quantity_sum_kurtosis',
                    'BYMODELsales_quantity_sum_skewness'
                    ]

train = train.merge(XX_sales_BYMODEL, on='model', how='left')
test  = test.merge(XX_sales_BYMODEL,  on='model', how='left')
print('train ',train.shape)
print('test ',test.shape)


# In[ ]:


def extract_features(df):
    df['month_real'] = (df['month']+df['Month_transaction_last']).apply(lambda x: x%12)
    pass

print('Extracting month_real features for train:')
extract_features(train)
print('Extracting month_real features for test:')
extract_features(test)
print('train ',train.shape)
print('test ',test.shape)


# In[ ]:


train.columns


# In[ ]:


y = np.log1p(train['target'].values)

# the list of feature to keep 
f = [
     'en_US_description_tfidf_0', 'en_US_description_tfidf_1', 'en_US_description_tfidf_2', 'en_US_description_tfidf_3', 'en_US_description_tfidf_4',
     'dim_pca_0', 'dim_pca_1', 'dim_pca_2', 'dim_pca_3','dim_pca_4',
      'dim_pca_5', 'dim_pca_6', 'dim_pca_7', 'dim_pca_8','dim_pca_9',

    'sales_quantity_sum',
    'sales_quantity_mean',
    # 'sales_quantity_std',
    # 'last_sales_quantity',
    # 'addtocart_sum',
    'fr_FR_price',
    'product_type',
    'month',
    'month_real',
    'product_gender',
    'macro_function',
    'function',
    # 'sub_function',
    #'model',
    'macro_material',
    'color',
    # 'en_US_description',
    'Month_transaction_last',
    'sales_quantity_sum_linear_trend_rvalue',
    'sales_quantity_sum_linear_trend_intercept',
    'sales_quantity_sum_linear_trend_attr_slope',
    'sales_quantity_sum_linear_trend_attr_stderr',
    'sales_quantity_sum_kurtosis',
    'sales_quantity_sum_skewness',
     'BYMODELsales_quantity_sum_linear_trend_rvalue',
     'BYMODELsales_quantity_sum_linear_trend_intercept',
     'BYMODELsales_quantity_sum_linear_trend_attr_slope',
     'BYMODELsales_quantity_sum_linear_trend_attr_stderr',
     'BYMODELsales_quantity_sum_kurtosis',
     'BYMODELsales_quantity_sum_skewness',
       ]


# In[ ]:


clfs, importances, err, oof_predict = train_classifiers(train,y,f)


# In[ ]:


plot_importances(importances)


# In[ ]:


preds_ = None
for clf in clfs:
        if preds_ is None:
            preds_  = clf.predict(test[f], num_iteration=clf.best_iteration_)
        else:
            preds_ += clf.predict(test[f], num_iteration=clf.best_iteration_)
            
preds_ = preds_ / len(clfs)
preds_ = np.expm1(preds_)

# Prepare submission
subm = pd.DataFrame()
subm['ID'] = test['ID'].values
subm['target'] = preds_
subm.to_csv('submission{}.csv'.format(err), index=False) 

