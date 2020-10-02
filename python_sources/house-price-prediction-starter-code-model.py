#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction

# ### Introduction

# The dataset provided contains a set of variables that could determine the housing prices in a locality . The challenge here is to build a model which could predict the prices accurately . Its a classic regression problem and the metric to be used here is RMSE score as per the competition page . We have been provided with the train and test dataset to play with . Lets get started.
# 
# **This is my first compeition kernel .The kernel has been created based on my learnings from various top GM's in kaggle .If you have any suggestions for improvement they are welcome .**

# ### Loading the required libraries and dataset 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
import gc
import datetime as dt
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

### Modelling:
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


# Going by the description of the dataset I see that most of the columns should be of categorical datatype . I create a custom dictionary to define the datatypes and read the dataset assigning this datatype.

# In[ ]:


dtypes = {'Id':'int64',
         'MSSubClass':'category',
         'MSZoning':'category',
         'LotFrontage':'object',
         'LotArea':'float64',
         'Street':'category',
         'Alley':'category',
         'LotShape':'category',
         'LandContour':'category',
         'Utilities':'category',
         'LotConfig':'category',
         'LandSlope':'category',
         'Neighborhood':'category',
         'Condition1':'category',
         'Condition2':'category',
         'BldgType':'category',
         'HouseStyle':'category',
         'OverallQual':'category',
         'OverallCond':'category',
         'YearBuilt':'int64',
         'YearRemodAdd':'int64',
         'RoofStyle':'category',
         'RoofMatl':'category',
         'Exterior1st':'category',
         'Exterior2nd':'category',
         'MasVnrType':'category',
         'MasVnrArea':'object',
         'ExterQual':'category',
         'ExterCond':'category',
         'Foundation':'category',
         'BsmtQual':'category',
         'BsmtCond':'category',
         'BsmtExposure':'category',
         'BsmtFinType1':'category',
         'BsmtFinSF1':'float32',
         'BsmtFinType2':'category',
         'BsmtFinSF2':'float32',
         'BsmtUnfSF':'float32',
         'TotalBsmtSF':'float32',
         'Heating':'category',
         'HeatingQC':'category',
         'CentralAir':'category',
         'Electrical':'category',
         '1stFlrSF':'int64',
         '2ndFlrSF':'int64',
         'LowQualFinSF':'float32',
         'GrLivArea':'float32',
         'BsmtFullBath':'float32',
         'BsmtHalfBath':'float32',
         'FullBath':'int64',
         'HalfBath':'int64',
         'BedroomAbvGr':'int64',
         'KitchenAbvGr':'int64',
         'KitchenQual':'category',
         'TotRmsAbvGrd':'int64',
         'Functional':'category',
         'Fireplaces':'int8',
         'FireplaceQu':'category',
         'GarageType':'category',
         'GarageYrBlt':'object',
         'GarageFinish':'category',
         'GarageCars':'float32',
         'GarageArea':'float32',
         'GarageQual':'category',
         'GarageCond':'category',
         'PavedDrive':'category',
         'WoodDeckSF':'float32',
         'OpenPorchSF':'float32',
         'EnclosedPorch':'float32',
         '3SsnPorch':'float32',
         'ScreenPorch':'float16',
         'PoolArea':'float16',
         'PoolQC':'category',
         'Fence':'category',
         'MiscFeature':'category',
         'MiscVal':'float16',
         'MoSold':'float16',
         'YrSold':'float16',
         'SaleType':'category',
         'SaleCondition':'category',
         'SalePrice':'int64'}


# In[ ]:


kaggle=1
if kaggle==0:
    train=pd.read_csv('train.csv',dtype=dtypes)
    test=pd.read_csv('test.csv',dtype=dtypes)
else:
    train=pd.read_csv('../input/train.csv',dtype=dtypes)
    test=pd.read_csv('../input/test.csv',dtype=dtypes)


# In[ ]:


print("Train dataset has {} rows and {} columns".format(train.shape[0],train.shape[1]))


# In[ ]:


print("Test dataset has {} rows and {} columns".format(test.shape[0],test.shape[1]))


# In[ ]:


train.info()


# The predictor variable here is the saleprice . Lets check the distribution of the saleprice in the training dataset.

# ### Exploratory Data Analysis

# In[ ]:


plt.figure(figsize=(8,8))
ax=sns.distplot(train['SalePrice'],color='red')
ax.set_xlabel('SalePrice')
ax.set_title('Distribution of SalePrice')


# The distribution is a perfect normal curve with some outliers in the extremes.Lets check the median value of the sale price with respect to the year built.

# In[ ]:


print("Train data has sale price from staring year {} to ending year {}".format(min(train['YearBuilt']),max(train['YearBuilt'])))


# In[ ]:


print("Test data has sale price from staring year {} to ending year {}".format(min(test['YearBuilt']),max(test['YearBuilt'])))


# In[ ]:


year_sale=train.groupby('YearBuilt')['SalePrice'].median()
year_sale.plot()


# The sale price seem to be showing a fluctuating trend over the year.

# ### Dealing with Null Values

# Lets check if there are any null values in the dataset.

# In[ ]:


train.isnull().values.any()


# In[ ]:


test.isnull().values.any()


# Thus we see that there are columns both in train and test data having null values . The total number of rows present is 1460 . Therefore we consider a threashold of 500 and check the columns having more than 500 null values and remove them.

# In[ ]:


def check_null(df):
    null=df.isnull().sum()
    focus_columns=null[null>0]
    
    return(focus_columns)


# In[ ]:


focus_columns_train=check_null(train)
focus_columns_test=check_null(test)
print("Columns to focus in train dataset:\n {} \nColumns to focus in test dataset:\n {}".format(focus_columns_train,focus_columns_test))


# Thus we see that there are 5 columns in train and test having null values in more than 500 rows . Lets remove them.

# In[ ]:


train_clean=train.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1)
test_clean=test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1)


# Other columns having null values need to be imputed before we build a basic model.Lets check the train dataset first.

# In[ ]:


impute_columns_train=focus_columns_train[focus_columns_train<500]
print("There are {} columns to be imputed \n The columns are \n{}".format(len(impute_columns_train),impute_columns_train.index))


# Lets check the description of these columns to decide on the imputing criteria.

# In[ ]:


train_clean[['LotFrontage', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical',
       'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual',
       'GarageCond']].info()


# Only 3 variables are numerical variables while others are categorical.We try to impute the numeric missing values with median values in that column.

# In[ ]:


### Inspited from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
###https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44341-2
for col in ['LotFrontage','MasVnrArea','GarageYrBlt']:
    train_clean[col]=train_clean[col].fillna(train_clean[col].median()).astype('float16')


# All the other categorical variables except electrical can be imputed with NA since in the data description there is a separate category 'NA'.

# In[ ]:


for col in ['GarageType','GarageFinish','GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType']:
    train_clean[col]=train_clean[col].cat.add_categories('NA')
    train_clean[col]=train_clean[col].fillna('NA')


# Now for electrical , lets check the cardinals and impute the missing value with most frequent cardinality

# In[ ]:


train_clean['Electrical'].value_counts()


# Standard Circuit Breakers & Romex is the type most used for the electrical system and hence we impute that value.

# In[ ]:


train_clean['Electrical']=train_clean['Electrical'].fillna('SBrkr')


# Now lets check if there are any more null values.

# In[ ]:


del impute_columns_train
del focus_columns_train
train_clean.isnull().values.any()


# Now lets take the test dataset and impute the missing values.

# In[ ]:


impute_columns_test=focus_columns_test[focus_columns_test<500]
print("There are {} columns to be imputed \n The columns are \n{}".format(len(impute_columns_test),impute_columns_test.index))


# In[ ]:


test_clean[['MSZoning', 'LotFrontage', 'Utilities', 'Exterior1st', 'Exterior2nd',
       'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
       'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual',
       'Functional', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
       'GarageArea', 'GarageQual', 'GarageCond', 'SaleType']].info()


# In[ ]:


for col in ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea','LotFrontage','MasVnrArea','GarageYrBlt']:
    test_clean[col]=test_clean[col].fillna(test_clean[col].median()).astype('float16')


# In[ ]:


for col in ['GarageType','GarageFinish','GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType']:
    test_clean[col]=test_clean[col].cat.add_categories('NA')
    test_clean[col]=test_clean[col].fillna('NA')


# The other columns - Exterior1st,Exterior2nd,Utilities,MSZoning,saletype,functional and Kitchen Qual do have have inherent NA value and hence they will be imputed with maximum value.

# In[ ]:


## https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn

for col in ['Exterior1st','Exterior2nd','Utilities','MSZoning','SaleType','Functional','KitchenQual']:
    print(f'Imputing col {col} with {test_clean[col].value_counts().index[0]}')
    test_clean[col]=test_clean[col].fillna(test_clean[col].value_counts().index[0])


# Check if there are any more missing values,

# In[ ]:


test_clean.isnull().values.any()


# In[ ]:


#test_clean.info()


# In[ ]:


del train
del test


# Before building our model , it is necessary to encode the categorical variables. Lets concatenate the train and test set for encoding.

# In[ ]:



ntrain=train_clean.shape[0]
all_df=pd.concat([train_clean,test_clean])


# In[ ]:


for col in ['Condition2', 'Electrical', 'Exterior1st', 'Exterior2nd', 'GarageQual','Heating', 'HouseStyle', 'RoofMatl', 'Utilities']:
    print(f'Converting {col} from object to category datatype')
    all_df[col]=all_df[col].astype('category')


# In[ ]:


categories = all_df.select_dtypes('category')


# Lets check the cardinality of each column,

# In[ ]:


for  cat in categories:
    print(f'\n{cat} has {all_df[cat].nunique()} categories')


# Defining a function for encoding the categorical columns,

# In[ ]:


def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype.name == 'category']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Now we encode the columns,

# In[ ]:


all_df,oh_new_cols=one_hot_encoder(all_df)


# In[ ]:


all_df.info()


# In[ ]:


all_df.head()


# Thus all the columns have been encoded.Lets split the train and test back.

# In[ ]:


train=all_df[:ntrain]
train.shape


# As a precaution , lets check if there are any more null values,

# In[ ]:


train.isnull().values.any()


# In[ ]:


test=all_df[ntrain:]
test.shape


# In[ ]:


test=test.drop(['SalePrice'],axis=1)


# In[ ]:


test.isnull().values.any()


# In[ ]:


test.head()


# ### Basic Modelling 

# The metric used for modelling is RMSLE score .Lets create a function to calculate the score.

# In[ ]:


def rmsle(x,y): return math.sqrt(((np.log1p(x)-np.log1p(y))**2).mean())


# We notice that the train data has features of houses built between 1872 to 2010 whereas the test data has houses built between 1879 till 2010. Therefore an effective train-valid-test split would be to split based on this time based feature since it is a common fact that the house prices tend to depreciate as the house gets older and older.We take  1000 rows as train data and 460 rows as validation set.

# In[ ]:


train.sort_values('YearBuilt',inplace=True)


# In[ ]:


X=train.drop(['SalePrice'],axis=1)
y=train['SalePrice']


# In[ ]:


from sklearn.model_selection import KFold, StratifiedKFold


# In[ ]:


num_folds=5


# In[ ]:


folds = KFold(n_splits= num_folds, shuffle=True, random_state=1054)


# We build a random forest model with the dataset and check the baseline accuracy.

# In[ ]:


oof_preds = np.zeros(train.shape[0])
feature_importance_df = pd.DataFrame()
sub_preds = np.zeros(test.shape[0])


# In[ ]:


feat=[f for f in X.columns if f in test.columns]


# In[ ]:


##https://www.kaggle.com/tunguz/xgb-simple-features
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X,y)):
        
            train_x, train_y = X[feat].iloc[train_idx], y.iloc[train_idx]
            valid_x, valid_y = X[feat].iloc[valid_idx], y.iloc[valid_idx]
            rfm = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True,random_state=100)
            rfm.fit(train_x, train_y)
            

            oof_preds[valid_idx] = rfm.predict(valid_x)[:1]
            sub_preds += rfm.predict(test[feat]) / folds.n_splits # - Uncomment for K-fold 

            fold_importance_df = pd.DataFrame()
            fold_importance_df["cols"] = X.columns
            fold_importance_df["importance"] = rfm.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmsle(valid_y, oof_preds[valid_idx])))
            del rfm, train_x, train_y, valid_x, valid_y
            gc.collect()


# In[ ]:


pred_df = pd.DataFrame(sub_preds, index=test["Id"], columns=["SalePrice"])
pred_df.to_csv('output_RF1.csv', header=True, index_label='Id')


# The model scored 0.15915 on LB . Lets try to improve our model.

# ### Improving our model

# One reason we are getting a high RMSLE error is due to presence of all the features .Lets build a RF model and check the permutation importance to know which features are useful in predicting the sale price.

# In[ ]:


from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance
import lightgbm as lgb


# In[ ]:


## From fast.ai library,
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


# We can now split the train and validation data without a time based split as we did earlier.

# In[ ]:


train_x,valid_x,train_y,valid_y=train_test_split(X,y,test_size=0.2,random_state=100)


# In[ ]:


print(f'Shape of train_x is {train_x.shape} Shape of train_y is {train_y.shape} Shape of valid_x is {valid_x.shape} Shape of valid_y is {valid_y.shape}')


# In[ ]:


rfm = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True,random_state=100)


# In[ ]:


rfm.fit(train_x,train_y)


# In[ ]:


print(f"RMSLE of the training model {rmsle(train_y,rfm.predict(train_x))} \n RMSLE of the validation model {rmsle(valid_y,rfm.predict(valid_x))}")


# From the RMSLE scores , we understand that the model has been overfit since the training and validation scores difference is more.

# In[ ]:


perm = PermutationImportance(rfm, random_state=100).fit(valid_x,valid_y)
eli5.show_weights(perm, feature_names = valid_x.columns.tolist())


# From the permutation importance we see that GrLivArea (Above grade (ground) living area square feet) seems to be most important in predicting the sale price followed by garage cars and TotalBsmtSF . It is interesting to note that Fireplaces is also noted as important feature.The external material quality of average/typical is also rated as important factor.Lets take the weights till 0.0115 and build our model .Lets check if there is any improvement in our predictions.

# In[ ]:


to_keep=['GrLivArea','GarageArea','YearBuilt','TotalBsmtSF','GarageCars']


# In[ ]:


X_keep=X[to_keep].copy()


# In[ ]:


train_x,valid_x,train_y,valid_y=train_test_split(X_keep,y,test_size=0.2)


# In[ ]:


print(f'Shape of train_x is {train_x.shape} Shape of train_y is {train_y.shape} Shape of valid_x is {valid_x.shape} Shape of valid_y is {valid_y.shape}')


# Lets try random forest model without cv again.

# In[ ]:


rfm = RandomForestRegressor(n_estimators=30, min_samples_leaf=2, max_features=0.5, n_jobs=-1, oob_score=True,random_state=100)


# In[ ]:


rfm.fit(train_x,train_y)


# In[ ]:


oof_rfm=rfm.predict(valid_x)


# In[ ]:


print(f"RMSLE of the training model {rmsle(train_y,rfm.predict(train_x))} \n RMSLE of the validation model {rmsle(valid_y,oof_rfm)}")


# The RMSLE is quite high .Lets try the lightgbm and see if there is any improvement in our validation scores.

# In[ ]:


param = {'num_leaves': 120,
         'min_data_in_leaf': 3, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.008,
         "min_child_samples": 5,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 2,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}


# In[ ]:


train_lgb=lgb.Dataset(train_x,label=train_y)
valid_lgb=lgb.Dataset(valid_x,label=valid_y)


# In[ ]:


num_round=5000
clf = lgb.train(param, train_lgb, num_round,valid_sets=[train_lgb,valid_lgb], verbose_eval=100,early_stopping_rounds=200)
oof_lgb= clf.predict(valid_x, num_iteration=clf.best_iteration)


# In[ ]:


print(f"RMSLE of the model {rmsle(valid_y,oof_lgb)}")


# In[ ]:


predictions=clf.predict(test[to_keep])


# In[ ]:


pred_df = pd.DataFrame(predictions, index=test["Id"], columns=["SalePrice"])


# In[ ]:


pred_df.head()


# Lets submit our model,

# In[ ]:


pred_df.to_csv('output_lgb.csv', header=True, index_label='Id')


# ### Reference

# 1.https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-zillow-prize
# 2.https://www.kaggle.com/humananalog/xgboost-lasso
# 3.https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# 4.https://www.kaggle.com/tunguz/xgb-simple-features
# 5.https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44341-2
