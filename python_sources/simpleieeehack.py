#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix
from sklearn import model_selection  
import lightgbm as lgb
import time
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# In[ ]:


# Transaction CSVs
train_identity= pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")
train_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")
test_transaction = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")
test_identity= pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")


# In[ ]:


train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
#Keep every row in the left dataframe.
test  = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')


# In[ ]:


train.head()


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


# This Step is done to reduce memory by conversion
# convert it to the low memory to fit the RAM
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            #Downsizing 
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    #iinfo-Machine limits for integer datatype
                    df[col] = df[col].astype(np.int8)
                    #Casting pandas object to a certain type-to int8,int16,int32 &int64 depending on size
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
# Using  above function to reduce memory usage for Train test df
train_df=reduce_mem_usage(train)
test_df=reduce_mem_usage(test)


# In[ ]:


print("Train shape: ", train.shape)
print("Test shape:", test.shape)


# In[ ]:


# Explore Categorical features
print('Training set:')
for col_name in train_df.columns:
    if train_df[col_name].dtypes == 'object' :
        unique_cat = len(train_df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

print("****************")
    
print('Test set:')
for col_name in test_df.columns:
    if test_df[col_name].dtypes == 'object' :
        unique_cat = len(test_df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))


# In[ ]:


cat_cols = [col for col in train_df.columns if train_df[col].dtype in ['object']]
cat_cols


# In[ ]:


v_feat = train_df.columns[55:394]
v_feat


# In[ ]:


v_test_feat=test_df.columns[54:393]


# In[ ]:


v_test_feat


# In[ ]:


from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA


# In[ ]:


#Replacing NULL values
for col in v_feat:
    train_df[col].fillna((train_df[col].min() - 2), inplace=True)
    train_df[col] = (minmax_scale(train_df[col], feature_range=(0,1)))


# In[ ]:


pca = PCA(n_components=35,random_state=5)
principalComponents = pca.fit_transform(train_df[v_feat])


# In[ ]:


principalComponents


# In[ ]:


principalDf = pd.DataFrame(principalComponents)


# In[ ]:


principalDf.head()


# In[ ]:


train_df.drop(v_feat,axis=1,inplace=True)


# In[ ]:


train_df.shape


# In[ ]:


principalDf.rename(columns=lambda x: str('V')+str(x), inplace=True)


# In[ ]:


principalDf.head()


# In[ ]:


train_df = pd.concat([train_df, principalDf], axis=1)


# In[ ]:


train_df.shape


# In[ ]:


#Replacing NULL values for test data
for col in v_test_feat:
    test_df[col].fillna((test_df[col].min() - 2), inplace=True)
    test_df[col] = (minmax_scale(test_df[col], feature_range=(0,1)))


# In[ ]:


pca = PCA(n_components=35,random_state=5)
pComponents = pca.fit_transform(test_df[v_test_feat])


# In[ ]:


test_principalDf = pd.DataFrame(pComponents)


# In[ ]:


test_df.drop(v_test_feat,axis=1,inplace=True)


# In[ ]:


test_principalDf.rename(columns=lambda x: str('V')+str(x), inplace=True)


# In[ ]:


test_df = pd.concat([test_df, test_principalDf], axis=1)


# In[ ]:


test_df.shape


# In[ ]:


train_df['DeviceInfo'].describe()


# In[ ]:


#Frequency Table-CrossTab
data_freq=pd.crosstab(index=train_df['DeviceInfo'], columns="count")    


# In[ ]:


data_freq=data_freq[data_freq['count']>350]


# In[ ]:


data_freq.shape


# In[ ]:


data_freq


# In[ ]:


def change_value_Dev_Info(x) :
    if x in ['ALE-L23 Build/HuaweiALE-L23', 'MacOS', 'Trident/7.0' ,'Windows','iOS Device','rv:11.0	','rv:57.0','rv:59.0','SM-G531H Build/LMY48B','SM-G610M Build/MMB29K','SM-J700M Build/MMB29K'] :
        return 0
    else :
        return 1
    
train_df.loc[:,'DeviceInfo'] = train_df['DeviceInfo'].apply(lambda x : change_value_Dev_Info(x))
test_df.loc[:,'DeviceInfo'] = test_df['DeviceInfo'].apply(lambda x : change_value_Dev_Info(x))


# In[ ]:


print(train_df.shape)
print(test_df.shape)


# In[ ]:


def change_value_P_emaildomain(x) :
    if x in ['gmail.com', 'icloud.com', 'mail.com' , 'outlook.es', 'protonmail.com'] :
        return x
    else :
        return 'etc'
    
train_df.loc[:,'P_emaildomain'] = train_df['P_emaildomain'].apply(lambda x : change_value_P_emaildomain(x))
test_df.loc[:,'P_emaildomain'] = test_df['P_emaildomain'].apply(lambda x : change_value_P_emaildomain(x))


# In[ ]:


def change_value_R_emaildomain(x) :
    if x in ['gmail.com', 'icloud.com', 'mail.com' , 'outlook.es', 'protonmail.com'] :
        return x
    else :
        return 'etc'
    
train_df.loc[:,'R_emaildomain'] = train_df['R_emaildomain'].apply(lambda x : change_value_P_emaildomain(x))
test_df.loc[:,'R_emaildomain'] = test_df['R_emaildomain'].apply(lambda x : change_value_P_emaildomain(x))


# In[ ]:


data_news=pd.crosstab(index=train_df['DeviceType'], columns="count")    


# In[ ]:


data_news


# In[ ]:


def change_value_Dev_Type(x) :
    if x in ['desktop', 'mobile'] :
        return 0
    else :
        return 1
    
train_df.loc[:,'DeviceType'] = train_df['DeviceType'].apply(lambda x : change_value_Dev_Type(x))
test_df.loc[:,'DeviceType'] = test_df['DeviceType'].apply(lambda x : change_value_Dev_Type(x))


# In[ ]:


#Replacing NULL values in Device Info and Device Type with 0 i.e-Unrecognized Activity
train_df[['DeviceType', 'DeviceInfo']] = train_df[['DeviceType','DeviceInfo']].fillna(value=0)
test_df[['DeviceType', 'DeviceInfo']] = test_df[['DeviceType','DeviceInfo']].fillna(value=0)


# In[ ]:


m_feat = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

for col in m_feat:
    train_df[col].fillna('None', inplace=True)
    test_df[col].fillna('None',inplace=True)


# In[ ]:


train_df[m_feat].dtypes


# In[ ]:


train_df[m_feat].describe()


# In[ ]:


d_feat =['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11','D12', 'D13', 'D14', 'D15']


# In[ ]:


train_df[d_feat].describe()


# In[ ]:


train_df[d_feat].isnull().sum()


# In[ ]:


train_df[d_feat].describe()


# In[ ]:


for col in d_feat:
    train_df[col] = (minmax_scale(train_df[col], feature_range=(0,1)))
    train_df[col] = train_df[col].fillna(-1)


# In[ ]:


for col in d_feat:
    test_df[col] = (minmax_scale(test_df[col], feature_range=(0,1)))
    test_df[col] = test_df[col].fillna(-1)


# In[ ]:


print(train_df.shape)
print(test_df.shape)


# In[ ]:


c_feat = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7','C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']

for col in c_feat:
    train_df[col] = train_df[col].fillna((train_df[col].min() - 1))
    train_df[col] = (minmax_scale(train_df[col], feature_range=(0,1)))


# In[ ]:


for col in c_feat:
    test_df[col] = test_df[col].fillna((test_df[col].min() - 1))
    test_df[col] = (minmax_scale(test_df[col], feature_range=(0,1)))


# In[ ]:


id_cols=['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08',
       'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',
       'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',
       'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
       'id_33', 'id_34','id_35','id_36','id_37','id_38']


# In[ ]:


for col in id_cols:
     print("Feature '{col_name}' has {unique_cat} ".format(col_name=col, unique_cat=train_df[col].dtype))


# In[ ]:


cat_id_cols = [col for col in id_cols if train_df[col].dtype in ['object']]

cat_id_cols


# In[ ]:


test_cat_id=[col for col in id_cols if test_df[col].dtype in ['object']]
test_cat_id


# In[ ]:


tr_cat=train_df[cat_id_cols]
tr_cat.isnull().sum()


# In[ ]:


train_df.drop(cat_id_cols,axis=1,inplace=True)
test_df.drop(cat_id_cols,axis=1,inplace=True)


# In[ ]:


id_cols=train_df.columns[55:78]
id_cols


# In[ ]:


for col in id_cols:
    train_df[col] = (minmax_scale(train_df[col], feature_range=(0,1)))
    train_df[col].fillna(-1, inplace=True)


# In[ ]:


test_id_cols=test_df.columns[54:77]
test_id_cols


# In[ ]:


for col in test_id_cols:
    test_df[col] = (minmax_scale(test_df[col], feature_range=(0,1)))
    test_df[col].fillna(-1, inplace=True)


# In[ ]:


train_df['addr1'].describe()


# In[ ]:


train_df['addr1'] = (minmax_scale(train_df['addr1'], feature_range=(0,1)))
train_df['addr1'].fillna((train_df['addr1'].max() - 200), inplace=True)


# In[ ]:


test_df['addr1'].describe()


# In[ ]:


test_df['addr1'] = (minmax_scale(test_df['addr1'], feature_range=(0,1)))
test_df['addr1'].fillna((test_df['addr1'].max() - 200), inplace=True)


# In[ ]:


train_df['addr2'].describe()


# In[ ]:


train_df['addr2'].fillna(87,inplace=True)


# In[ ]:


test_df['addr2'].describe()


# In[ ]:


test_df['addr2'].fillna(87,inplace=True)


# In[ ]:


print(train_df.shape)
print(test_df.shape)


# In[ ]:


card_feat=['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
card_feat


# In[ ]:


for col in card_feat:
     print("Feature '{col_name}' has {unique_cat} ".format(col_name=col, unique_cat=train_df[col].dtype))


# In[ ]:


train_cat_card=[col for col in card_feat if train_df[col].dtype in ['object']]
train_cat_card


# In[ ]:


num_card=[col for col in card_feat if train_df[col].dtype not in ['object']]
num_card


# In[ ]:


card_types=pd.crosstab(index=train_df['card4'], columns="count")    


# In[ ]:


card_types


# In[ ]:


new_card_types=pd.crosstab(index=train_df['card6'], columns="count")   


# In[ ]:


new_card_types


# In[ ]:


for col in train_cat_card:
    train_df[col] = train_df[col].fillna('None')


# In[ ]:


for col in train_cat_card:
    test_df[col] = test_df[col].fillna('None')


# In[ ]:


for col in num_card:
    train_df[col] = train_df[col].fillna((train_df[col].min() - 1))
    train_df[col] = (minmax_scale(train_df[col], feature_range=(0,1)))


# In[ ]:


for col in num_card:
    test_df[col] = test_df[col].fillna((test_df[col].min() - 1))
    test_df[col] = (minmax_scale(test_df[col], feature_range=(0,1)))


# In[ ]:


# Check missing data - Many Columns have more than 50% NA/Null records
def missing_data(df) :
    count = df.isnull().sum()
    percent = (df.isnull().sum()) / (df.isnull().count()) * 100
    total = pd.concat([count, percent], axis=1, keys = ['Count', 'Percent'])
    types = []
    for col in df.columns :
        dtypes = str(df[col].dtype)
        types.append(dtypes)
    total['dtypes'] = types
    
    return np.transpose(total)

total=missing_data(train_df)
print(total)


# In[ ]:


## WE will Drop columns with more 50% Null value.
null_percent = train_df.isnull().sum()/train_df.shape[0]*100

cols_to_drop = np.array(null_percent[null_percent > 50].index)

cols_to_drop


# In[ ]:


# Drop Columns (cols_to_drop) from train and test
train_df = train_df.drop(cols_to_drop, axis=1)
test_df = test_df.drop(cols_to_drop,axis=1)


# In[ ]:


train_df.fillna(-999)
test_df.fillna(-999)


# In[ ]:


train_y = train_df['isFraud']
train_X = train_df.drop('isFraud', axis=1)


# In[ ]:


print(train_X.shape)
print(test_df.shape)


# In[ ]:


null_columns=train_df.columns[train_df.isnull().any()]
train_df[null_columns].isnull().sum()


# In[ ]:


null_columns=test_df.columns[test_df.isnull().any()]
test_df[null_columns].isnull().sum()


# In[ ]:


# Label Encoding for categorical variables.
for f in train_X.columns:
    if train_X[f].dtype=='object' or test_df[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_X[f].values) + list(test_df[f].values))
        train_X[f] = lbl.transform(list(train_X[f].values))
        test_df[f] = lbl.transform(list(test_df[f].values))


# In[ ]:


## Create Base Model - LogisticRegression

logreg = LogisticRegression()
logreg.fit(train_X, train_y)


# In[ ]:


submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv',index_col='TransactionID')
submission['isFraud'] = logreg.predict_proba(test_df)[:,1]
submission.to_csv('Logreg_submission.csv')
submission.head()


# In[ ]:


X=train_X 
X_test=test_df
y=train_df['isFraud']


# In[ ]:


n_fold = 5
folds = TimeSeriesSplit(n_splits=n_fold)
folds = model_selection.KFold(n_splits=5)


# In[ ]:


def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true, y_pred), True


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()
    

def train_model_regression(X, X_test, y, params, folds=None, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3):
    """
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]
    splits = folds.split(X) if splits is None else splits
    n_splits = folds.n_splits if splits is None else n_folds
    
    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                        'catboost_metric_name': 'MSE',
                        'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    
    result_dict = {}
    
    # out-of-fold predictions on train data
    oof = np.zeros(len(X))
    
    # averaged predictions on train data
    prediction = np.zeros(len(X_test))
    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(splits):
        if verbose:
            print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred    
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_splits
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
        
    return result_dict


# In[ ]:


def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3, averaging='usual', n_jobs=-1):
    """ 
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    n_splits = folds.n_splits if splits is None else n_folds
    X_test = X_test[columns]
    
    # to set up scoring parameters
    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,
                        'catboost_metric_name': 'AUC',
                        'sklearn_scoring_function': metrics.roc_auc_score},
                    }
    
    result_dict = {}
    if averaging == 'usual':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))
        
    elif averaging == 'rank':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))

    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = n_jobs)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
        
        if averaging == 'usual':
            
            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
            
            prediction += y_pred.reshape(-1, 1)

        elif averaging == 'rank':
                                  
            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
                                  
            prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)        
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_splits
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
            result_dict['top_columns'] = cols
        
    return result_dict


# In[ ]:


params = {'num_leaves': 256,
          'min_child_samples': 79,
          'objective': 'binary',
          'max_depth': 13,
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9,
          #'categorical_feature': cat_cols
         }
result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb', eval_metric='auc', plot_feature_importance=True,
                                                      verbose=500, early_stopping_rounds=200, n_estimators=5000, averaging='usual', n_jobs=-1)


# In[ ]:


sub = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
sub['isFraud'] = result_dict_lgb['prediction']
sub.to_csv('submission.csv', index=False)
sub.head()


# In[ ]:




