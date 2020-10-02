#!/usr/bin/env python
# coding: utf-8

# # Categorical Feature Encoding Challenge II
# 
# 
# There is little explanation in the kernel. sorry.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os,gc
import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_columns = 50


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')\ntest  = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')\nBIN_COL  = [s for s in train.columns if 'bin_' in s]\nNOM_COL  = [s for s in train.columns if 'nom_' in s]\nORD_COL  = [s for s in train.columns if 'ord_' in s]\nNOM_5_9  = ['nom_5','nom_6','nom_7','nom_8','nom_9']\nDATE_COL = ['day','month']")


# In[ ]:


def plot_corr(annot=False):
    plt.figure(figsize=(12,6))
    plt.title(f'Correlation coefficient heatmap(train data)')
    sns.heatmap(train.drop(columns='id').corr(), annot=annot, vmin=-1.0, cmap='Spectral')
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.title(f'Correlation coefficient heatmap(test data)')
    sns.heatmap(test.drop(columns='id').corr(), annot=annot, vmin=-1.0, cmap='Spectral')
    
def plot_line(col_name):
    plt.figure(figsize=(12,3))
    sns.lineplot(train[col_name], train.target)
    plt.show()

def plot_traintest(col_name, h):
#     col_name = f'ord_{i}'
    fig, ax = plt.subplots(1, 3, figsize=(15, h))
    _order = sorted(list(set(train[col_name].dropna().unique()) &                   set(test[col_name].dropna().unique())))
    ax[0].set_title(f'train data {col_name}')
    ax[1].set_title(f'test data {col_name}')
    ax[2].set_title(f'train data {col_name} per target')
    if train[col_name].nunique()>8:
        sns.countplot(y=train[col_name], order=_order, ax=ax[0])
        sns.countplot(y=test[col_name],  order=_order, ax=ax[1])
    else:
        tmp = train[col_name].value_counts(dropna=False)
        ax[0].pie(tmp, labels= tmp.index, autopct='%1.1f%%',
                 shadow=True, startangle=90)
        tmp = test[col_name].value_counts(dropna=False)
        ax[1].pie(tmp, labels= tmp.index, autopct='%1.1f%%',
                  shadow=True, startangle=90)
    
    sns.countplot(y=train[col_name], order=_order, hue=train['target'], ax=ax[2])
    plt.tight_layout()


# # simple Data Check

# In[ ]:


train.info()
test.info()


# # NaN Check
# 
# 

# In[ ]:


import missingno as msno
msno.heatmap(train.drop(columns=['id','target']))
plt.show()
msno.heatmap(test.drop(columns=['id']))


# About 3% of data is NaN for each feature

# In[ ]:


feats = test.drop(columns=['id']).columns
print('1.null data size --')
tmp_df = pd.concat([train[feats].isnull().sum(),
                 test[feats].isnull().sum()],axis=1).rename(columns={0:'train',1:'test'})
print(tmp_df)
print()
print('2.null data rate -------')
tmp_df['train'] = tmp_df['train']/len(train)*100
tmp_df['test']  = tmp_df['test']/len(test)*100
print(tmp_df)
del tmp_df;gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "train['null_count'] = train.isnull().sum(axis=1)\ntest['null_count']  = test.isnull().sum(axis=1)\nprint('train null_count value_counts-----------')\nprint(train['null_count'].value_counts())\nprint('test null_count value_counts-----------')\nprint(test['null_count'].value_counts())\ntrain.null_count = np.clip(train.null_count,0,6) \n# train.null_count = np.clip(train.null_count,0,2) \n\nfor col in test.drop(columns=['id','null_count']).columns.tolist():\n    train[f'{col}_missing'] = train[col].isnull().astype(int)\n    test[f'{col}_missing']  = test[col].isnull().astype(int)    ")


# In[ ]:


sns.countplot(x='null_count', data=train,hue='target')


# In[ ]:


# feats = test.drop(columns=['id','null_count']).columns
# for col1 in feats:
#     for col2 in feats.drop(col1):
#         if len(train[feats].dropna(subset=[col1])[col2].value_counts(dropna=False))<=3:
#             print(f'{col1}, {col2}-----------------------------')
#             print(train[feats].dropna(subset=[col1])[col2].value_counts(dropna=False).head(20))


# In[ ]:


# feats = test.drop(columns=['id','null_count']).columns#.tolist()
# for col1 in feats:
#     for col2 in feats.drop(col1):
#         if len(train.drop(columns=['id','null_count']).loc[
#             train[col1].isnull(), col2].value_counts(dropna=False))<=3:
#             print(f'{col1}, {col2}-----------------------------')
#             print(train.drop(columns=['id','null_count']).loc[
#                 train[col1].isnull(), col2].value_counts(dropna=False).head(2))
#             print(test.drop(columns=['id','null_count']).loc[
#                 train[col1].isnull(), col2].value_counts(dropna=False).head(2))


# # Imbalanced data
# 
# pending

# In[ ]:


plt.title(f'target==1 ratio {len(train[train.target==1]) / len(train)}' )
sns.countplot(train[f'target'])


# In[ ]:


display(train.head(5))
display(test.head(5))


# # Correlation

# In[ ]:


plot_corr(annot=False)


# # unique value check
# 
# Checked for each feature.  
# Few data exists only in training data / test data.

# In[ ]:


gc.collect()


# In[ ]:


get_ipython().run_line_magic('time', '')
tmp_df = pd.concat([train.drop(columns=['target']).nunique(),
                    test.nunique()],axis=1)
tmp_df = pd.concat([tmp_df, pd.concat([train,test], axis=0).drop(columns=['target']).nunique()], axis=1)
tmp_df = tmp_df.reset_index()
tmp_df.columns = ['feature','train','test','all']
tmp_df = tmp_df.loc[tmp_df.feature!='id'].reset_index(drop=True)
print(tmp_df)


# In[ ]:


for col in test.drop(columns='id').columns:
    if len(set(train[col].dropna().unique().tolist())^ set(test[col].dropna().unique().tolist()))>0:
        print(col, 
              '(train only)', set(train[col].dropna().unique().tolist()) - set(test[col].dropna().unique().tolist()),    
              '(test only)',  set(test[col].dropna().unique().tolist()) - set(train[col].dropna().unique().tolist())) 

print(f'train only nom_5 count:', len(train[train['nom_5'].isin(['b3ad70fcb'])]))
print(f'train only nom_6 count:', len(train[train['nom_6'].isin(['f0732a795', 'ee6983c6d', '3a121fefb'])]))
print(f'test only nom_6 count:', len(test[test['nom_6'].isin(['a885aacec'])]))
print(f'train only nom_9 count:', len(train[train['nom_9'].isin(['3d19cd31d', '1065f10dd'])]))


# # binary features

# In[ ]:


get_ipython().run_cell_magic('time', '', "for df in [train, test]:\n    df.bin_3.replace({'F':0, 'T':1}, inplace=True)\n    df.bin_4.replace({'N':0, 'Y':1}, inplace=True)")


# In[ ]:


for col in BIN_COL:
    fig, ax = plt.subplots(1, 3, figsize=(15,6))
#     sns.countplot(y=train[col], ax=ax[0])
    tmp = train[col].value_counts(dropna=False)
    ax[0].set_title(f'train data {col}')
    ax[0].pie(tmp, labels= tmp.index, autopct='%1.1f%%',
             shadow=True, startangle=90,
             labeldistance=0.8)
#     sns.countplot(y=test[col],  ax=ax[1])
    tmp = test[col].value_counts(dropna=False)
    ax[1].set_title(f'test data {col}')
    ax[1].pie(tmp, labels= tmp.index, autopct='%1.1f%%',
              shadow=True, startangle=90,
              labeldistance=0.8,)
    ax[2].set_title(f'train data {col} per target')
    sns.countplot(train[col], hue=train['target'], ax=ax[2])

    plt.show()


# # ordinal features
# - ord_0
# 
# - ord_1/ord_2  
# 
# - ord_3/ord_4
# 
# - ord_5

# In[ ]:


get_ipython().run_cell_magic('time', '', "ord_1_map = {'Novice':1,'Contributor':2,'Expert':3,'Master':4,'Grandmaster':5}\nord_2_map = {'Freezing':1, 'Cold':2,'Warm':3,'Hot':4, 'Boiling Hot':5,'Lava Hot':6}\n\ntrain.loc[train['ord_1'].notnull(),'ord_1'] = train.loc[train['ord_1'].notnull(),'ord_1'].map(ord_1_map)\ntrain.loc[train['ord_2'].notnull(),'ord_2'] = train.loc[train['ord_2'].notnull(),'ord_2'].map(ord_2_map)\ntrain.loc[train['ord_3'].notnull(),'ord_3'] = train.loc[train['ord_3'].notnull(),'ord_3'].apply(lambda c: ord(c) - ord('a') + 1)\ntrain.loc[train['ord_4'].notnull(),'ord_4'] = train.loc[train['ord_4'].notnull(),'ord_4'].apply(lambda c: ord(c) - ord('A') + 1)\ntest.loc[test['ord_1'].notnull(),'ord_1']   = test.loc[test['ord_1'].notnull(),'ord_1'].map(ord_1_map)\ntest.loc[test['ord_2'].notnull(),'ord_2']   = test.loc[test['ord_2'].notnull(),'ord_2'].map(ord_2_map)\ntest.loc[test['ord_3'].notnull(),'ord_3']   = test.loc[test['ord_3'].notnull(),'ord_3'].apply(lambda c: ord(c) - ord('a') + 1)\ntest.loc[test['ord_4'].notnull(),'ord_4']   = test.loc[test['ord_4'].notnull(),'ord_4'].apply(lambda c: ord(c) - ord('A') + 1)\nfor i in range(1,5):\n    train[f'ord_{i}'] = train[f'ord_{i}'].astype(float)\n    test[f'ord_{i}']  = test[f'ord_{i}'].astype(float)")


# In[ ]:


for i, h in zip(range(5),[6,6,6,6,6]):
    plot_traintest(f'ord_{i}',h)


# target meaning distribution

# In[ ]:


# for i in range(5):
for i in range(6):
    plot_line(f'ord_{i}')


# ### ord_5

# In[ ]:


# %%time
# for col in ['ord_5']:
#     _map = pd.concat([train[col], test[col]]).value_counts().rank().to_dict()
#     train[f'{col}_freq'] = train[col].map(_map)
#     test[f'{col}_freq']  = test[col].map(_map)


# In[ ]:


get_ipython().run_cell_magic('time', '', "for df in [train, test]:\n    df.loc[df.ord_5.notnull(), 'ord_5_1'] = df.loc[df.ord_5.notnull(), 'ord_5'].apply(lambda x: x[0])\n    df.loc[df.ord_5.notnull(), 'ord_5_2'] = df.loc[df.ord_5.notnull(), 'ord_5'].apply(lambda x: x[1])")


# In[ ]:


for i in range(1,3):
    plot_line(f'ord_5_{i}')


# In[ ]:


get_ipython().run_cell_magic('time', '', "for col in ['ord_5_1', 'ord_5_2']:\n    _map = pd.concat([train[col], test[col]]).value_counts().rank().to_dict()\n    train[f'{col}_freq'] = train[col].map(_map)\n    test[f'{col}_freq']  = test[col].map(_map)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "train.loc[train['ord_5_1'].notnull(),'ord_5_1'] = train.loc[train['ord_5_1'].notnull(),'ord_5_1'].apply(lambda c: ord(c) - ord('a') + 33).astype(float)\ntest.loc[test['ord_5_1'].notnull(),'ord_5_1']   = test.loc[test['ord_5_1'].notnull(),'ord_5_1'].apply(lambda c: ord(c) - ord('a') + 33).astype(float)\ntrain.loc[train['ord_5_2'].notnull(),'ord_5_2'] = train.loc[train['ord_5_2'].notnull(),'ord_5_2'].apply(lambda c: ord(c) - ord('a') + 33).astype(float)\ntest.loc[test['ord_5_2'].notnull(),'ord_5_2']   = test.loc[test['ord_5_2'].notnull(),'ord_5_2'].apply(lambda c: ord(c) - ord('a') + 33).astype(float)")


# In[ ]:


for i in range(1,3):
    plot_line(f'ord_5_{i}')


# In[ ]:


for i in range(1,3):
    plot_line(f'ord_5_{i}_freq')


# # nominal features
# ### nom_0 - nom_4

# In[ ]:


for i, h in zip(range(5),[6,6,6,6,6]):
    plot_traintest(f'nom_{i}',h)


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(5):\n    _map = pd.concat([train[f'nom_{i}'], test[f'nom_{i}']]).value_counts().rank().to_dict()\n    train[f'nom_{i}_freq'] = train[f'nom_{i}'].map(_map)\n    test[f'nom_{i}_freq']  = test[f'nom_{i}'].map(_map)")


# In[ ]:


for i in range(5):
    plot_line(f'nom_{i}_freq')


# ### nom_5 - nom_9
# 
# - Features with high cardinality
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfor i in range(5,10):\n    col_name = f'nom_{i}'\n    print(f'{col_name} train data ---------------')\n    print(train[col_name].value_counts(dropna=False, normalize=False)[:20])\n    print(f'{col_name} test data ---------------')\n    print(test[col_name].value_counts(dropna=False, normalize=False)[:20])")


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(5, 10):\n    col_name = f'nom_{i}'\n    print(f'{col_name} train data ---------------')\n    print(train[col_name].value_counts(dropna=False, normalize=True)[:20])\n    print(f'{col_name} test data ---------------')\n    print(test[col_name].value_counts(dropna=False, normalize=True)[:20])    ")


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(5, 10):\n    plot_line(f'nom_{i}')")


# In[ ]:


# %%time
# for i in range(5, 10):
#     _map = pd.concat([train[f'nom_{i}'], test[f'nom_{i}']]).value_counts().to_dict()
#     train[f'nom_{i}_freq'] = train[f'nom_{i}'].map(_map)
#     test[f'nom_{i}_freq']  = test[f'nom_{i}'].map(_map)


# In[ ]:


# for i in range(5, 10):
#     plot_line(f'nom_{i}_freq')


# In[ ]:


train.info()
test.info()


# # Cyclical features
# 
# 
# 

# target meaning distribution
# - month
# - day  
#   The distributions of 3 and 5, 2 and 6, 1 and 7 are similar.
#  
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfig, ax = plt.subplots(1, 2, figsize=(12,6))\nsns.lineplot(train.month, train.target, ax=ax[0])\nsns.lineplot(train.day,   train.target, ax=ax[1])')


# In[ ]:


train.day = train.day.replace({3:5,2:6,1:7})
test.day  = test.day.replace({3:5,2:6,1:7})


# In[ ]:


sns.lineplot(train.day, train.target)


# In[ ]:


# %%time
# train['sin_day'] = np.sin(train['day']*np.pi/3.5).astype(float)
# train['cos_day'] = np.cos(train['day']*np.pi/3.5).astype(float)
# test['sin_day']  = np.sin(test['day']*np.pi/3.5).astype(float)
# test['cos_day']  = np.cos(test['day']*np.pi/3.5).astype(float)


# In[ ]:


display(train.day.value_counts(dropna=False))
display(test.day.value_counts(dropna=False))


# # NaN Cleaning
# 
# 
# pending

# In[ ]:


# %%time
# threthold = 0.9#0.8#0.87
# for col in test.columns:
#     if train[col].value_counts().tolist()[0] >len(train.dropna())*threthold:
#         print('-'*20,'\ntrain:', train[col].value_counts().head(1))
#     if test[col].value_counts().tolist()[0] >len(test.dropna())*threthold:
#         print('test:',  test[col].value_counts().head(1))


# In[ ]:


get_ipython().run_cell_magic('time', '', "# train.loc[train['nom_5'].isin(['b3ad70fcb']),'nom_5'] = np.NaN\n# train.loc[train['nom_6'].isin(['f0732a795', 'ee6983c6d', '3a121fefb']),'nom_6'] = np.NaN\n# test.loc[test['nom_6'].isin(['a885aacec']),'nom_6'] = np.NaN\n# train.loc[train['nom_9'].isin(['3d19cd31d', '1065f10dd']),'nom_9'] = np.NaN\nfrom scipy import stats\n\nfor col in NOM_5_9:\n    train.loc[train[col].notnull(), col] = train.loc[train[col].notnull(), col].astype(str).apply(int, base=16)\n    test.loc[test[col].notnull(), col]   = test.loc[test[col].notnull(), col].astype(str).apply(int, base=16)\nfor col in NOM_5_9:\n    train[col] = train[col].astype(float)\n    test[col]  = test[col].astype(float)\n\n# exclude_cols = ['sin_day', 'cos_day'] + [s for s in train.columns if '_mean' in s]\nfeats = train.select_dtypes(float).columns.drop([]).tolist()#\nfor col in feats:\n    if col in ['bin_0']:\n        train[col].fillna(0, inplace=True)\n        test[col].fillna(0,  inplace=True)\n    else:\n        train[col].fillna(-1, inplace=True)\n        test[col].fillna(-1,  inplace=True)\n\n    \n    \nfor col in feats:    \n    train[col] = train[col].astype(int)\n    test[col]  = test[col].astype(int)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "for col in NOM_COL+['ord_5']:\n    _map = train.groupby(col).mean()['target'].to_dict()\n    train[f'{col}_mean'] = train[col].map(_map)\n    test[f'{col}_mean']  = test[col].map(_map)")


# In[ ]:


# fig, ax = plt.subplots(1,2,figsize=(12,6))
# sns.scatterplot(train.sin_day, train.cos_day, ax=ax[0])
# sns.scatterplot(test.sin_day,  test.cos_day,  ax=ax[1])


# # Correlation (Immediately before training)

# In[ ]:


plot_corr()


# In[ ]:


tmp_df = train.drop(columns='id').corr().sort_values(['target'], ascending=False)[1:]
plt.figure(figsize=(12,10))
plt.title('Correlation plot(per target)')
sns.barplot(x=tmp_df['target'], y=tmp_df.index)


# In[ ]:


train.info(null_counts=True)
test.info(null_counts=True)


# # train

# In[ ]:


from sklearn.model_selection import train_test_split,KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
feats = train.select_dtypes(int).columns.drop([
    'id','target',#'bin_3','day',
    'nom_0_freq', 'nom_1_freq', 'nom_2_freq', 'nom_3_freq', 'nom_4_freq',
]).tolist()+['nom_0_mean','nom_1_mean','nom_2_mean','nom_3_mean', 'nom_4_mean']#+['sin_day', 'cos_day',]#+[s for s in train.columns if '_mean' in s]#train.select_dtypes(float).columns.tolist()#train.drop(columns=['id','target']).columns#

islgb, isxgb, isctb = False, False, True

X = train[feats]
y = train.target
X_test = test[feats]
(X_train,X_val, y_train, y_val) = train_test_split(X, y, stratify=y, random_state=42)
print(X_train.shape,X_val.shape, y_train.shape, y_val.shape)
feats


# In[ ]:


def create_categorical_feats(df, category_columns):
    index_df = pd.DataFrame([df.columns, df.dtypes]).T

    index_df = index_df.rename(columns={0:"column_name", 1:'dtype',})
#     categorical_feats = index_df[index_df.column_name.isin(category_columns)].index.tolist()
    print(categorical_feats)
    return categorical_feats   


# In[ ]:


import lightgbm as lgb
import xgboost  as xgb
import catboost as ctb
from sklearn.metrics import confusion_matrix

categorical_feats = [
#     'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 
    'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',
#     'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4',
    'day', 
    'month',
#     'nom_0_freq', 'nom_1_freq', 'nom_2_freq', 'nom_3_freq', 'nom_4_freq', 
#     'ord_5_1_freq', 'ord_5_2_freq'
]

lgb_params = {
    'boosting_type':'gbdt', 
    'num_leaves':2**4-1,#2**5-1,
    'learning_rate':0.05,#0.1, 
    'n_estimators':3000,#1000,#100, 
#     subsample_for_bin=200000, 
    'objective':'binary',#=None,
    'metrics':'auc',
    'feature_fraction':0.8,
    'reg_alpha':0.9,#0.1, 
    'reg_lambda':0.5,#0.1, 
    'random_state':42,
#     'verbosity':100,
    'early_stopping_rounds':100
}

xgb_params = {
    'learning_rate':0.1, 
    'n_estimators':3000,#100, 
#     subsample_for_bin=200000, 
    'objective':'binary:logistic',
    'eval_metric':'auc',
    'random_state':42, 
}

ctb_params = {
    'task_type':'GPU',
    'learning_rate':0.1, 
    'n_estimators':10000,#3000,#100, 
    'objective':'Logloss',
    'eval_metric':'AUC',
    'random_state':10372,#42, 
    'use_best_model':True,
    'verbose':1000,
    'early_stopping_rounds':100,
    'l2_leaf_reg':0.9,
#     'silent':False,
#     'plot':True,
    'cat_features':create_categorical_feats(X, categorical_feats)
}


# # Easy training(lightgbm)

# In[ ]:


if islgb:
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train, 
              eval_set=[(X_train, y_train),(X_val, y_val)],
              verbose=100)
    oof_preds = model.predict_proba(
        X_val, num_iteration=model.best_iteration_)[:,1]
    print(roc_auc_score(y_val, oof_preds))
    print(confusion_matrix(y_val, np.round(oof_preds).astype(np.int8)))  
    
    plt.figure(figsize=(12,10))
    lgb.plot_importance(model)


# # Easy training(xgboost)

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif isxgb:\n    model = xgb.XGBClassifier(**xgb_params)\n    model.fit(X_train, y_train, eval_set=[(X_train, y_train),\n                                          (X_val, y_val)], \n              verbose=10, early_stopping_rounds=100)\n    oof_preds = model.predict_proba(X_val)[:,1]\n    print(roc_auc_score(y_val, oof_preds))\n    print(confusion_matrix(y_val, np.round(oof_preds).astype(np.int8)))\n    \n    plt.figure(figsize=(12,10))\n    xgb.plot_importance(model)')


# # Easy training(catboost)

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nif isctb:\n    model = ctb.CatBoostClassifier(**ctb_params)\n    model.fit(X_train, y_train, \n              eval_set=[#(X_train, y_train), \n                        (X_val, y_val)],#Multiple eval sets are not supported on GPU\n              plot=True)\n    oof_preds = model.predict_proba(X_val)[:,1]\n    print(roc_auc_score(y_val, oof_preds))\n    \n    feature_importance_df = pd.DataFrame(np.log1p(model.get_feature_importance()), model.feature_names_).reset_index() \n    feature_importance_df = feature_importance_df.rename(columns={'index':'feature',0:'importance'}).sort_values('importance',ascending=False)\n    plt.figure(figsize=(12,10))\n    display(sns.barplot(feature_importance_df['importance'], feature_importance_df['feature']))\n\n    print(confusion_matrix(y_val, np.round(oof_preds).astype(np.int8)))")


# In[ ]:


def display_importances(feature_importance_df_,height=50,title='catboost'):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False).index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

#     plt.figure(figsize=(8, height))
    plt.figure(figsize=(15, height))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title(f'{title} Features (avg over folds)')
    plt.tick_params(labelcolor='Red')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


# In[ ]:


def set_importance(model, feature_importance_df, fold_idx, feats, modeltype): 
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    if modeltype=='lgb':
        fold_importance_df["importance"] = np.log1p(model.feature_importance(
            importance_type='gain',iteration=model.best_iteration)) 
    else:
        fold_importance_df["importance"] = np.log1p(model.feature_importances_)
        
    fold_importance_df["fold"] = fold_idx + 1
    feature_importance_df      = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    return feature_importance_df   


# # Training(KFold)

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nkf = StratifiedKFold(n_splits=10,#5, \n                     shuffle=True, random_state=42)\n# kf = KFold(n_splits=5, shuffle=True,#False, \n#            random_state=42)\noof_preds = np.zeros(len(X)).astype(np.float32)\nsub_preds = np.zeros(len(X_test)).astype(np.float32)\nfeature_importance_df = pd.DataFrame()\nfor fold_, (train_idx, val_idx) in enumerate(kf.split(X,y=y)):\n#     print("train:", train_idx, "val:", val_idx)\n    X_train = X.loc[train_idx] \n    y_train = y.loc[train_idx]\n    X_val, y_val = X.loc[val_idx], y.loc[val_idx]\n    if islgb:\n        model = lgb.LGBMClassifier(**lgb_params)\n        model.fit(X_train, y_train, \n                  eval_set=[(X_train, y_train),(X_val, y_val)], \n                  verbose=100)\n        oof_preds[val_idx] = model.predict_proba(\n            X_val, num_iteration=model.best_iteration_)[:,1]\n        sub_preds += model.predict_proba(\n            X_test, num_iteration=model.best_iteration_)[:,1] / kf.n_splits\n\n        plt.figure(figsize=(12,10))\n        lgb.plot_importance(model)\n        feature_importance_df = set_importance(model, feature_importance_df, fold_, feats, \'lgb\')\n    \n    if isctb:\n        model = ctb.CatBoostClassifier(**ctb_params)\n        model.fit(X_train, y_train, \n                  eval_set=[#(X_train, y_train), \n                      (X_val, y_val)],\n                  plot=True)\n        oof_preds[val_idx] = model.predict_proba(X_val)[:,1]\n        sub_preds += model.predict_proba(X_test)[:,1] / kf.n_splits\n        \n        feature_importance_df = set_importance(\n            model, feature_importance_df, fold_, feats, \'ctb\')\n\nplt.title(f\'auc_score:{roc_auc_score(y, oof_preds)}\')\nsns.distplot(oof_preds)\nsns.distplot(sub_preds)\nplt.legend([\'train\',\'test\'])\nplt.show()        ')


# In[ ]:


plt.title(f'auc_score:{roc_auc_score(y, oof_preds)}')
sns.distplot(oof_preds)
sns.distplot(sub_preds)
plt.legend(['train','test'])
plt.show()        


# In[ ]:


display_importances(feature_importance_df, height=len(feats), title='catboost(training cv)')


# In[ ]:


submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')
submission['target'] = sub_preds


# In[ ]:


submission


# In[ ]:


submission.describe()


# In[ ]:


get_ipython().run_cell_magic('time', '', "submission.to_csv('submission.csv', index=False)")


# In[ ]:


gc.collect()

