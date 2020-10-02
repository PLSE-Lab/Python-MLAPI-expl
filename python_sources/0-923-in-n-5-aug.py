#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook as tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import random
random.seed(2019)
np.random.seed(2019)
import os,math
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:



df_train = pd.read_csv('../input/train.csv')

test_path = '../input/test.csv'

df_test = pd.read_csv(test_path)
df_test.drop(['ID_code'], axis=1, inplace=True)
df_test = df_test.values

unique_samples = []
unique_count = np.zeros_like(df_test)
for feature in tqdm(range(df_test.shape[1])):
    _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], feature] += 1

# Samples which have unique values are real the others are fake
real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

print(len(real_samples_indexes))
print(len(synthetic_samples_indexes))


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


def findmeancounts(bins,values,x):
    for i in range(len(bins)):
        if x < bins[i]:
            return values[i]
    return values[-1]
 
def magic2(): 
    feats = ["var_"+str(i) for i in range(200)] 
    df = pd.concat([df_train,df_test.ix[real_samples_indexes]]) 
    for feat in feats:
#     for feat in ["var_170"]:
#         m = df[feat].mean()
#         s = df[feat].std()
#         temp = np.random.normal(m, s, 300000) 
        
#         df[feat+"_temp"] = temp
#         df[feat+"_temp"] = df[feat+"_temp"].apply(lambda x:round(x,4))
#         temp2 = df[feat+"_temp"].value_counts(dropna = True) 
#         bins = []  
#         values = []
#         count = 1
#         temp_values = []
#         alpha = min(200,len(temp2)//50)
#         for k,v in sorted(temp2.items(),key = lambda x:x[0]):
#             if count % alpha == 0:
#                 bins.append(k)
#                 values.append(sum(temp_values)*1.0/len(temp_values))
#                 temp_values = []
#             count += 1
#             temp_values.append(v)
#         print(m,s,count,feat,len(values))
#         df_train[feat+"_vc_expect"] = df_train[feat].apply(lambda x:findmeancounts(bins,values,x)) 
#         df_test[feat+"_vc_expect"] = df_test[feat].apply(lambda x:findmeancounts(bins,values,x))
# #         print(temp2,temp_values[10:20])
#         del df[feat+"_temp"]
        temp = df[feat].value_counts(dropna = True) 
        
#         print(temp)
        df_train[feat+"vc"] = df_train[feat].map(temp).map(lambda x:min(10,x)).astype(np.uint8)
        df_test[feat+"vc"] = df_test[feat].map(temp).map(lambda x:min(10,x)).astype(np.uint8)
        print(feat,temp.shape[0],df_train[feat+"vc"].map(lambda x:int(x>2)).sum(),df_train[feat+"vc"].map(lambda x:int(x>3)).sum())
        df_train[feat+"sum"] = ((df_train[feat] - df[feat].mean()) * df_train[feat+"vc"].map(lambda x:int(x>1))).astype(np.float32)
        df_test[feat+"sum"] = ((df_test[feat] - df[feat].mean()) * df_test[feat+"vc"].map(lambda x:int(x>1))).astype(np.float32)
        df_train[feat+"sum2"] = ((df_train[feat]) * df_train[feat+"vc"].map(lambda x:int(x>2))).astype(np.float32)
        df_test[feat+"sum2"] = ((df_test[feat]) * df_test[feat+"vc"].map(lambda x:int(x>2))).astype(np.float32)
#         if df_train[feat+"vc"].map(lambda x:int(x>4)).sum() > 20000:
        df_train[feat+"sum3"] = ((df_train[feat]) * df_train[feat+"vc"].map(lambda x:int(x>4))).astype(np.float32) 
        df_test[feat+"sum3"] = ((df_test[feat]) * df_test[feat+"vc"].map(lambda x:int(x>4))).astype(np.float32) 
#         if df_train[feat+"vc"].map(lambda x:int(x>6)).sum() > 20000:
#             df_train[feat+"sum4"] = ((df_train[feat]) * df_train[feat+"vc"].map(lambda x:int(x>6))).astype(np.float32) 
#             df_test[feat+"sum4"] = ((df_test[feat]) * df_test[feat+"vc"].map(lambda x:int(x>6))).astype(np.float32) 
        #temp = df_train[feat].value_counts(dropna = True) 
        #df_train[feat+"sum4"] = ((df_train[feat] - df[feat].mean()) * df_train[feat+"vc"].map(lambda x:int(x>1))).astype(np.float32)
        #df_test[feat+"sum4"] = ((df_test[feat] - df[feat].mean()) * df_test[feat+"vc"].map(lambda x:int(x>1))).astype(np.float32)
        
        # TODO
#         if df_train[feat+"vc"].map(lambda x:int(x>2)).sum() > 10000:
#             df_train[feat+"sum2"] = (df_train[feat]) * df_train[feat+"vc"].map(lambda x:int(x>2)) 
#             df_test[feat+"sum2"] = (df_test[feat]) * df_test[feat+"vc"].map(lambda x:int(x>2))
#         else:
#             df_train[feat+"sum2"] = (df_train[feat]) * df_train[feat+"vc"].map(lambda x:int(x>1)) 
#             df_test[feat+"sum2"] = (df_test[feat]) * df_test[feat+"vc"].map(lambda x:int(x>1))
#         df_train[feat+"_filter1"] = (df_train[feat] * (df_train[feat+"vc"].apply(lambda x:min(3,math.sqrt(x))) - df_train[feat+"_vc_expect"]).apply(lambda x: int(x > 0)) * df_train[feat+"vc"].apply(lambda x: int(x > 1))).astype(np.float32) 
#         df_test[feat+"_filter1"] = (df_test[feat] * (df_test[feat+"vc"].apply(lambda x:min(3,math.sqrt(x))) - df_test[feat+"_vc_expect"]).apply(lambda x: int(x > 0)) * df_test[feat+"vc"].apply(lambda x: int(x > 1))).astype(np.float32) 
#         df_train[feat+"_filter1"] = (df_train[feat] * (df_train[feat+"vc"] - df_train[feat+"_vc_expect"].apply(lambda x:1+math.sqrt(x-1))).apply(lambda x: int(x > 0)) * df_train[feat+"vc"].apply(lambda x: int(x > 1))).astype(np.float32) 
#         df_test[feat+"_filter1"] = (df_test[feat] * (df_test[feat+"vc"] - df_test[feat+"_vc_expect"].apply(lambda x:1+math.sqrt(x-1))).apply(lambda x: int(x > 0)) * df_test[feat+"vc"].apply(lambda x: int(x > 1))).astype(np.float32) 

#         df_train[feat+"vc2"] = df_train[feat+"vc"].map(lambda x:int(math.sqrt(x))).astype(np.int8) 
#         df_test[feat+"vc2"] = df_test[feat+"vc"].map(lambda x:int(math.sqrt(x))).astype(np.int8) 
#         df_train[feat+"vc"] = df_train[feat+"vc"].map(lambda x:int(x>1)).astype(np.int8) 
#         df_test[feat+"vc"] = df_test[feat+"vc"].map(lambda x:int(x>1)).astype(np.int8)
magic2()

df_train.iloc[:5,:]


# In[ ]:


def augment(x,y,t=2):
    xs,xn = [],[]


    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        featnum = x1.shape[1]//200 - 1
        for c in range(200):
            np.random.shuffle(ids)
            x1[:,[c] + [200 + featnum * c + idc for idc in range(featnum)]] = x1[ids][:,[c] + [200 + featnum * c + idc for idc in range(featnum)]]
        xn.append(x1)

    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        featnum = x1.shape[1]//200 - 1
        for c in range(200):
            np.random.shuffle(ids)
            x1[:,[c] + [200 + featnum * c + idc for idc in range(1)]] = x1[ids][:,[c] + [200 + featnum * c + idc for idc in range(1)]]
        xs.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y

features = [col for col in df_train.columns if col not in ['target', 'ID_code']]
# X_train, y_train = df_train[features], df_train['target']
# augment(X_train.values, y_train.values)


# In[ ]:


print(df_train['target'].value_counts())
print(df_train.groupby('var_80vc')['target'].value_counts())
print(df_train.groupby('var_81vc')['target'].value_counts())
print(df_train.groupby('var_82vc')['target'].value_counts())
# print(df_train[df_train['var_82vc']>2].groupby('var_82')['target'].value_counts())
# print(df_train[df_train['var_68vc']>2].groupby('var_68')['target'].value_counts())
# print(df_train.groupby('var_68')['target'].value_counts())


# In[ ]:


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 15,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.6,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 50,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    "lambda_l1" : 1.,
#     "lambda_l2" : 0.5,
    "bagging_seed" : 2007,
    "verbosity" : 1,
    "seed": 2007
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2007)
oof = df_train[['ID_code', 'target']]
oof['predict'] = 0
predictions = np.zeros((df_test.shape[0],5))
val_aucs = []
feature_importance_df = pd.DataFrame()
features = [col for col in df_train.columns if col not in ['target', 'ID_code']]
X_test = df_test[features].values

for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):
    X_train, y_train = df_train.iloc[trn_idx][features], df_train.iloc[trn_idx]['target']
    X_valid, y_valid = df_train.iloc[val_idx][features], df_train.iloc[val_idx]['target']
    
    N = 1
    p_valid,yp = 0,0
    for i in range(N):
        X_t, y_t = augment(X_train.values, y_train.values)
        weights = np.array([0.8] * X_t.shape[0])
        weights[:X_train.shape[0]] = 1.0
        print(X_t.shape)
#         X_t = pd.DataFrame(X_t)
        trn_data = lgb.Dataset(X_t, label=y_t, weight = weights)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        evals_result = {}
        lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=5000,
                        verbose_eval = 1000,
                        evals_result=evals_result
                       )
        p_valid += lgb_clf.predict(X_valid)
        yp += lgb_clf.predict(X_test)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    oof['predict'][val_idx] = p_valid/N
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)
    
    predictions[:,fold] = yp/N


# After collecting the "verified generators" for each fake sample, finding the Public/Private LB split is no more than a few set operations.

# In[ ]:


mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))



cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

print(cols[:50])
print(cols[-50:])
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]




# In[ ]:


cols[:30]


# In[ ]:


##submission
sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
sub_df["target"] = np.mean(predictions,axis = 1)
sub_df.to_csv("lgb_submission.csv", index=False)

