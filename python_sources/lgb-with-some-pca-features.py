#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
sns.set(font_scale=1)


# In[2]:


random_state = 42
np.random.seed(random_state)
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[3]:


train_X = df_train.drop(['ID_code', 'target'], axis = 1)
test_X = df_test.drop(['ID_code'], axis = 1)


# In[4]:


def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


# In[5]:


# Scaling
mmscale = MinMaxScaler()  
train_X_scaled = mmscale.fit_transform(train_X)  
test_X_scaled = mmscale.transform(test_X)


# In[6]:


# PCA
pca = PCA()  
factors_train = pca.fit_transform(train_X_scaled) 
factors_test = pca.transform(test_X_scaled)


# In[7]:


# add 200 new PCA features
pca_columns_name = ["pca_" + str(col) for col in range(0, 200)]
factors_train = pd.DataFrame(factors_train, columns = pca_columns_name)
factors_test = pd.DataFrame(factors_test, columns = pca_columns_name)
train_pca = df_train.merge(factors_train, left_index = True, right_index = True)
test_pca = df_test.merge(factors_test, left_index = True, right_index = True)


# In[9]:


# we choose the most correlated factors with the target (>0.05 and <-0.05) and the less correlated factors with the initial vars

# Correlations of factors with the target:
# pca_2 +0,218 | pca_7 +0,113 | pca_9 +0,050 | pca_14 +0,135 | pca_19 +0,103 | pca_28 +0,070 | pca_42 +0,050
# pca_6 -0,064 | pca_34 -0,051 | pca_39 -0,062 | pca_46 -0,055

# Correlations of these pca with some vars are around 0.3 - 0.6, which is ok

important_factors = ['pca_2', 'pca_7', 'pca_14', 'pca_19']
vars_to_exclude = ['var_9', 'var_94', 'var_109', 'var_151']
df_train_columns = list(df_train.columns)
df_test_columns = list(df_test.columns)
train_columns_with_factors = important_factors + df_train_columns
test_columns_with_factors = important_factors + df_test_columns
train_main_pca = train_pca[train_columns_with_factors].drop(vars_to_exclude, axis = 1)
test_main_pca = test_pca[test_columns_with_factors].drop(vars_to_exclude, axis = 1)


# In[ ]:


df_train = train_main_pca
df_test = test_main_pca


# # LGB with KFolds

# In[ ]:


lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
}


# In[ ]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
oof = df_train[['ID_code', 'target']]
oof['predict'] = 0
predictions = df_test[['ID_code']]
val_aucs = []
feature_importance_df = pd.DataFrame()


# In[ ]:


features = [col for col in df_train.columns if col not in ['target', 'ID_code']]
X_test = df_test[features].values


# In[ ]:


for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):
    X_train, y_train = df_train.iloc[trn_idx][features], df_train.iloc[trn_idx]['target']
    X_valid, y_valid = df_train.iloc[val_idx][features], df_train.iloc[val_idx]['target']
    
    N = 3
    p_valid,yp = 0,0
    for i in range(N):
        X_t, y_t = augment(X_train.values, y_train.values)
        X_t = pd.DataFrame(X_t)
        X_t = X_t.add_prefix('var_')
    
        trn_data = lgb.Dataset(X_t, label=y_t)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        evals_result = {}
        lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 500,
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
    
    predictions['fold{}'.format(fold+1)] = yp/N


# In[ ]:


mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))


# In[ ]:


cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# In[ ]:


# submission
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
predictions.to_csv('lgb_all_predictions.csv', index=None)
sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv("lgb_submission.csv", index=False)
oof.to_csv('lgb_oof.csv', index=False)

