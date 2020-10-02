#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
pd.set_option('display.float_format', '{:.3f}'.format)
x_min, x_max = -1., 1.
y_min, y_max = -.3, .3


# In[2]:


trn_index_list = np.load("../input/eyn-folds/trn_index_list_10f.npy", allow_pickle=True)
val_index_list = np.load("../input/eyn-folds/val_index_list_10f.npy", allow_pickle=True)
print([np.shape(trn_index_list[i]) for i in range(np.shape(trn_index_list)[0])])
print([np.shape(val_index_list[i]) for i in range(np.shape(val_index_list)[0])])
train_targets_inside = np.load("../input/eyn-original/train_targets_inside.npy")


# In[3]:


df_train = pd.read_pickle("../input/eyn-pre-unravel-df/df_train.pickle")
df_test = pd.read_pickle("../input/eyn-pre-unravel-df/df_test.pickle")
print(df_train.shape, df_test.shape)
df_train.head(7)


# In[4]:


df_train_cluster = pd.read_pickle("../input/eyn-pre-cluster-calc/df_train_cluster_s.pickle")
df_test_cluster = pd.read_pickle("../input/eyn-pre-cluster-calc/df_test_cluster_s.pickle")
print(df_train_cluster.shape, df_test_cluster.shape)
df_train_cluster = df_train_cluster.astype(float)
df_test_cluster = df_test_cluster.astype(float)
df_train_cluster.head(7)


# In[5]:


df_train = pd.concat([df_train, df_train_cluster], ignore_index=False, axis=1, sort=True)
df_test = pd.concat([df_test, df_test_cluster], ignore_index=False, axis=1, sort=True)
print(df_train.shape)
df_train.head(7)


# In[6]:


# not using the individual data points
# df_train_cluster = pd.read_pickle("../input/eyn-pre-cluster-calc/df_train_cluster_w.pickle")
# df_test_cluster = pd.read_pickle("../input/eyn-pre-cluster-calc/df_test_cluster_w.pickle")
# print(df_train_cluster.shape, df_test_cluster.shape)
# df_train_cluster = df_train_cluster.astype(float)
# df_test_cluster = df_test_cluster.astype(float)
# df_train_cluster.head(7)


# In[7]:


# df_train = pd.concat([df_train, df_train_cluster], ignore_index=False, axis=1, sort=True)
# df_test = pd.concat([df_test, df_test_cluster], ignore_index=False, axis=1, sort=True)
# print(df_train.shape)
# df_train.head(7)


# In[8]:


import math
time_dummy_train = []
for i in df_train["t_entry"]:
    entry = np.repeat(0, 16)
    if math.isnan(i):
        time_dummy_train.append([np.nan]*16)
    else:
        num = int((i + 1.5) // 0.1)
        if num != 0:
            entry[num-1] = 1
        time_dummy_train.append(entry)
        
import math
time_dummy_test = []
for i in df_test["t_entry"]:
    entry = np.repeat(0, 16)
    if math.isnan(i):
        time_dummy_test.append([np.nan]*16)
    else:
        num = int((i + 1.5) // 0.1)
        if num != 0:
            entry[num-1] = 1
        time_dummy_test.append(entry)


# In[9]:


time_dummy_train = np.array(time_dummy_train)
time_dummy_test = np.array(time_dummy_test)

time_dummy_col_name = ["td{}".format(i) for i in range(16)]

for i,col_name in enumerate(time_dummy_col_name):
    df_train[col_name] = time_dummy_train[:,i]
    df_test[col_name] = time_dummy_test[:,i]
    
df_train[time_dummy_col_name] = df_train[time_dummy_col_name].astype('category')
df_test[time_dummy_col_name] = df_test[time_dummy_col_name].astype('category')
df_train.head(7)


# In[10]:


cat_col = ['entry_in', 'exit_in', 'tid_0', 'tid_1']
df_train[cat_col] = df_train[cat_col].astype('category')
df_test[cat_col] = df_test[cat_col].astype('category')
# cluster targets has been made categorical in eyn-pre-cluster-unif, but can be ensured categorical again here


# In[11]:


drop_col = ['x_exit_0', 'y_exit_0', 'vmean_0', 'vmax_0', 'vmin_0', 
             'exit_in_0', 'dist_0', 'speed_0', 'dir_x_0', 'dir_y_0']

df_columns = df_train.columns
flatten = lambda l: [item for sublist in l for item in sublist]
drop_col += flatten([[name + "_" + str(i) for name in df_columns] for i in range(15,21)])
# drop_col = []  # not dropping any columns, if you choose


# In[12]:


df_train_unstack = df_train.unstack()
unstack_col_names = ["_".join([tup[0],str(tup[1])]) for tup in df_train_unstack.columns.values]
df_train_unstack.columns = unstack_col_names
df_train_unstack = df_train_unstack.drop(drop_col, axis=1)
df_train_unstack.head()


# In[ ]:


df_test_unstack = df_test.unstack()
unstack_col_names = ["_".join([tup[0],str(tup[1])]) for tup in df_test_unstack.columns.values]
df_test_unstack.columns = unstack_col_names
df_test_unstack = df_test_unstack.drop(drop_col, axis=1)
df_test_unstack.head()


# In[ ]:


trn_embedding = np.load("../input/eynembedding/train_lstm.npy")
test_embedding = np.load("../input/eynembedding/test_lstm.npy")
for e in range(len(trn_embedding[0])):
    df_train_unstack["e{}".format(e)] = trn_embedding[:,e]
    df_test_unstack["e{}".format(e)] = test_embedding[:,e]
df_train_unstack.head()


# In[ ]:


# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def scoring_package(y_true, y_pred, plotting=False):
    threshold_search = np.arange(0, 1., 0.01)
    f1_arr = [f1_score(y_true, [k > threshold for k in y_pred]) for threshold in threshold_search]
    precision_arr = [precision_score(y_true, [k > threshold for k in y_pred]) for threshold in threshold_search]
    recall_arr = [recall_score(y_true, [k > threshold for k in y_pred]) for threshold in threshold_search]
    
    if plotting:
        plt.figure(figsize=(24,3))
        plt.plot(threshold_search, f1_arr)
        plt.plot(threshold_search, precision_arr)
        plt.plot(threshold_search, recall_arr)
        plt.show()
    
    threshold = threshold_search[np.argmax(f1_arr)]
    y_pred_class = np.array([k > threshold for k in y_pred])
    f1 = f1_score(y_true, y_pred_class)
    precision = precision_score(y_true, y_pred_class)
    recall = recall_score(y_true, y_pred_class)
    roc = roc_auc_score(y_true, y_pred)
    return f1, precision, recall, roc, threshold


# In[ ]:


df_train_values = df_train.values
df_train_columns = list(df_train.columns)
df_test_values = df_test.values
df_test_columns = list(df_test.columns)
train_last_is_stationary = np.argwhere(df_train_values[::21,df_train_columns.index("dur")] == 0)[:,0]
train_last_not_stationary = np.argwhere(df_train_values[::21,df_train_columns.index("dur")] != 0)[:,0]
test_last_is_stationary = np.argwhere(df_test_values[::21,df_test_columns.index("dur")] == 0)[:,0]
test_last_not_stationary = np.argwhere(df_test_values[::21,df_test_columns.index("dur")] != 0)[:,0]
print(train_last_is_stationary.shape, train_last_not_stationary.shape)
print(test_last_is_stationary.shape, test_last_not_stationary.shape)

train_last_seen_is_inside = np.argwhere(df_train_values[::21,df_train_columns.index("entry_in")] == 1)[:,0]
train_last_seen_not_inside = np.argwhere(df_train_values[::21,df_train_columns.index("entry_in")] == 0)[:,0]
test_last_seen_is_inside = np.argwhere(df_test_values[::21,df_test_columns.index("entry_in")] == 1)[:,0]
test_last_seen_not_inside = np.argwhere(df_test_values[::21,df_test_columns.index("entry_in")] == 0)[:,0]
print(train_last_seen_is_inside.shape, train_last_seen_not_inside.shape)
print(test_last_seen_is_inside.shape, test_last_seen_not_inside.shape)

y_pred_full = np.zeros(np.shape(train_targets_inside))
y_pred_full[np.intersect1d(train_last_is_stationary, train_last_seen_is_inside)] = 1
y_pred_full[np.intersect1d(train_last_is_stationary, train_last_seen_not_inside)] = 0
# print('The F1-PC-RC-ROC of full prediction is: {:.5f}-{:.5f}-{:.5f}-{:.5f} at threshold {:.3f}'
#       .format(*scoring_package(train_targets_inside, y_pred_full)))
test_preds = []


# In[ ]:


# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 63,
    'learning_rate': 0.05,  # dynamic one below
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': 0,
    'lambda': 0.1,
    'num_threads': 4,
    'seed' : 42,
#     'histogram_pool_size' : 2048  # to restrict memory usage
#     'boost_from_average' : False  # as per warning message
}
num_boost_round = 3000


# In[ ]:


# inputs: train_pd, train_targets_inside, and test_pd
# probably try https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.cv, but cluster complications
for fold_num, (trn_index, val_index) in enumerate(zip(trn_index_list, val_index_list)):
    print("Training fold {}, hash: {}".format(fold_num, np.sum(val_index)%999))
    
    x_trn = df_train_unstack.iloc[trn_index]
    x_val = df_train_unstack.iloc[val_index]
    y_trn = pd.DataFrame(train_targets_inside).astype('bool').iloc[trn_index]
    y_val = pd.DataFrame(train_targets_inside).astype('bool').iloc[val_index]
    
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(x_trn, y_trn)
    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)
    
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_eval],
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=25,
                    learning_rates=lambda iter: 0.1 * (0.995 ** iter),
                    verbose_eval=50)

    # eval
    y_pred = gbm.predict(x_val, num_iteration=gbm.best_iteration)
    print('The F1-PC-RC-ROC score of fold {} is: {:.5f}-{:.5f}-{:.5f}-{:.5f} at threshold {:.3f}'
          .format(fold_num, *scoring_package(y_val, y_pred)))
    y_pred_full[val_index] = y_pred
    
    # testing
    test_pred = gbm.predict(df_test_unstack, num_iteration=gbm.best_iteration)
    test_preds.append(test_pred)


# In[ ]:


scoring_results = scoring_package(train_targets_inside, y_pred_full, plotting=True)
print('The F1-PC-RC-ROC of full prediction is: {:.5f}-{:.5f}-{:.5f}-{:.5f} at threshold {:.3f}'
      .format(*scoring_results))


# In[ ]:


plt.figure(figsize = (30,5))
for i in range(gbm.num_trees())[:2]:
    plt.plot([gbm.get_leaf_output(i,j) for j in range(gbm.params['num_leaves'])])


# In[ ]:


# very blur cannot see, don't know how to plot just a small part of it
# lags when asked to plot everything
# ax = lgb.plot_tree(gbm, tree_index=0, figsize=(30, 10), 
#               show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])

# doesn't work on Kaggle
# graph = lgb.create_tree_digraph(gbm, tree_index=0, name='Tree54')
# graph.render(view=True)


# In[ ]:


num_features = len(gbm.feature_importance())
plt.figure(figsize=(100,3))
plt.bar(np.arange(num_features), gbm.feature_importance(importance_type='split'), align='center', width=0.4)
plt.xticks(np.arange(0,num_features,21), gbm.feature_name()[::21], rotation='vertical', fontsize=20)
plt.title('Feature Importance - Split')
plt.yscale('log') 
plt.show()


# In[ ]:


plt.figure(figsize=(100,3))
plt.bar(np.arange(num_features), gbm.feature_importance(importance_type='gain'), align='center', width=0.4)
plt.xticks(np.arange(0,num_features,21), gbm.feature_name()[::21], rotation='vertical', fontsize=20)
plt.title('Feature Importance - Gain')
plt.yscale('log') 
plt.show()


# # Submission

# In[ ]:


print([i[2] for i in test_preds])
print([i[-4] for i in test_preds])
test_preds


# In[ ]:


np.save("train_preds", y_pred_full)
np.save("test_preds", test_preds)


# In[ ]:


test_preds_mean = np.mean(test_preds, axis=0)
test_preds_mean = np.array([1 if pred>scoring_results[-1] else 0 for pred in test_preds_mean])
print(np.sum(test_preds_mean))


# In[ ]:


test_preds_mean[np.intersect1d(test_last_is_stationary, test_last_seen_is_inside)] = 1.
test_preds_mean[np.intersect1d(test_last_is_stationary, test_last_seen_not_inside)] = 0.
print(np.sum(test_preds_mean))


# In[ ]:


df_submit = pd.read_csv("../input/ey-nextwave/data_test/data_test.csv")
df_submit = df_submit[df_submit['x_exit'].isnull()]
df_submit = df_submit[['trajectory_id']].copy()
df_submit = df_submit.rename(columns = {'trajectory_id':'id'})
df_submit['target'] = test_preds_mean
df_submit.to_csv('submission.csv', index=False)


# In[ ]:


df_submit = pd.read_csv("submission.csv")
df_submit.head()


# In[ ]:


df_submit.tail()


# In[ ]:


print(scoring_results)
print(params)


# In[ ]:




