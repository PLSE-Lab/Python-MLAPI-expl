#!/usr/bin/env python
# coding: utf-8

# # LGB Parameter_SimpleVersion <section id="section_top" />
# 
# - [Loading Libraries](#section_LL)
# - [Defining Loss](#section_DL)
# - [Extracting Useful Features](#section_EUF)
# - [Parameter_tuning](#section_pt)
# - [Examination](#section_ex)
# - [Convergence Plot](#section_CPlot)
# - [Training LGB Classifier with tuned Parameters](#section_train)
# - [Making Predictions](#section_pred)
# 
# ---------------------
# 

# This kernel was created with reference to the following.  
# 
# ref.    
# https://www.kaggle.com/meaninglesslives/lgb-parameter-tuning  
# https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data  
# https://www.kaggle.com/ashishpatel26/can-this-make-sense-of-the-universe-tuned  
# 
# - In this kernel, passband and object_id are used as the grouping key.  
# - aggs: Deleted except flux, flux_err   
# - etc...
# 

# # Loading Libraries  <section id="section_LL" />
# 
# [return](#section_top)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import lightgbm as lgb

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt

from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
notebookstart= time.time()
pd.set_option("display.max_rows", 101)
isDataCheck=False


# # Defining Loss <section id="section_DL" />
# 
# [return](#section_top)

# In[ ]:


def lgb_multi_weighted_logloss(y_true, y_preds):#use by eval_metric
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False

def multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


# # Extracting Useful Features <section id="section_EUF" />
# 
# [return](#section_top)

# In[ ]:


#grouping(object_id,passband)
aggs = {
    'flux': ['min', 'max', 'mean', 'median', 'std'],
    'flux_err': ['median', 'std'],
} 

grp_col=['object_id','passband']#'object_id'    


# In[ ]:


get_ipython().run_cell_magic('time', '', 'gc.enable()\n\n# train = pd.read_csv(\'../input/training_set.csv\')\ntrain = pd.read_csv(\'../input/training_set.csv\',\n                   dtype = {\n                       \'object_id\':np.int32,\n                       \'mjd\':np.float64,\n                       \'passband\':np.int8,\n                       \'flux\':np.float32,\n                       \'flux_err\':np.float32,\n                       \'detected\':np.int32})\n\n# agg_train = train.groupby(\'object_id\').agg(aggs)\nagg_train = train.groupby(grp_col).agg(aggs)\nnew_columns = [k + \'_\' + agg for k in aggs.keys() for agg in aggs[k]]\nagg_train.columns = new_columns\nagg_train=pd.pivot_table(agg_train, index=\'object_id\', columns=\'passband\')\ndel train\n\ndisplay(agg_train.head(10))\nprint("gc.collect:",gc.collect())')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# meta_train = pd.read_csv('../input/training_set_metadata.csv')\nmeta_train = pd.read_csv('../input/training_set_metadata.csv',\n                         dtype = {\n                             'object_id':np.int32,\n                             'ra':np.float32,\n                             'decl':np.float32,                 \n                             'gal_l':np.float32,           \n                             'gal_b':np.float32,           \n                             'ddf':np.int8,#bool\n                             'hostgal_specz':np.float32,         \n                             'hostgal_photoz':np.float32,        \n                             'hostgal_photoz_err':np.float32,    \n                             'distmod':np.float32,          \n                             'mwebv':np.float32,            \n                             'target':np.int8})\n\ndisplay(meta_train.head())\n\nfull_train = agg_train.reset_index().merge(\n    right=meta_train,\n    how='outer',\n    on='object_id'\n)\nfull_train=full_train.drop( columns=[('object_id', '')])\n# print(full_train.columns)\n\nif 'target' in full_train:\n    y = full_train['target']\n    del full_train['target']\nclasses = sorted(y.unique())\n\n# Taken from Giba's topic : https://www.kaggle.com/titericz\n# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194\n# with Kyle Boone's post https://www.kaggle.com/kyleboone\nclass_weight = {\n    c: 1 for c in classes\n}\nfor c in [64, 15]:\n    class_weight[c] = 2\n\nprint('Unique classes : ', classes)")


# In[ ]:


del agg_train
print("gc.collect:",gc.collect())


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nisfillNaN=False#True\n\nif 'object_id' in full_train:\n    oof_df = full_train[['object_id']]\n    del full_train['object_id'], full_train['hostgal_specz'],full_train['ddf']\n    \nif isfillNaN:    \n    train_mean = full_train.mean(axis=0)\n    full_train.fillna(train_mean, inplace=True)\n\nfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)\nclfs = []\nimportances = pd.DataFrame()")


#  

# # Parameter Tuning <section id="section_pt" />
# 
# [return](#section_top)

# In[ ]:


get_ipython().run_cell_magic('time', '', "dim_learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform',name='learning_rate')\ndim_estimators = Integer(low=800, high=2000,name='n_estimators')\ndim_max_depth = Integer(low=3, high=6,name='max_depth')\n\ndimensions = [dim_learning_rate,\n              dim_estimators,\n              dim_max_depth]\n\ndefault_parameters = [0.03,1000,3]\n\nlgb_params = {\n    'boosting_type': 'gbdt',\n    'objective': 'multiclass',\n    'num_class': 14,\n    'metric': 'multi_logloss',\n    'subsample': .9,\n    'colsample_bytree': .7,\n    'reg_alpha': .01,#L1\n    'reg_lambda': .02,#01,#L2\n#     'num_leaves': 31,#63,# Add 2^(max_depth) > num_leaves warning\n    'min_split_gain': 0.01,\n    'min_child_weight': 10,\n    'silent':True,\n    'verbosity':-1,\n}")


# In[ ]:


get_ipython().run_cell_magic('time', '', "def createModel(learning_rate,n_estimators,max_depth):       \n\n    oof_preds = np.zeros((len(full_train), len(classes)))\n    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):\n        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]\n        val_x, val_y = full_train.iloc[val_], y.iloc[val_]\n\n        clf = lgb.LGBMClassifier(**lgb_params,learning_rate=learning_rate,\n                                n_estimators=n_estimators,max_depth=max_depth)\n        clf.fit(\n            trn_x, trn_y,\n            eval_set=[(trn_x, trn_y), (val_x, val_y)],\n            eval_metric=lgb_multi_weighted_logloss,\n            verbose=False,#True,\n            early_stopping_rounds=50\n        )\n        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)\n        print('fold',fold_+1,multi_weighted_logloss(val_y, clf.predict_proba(val_x, num_iteration=clf.best_iteration_)))\n\n        clfs.append(clf)\n    \n    loss = multi_weighted_logloss(y_true=y, y_preds=oof_preds)\n    print('MULTI WEIGHTED LOG LOSS : %.5f ' % loss)\n    \n    return loss")


# In[ ]:


get_ipython().run_cell_magic('time', '', '@use_named_args(dimensions=dimensions)\ndef fitness(learning_rate,n_estimators,max_depth):\n    """\n    Hyper-parameters:\n    learning_rate:     Learning-rate for the optimizer.\n    n_estimators:      Number of estimators.\n    max_depth:         Maximum Depth of tree.\n    """\n\n    # Print the hyper-parameters.\n    print(\'learning rate: {0:.2e}\'.format(learning_rate))\n    print(\'estimators:\', n_estimators)\n    print(\'max depth:\', max_depth)\n    \n    lv= createModel(learning_rate=learning_rate,\n                    n_estimators=n_estimators,\n                    max_depth = max_depth)\n    return lv')


# ## Examination
# <section id="section_ex" />
# 
# [return](#section_top)
#     

# In[ ]:


get_ipython().run_cell_magic('time', '', 'folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)')


# ----------------------------------------

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nisSearchForHyperparameters=False\n\nif isSearchForHyperparameters:\n    search_result = gp_minimize(func=fitness,\n                                dimensions=dimensions,\n                                acq_func='EI', \n                                n_calls=20,\n                                x0=default_parameters,n_jobs=-1)")


# # Convergence Plot
# <section id="section_CPlot" />
# 
# [return](#section_top)

# In[ ]:


if isSearchForHyperparameters:
    plot_convergence(search_result)
    plt.show()

# optimal parameters found using scikit optimize. use these parameter to initialize the 2nd level model.
if isSearchForHyperparameters:
    print(search_result.x)
    learning_rate = search_result.x[0]
    n_estimators = search_result.x[1]
    max_depth = search_result.x[2]
else:
    learning_rate = default_parameters[0]
    n_estimators = default_parameters[1]
    max_depth = default_parameters[2] 
print("learning_rate:",learning_rate)
print("n_estimators:",n_estimators)
print("max_depth:",max_depth)


# In[ ]:


if isSearchForHyperparameters:
    del search_result,plot_convergence


# # Training LGB Classifier with tuned Parameters <section id="section_train" />
# 
# [return](#section_top)

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfolds = StratifiedKFold(n_splits=5, \n                        shuffle=True, random_state=1)\nclfs = []\nimportances = pd.DataFrame()\n\noof_preds = np.zeros((len(full_train), len(classes)))\nfor fold_, (trn_, val_) in enumerate(folds.split(y, y)):\n    trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]\n    val_x, val_y = full_train.iloc[val_], y.iloc[val_]\n    \n    clf = lgb.LGBMClassifier(\n        **lgb_params,\n        learning_rate=learning_rate,\n        n_estimators=n_estimators,max_depth=max_depth)\n    clf.fit(\n        trn_x, trn_y,\n        eval_set=[(trn_x, trn_y), (val_x, val_y)],\n        eval_metric=lgb_multi_weighted_logloss,\n        verbose=100,\n        early_stopping_rounds=50)\n    oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)\n    print(multi_weighted_logloss(val_y, clf.predict_proba(val_x, num_iteration=clf.best_iteration_)))\n    \n    imp_df = pd.DataFrame()\n    imp_df['feature'] = full_train.columns\n    imp_df['gain'] = clf.feature_importances_\n    imp_df['fold'] = fold_ + 1\n    importances = pd.concat([importances, imp_df], axis=0, sort=False)\n    \n    clfs.append(clf)\n\nprint('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_true=y, y_preds=oof_preds))\n\n\nmean_gain = importances[['gain', 'feature']].groupby('feature').mean()\nimportances['mean_gain'] = importances['feature'].map(mean_gain['gain'])\n# plt.figure(figsize=(8, 12))\n# sns.barplot(x='gain', y='feature', data=importances.sort_values('mean_gain', ascending=False))\n# plt.tight_layout()\n# plt.savefig('importances.png')")


# In[ ]:


importances.loc[:,['feature','mean_gain']].groupby(
    'feature').mean().sort_values('mean_gain',ascending=False)


# In[ ]:


# lgb.plot_tree(clf,figsize=(18,10))


# In[ ]:


importances.loc[importances.fold==1].sort_values('gain',ascending=False)


# In[ ]:


importances.loc[importances.fold==2].sort_values('gain',ascending=False)


# In[ ]:


importances.loc[importances.fold==3].sort_values('gain',ascending=False)


# In[ ]:


importances.loc[importances.fold==4].sort_values('gain',ascending=False)


# In[ ]:


importances.loc[importances.fold==5].sort_values('gain',ascending=False)


# In[ ]:


importances.loc[:,['feature','mean_gain']].groupby('feature').mean().sort_values('mean_gain',ascending=False)


# In[ ]:


del oof_preds,importances,mean_gain
print("gc.collect:",gc.collect())


# # Making Predictions <section id="section_pred" />
# 
# [return](#section_top)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("read test_set_metadata.csv")\n# meta_test = pd.read_csv(\'../input/test_set_metadata.csv\')\nmeta_test = pd.read_csv(\'../input/test_set_metadata.csv\',\n                        dtype = {\'object_id\':np.int32,\n                                 \'ra\':np.float32,\n                                 \'decl\':np.float32,                 \n                                 \'gal_l\':np.float32,           \n                                 \'gal_b\':np.float32,           \n                                 \'ddf\':np.int8,#bool\n                                 \'hostgal_specz\':np.float32,         \n                                 \'hostgal_photoz\':np.float32,        \n                                 \'hostgal_photoz_err\':np.float32,    \n                                 \'distmod\':np.float32,          \n                                 \'mwebv\':np.float32, } )')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# isDebug=False\n\nimport time\n\nstart = time.time()\n# chunks = 20_000_000\nchunks = 5_000_000\n\npreds_1 = 14\nfrom tqdm import tqdm\nprint("read test_set.csv")\ncolumnslist=full_train.columns\ndel full_train\nprint("gc.collect",gc.collect())\nfor i_c, df in enumerate(tqdm(pd.read_csv(\'../input/test_set.csv\', \n                                     chunksize=chunks, iterator=True,\n                                     dtype = {\'object_id\':np.int32,\n                                              \'mjd\':np.float64,\n                                              \'passband\':np.int8,\n                                              \'flux\':np.float32,\n                                              \'flux_err\':np.float32,\n                                              \'detected\':np.int32}))):\n    agg_test = df.groupby(grp_col).agg(aggs)\n    agg_test.columns = new_columns \n    agg_test=pd.pivot_table(agg_test, index=\'object_id\', columns=\'passband\')#.reset_index()\n    full_test = agg_test.reset_index().merge(right=meta_test, how=\'left\', on=\'object_id\')\n    if isfillNaN:\n        full_test = full_test.fillna(train_mean)\n\n    # Make predictions\n    preds = None\n    for clf in clfs:\n        if preds is None:\n#             preds = clf.predict_proba(full_test[full_train.columns]) / folds.n_splits\n            preds = clf.predict_proba(full_test[columnslist]) / folds.n_splits\n        else:\n#             preds += clf.predict_proba(full_test[full_train.columns]) / folds.n_splits\n            preds += clf.predict_proba(full_test[columnslist]) / folds.n_splits\n\n    # preds_99 = 0.1 gives 1.769\n    preds_99 = np.ones(preds.shape[0])\n    #     for i in range(preds.shape[1]):\n    #         preds_99 *= (1 - preds[:, i])\n    #     preds_1 = preds.shape[1]\n    for i in range(preds_1):\n        preds_99 *= (1 - preds[:, i])\n\n    # Store predictions\n    preds_df = pd.DataFrame(preds, columns=[\'class_\' + str(s) for s in clfs[0].classes_])\n    preds_df[\'object_id\'] = full_test[\'object_id\']\n    #     https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data/code\n    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/68943\n    #     preds_df[\'class_99\'] = preds_99\n    preds_df[\'class_99\'] = 0.14 * preds_99 / np.mean(preds_99) \n#     if isDebug:\n#         print(preds_df[\'class_99\'].mean(),np.mean(preds_99))\n#         print(np.mean(0.14 * preds_99))\n    \n    if i_c == 0:\n        preds_df.to_csv(\'predictions.csv\',  header=True, mode=\'a\', index=False)\n    else: \n        preds_df.to_csv(\'predictions.csv\',  header=False, mode=\'a\', index=False)\n        \n    del agg_test, full_test, preds_df, preds\n    gc.collect()\n    \n    if (i_c + 1) % 10 == 0:\n        print(\'%15d done in %5.1f\' % (chunks * (i_c + 1), (time.time() - start) / 60))')


# In[ ]:


get_ipython().run_cell_magic('time', '', "z = pd.read_csv('predictions.csv')\n\nprint(z.groupby('object_id').size().max())\nprint((z.groupby('object_id').size() > 1).sum())\n\nz = z.groupby('object_id').mean()\n\nz.to_csv('single_predictions.csv', index=True)")


# In[ ]:


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

