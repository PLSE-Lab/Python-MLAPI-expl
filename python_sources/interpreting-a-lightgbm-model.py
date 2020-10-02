#!/usr/bin/env python
# coding: utf-8

# # Interpreting a LightGBM model
# 
# This notebook explores the patterns identified by a straightforward LightGBM model. Many thanks to Olivier and the Good_fun_with_LigthGBM kernel where all the LightGBM training code is from. This notebook is just meant to extend that kernel and examine it using individualized feature importances. You will need to scroll a bit to get to the pretty pictures :)

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold, train_test_split
from lightgbm import LGBMClassifier
import matplotlib.pyplot as pl
import gc
import shap


# ## Build the dataset

# In[2]:


def build_model_input():
    buro_bal = pd.read_csv('../input/bureau_balance.csv')
    print('Buro bal shape : ', buro_bal.shape)
    
    print('transform to dummies')
    buro_bal = pd.concat([buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')], axis=1).drop('STATUS', axis=1)
    
    print('Counting buros')
    buro_counts = buro_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
    buro_bal['buro_count'] = buro_bal['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])
    
    print('averaging buro bal')
    avg_buro_bal = buro_bal.groupby('SK_ID_BUREAU').mean()
    
    avg_buro_bal.columns = ['avg_buro_' + f_ for f_ in avg_buro_bal.columns]
    del buro_bal
    gc.collect()
    
    print('Read Bureau')
    buro = pd.read_csv('../input/bureau.csv')
    
    print('Go to dummies')
    buro_credit_active_dum = pd.get_dummies(buro.CREDIT_ACTIVE, prefix='ca_')
    buro_credit_currency_dum = pd.get_dummies(buro.CREDIT_CURRENCY, prefix='cu_')
    buro_credit_type_dum = pd.get_dummies(buro.CREDIT_TYPE, prefix='ty_')
    
    buro_full = pd.concat([buro, buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum], axis=1)
    # buro_full.columns = ['buro_' + f_ for f_ in buro_full.columns]
    
    del buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum
    gc.collect()
    
    print('Merge with buro avg')
    buro_full = buro_full.merge(right=avg_buro_bal.reset_index(), how='left', on='SK_ID_BUREAU', suffixes=('', '_bur_bal'))
    
    print('Counting buro per SK_ID_CURR')
    nb_bureau_per_curr = buro_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
    buro_full['SK_ID_BUREAU'] = buro_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])
    
    print('Averaging bureau')
    avg_buro = buro_full.groupby('SK_ID_CURR').mean()
    print(avg_buro.head())
    
    del buro, buro_full
    gc.collect()
    
    print('Read prev')
    prev = pd.read_csv('../input/previous_application.csv')
    
    prev_cat_features = [
        f_ for f_ in prev.columns if prev[f_].dtype == 'object'
    ]
    
    print('Go to dummies')
    prev_dum = pd.DataFrame()
    for f_ in prev_cat_features:
        prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_).astype(np.uint8)], axis=1)
    
    prev = pd.concat([prev, prev_dum], axis=1)
    
    del prev_dum
    gc.collect()
    
    print('Counting number of Prevs')
    nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])
    
    print('Averaging prev')
    avg_prev = prev.groupby('SK_ID_CURR').mean()
    #print(avg_prev.head())
    del prev
    gc.collect()
    
    print('Reading POS_CASH')
    pos = pd.read_csv('../input/POS_CASH_balance.csv')
    
    print('Go to dummies')
    pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'])], axis=1)
    
    print('Compute nb of prevs per curr')
    nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    
    print('Go to averages')
    avg_pos = pos.groupby('SK_ID_CURR').mean()
    
    del pos, nb_prevs
    gc.collect()
    
    print('Reading CC balance')
    cc_bal = pd.read_csv('../input/credit_card_balance.csv')
    
    print('Go to dummies')
    cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='cc_bal_status_')], axis=1)
    
    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    
    print('Compute average')
    avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
    avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]
    
    del cc_bal, nb_prevs
    gc.collect()
    
    print('Reading Installments')
    inst = pd.read_csv('../input/installments_payments.csv')
    nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    
    avg_inst = inst.groupby('SK_ID_CURR').mean()
    avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]
    
    print('Read data and test')
    data = pd.read_csv('../input/application_train.csv')
    test = pd.read_csv('../input/application_test.csv')
    print('Shapes : ', data.shape, test.shape)
    
    y = data['TARGET']
    del data['TARGET']
    
    categorical_feats = [
        f for f in data.columns if data[f].dtype == 'object'
    ]
    categorical_feats
    for f_ in categorical_feats:
        data[f_], indexer = pd.factorize(data[f_])
        test[f_] = indexer.get_indexer(test[f_])
        
    data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    
    del avg_buro, avg_prev
    gc.collect()

    return data, test, y


# In[3]:


data, test, y = build_model_input()
data_train, data_valid, y_train, y_valid = train_test_split(data, y, test_size=0.2, random_state=0)


# ## Train the LightGBM model
# 
# Here we just use a simple train/validation split.

# In[4]:


clf = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=30,
    colsample_bytree=.8,
    subsample=.9,
    max_depth=7,
    reg_alpha=.1,
    reg_lambda=.1,
    min_split_gain=.01,
    min_child_weight=2,
    silent=-1,
    verbose=-1,
)

clf.fit(
    data_train, y_train, 
    eval_set= [(data_train, y_train), (data_valid, y_valid)], 
    eval_metric='auc', verbose=100, early_stopping_rounds=30  #30
)


# # Explain the model
# 
# SHAP values are fair allocation of credit among the features and have theoretical garuntees about consistency from game theory (so you can trust them). There is a high speed algorithm to compute SHAP values for LightGBM (and XGBoost and CatBoost), so they are particularly helpful when interpreting predictions from gradient boosting tree models. While typical feature importances are for the whole dataset (and often [have consistency problems](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27) these values are computed for every single prediction, which opens up new ways to understand the model. For more details check out the [docs](https://github.com/slundberg/shap).**

# In[5]:


# explain 10000 examples from the validation set
# each row is an explanation for a sample, and the last column in the base rate of the model
# the sum of each row is the margin (log odds) output of the model for that sample
shap_values = shap.TreeExplainer(clf.booster_).shap_values(data_valid.iloc[:10000,:])
shap_values.shape


# ### Summarize the feature imporances with a bar chart

# In[10]:


# compute the global importance of each feature as the mean absolute value
# of the feature's importance over all the samples
global_importances = np.abs(shap_values).mean(0)[:-1]


# In[50]:


# make a bar chart that shows the global importance of the top 20 features
inds = np.argsort(-global_importances)
f = pl.figure(figsize=(5,10))
y_pos = np.arange(20)
inds2 = np.flip(inds[:20], 0)
pl.barh(y_pos, global_importances[inds2], align='center', color="#1E88E5")
pl.yticks(y_pos, fontsize=13)
pl.gca().set_yticklabels(data.columns[inds2])
pl.xlabel('mean abs. SHAP value (impact on model output)', fontsize=13)
pl.gca().xaxis.set_ticks_position('bottom')
pl.gca().yaxis.set_ticks_position('none')
pl.gca().spines['right'].set_visible(False)
pl.gca().spines['top'].set_visible(False)


# ### Summarize the feature importances with a density scatter plot
# 
# Just looking a single number hides a lot of information about the model. So instead we plot the SHAP values for each feature for every sample on the x-axis and then let them pil up when there is not space to show density. If we then color each dot by the value of the original feature we can see how a low feature value or high feature value effects the model output. For example EXT_SOURCE_2 is the most important feature (because they are sorted by mean abolute SHAP value) and the red colored dots (samples) have low SHAP values with the blue colored dots have high SHAP values. Since for a logistic regression model the SHAP values are in log odds, this color spread means low values of EXT_SOURCE_2 raise the log odds output of the model significantly.
# 
# Note that some features like SK_DPD_DEF are not important for most people, but have a very large impact for a subset of the people in the data set. This highlights how a globally important feature is not nessecarrily the most importance feature for each person.

# In[6]:


shap.summary_plot(shap_values, data_valid.iloc[:10000,:])


# ### Investigate the dependence of the model on each feature
# 
# The summary plot above gives a lot of information. But we can learn more by examining a single feature and how it impacts the model's predicition across all the samples. To do this we plot the SHAP value for a feature on the y-axis and the original value of the feature on the x-axis. Doing this for all samples for EXT_SOURCE_2 shows a fairly linear relationship that flattens near high values. Note that the impact of EXT_SOURCE_2 on the model output is different for different people that have the same value of EXT_SOURCE_2, as shown by the vertical spread of the dots at a single point on the x axis. This is because of interaction effects (and would disappear if the trees were all depth 1 in the LightGBM model). To highlight what interactions may be driving this vertical spread shap colors the dots by another feature that seems to explain some of the dispersion (see the docs for more details on getting the exact interaction effects). For EXT_SOURCE_2 we see gender seems to have some effect.

# In[54]:


shap.dependence_plot("EXT_SOURCE_2", shap_values, data_valid.iloc[:10000,:])


# In[58]:


shap.dependence_plot("SK_DPD_DEF", shap_values, data_valid.iloc[:10000,:], show=False)
pl.xlim(0,5)
pl.show()


# ## Plot the SHAP dependence plots for the top 20 features
# 
# Here you can scroll through the top 20 features and see lots of interesting patterns and interactions. Like how having high EXT_SOURCE_1 is even more important if you also have a small DAYS_BIRTH magnitude (see the EXT_SOURCE_1 plot).

# In[67]:


for i in reversed(inds2):
    shap.dependence_plot(i, shap_values, data_valid.iloc[:10000,:])


# In[ ]:




