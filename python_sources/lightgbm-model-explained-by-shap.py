#!/usr/bin/env python
# coding: utf-8

#  In this notebook we will try to gain insight into a tree model based on the shap package. To understand why current feature importances calculated by lightGBM, Xgboost and other tree based models have issues read this article:[ Interpretable Machine Learning with XGBoost]( Interpretable Machine Learning with XGBoost). The shap library [https://github.com/slundberg/shap](https://github.com/slundberg/shap) can be used by going to the settings of the notebook (upper right corner,) and  "add a custom package" in the settings tab.
#  
#  The most important plot is the summary plot (below in this notebook), that shows the 30 most important features. For each feature a distribution is plotted on how the train samples influence the model outcome. The more red the dots, the higher the feature value, the more blue the lower the feature value.
# 
# In this case, the feature EXT_SOURCE_2 is the feature that has the most impact on the model output. Train samples with low EXT_SOURCE_2 have higher probability upon obtaining a loan. If the client has a high EXT_SOURCE_2 value, the probability of getting a loan is low. For the red blob on the left, we see that a lot clients are in this case.
# 
# In the dependence plot of EXT_SOURCE_2, we see that if this value is between 0 and 0.2 the model output is higher especially when CODE_GENDER is zero (the blue dots). The feature CODE_GENDER is automatically chosen by the shap dependence plot function.
# 
# In the cc_bal_CNT_DRAWINGS_ATM_CURRENT dependence plot we see a few outliers. Maybe we should remove them ? The are training samples with CODE_GENDER equal to 2. Are that transgenders ? Also dependence plot of SK_DPD_DEF show that most samples are zero except a few exceptions. Maybe we need some feature engineering here. Chances that the model influence of this feature does not generalise to the test set is very high. In other words, overfitting is likely. I would not recommend including this feature. As we can derive from the depence plot, the impact of leaving this feature out will be low on the model performance.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings('ignore')

gc.enable()


# In[2]:


print('Importing data...')
lgbm_submission = pd.read_csv('../input/sample_submission.csv')


# # Feature enginering based on [https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm](https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm)

# In[3]:


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
#print(avg_buro.head())

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


# In[4]:


#Create train and validation set
train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.2, shuffle=True, stratify=y, random_state=1301)


# In[5]:


#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_x,label=train_y)
valid_data=lgb.Dataset(valid_x,label=valid_y)


# In[6]:


#Select Hyper-Parameters
params = {'metric' : 'auc',
          'boosting_type' : 'gbdt',
          'colsample_bytree' : 0.9234,
          'num_leaves' : 13,
          'max_depth' : -1,
          'n_estimators' : 200,
          'min_child_samples': 399, 
          'min_child_weight': 0.1,
          'reg_alpha': 2,
          'reg_lambda': 5,
          'subsample': 0.855,
          'verbose' : -1,
          'num_threads' : 4
}


# In[7]:


#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 2500,
                 valid_sets=valid_data,
                 early_stopping_rounds= 30,
                 verbose_eval= 10
                 )


# In[8]:


y_hat = lgbm.predict(data)
score = roc_auc_score(y, y_hat)
print("Overall AUC: {:.3f}" .format(score))


# # Explaining the lightgbm model with shap
# The advantage of using lightgbm over sklearn random forrest classifier is that lightGBM can deal with the Nan.

# In[9]:


import shap


# In[10]:


get_ipython().run_line_magic('time', 'shap_values = shap.TreeExplainer(lgbm).shap_values(valid_x)')


# In[11]:


shap.summary_plot(shap_values, valid_x)


# In[12]:


shap.dependence_plot("EXT_SOURCE_2", shap_values, valid_x)


# In[13]:


shap.dependence_plot("EXT_SOURCE_3", shap_values, valid_x)


# In[14]:


shap.dependence_plot("EXT_SOURCE_1", shap_values, valid_x)


# In[15]:


shap.dependence_plot("AMT_GOODS_PRICE_x", shap_values, valid_x)


# In[16]:


shap.dependence_plot("CODE_GENDER", shap_values, valid_x)


# There are CODER_GENDER = 2 ?

# In[19]:


data['CODE_GENDER'].value_counts()


# In[21]:


# Who are those four ?
data[data['CODE_GENDER']==2]


# In[ ]:


# Did those got a loan ?
y[data['CODE_GENDER']==2].describe()


# In[23]:


# Do we have those in test as well ?
test['CODE_GENDER'].value_counts()


# Since CODE_GENDER does not appear in the test set, we can drop them from the train samples ?

# In[24]:


shap.dependence_plot("AMT_ANNUITY_x", shap_values, valid_x)


# In[25]:


shap.dependence_plot("inst_AMT_PAYMENT", shap_values, valid_x)


# In[39]:


# How many outliers are they and do they get a loan ?
y[data['inst_AMT_PAYMENT'] > 500000].describe()


# In[ ]:


# Do those outliers also occur in the test set ?
test[test['inst_AMT_PAYMENT'] > 500000].filter(regex='EXT_SOURCE_.', axis=1).describe()


# In[26]:


shap.dependence_plot("AMT_CREDIT_x", shap_values, valid_x)


# In[27]:


shap.dependence_plot("DAYS_BIRTH", shap_values, valid_x)


# In[28]:


shap.dependence_plot("CNT_INSTALMENT_FUTURE", shap_values, valid_x)


# In[29]:


shap.dependence_plot("DAYS_EMPLOYED", shap_values, valid_x)


# In[56]:


y[data['DAYS_EMPLOYED'] > 0].describe()


# In[57]:


data[data['DAYS_EMPLOYED'] <= 0].DAYS_EMPLOYED.hist()


# In[58]:


y[data['DAYS_EMPLOYED'] <= 0].describe()


# If DAYS_EMPLOYED is a large positive number means the client is unemployed ? Maybe extraxt those with dummy variable ? It would applying for a loan unemployed lowers your approval from 8.6% downto 5.4%.

# In[59]:


data[data['DAYS_EMPLOYED'] > 0].filter(regex='EXT_SOURCE_.', axis=1).describe()


# In[60]:


test[test['DAYS_EMPLOYED'] <= 0].filter(regex='EXT_SOURCE_.', axis=1).describe()


# In[61]:


# how many unemployed in the test set ?
test[test['DAYS_EMPLOYED'] > 0].filter(regex='EXT_SOURCE_.', axis=1).describe()


# In[30]:


shap.dependence_plot("CNT_PAYMENT", shap_values, valid_x)


# In[31]:


shap.dependence_plot("NAME_EDUCATION_TYPE", shap_values, valid_x)


# In[32]:


shap.dependence_plot("SK_ID_PREV_y", shap_values, valid_x)


# In[33]:


shap.dependence_plot("NAME_CONTRACT_STATUS_Refused", shap_values, valid_x)


# In[34]:


shap.dependence_plot("OWN_CAR_AGE", shap_values, valid_x)


# In[35]:


shap.dependence_plot("cc_bal_CNT_DRAWINGS_ATM_CURRENT", shap_values, valid_x)


# In[50]:


y[data['cc_bal_CNT_DRAWINGS_ATM_CURRENT'] > 10].describe()


# In[45]:


test[test['cc_bal_CNT_DRAWINGS_ATM_CURRENT'] > 10].shape


# In[36]:


shap.dependence_plot("AMT_CREDIT_SUM_DEBT", shap_values, valid_x)


# In[37]:


shap.dependence_plot("SK_DPD_DEF", shap_values, valid_x)


# In[49]:


y[data['SK_DPD_DEF'] > 0].describe()


# In[38]:


shap.dependence_plot("DAYS_CREDIT", shap_values, valid_x)


# # Visualize many predictions
# To keep the browser happy we only visualize 1,000 individuals.

# In[32]:


# shap.initjs()


# In[33]:


# Makes the browser to slow
# shap.force_plot(shap_values[:1000,:], valid_x.iloc[:1000,:])


# # Prepare for submission

# In[34]:


sub_pred = lgbm.predict(test)


# In[38]:


sub_pred = np.clip(sub_pred, 0, 1)


# In[40]:


sub_pred.min(), sub_pred.max()


# In[35]:


lgbm_submission.TARGET = sub_pred
lgbm_submission.to_csv('subm_lgbm_auc{:.8f}.csv'.format(score), index=False, float_format='%.8f')


# In[ ]:




