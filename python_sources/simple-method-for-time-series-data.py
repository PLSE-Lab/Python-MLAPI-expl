#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd

import seaborn as sns
import lightgbm as lgb


# I wanted to share a simple and effective way of dealing with the time-series element of this competition. I've found that exponential weighting of values works quite nicely. It is also computationally inexpensive and quite easy to interpret.
# 
# I'll show a simple example from the installments_payments data, and leave it to others to refine and apply more widely in their own solutions. Note that I'm showing this for weighting events that happened prior to the application (i.e. negative time values) but you might also want to apply this to features with positive time values (remaining payments on loans for example).

# In[ ]:


ins = pd.read_csv('../input/installments_payments.csv')
ins.head()


# First we'll make some time weights for our features. Intuitively, we want more-recent observations to be wighted more strongly, but we don't know how quickly we want the weights to tail off for older observations. We'll try a few different exponential weights. The higher the multiplier in the exponential the quicker the results will taper off. Note that all of the time values in the data are negative.

# In[ ]:


ins['tw1']=np.exp(ins['DAYS_ENTRY_PAYMENT']*0.01)
ins['tw2']=np.exp(ins['DAYS_ENTRY_PAYMENT']*0.05)
ins['tw3']=np.exp(ins['DAYS_ENTRY_PAYMENT']*0.25)


# Let's see what these weights look like

# In[ ]:


ts_view=ins[['DAYS_ENTRY_PAYMENT', 'tw1', 'tw2', 'tw3']][ins['DAYS_ENTRY_PAYMENT']>-100]
ts_view.drop_duplicates('DAYS_ENTRY_PAYMENT', inplace=True)
sns.regplot('DAYS_ENTRY_PAYMENT', 'tw1', data=ts_view, fit_reg=False)
sns.regplot('DAYS_ENTRY_PAYMENT', 'tw2', data=ts_view, fit_reg=False)
sns.regplot('DAYS_ENTRY_PAYMENT', 'tw3', data=ts_view, fit_reg=False)


# Great, this looks like a reasonable selection of weightings. Note that more-recent data are closer to 0 (the right-hand side of this plot). Let's create some features and apply some weighting.
# 
# I've borrowed heavily from the following kernel for this part of the code:
# 
# https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features

# In[ ]:


#Are the payments late/early?

ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

#Now let's weight these new values

ins['DPD_tw1']=ins['DPD']*np.exp(ins['DAYS_ENTRY_PAYMENT']*ins['tw1'])
ins['DPD_tw2']=ins['DPD']*np.exp(ins['DAYS_ENTRY_PAYMENT']*ins['tw2'])
ins['DPD_tw3']=ins['DPD']*np.exp(ins['DAYS_ENTRY_PAYMENT']*ins['tw3'])

ins['DBD_tw1']=ins['DBD']*np.exp(ins['DAYS_ENTRY_PAYMENT']*ins['tw1'])
ins['DBD_tw2']=ins['DBD']*np.exp(ins['DAYS_ENTRY_PAYMENT']*ins['tw2'])
ins['DBD_tw3']=ins['DBD']*np.exp(ins['DAYS_ENTRY_PAYMENT']*ins['tw3'])

#Let's weight the value of payments made

ins['AMT_PAYMENT_tw1']=ins['AMT_PAYMENT']*ins['tw1']
ins['AMT_PAYMENT_tw2']=ins['AMT_PAYMENT']*ins['tw2']
ins['AMT_PAYMENT_tw3']=ins['AMT_PAYMENT']*ins['tw3']





# Now we do the aggregations of the weighted features, and the weights themselves

# In[ ]:


# Features: Perform aggregations
# The natural thing to want to do here is to sum the time-weights and weighted values so that we can get weighted-averages
# You might want to experiment with this a bit more - perhaps other aggregations will prove useful?
aggregations = {
    'NUM_INSTALMENT_VERSION': ['nunique'],
    'DPD': ['max', 'mean', 'sum'],
    'DBD': ['max', 'mean', 'sum'],
    'AMT_INSTALMENT': ['max', 'mean', 'sum'],
    'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
    'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
    'DPD_tw1':['sum'],
    'DPD_tw2':['sum'],
    'DPD_tw3':['sum'],
    'DBD_tw1':['sum'],
    'DBD_tw2':['sum'],
    'DBD_tw3':['sum'],
    'tw1':['sum'],
    'tw2':['sum'],
    'tw3':['sum'],


    'AMT_PAYMENT_tw1':['sum'],
    
    'AMT_PAYMENT_tw2':['sum'],
    'AMT_PAYMENT_tw3':['sum']
    
}

    
ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])


# Divide the aggregated weighted features by the sum of the weights for some time-decay-weighted features

# In[ ]:



ins_agg['INSTAL_weighted_tw1_DPD_SUM']=ins_agg['INSTAL_DPD_tw1_SUM']/ins_agg['INSTAL_tw1_SUM']
ins_agg['INSTAL_weighted_tw2_DPD_SUM']=ins_agg['INSTAL_DPD_tw2_SUM']/ins_agg['INSTAL_tw2_SUM']
ins_agg['INSTAL_weighted_tw3_DPD_SUM']=ins_agg['INSTAL_DPD_tw3_SUM']/ins_agg['INSTAL_tw3_SUM']

ins_agg['INSTAL_weighted_tw1_DBD_SUM']=ins_agg['INSTAL_DBD_tw1_SUM']/ins_agg['INSTAL_tw1_SUM']
ins_agg['INSTAL_weighted_tw2_DBD_SUM']=ins_agg['INSTAL_DBD_tw2_SUM']/ins_agg['INSTAL_tw2_SUM']
ins_agg['INSTAL_weighted_tw3_DBD_SUM']=ins_agg['INSTAL_DBD_tw3_SUM']/ins_agg['INSTAL_tw3_SUM']

ins_agg['INSTAL_weighted_tw1_AMTPAY_SUM']=ins_agg['INSTAL_AMT_PAYMENT_tw1_SUM']/ins_agg['INSTAL_tw1_SUM']
ins_agg['INSTAL_weighted_tw2_AMTPAY_SUM']=ins_agg['INSTAL_AMT_PAYMENT_tw2_SUM']/ins_agg['INSTAL_tw2_SUM']
ins_agg['INSTAL_weighted_tw3_AMTPAY_SUM']=ins_agg['INSTAL_AMT_PAYMENT_tw3_SUM']/ins_agg['INSTAL_tw3_SUM']


# In[ ]:


ins_agg.head()


# How do we know if the features are any good? Let's try to predict the target values based only on the data in the frame.
# 
# Note - this is only a quick-and-dirty lightgbm run with no tuning, regularisation etc. as I only want a rough idea of feature importance.

# In[ ]:


targets=pd.read_csv('../input/application_train.csv')[['SK_ID_CURR', 'TARGET']]

ins_agg=ins_agg.join(targets.set_index('SK_ID_CURR'))

train=ins_agg[pd.notnull(ins_agg['TARGET'])]
train=train.fillna(0)
target = np.array(train['TARGET'])
train=train.drop(['TARGET'],axis=1)


# In[ ]:


estimator=lgb.LGBMClassifier()


# In[ ]:


estimator.fit(train, target)


# In[ ]:


import shap

shap_values = shap.TreeExplainer(estimator).shap_values(train[0:2000])
shap.summary_plot(shap_values, train[0:2000], max_display=800)


# Great, some of our time-weighted features look to be pretty useful. There's plenty of scope to improve this method, and to apply it to other data. Good luck!

# 

# 
