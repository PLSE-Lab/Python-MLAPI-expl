#!/usr/bin/env python
# coding: utf-8

# In[27]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[28]:


from fastai.imports import *
from fastai.structured import *
from fastai.column_data import *
from sklearn import metrics


# In[29]:


app_train = pd.read_csv("../input/application_train.csv")
app_test = pd.read_csv("../input/application_test.csv")


# In[30]:


app_train = app_train.fillna(0)
app_test = app_test.fillna(0)


# In[31]:


app_test["TARGET"] = 0


# In[32]:


cat_vars = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_TYPE_SUITE", 
            "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "FLAG_MOBIL",
            "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL", "OCCUPATION_TYPE",
            "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START",
            "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
            "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "ORGANIZATION_TYPE", 
            'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',
           'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
           'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
           'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
           'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
           'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', "AMT_REQ_CREDIT_BUREAU_HOUR", 
           'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
           'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
           ]

contin_vars = ["CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "REGION_POPULATION_RELATIVE", 
               "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE", "CNT_FAM_MEMBERS",
               'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG',
               'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
               'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG',
               'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
               'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE',
               'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
               'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI',
               'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI',
               'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI',
               'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
               'TOTALAREA_MODE', 
               'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
               'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE'
               ]


# In[33]:


index="SK_ID_CURR"
n = len(app_train); n
for df in (app_test,app_train):
    df.set_index(index)


# In[34]:


for v in cat_vars: app_train[v] = app_train[v].astype('category').cat.as_ordered()


# In[35]:


apply_cats(app_test, app_train)


# In[36]:


for v in contin_vars:
    app_train[v] = app_train[v].astype('float32')
    app_test[v] = app_test[v].astype('float32')


# In[37]:


df, y, nas, mapper = proc_df(app_train, 'TARGET', do_scale=True)
df_test, _, _, _ = proc_df(app_test, 'TARGET', do_scale=True, mapper=mapper, na_dict=nas)


# In[38]:


train_ratio = 0.9
train_size = int(len(df) * train_ratio)
val_idx = list(range(train_size, len(df)))


# In[39]:


md = ColumnarModelData.from_data_frame("", val_idx, df, y, cat_flds=cat_vars, bs=128, is_reg=None, test_df=df_test)


# In[40]:


cat_sz = [(c, len(app_train[c].cat.categories)+1) for c in cat_vars]


# In[41]:


emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]


# In[42]:


m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars), 0.06, 2,  [100,50], [0.03,0.06], None, True)


# In[43]:


from sklearn.metrics import roc_auc_score
def imbalanced_loss(inp,targ):
    return F.nll_loss(inp,targ,weight=T([.1,.9]))
def auc(inp,targ):
    return roc_auc_score(to_np(targ),to_np(np.exp(inp[:,1])))


# In[44]:


m.crit = imbalanced_loss


# In[45]:


m.lr_find()


# In[46]:


m.sched.plot(100)


# In[47]:


lr = 1e-2


# In[ ]:


m.fit(lr, 1, metrics=[auc])


# In[ ]:


m.fit(lr, 3, cycle_len=1, cycle_mult=2, metrics=[auc])


# In[ ]:


preds = np.exp(m.predict(True))[:,1]


# In[ ]:


app_test["TARGET"] = preds


# In[ ]:


app_test[['SK_ID_CURR', 'TARGET']].to_csv("submit_fastai_trial.csv", index=False, float_format='%.8f')


# In[ ]:




