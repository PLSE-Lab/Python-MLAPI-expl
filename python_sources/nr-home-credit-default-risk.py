#!/usr/bin/env python
# coding: utf-8

# ### Loading Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,f1_score
import gc


# ### Loading Dataset

# In[2]:


application_test=pd.read_csv('../input/application_test.csv')
application_train=pd.read_csv('../input/application_train.csv')
bureau=pd.read_csv('../input/bureau.csv')
bureau_balance=pd.read_csv('../input/bureau_balance.csv')
credit_card_balance=pd.read_csv('../input/credit_card_balance.csv')
installments_payments=pd.read_csv('../input/installments_payments.csv')
POS_CASH_balance=pd.read_csv('../input/POS_CASH_balance.csv')
previous_application=pd.read_csv('../input/previous_application.csv')


# ### Defining some useful functions

# In[3]:


def basic_info(df):
    print('Num of rows and columns: ',df.shape)
    print('Missing value status: ',df.isnull().values.any())
    print('Columns names:\n ',df.columns.values)
    return df.head()


# In[4]:


def check_missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# In[5]:


def categorical_features(df):
    cat_features=df.columns[df.dtypes=='object']
    return list(cat_features)


# In[6]:


def onehot_encoding(df,cat_features_name):
    df=pd.get_dummies(df,columns=cat_features_name)
    return df


# ### pre-processing bureau

# In[7]:


basic_info(bureau)


# In[8]:


categorical_features(bureau)


# In[9]:


bureau.CREDIT_ACTIVE.value_counts()


# In[10]:


bureau.CREDIT_CURRENCY.value_counts()


# In[11]:


bureau.CREDIT_TYPE.value_counts()


# In[12]:


check_missing_data(bureau)[0:9]


# In[13]:


bureau.AMT_CREDIT_SUM.plot()


# In[14]:


bureau.AMT_CREDIT_SUM.plot(kind='box')


# In[15]:


bureau.AMT_CREDIT_SUM.describe()


# In[16]:


print(bureau.AMT_CREDIT_SUM.max())
print(bureau.AMT_CREDIT_SUM.mean())
print(bureau.AMT_CREDIT_SUM.median())


# Filling the NAN values in AMT_CREDIT_SUM column by median value.

# In[17]:


bureau.AMT_CREDIT_SUM.fillna(value=bureau.AMT_CREDIT_SUM.median(),inplace=True)


# In[18]:


bureau.DAYS_CREDIT_ENDDATE.describe()


# Filling the NAN values in 'DAYS_CREDIT_ENDDATE' column by values in "DAYS_ENDDATE_FACT", if it is posible. 

# In[19]:


bureau['DAYS_CREDIT_ENDDATE']=np.where(bureau.DAYS_CREDIT_ENDDATE.isnull(),bureau.DAYS_ENDDATE_FACT,bureau.DAYS_CREDIT_ENDDATE)


# In[20]:


bureau.DAYS_CREDIT_ENDDATE.describe()


# In[21]:


bureau.DAYS_CREDIT_ENDDATE.plot(kind='box')


# Filling the remaining NAN values in DAYS_CREDIT_ENDDATE by 0

# In[22]:


bureau.DAYS_CREDIT_ENDDATE.fillna(value=0.0,inplace=True)


# In[23]:


bureau.drop('DAYS_ENDDATE_FACT',axis=1,inplace=True)


# In[24]:


bureau[['AMT_ANNUITY','AMT_CREDIT_MAX_OVERDUE','AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_DEBT']].describe()


# In[25]:


bureau.AMT_CREDIT_MAX_OVERDUE.fillna(0.0,inplace=True)


# In[26]:


bureau.AMT_CREDIT_SUM_LIMIT.fillna(0.0,inplace=True)


# In[27]:


bureau.AMT_CREDIT_SUM_DEBT.fillna(0.0,inplace=True)


# In[28]:


bureau.drop('AMT_ANNUITY',axis=1,inplace=True)


# In[29]:


check_missing_data(bureau).head()


# In[30]:


bureau_onehot=onehot_encoding(bureau,categorical_features(bureau))
bureau_onehot.head()


# In[31]:


check_missing_data(bureau_onehot).head()


# In[32]:


# clean up. Free RAM space
del bureau
gc.collect()


# ### pre-processing bureau_balance

# In[33]:


basic_info(bureau_balance)


# In[34]:


month_count=bureau_balance.groupby('SK_ID_BUREAU').size()


# In[35]:


bureau_balance.STATUS.value_counts()


# In[36]:


bureau_balance_unstack=bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize = False).unstack('STATUS')
bureau_balance_unstack.columns=['status_DPD0','status_DPD1','status_DPD2','status_DPD3','status_DPD4','status_DPD5','status_closed','status_X']
bureau_balance_unstack['month_count']=month_count
bureau_balance_unstack.fillna(value=0,inplace=True)
bureau_balance_unstack.head()


# In[37]:


# clean up. Free RAM space
del bureau_balance
gc.collect()


# ### Merge bureau and bureau_balance

# In[38]:


bureau_merge=bureau_onehot.merge(bureau_balance_unstack,how='left',on='SK_ID_BUREAU')


# In[39]:


cnt_id_bureau=bureau_merge[['SK_ID_CURR','SK_ID_BUREAU']].groupby('SK_ID_CURR').size()


# In[40]:


bureau_final_median=bureau_merge.groupby('SK_ID_CURR').median().drop('SK_ID_BUREAU',axis=1)
bureau_final_median['cnt_id_bureau']=cnt_id_bureau
bureau_final_median.fillna(0,inplace=True)
bureau_final_median.head()


# In[41]:


# clean up. Free RAM space
del bureau_merge,bureau_onehot,bureau_balance_unstack
gc.collect()


# ### pre-processing previous_application.csv

# In[42]:


basic_info(previous_application)


# In[43]:


categorical_features(previous_application)


# In[44]:


check_missing_data(previous_application).head(10)


# In[45]:


previous_application.drop(['RATE_INTEREST_PRIVILEGED','RATE_INTEREST_PRIMARY'],axis=1,inplace=True)


# In[46]:


previous_application.AMT_CREDIT.fillna(previous_application.AMT_CREDIT.median(),inplace=True)


# In[47]:


previous_application.CHANNEL_TYPE.value_counts()


# In[48]:


previous_application.drop(['PRODUCT_COMBINATION','NAME_TYPE_SUITE',],axis=1,inplace=True)


# In[49]:


previous_application.RATE_DOWN_PAYMENT.plot()


# In[50]:


previous_application.RATE_DOWN_PAYMENT.describe()


# In[51]:


previous_application.RATE_DOWN_PAYMENT.fillna(previous_application.RATE_DOWN_PAYMENT.median(),inplace=True)


# In[52]:


previous_application.AMT_DOWN_PAYMENT.describe()


# In[53]:


previous_application.AMT_DOWN_PAYMENT.plot()


# In[54]:


previous_application.AMT_DOWN_PAYMENT.fillna(0.0,inplace=True)


# In[55]:


previous_application.NFLAG_INSURED_ON_APPROVAL.fillna(0,inplace=True)


# In[56]:


previous_application.AMT_GOODS_PRICE.plot()


# In[57]:


previous_application.AMT_GOODS_PRICE.describe()


# In[58]:


previous_application.AMT_GOODS_PRICE.fillna(previous_application.AMT_GOODS_PRICE.mean(),inplace=True)


# In[59]:


previous_application.AMT_ANNUITY.plot()


# In[60]:


previous_application.AMT_ANNUITY.describe()


# In[61]:


previous_application.AMT_ANNUITY.fillna(previous_application.AMT_ANNUITY.mean(),inplace=True)


# In[62]:


previous_application.CNT_PAYMENT.describe()


# In[63]:


previous_application.CNT_PAYMENT.fillna(previous_application.CNT_PAYMENT.median(),inplace=True)


# In[64]:


previous_application.head()


# In[65]:


previous_application_onehot=onehot_encoding(previous_application,categorical_features(previous_application))


# In[66]:


cnt_id_prev1=previous_application_onehot[['SK_ID_CURR','SK_ID_PREV']].groupby('SK_ID_CURR').size()


# In[67]:


previous_application_mean=previous_application_onehot.groupby('SK_ID_CURR').mean().drop('SK_ID_PREV',axis=1)
previous_application_mean['cnt_id_prev1']=cnt_id_prev1
previous_application_mean.fillna(0,inplace=True)
previous_application_mean.head()


# In[68]:


previous_application_min=previous_application_onehot.groupby('SK_ID_CURR').min().drop('SK_ID_PREV',axis=1)

previous_application_max=previous_application_onehot.groupby('SK_ID_CURR').max().drop('SK_ID_PREV',axis=1)

previous_application_median=previous_application_onehot.groupby('SK_ID_CURR').median().drop('SK_ID_PREV',axis=1)


# In[69]:


previous_application_merge=previous_application_mean.merge(previous_application_min,on='SK_ID_CURR').merge(previous_application_max,on='SK_ID_CURR').merge(previous_application_median,on='SK_ID_CURR')
previous_application_merge['cnt_id_prev1']=cnt_id_prev1
previous_application_merge.fillna(0,inplace=True)
previous_application_merge.head()


# In[70]:


# clean up. Free RAM space
del previous_application,previous_application_max,previous_application_mean,previous_application_min,previous_application_onehot

gc.collect()


# ### pre-processing POS_CASH_balance.csv

# In[71]:


basic_info(POS_CASH_balance)


# In[72]:


POS_CASH_balance.NAME_CONTRACT_STATUS.value_counts()


# In[73]:


check_missing_data(POS_CASH_balance)


# In[74]:


POS_CASH_balance.CNT_INSTALMENT_FUTURE.describe()


# In[75]:


POS_CASH_balance.CNT_INSTALMENT_FUTURE.plot(kind='box')


# In[76]:


POS_CASH_balance.CNT_INSTALMENT_FUTURE.fillna(POS_CASH_balance.CNT_INSTALMENT_FUTURE.median(),inplace=True)


# In[77]:


POS_CASH_balance.CNT_INSTALMENT.plot(kind='box')


# In[78]:


POS_CASH_balance.CNT_INSTALMENT.describe()


# In[79]:


POS_CASH_balance.drop('CNT_INSTALMENT',axis=1,inplace=True)


# In[80]:


POS_CASH_balance_onehot=onehot_encoding(POS_CASH_balance,categorical_features(POS_CASH_balance))
POS_CASH_balance_onehot.head()


# In[81]:


cnt_id_prev2=POS_CASH_balance_onehot[['SK_ID_CURR','SK_ID_PREV']].groupby('SK_ID_CURR').size()


# In[82]:


POS_CASH_balance_median=POS_CASH_balance_onehot.groupby('SK_ID_CURR').median().drop('SK_ID_PREV',axis=1)
POS_CASH_balance_median['cnt_id_prev2']=cnt_id_prev2
POS_CASH_balance_median.fillna(0,inplace=True)
POS_CASH_balance_median.head()


# In[83]:


# clean up. Free RAM space
del POS_CASH_balance,POS_CASH_balance_onehot
gc.collect()


# ### Pre-processing credit_card_balance.csv

# In[84]:


basic_info(credit_card_balance)


# In[85]:


check_missing_data(credit_card_balance).head()


# In[86]:


categorical_features(credit_card_balance)


# In[87]:


credit_card_balance.NAME_CONTRACT_STATUS.value_counts()


# In[88]:


credit_card_balance_onehot=onehot_encoding(credit_card_balance,categorical_features(credit_card_balance))


# In[89]:


credit_card_balance_onehot.fillna(credit_card_balance_onehot.median(),inplace=True)
credit_card_balance.head()


# In[90]:


cnt_id_prev3=credit_card_balance_onehot[['SK_ID_CURR','SK_ID_PREV']].groupby('SK_ID_CURR').size()


# In[91]:


credit_card_balance_median=credit_card_balance_onehot.groupby('SK_ID_CURR').median().drop('SK_ID_PREV',axis=1)
credit_card_balance_median['cnt_id_prev3']=cnt_id_prev3
credit_card_balance_median.fillna(0,inplace=True)
credit_card_balance_median.head()


# In[92]:


# clean up. Free RAM space
del credit_card_balance,credit_card_balance_onehot
gc.collect()


# ### pre-processing installments_payments.csv

# In[93]:


basic_info(installments_payments)


# In[94]:


check_missing_data(installments_payments)


# In[95]:


categorical_features(installments_payments)


# In[96]:


installments_payments.dropna(inplace=True)


# In[97]:


cnt_id_prev4=installments_payments[['SK_ID_CURR','SK_ID_PREV']].groupby('SK_ID_CURR').size()


# In[98]:


installments_payments_min=installments_payments.groupby('SK_ID_CURR').min().drop('SK_ID_PREV',axis=1)
installments_payments_max=installments_payments.groupby('SK_ID_CURR').max().drop('SK_ID_PREV',axis=1)
installments_payments_median=installments_payments.groupby('SK_ID_CURR').median().drop('SK_ID_PREV',axis=1)


# In[99]:


installments_payments_merge=installments_payments_min.merge(installments_payments_max,on='SK_ID_CURR').merge(installments_payments_median,on='SK_ID_CURR')


# In[100]:


installments_payments_merge['cnt_id_prev4']=cnt_id_prev4
installments_payments_merge.fillna(0,inplace=True)
installments_payments_merge.head()


# In[101]:


# clean up. Free RAM space
del installments_payments,installments_payments_max,installments_payments_min
gc.collect()


# ### Pre-processing application_{train|test}.csv

# In[102]:


application_train.head()


# In[103]:


application_test.tail()


# In[104]:


target=application_train['TARGET']


# In[105]:


application_train.drop('TARGET',axis=1,inplace=True)


# In[106]:


application_train['TARGET']=target
application_train.head()


# In[107]:


application_test['TARGET']=-999


# In[108]:


df=pd.concat([application_train,application_test])


# In[109]:


check_missing_data(df).head()


# In[110]:


categorical_features(df)


# In[111]:


df_onehot=onehot_encoding(df,categorical_features(df))
df_onehot.shape


# In[112]:


df_onehot.fillna(0,inplace=True)


# In[113]:


check_missing_data(df_onehot).head()


# In[114]:


# clean up. Free RAM space
del application_test,application_train,df
gc.collect()


# ### Merging All the dataset

# In[115]:


total=df_onehot.merge(right=bureau_final_median,on='SK_ID_CURR',how='left').merge(right=previous_application_median,on='SK_ID_CURR',how='left').merge(right=POS_CASH_balance_median,on='SK_ID_CURR',how='left').merge(right=credit_card_balance_median,on='SK_ID_CURR',how='left').merge(right=installments_payments_merge,on='SK_ID_CURR',how='left')

total.shape


# In[116]:


df_total=total.fillna(0)
df_total.head()


# In[117]:


# clean up. Free RAM space
del total,df_onehot,bureau_final_median,previous_application_merge,previous_application_median
del POS_CASH_balance_median,credit_card_balance_median,installments_payments_median,installments_payments_merge
gc.collect()


# ### Final dataset for Machine Learning

# In[118]:


df_train=df_total[df_total.TARGET!=-999]
# print(df_train.shape)
# print(application_train.shape)
# df_train.head()


# In[119]:


df_test=df_total[df_total.TARGET==-999]
# print(df_test.shape)
# print(application_test.shape)
# df_test.head()


# In[120]:


test=df_test.drop(columns=["SK_ID_CURR",'TARGET'],axis=1)
test.shape


# In[121]:


y=df_train['TARGET'].values
y


# In[122]:


train=df_train.drop(columns=["SK_ID_CURR",'TARGET'],axis=1).values
train.shape


# In[123]:


# clean up. Free RAM space
del df_train,df_test,df_total
gc.collect()


# In[125]:


gc.collect()


# ### Splitting into train and test set

# In[126]:


from sklearn.model_selection import train_test_split


# In[127]:


X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2)


# In[128]:


# clean up. Free RAM space
del train
gc.collect()


# In[129]:


import lightgbm


# In[130]:


train_data=lightgbm.Dataset(X_train,label=y_train)
valid_data=lightgbm.Dataset(X_test,label=y_test)


# In[131]:



params = {'boosting_type': 'gbdt',
          'max_depth' : 10,
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 64,
          'learning_rate': 0.1,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.005,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'auc'
          }


# In[132]:


lgbm = lightgbm.train(params,
                 train_data,
                 25000,
                 valid_sets=valid_data,
                 early_stopping_rounds= 80,
                 verbose_eval= 10
                 )


# In[133]:


#Predict on test set and write to submit
predictions_lgbm_prob = lgbm.predict(test.values)


# In[134]:


sub=pd.read_csv('../input/sample_submission.csv')


# In[135]:


sub.TARGET=predictions_lgbm_prob


# In[136]:


sub.to_csv('sub.csv',index=False)


# In[ ]:




