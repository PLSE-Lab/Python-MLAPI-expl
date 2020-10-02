#!/usr/bin/env python
# coding: utf-8

# <font color='blue'><I>Please upvote if you find this Notebook useful</I></font><br><br><br>
# 
# <h1>My Learnings and mistakes in this competition</h1>
# <h2><u>Learnings</u></h2>
# All Learning points are covered in this notebook
# <ol>
#     <li>Handling large number of features
#         <ul>
#             <li>Had around 1000 features
#             <li>Identification of most valuable features
#             <li>Feature Engineering and Feature selection are most important aspects of Machine Learning
#             <li>Good Feature Engineering and Feature selection super seeds superior hardware
#             <li>Different set of features perform better with different algorithms. 
#         </ul>
#     <li>Combining different tables together and preparing them for model fitting
#         <ul>
#             <li>Tables have well defined relationships for creating joins
#             <li>Data from supporting tables is grouped before joining it with final tables
#             <li>Grouping process is a very good application of <b>groupby aggregation functionality </b>
#         </ul>
#     <li>Extensive use of Groupby and aggregation on supporting table
#     <li>Manual feature engineering
#     <li>Dropping highly correlated features.
#    <li>Categorization of observations can help improving score. 
#     <li>Feature selection and feature exclusion. Check following Notebooks for Feature selection and exclusion:
#         <ul>
#       <li> <a href="https://www.kaggle.com/rahullalu/hcdr-installments-table-feature-selection">https://www.kaggle.com/rahullalu/hcdr-installments-table-feature-selection</a>
#         <li><a href="https://www.kaggle.com/rahullalu/hcdr-feature-selection-for-pos-table">https://www.kaggle.com/rahullalu/hcdr-feature-selection-for-pos-table</a>
#        <li><a href="https://www.kaggle.com/rahullalu/home-credit-default-risk-preparing-bureau-data">https://www.kaggle.com/rahullalu/home-credit-default-risk-preparing-bureau-data</a>
#        <li><a href="https://www.kaggle.com/rahullalu/hcdr-feature-selection-for-creditcard-table">https://www.kaggle.com/rahullalu/hcdr-feature-selection-for-creditcard-table</a>
#         </ul>
#     <li> Ensembling: With Private score of 0.79134
#     
#  </ol>
# <h2><u>Mistakes</u></h2>
# Mistake of not exploring all available Boosting models. Biggest mistake :(
# <ol>
#     <li>Extensively worked with XGBoost and LGBM. But not worked extensively with CatBoost.
#     <li>Got best result with CatBoost. Realized pretty late.
#     <li>Thought LGBM is the best model. Got CV score of 0.7869.
#     <li>Realized this mistake on last day. So was not able to submit best score with CatBoost.
#     <li>Didn't worked extensively on ensemble. Ensemble techinques helped a lot in improving scores.
#     
# </ol>

# ![](http://)<h2>Problem Statement</h2>
# The objective of this competition is to use historical loan application data to predict whether or not an applicant will be able to repay a loan. This is a standard supervised classification task:
# 
# **Supervised:** The labels are included in the training data and the goal is to train a model to learn to predict the labels from the features<br>
# **Classification:** The label is a binary variable, 0 (will repay loan on time), 1 (will have difficulty repaying loan)

# 
# <img src="https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png" alt="Count of Operation" height="800" width="800"></img>
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import KFold,StratifiedKFold

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# <h2>Dataset with file size</h2>

# In[ ]:


#DATASET VIEW
path1="../input/"
data_files=list(os.listdir(path1))
df_files=pd.DataFrame(data_files,columns=['File_Name'])
df_files['Size_in_MB']=df_files.File_Name.apply(lambda x:round(os.stat(path1+x).st_size/(1024*1024),2))
df_files


# In[ ]:


#ALL FUNCTIONS

#FUNCTION FOR PROVIDING FEATURE SUMMARY
def feature_summary(df_fa):
    print('DataFrame shape')
    print('rows:',df_fa.shape[0])
    print('cols:',df_fa.shape[1])
    col_list=['Null','Unique_Count','Data_type','Max/Min','Mean','Median','Std','Skewness','Sample_values']
    df=pd.DataFrame(index=df_fa.columns,columns=col_list)
    df['Null']=list([len(df_fa[col][df_fa[col].isnull()]) for i,col in enumerate(df_fa.columns)])
    #df['%_Null']=list([len(df_fa[col][df_fa[col].isnull()])/df_fa.shape[0]*100 for i,col in enumerate(df_fa.columns)])
    df['Unique_Count']=list([len(df_fa[col].unique()) for i,col in enumerate(df_fa.columns)])
    df['Data_type']=list([df_fa[col].dtype for i,col in enumerate(df_fa.columns)])
    for i,col in enumerate(df_fa.columns):
        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):
            df.at[col,'Max/Min']=str(round(df_fa[col].max(),2))+'/'+str(round(df_fa[col].min(),2))
            df.at[col,'Mean']=df_fa[col].mean()
            df.at[col,'Median']=df_fa[col].median()
            df.at[col,'Std']=df_fa[col].std()
            df.at[col,'Skewness']=df_fa[col].skew()
        df.at[col,'Sample_values']=list(df_fa[col].unique())
           
    return(df.fillna('-'))

#FUNCTION TO IDENTIFY HIGHLY CORRELATED FEATURES
def drop_corr_col(df_corr):
    upper = df_corr.where(np.triu(np.ones(df_corr.shape),
                          k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.97
    to_drop = [column for column in upper.columns if any(upper[column] > 0.999)]
    return(to_drop)

#FUNCTION USED FOR GROUPING DATA AND COUNTING UNIQUE VALUES
#USED WITH GROUP BY
def cnt_unique(df):
    return(len(df.unique()))


# In[ ]:


get_ipython().run_cell_magic('time', '', "#READING TRAINGING DATA\ntrain=pd.read_csv(path1+'application_train.csv')\n#READING TEST DATA\ntest=pd.read_csv(path1+'application_test.csv')")


# In[ ]:


#TRAINING DATA VEIW
train.head()


# In[ ]:


#TESTING DATA VIEW
test.head()


# In[ ]:


#COMBINING TRAIN AND TEST DATA
print('Train shape:',train.shape,'Test shape:',test.shape)
df_comb=pd.concat([train.drop(['TARGET'],axis=1),test],ignore_index=True)
df_comb_fs=feature_summary(df_comb)

#IDENTIFYING CATEGORICAL FEATURES
cat_features=df_comb_fs[df_comb_fs.Data_type=='object'].index

#REPLACING SPACE WITH UNDERSCORE IN CATEGORICAL VALUES
#AS CATEGORICAL VALUES WILL APPEAR IN COLUMN NAMES WHEN CONVERTED TO DUMMIES
for col in cat_features:
    df_comb[col]=df_comb[col].apply(lambda x: str(x).replace(" ","_"))

print('categorical features',len(cat_features))


# In[ ]:


#CREATING DUMMIES
#NAN VALUES WILL BE TREATED AS A CATEGORY
cat_train=pd.DataFrame()
for col in cat_features:
    dummy=pd.get_dummies(df_comb[col],prefix='DUM_'+col)
    cat_train=pd.concat([cat_train,dummy],axis=1)
display(cat_train.head())
print('Newly created dummy columns:',cat_train.shape)


# In[ ]:


#REMOVING COLUMNS WILL VERY FEW VALUES OR HIGHLY CORRELATED
cat_train.drop(['DUM_CODE_GENDER_XNA','DUM_NAME_INCOME_TYPE_Businessman','DUM_NAME_INCOME_TYPE_Maternity_leave','DUM_NAME_INCOME_TYPE_Student',
                'DUM_NAME_INCOME_TYPE_Unemployed','DUM_NAME_FAMILY_STATUS_Unknown','DUM_OCCUPATION_TYPE_Security_staff','DUM_FONDKAPREMONT_MODE_nan',
                'DUM_OCCUPATION_TYPE_IT_staff','DUM_ORGANIZATION_TYPE_Police','DUM_ORGANIZATION_TYPE_Telecom','DUM_ORGANIZATION_TYPE_Business_Entity_Type_1'],
               axis=1,inplace=True)
print('Categorical feature count after dropping irrelevant features',cat_train.shape)


# In[ ]:


#COMBINING DUMMIES WITH NON CATEGORICAL FEATURES
df_final=pd.concat([df_comb.drop(cat_features,axis=1),cat_train],axis=1)
print('Final shape:',df_final.shape)
display(df_final.head())


# In[ ]:


#MANUAL FEATURE ENGINEERING ON APPLICATION DATA
df_final['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

df_final['CALC_PERC_DAYS_EMPLOYED'] = df_final['DAYS_EMPLOYED'] / df_final['DAYS_BIRTH']
df_final['CALC_PERC_INCOME_CREDIT'] = df_final['AMT_INCOME_TOTAL'] /df_final['AMT_CREDIT']
df_final['CALC_INCOME_PER_PERSON'] = df_final['AMT_INCOME_TOTAL'] / df_final['CNT_FAM_MEMBERS']
df_final['CALC_ANNUITY_INCOME_PERC'] = df_final['AMT_ANNUITY'] / df_final['AMT_INCOME_TOTAL']
df_final['CALC_PAYMENT_RATE'] = df_final['AMT_ANNUITY'] / df_final['AMT_CREDIT']
df_final['CALC_GOODS_PRICE_PER']=df_final['AMT_GOODS_PRICE']/df_final['AMT_CREDIT']
df_final['CALC_PERC_CHILDREN']=df_final['CNT_CHILDREN']/df_final['CNT_FAM_MEMBERS']
df_final['CALC_RATIO_CAR_TO_BRITH']=df_final['OWN_CAR_AGE'] / df_final['DAYS_BIRTH']
df_final['CALC_RATIO_CAR_TO_EMPLOY'] = df_final['OWN_CAR_AGE'] / df_final['DAYS_EMPLOYED']
df_final['CALC_INCOME_PER_CHILD'] = df_final['AMT_INCOME_TOTAL'] / (1 + df_final['CNT_CHILDREN'])
df_final['CALC_INCOME_PER_PERSON'] = df_final['AMT_INCOME_TOTAL'] / df_final['CNT_FAM_MEMBERS']
df_final['CALC_RATIO_PHONE_TO_BIRTH'] = df_final['DAYS_LAST_PHONE_CHANGE'] / df_final['DAYS_BIRTH']
df_final['CALC_RATIO_PHONE_TO_EMPLOY'] = df_final['DAYS_LAST_PHONE_CHANGE'] / df_final['DAYS_EMPLOYED']


# In[ ]:


#DROPING IRRELEVANT CONTINUOUS FEATURES
df_final.drop(['FLAG_MOBIL','FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_10','FLAG_DOCUMENT_12'],axis=1,inplace=True)


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del dummy
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "#READING BUREAU DATA\nbur=pd.read_csv(path1+'bureau.csv')\nprint('bureau set reading complete...')\n#READING BUREAU BALANCE DATA\nbur_bal=pd.read_csv(path1+'bureau_balance.csv')\nprint('bureau balance set reading complete...')")


# In[ ]:


get_ipython().run_cell_magic('time', '', '#CONVERTING CATEGORICAL FEATURES IN BUREAU DATA TO DUMMIES\nfor col in [\'CREDIT_CURRENCY\',\'CREDIT_TYPE\',\'CREDIT_ACTIVE\']:\n    bur[col]=bur[col].apply(lambda x: str(x).replace(" ","_")) \n\ndummy=pd.DataFrame()\nfor col in [\'CREDIT_CURRENCY\',\'CREDIT_TYPE\',\'CREDIT_ACTIVE\']:\n    dummy=pd.concat([dummy,pd.get_dummies(bur[col],prefix=\'DUM_\'+col)],axis=1)')


# In[ ]:


#COMBINING DUMMIES WITH CONTINUOUS BUREAU FEATURES
bur_f=pd.concat([bur.drop(['CREDIT_CURRENCY','CREDIT_TYPE','CREDIT_ACTIVE'],axis=1),dummy],axis=1)


# In[ ]:


#MANUAL FEATURE ENGINEERING WITH BUREAU DATA
bur_f['CALC_PER_CREDIT_MAX_OVERDUE']=bur_f['AMT_CREDIT_MAX_OVERDUE']/bur_f['AMT_CREDIT_SUM']
bur_f['CALC_PER_CREDIT_SUM_DEBT']=bur_f['AMT_CREDIT_SUM_DEBT']/bur_f['AMT_CREDIT_SUM']
bur_f['CALC_PER_CREDIT_SUM_LIMIT']=bur_f['AMT_CREDIT_SUM_LIMIT']/bur_f['AMT_CREDIT_SUM']
bur_f['CALC_PER_CREDIT_SUM_OVERDUE']=bur_f['AMT_CREDIT_SUM_OVERDUE']/bur_f['AMT_CREDIT_SUM']
bur_f['CALC_PER_ANNUITY']=bur_f['AMT_ANNUITY']/bur_f['AMT_CREDIT_SUM']
bur_f['CALC_CREDIT_LIMIT_CROSSED']=bur_f['AMT_CREDIT_SUM_LIMIT']-bur_f['AMT_CREDIT_SUM']
bur_f['CALC_CREDIT_PER_DAY']=bur_f['AMT_CREDIT_SUM']/bur_f['DAYS_CREDIT_ENDDATE'].abs()
bur_f['CALC_CREDIT_CLOSED']=(bur_f['DAYS_ENDDATE_FACT'] < 0).astype(int)


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del bur,dummy
gc.collect()


# In[ ]:


#CONVERTING MONTHS_BALANCE TO ITS ABSOLUTE VALUE
#FOR MONTHS BALANCE ABSOLUTE VALUE IS GIVING BETTER RESULTS
bur_bal['MONTHS_BALANCE']=bur_bal.MONTHS_BALANCE.abs()


# In[ ]:


get_ipython().run_cell_magic('time', '', "#AGGREGATING BUREAU_BALANCE ON 'SK_ID_BUREAU','STATUS'\n#THIS HELPS IN REDUCING NUMBER OF OBSERVATION AND WE HAVE TO JOIN BUREAU_BALANCE WITH BUREAU TABLE\nbur_bal_f=bur_bal.groupby(['SK_ID_BUREAU','STATUS']).aggregate({'STATUS':['count'],'MONTHS_BALANCE':['max','min']})\nbur_bal_f.reset_index(inplace=True)\nbur_bal_f.columns=['SK_ID_BUREAU','STATUS','STATUS_count','MONTHS_BALANCE_max','MONTHS_BALANCE_min']")


# In[ ]:


#CONVERTING STATUS INTO DUMMIES
dummy=pd.get_dummies(bur_bal_f['STATUS'],prefix='DUM_STATUS')


# In[ ]:


#CONCATENATING STATUS DUMMIES WITH OTHER FEATURES
bur_bal_ff=pd.concat([bur_bal_f.drop(['STATUS'],axis=1),dummy],axis=1)


# In[ ]:


get_ipython().run_cell_magic('time', '', "#DISTRIBUTING STATUS COUNT ON STATUS TYPE FEATURES\ndummy_col=[x for x in bur_bal_ff.columns if 'DUM_' in x]\nfor col in dummy_col:\n    bur_bal_ff[col]=bur_bal_ff.apply(lambda x: x.STATUS_count if x[col]==1 else 0,axis=1)")


# In[ ]:


bur_bal_ff.head(10)


# In[ ]:


#DROPPING STATUS_count FEATURE
bur_bal_ff.drop('STATUS_count',axis=1,inplace=True)


# In[ ]:


#DEFINING AGGREGATION RULES FOR GROUPING BUREAU_BALANCE DATA ON 'SK_ID_BUREAU'
bur_bal_cols=[x for x in list(bur_bal_ff.columns) if x not in ['SK_ID_BUREAU']]
bur_bal_agg={}
bur_bal_name=['SK_ID_BUREAU']
for col in bur_bal_cols:
    if 'DUM_' in col:
        bur_bal_agg[col]=['sum']
        bur_bal_name.append(col)
    elif '_max' in col:
        bur_bal_agg[col]=['max']
        bur_bal_name.append(col)
    elif '_min' in col:
        bur_bal_agg[col]=['min']
        bur_bal_name.append(col)
    else:
        bur_bal_agg[col]=['sum','mean']
        bur_bal_name.append(col+'_'+'sum')
        bur_bal_name.append(col+'_'+'mean')


# In[ ]:


get_ipython().run_cell_magic('time', '', "#GROUPING BUREAU_BALANCE DATA ON 'SK_ID_BUREAU'\nbur_bal_fg=bur_bal_ff.groupby('SK_ID_BUREAU').aggregate(bur_bal_agg)\nbur_bal_fg.reset_index(inplace=True)\nbur_bal_fg.columns=bur_bal_name")


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del bur_bal,bur_bal_f,bur_bal_ff
gc.collect()


# In[ ]:


#JOINING BUREAU AND BUREAU_BALANCE TABLES ON 'SK_ID_BUREAU'
bur_combi=bur_f.join(bur_bal_fg.set_index('SK_ID_BUREAU'),on='SK_ID_BUREAU',lsuffix='_BU', rsuffix='_BUB')


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del bur_bal_fg
gc.collect()


# In[ ]:


#DEFINING AGGREGATION RULES FOR BUREAU COMBINED DATA TO GROUP ON 'SK_ID_CURR','SK_ID_BUREAU'
bur_combi_cols=[x for x in list(bur_combi.columns) if x not in ['SK_ID_CURR','SK_ID_BUREAU']]
bur_combi_agg={}
bur_combi_name=['SK_ID_CURR','SK_ID_BUREAU']
for col in bur_combi_cols:
    if 'DUM_' in col:
        bur_combi_agg[col]=['sum']
        bur_combi_name.append(col+'_'+'sum')
    elif 'AMT_' in col:
        bur_combi_agg[col]=['sum','mean','max','min','var','std']
        bur_combi_name.append(col+'_'+'sum')
        bur_combi_name.append(col+'_'+'mean')
        bur_combi_name.append(col+'_'+'max')
        bur_combi_name.append(col+'_'+'min')
        bur_combi_name.append(col+'_'+'var')
        bur_combi_name.append(col+'_'+'std')
    elif 'CNT_' in col:
        bur_combi_agg[col]=['sum','max','min','count']
        bur_combi_name.append(col+'_'+'sum')
        bur_combi_name.append(col+'_'+'max')
        bur_combi_name.append(col+'_'+'min')
        bur_combi_name.append(col+'_'+'count')
    elif 'DAYS_' in col:
        bur_combi_agg[col]=['sum','max','min']
        bur_combi_name.append(col+'_'+'sum')
        bur_combi_name.append(col+'_'+'max')
        bur_combi_name.append(col+'_'+'min')
    elif 'CALC_' in col:
        bur_combi_agg[col]=['mean']
        bur_combi_name.append(col+'_'+'mean')
    else:
        bur_combi_agg[col]=['sum']
        bur_combi_name.append(col+'_'+'sum')
       


# In[ ]:


get_ipython().run_cell_magic('time', '', "#GROUPING BUREAU COMBINED DATA ON 'SK_ID_CURR','SK_ID_BUREAU'\nbur_combi_f=bur_combi.groupby(['SK_ID_CURR','SK_ID_BUREAU']).aggregate(bur_combi_agg)                 \nbur_combi_f.reset_index(inplace=True)\nbur_combi_f.columns=bur_combi_name")


# In[ ]:


#DEFINING AGGREGATION RULES FOR BUREAU COMBINED DATA TO GROUP ON 'SK_ID_CURR'
bur_combi_cols=list(bur_combi_f.columns)
bur_combi_agg={}
bur_combi_name=['SK_ID_CURR']
for col in bur_combi_cols:
    if 'SK_ID_CURR'==col:
        bur_combi_agg[col]=['count']
        bur_combi_name.append('SK_ID_BUREAU_count')
    elif '_sum'==col:
        bur_combi_agg[col]=['sum']
        bur_combi_name.append(col)
    elif '_mean' in col:
        bur_combi_agg[col]=['mean']
        bur_combi_name.append(col)
    elif '_max' in col:
        bur_combi_agg[col]=['max']
        bur_combi_name.append(col)
    elif '_min' in col:
        bur_combi_agg[col]=['min']
        bur_combi_name.append(col)
    elif '_count' in col:
        bur_combi_agg[col]=['sum']
        bur_combi_name.append(col)
    elif '_var' in col:
        bur_combi_agg[col]=['mean']
        bur_combi_name.append(col)
    elif '_std' in col:
        bur_combi_agg[col]=['mean']
        bur_combi_name.append(col)
    else:
        bur_combi_agg[col]=['sum']
        bur_combi_name.append(col)


# In[ ]:


get_ipython().run_cell_magic('time', '', "#GROUPING BUREAU COMBINED DATA ON 'SK_ID_CURR'\nbur_combi_fg=bur_combi_f.groupby(['SK_ID_CURR']).aggregate(bur_combi_agg)                 \nbur_combi_fg.reset_index(inplace=True)\nbur_combi_fg.columns=bur_combi_name")


# In[ ]:


#DROPPING IRRELEVANT FEATURES
bur_combi_fg.drop(['DUM_CREDIT_TYPE_Car_loan_sum', 'DUM_CREDIT_CURRENCY_currency_2_sum', 'AMT_CREDIT_SUM_OVERDUE_var'],axis=1,inplace=True)


# In[ ]:


#COMBINING BUREAUE COMBINED DATA WITH APPLICATION DATA
df_final=df_final.join(bur_combi_fg.set_index('SK_ID_CURR'),on='SK_ID_CURR',lsuffix='_AP', rsuffix='_BU')


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del bur_combi,bur_combi_f,bur_combi_fg
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "#READING PREVIOUS APPLICATION DATA\nprev_appl=pd.read_csv(path1+'previous_application.csv')\nprint('previous_application set reading complete...')")


# In[ ]:


#ENCODING CATEGORICAL FEATURES
prev_appl_encoding={'NAME_YIELD_GROUP':{'middle':3,'low_action':1,'high':4,'low_normal':1,'XNA':0},
                   'WEEKDAY_APPR_PROCESS_START':{'MONDAY':1,'TUESDAY':2,'WEDNESDAY':3,'THURSDAY':4,'FRIDAY':5,'SATURDAY':6,'SUNDAY':7},
                   'FLAG_LAST_APPL_PER_CONTRACT':{'Y':1,'N':0}}
prev_appl.replace(prev_appl_encoding,inplace=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "#REPLACING 365243 IN DAYS_ FEATURES WITH NAN\ndays_col=[x for x in prev_appl.columns if 'DAYS_' in x]\nfor col in days_col:\n    prev_appl.replace(365243,np.nan,inplace= True)")


# In[ ]:


#CREAGING FEATURE SUMMARY
prev_appl_fs=feature_summary(prev_appl)


# In[ ]:


#CATEGORICAL FEATURES
prev_appl_fs[prev_appl_fs.Data_type=='object']


# In[ ]:


get_ipython().run_cell_magic('time', '', '#CONVERTING CATEGORICAL FEATURES INTO DUMMIES\nprev_appl_cf=list(prev_appl_fs[prev_appl_fs.Data_type==\'object\'].index)\n\nfor col in prev_appl_cf:\n    prev_appl[col]=prev_appl[col].apply(lambda x: str(x).replace(" ","_"))    \n    \nprint(\'previous application categorical features\',len(prev_appl_cf))\nprev_appl_cf.remove(\'NAME_CONTRACT_STATUS\')\n\nprev_appl_cat=pd.DataFrame()\nfor col in prev_appl_cf:\n    dummy=pd.get_dummies(prev_appl[col],prefix=\'DUM_\'+col)\n    prev_appl_cat=pd.concat([prev_appl_cat,dummy],axis=1)\ndisplay(prev_appl_cat.head())')


# In[ ]:


#DROPPING IRRELEVANT FEATURES
prev_appl_cat.drop(['DUM_NAME_CASH_LOAN_PURPOSE_XAP', 'DUM_PRODUCT_COMBINATION_nan'],axis=1,inplace=True)


# In[ ]:


print('Newly created Dummy features:',len(prev_appl_cat.columns))


# In[ ]:


#COMBINING DUMMIES WITH OTHER CONTINUOUS FEATURES
prev_appl_f=pd.concat([prev_appl.drop(prev_appl_cf,axis=1),prev_appl_cat],axis=1)
# with pd.option_context('display.max_columns',prev_appl_f.shape[1]):
#     display(prev_appl_f.head(10))


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del prev_appl,prev_appl_cat,prev_appl_fs
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "#MANUAL FEATURE ENGINEERING ON PREVIOUS_APPLICATION DATA\nprev_appl_f['CALC_PERC_ANNUITY']=prev_appl_f['AMT_ANNUITY']/prev_appl_f['AMT_CREDIT']\nprev_appl_f['CALC_PERC_AMT_APPLICATION']=prev_appl_f['AMT_APPLICATION']/prev_appl_f['AMT_CREDIT']\nprev_appl_f['CALC_PERC_DOWN_PAYMENT']=prev_appl_f['AMT_DOWN_PAYMENT']/prev_appl_f['AMT_CREDIT']\nprev_appl_f['CALC_PERC_APPL_DOWN_PAYMENT']=prev_appl_f['AMT_DOWN_PAYMENT']/prev_appl_f['AMT_APPLICATION']\nprev_appl_f['CALC_PERC_GOODS_PRICE']=prev_appl_f['AMT_GOODS_PRICE']/prev_appl_f['AMT_CREDIT']\nprev_appl_f['CALC_FLAG_PRIVILEGED_CUSTOMER']=(prev_appl_f['RATE_INTEREST_PRIVILEGED']>0).astype(int)\nprev_appl_f['CALC_TERMINATED']=(prev_appl_f['DAYS_TERMINATION']<0).astype(int)\nprev_appl_f['CALC_AMT_ANNUITY_PER_PAY']=prev_appl_f['AMT_ANNUITY']/prev_appl_f['CNT_PAYMENT']\nprev_appl_f['CALC_AMT_APPLICATION_PER_PAY']=prev_appl_f['AMT_APPLICATION']/prev_appl_f['CNT_PAYMENT']\nprev_appl_f['CALC_AMT_DOWN_PAYMENT_PER_PAY']=prev_appl_f['AMT_DOWN_PAYMENT']/prev_appl_f['CNT_PAYMENT']\nprev_appl_f['CALC_AMT_APPL_DOWN_PAYMENT_PER_PAY']=prev_appl_f['AMT_DOWN_PAYMENT']/prev_appl_f['CNT_PAYMENT']\nprev_appl_f['CALC_AMT_GOODS_PRICE_PER_PAY']=prev_appl_f['AMT_GOODS_PRICE']/prev_appl_f['CNT_PAYMENT']\nprev_appl_f['CALC_AMT_CREDIT_APP_DIFF']=prev_appl_f['AMT_CREDIT']-prev_appl_f['AMT_APPLICATION']\nprev_appl_f['CALC_LAST_DUE_AFTER_APPL']=(prev_appl_f['DAYS_LAST_DUE_1ST_VERSION']>0).astype(int)")


# In[ ]:


print('Previoun application data shape:',prev_appl_f.shape)


# In[ ]:


#DEFINING AGGREGATION RULES FOR GROUPING PREVIOUS APPLICATION DATA ON 'SK_ID_CURR','SK_ID_PREV','NAME_CONTRACT_STATUS'
prev_cols=[x for x in list(prev_appl_f.columns) if x not in ['SK_ID_CURR','SK_ID_PREV','NAME_CONTRACT_STATUS']]
prev_agg={}
prev_name=['SK_ID_CURR','SK_ID_PREV','NAME_CONTRACT_STATUS']
for col in prev_cols:
    if 'DUM_' in col:
        prev_agg[col]=['sum','mean','std']
        prev_name.append(col+'_'+'sum')
        prev_name.append(col+'_'+'mean')
        prev_name.append(col+'_'+'std')
    elif 'AMT_' in col:
        prev_agg[col]=['sum','mean','max','min','var','std']
        prev_name.append(col+'_'+'sum')
        prev_name.append(col+'_'+'mean')
        prev_name.append(col+'_'+'max')
        prev_name.append(col+'_'+'min')
        prev_name.append(col+'_'+'var')
        prev_name.append(col+'_'+'std')
    elif 'CNT_' in col:
        prev_agg[col]=['sum','max','min','size','count']
        prev_name.append(col+'_'+'sum')
        prev_name.append(col+'_'+'max')
        prev_name.append(col+'_'+'min')
        prev_name.append(col+'_'+'size')
        prev_name.append(col+'_'+'count')
    elif 'DAYS_' in col:
        prev_agg[col]=['sum','max','min']
        prev_name.append(col+'_'+'sum')
        prev_name.append(col+'_'+'max')
        prev_name.append(col+'_'+'min')
    elif 'CALC_FLAG_' in col:
        prev_agg[col]=['sum']
        prev_name.append(col+'_'+'sum')
    elif 'CALC_' in col:
        prev_agg[col]=['mean']
        prev_name.append(col+'_'+'mean')
    else:
        prev_agg[col]=['sum','mean']
        prev_name.append(col+'_'+'sum')
        prev_name.append(col+'_'+'mean')


# In[ ]:


get_ipython().run_cell_magic('time', '', "#GROUPING PREVIOUS APPLICATION DATA ON 'SK_ID_CURR','SK_ID_PREV','NAME_CONTRACT_STATUS'\nprev_appl_ff=prev_appl_f.groupby(['SK_ID_CURR','SK_ID_PREV','NAME_CONTRACT_STATUS']).aggregate(prev_agg)\nprev_appl_ff.reset_index(inplace=True)\nprev_appl_ff.columns=prev_name")


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del prev_appl_f
gc.collect()


# In[ ]:


#CONVERTING 'NAME_CONTRACT_STATUS' TO DUMMIES
dummy=pd.get_dummies(prev_appl_ff['NAME_CONTRACT_STATUS'],prefix='DUM_'+'NAME_CONTRACT_STATUS')


# In[ ]:


#COMBINING DUMMIES WITH OTHER CONTINUOUS FEATURES 
prev_appl_fg=pd.concat([prev_appl_ff.iloc[:,:1],dummy,prev_appl_ff.iloc[:,3:]],axis=1)


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del prev_appl_ff,dummy
gc.collect()


# In[ ]:


#DEFINING AGGREGATION RULES FOR GROUPING PREVIOUS APPLICATION DATA ON 'SK_ID_CURR'
prev_cols=list(prev_appl_fg.columns)
prev_agg={}
prev_name=['SK_ID_CURR']
for col in prev_cols:
    if 'SK_ID_CURR'==col:
        prev_agg[col]=['count']
        prev_name.append('SK_ID_PREV_count')
    elif '_sum' in col:
        prev_agg[col]=['sum']
        prev_name.append(col)
    elif '_mean' in col:
        prev_agg[col]=['mean']
        prev_name.append(col)
    elif '_max' in col:
        prev_agg[col]=['max']
        prev_name.append(col)
    elif '_min' in col:
        prev_agg[col]=['min']
        prev_name.append(col)
    elif '_var' in col:
        prev_agg[col]=['mean']
        prev_name.append(col)
    elif '_std' in col:
        prev_agg[col]=['mean']
        prev_name.append(col)
    elif '_size' in col:
        prev_agg[col]=['mean']
        prev_name.append(col)
    elif '_count' in col:
        prev_agg[col]=['sum']
        prev_name.append(col)
    else:
        prev_agg[col]=['sum']
        prev_name.append(col)
       


# In[ ]:


get_ipython().run_cell_magic('time', '', "#GROUPING PREVIOUS APPLICATION DATA ON 'SK_ID_CURR'\nprev_appl_fgg=prev_appl_fg.groupby(['SK_ID_CURR']).aggregate(prev_agg)\nprev_appl_fgg.reset_index(inplace=True)\nprev_appl_fgg.columns=prev_name")


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del prev_appl_fg
gc.collect()


# In[ ]:


#JOINING PREVIOUS APPLICATION DATA WITH FINAL TABLE (CONTAINING APPLICATION AND BUREAU DATA)
df_final=df_final.join(prev_appl_fgg.set_index('SK_ID_CURR'),on='SK_ID_CURR',lsuffix='_AP', rsuffix='_PV')


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del prev_appl_fgg
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "#READING INSTALLMENTS_PAYMENTS DATA\ninst_pay=pd.read_csv(path1+'installments_payments.csv')\nprint('installments_payments set reading complete...')")


# In[ ]:


#TAKING ABSOLUTE VALUE FOR DAYS_ FEATURES
for col in inst_pay.columns:
    if 'DAYS_' in col:
        inst_pay[col]=inst_pay[col].abs()


# In[ ]:


get_ipython().run_cell_magic('time', '', "#MANUAL FEATURE ENGINEERING\n\ninst_pay['CALC_DAYS_LATE_PAYMENT']=inst_pay['DAYS_ENTRY_PAYMENT']-inst_pay['DAYS_INSTALMENT']\ninst_pay['CALC_PERC_LESS_PAYMENT']=inst_pay['AMT_PAYMENT']/inst_pay['AMT_INSTALMENT']\ninst_pay['CALC_PERC_LESS_PAYMENT'].replace(np.inf,0,inplace=True)\ninst_pay['CALC_DIFF_INSTALMENT']=inst_pay['AMT_INSTALMENT']-inst_pay['AMT_PAYMENT']\ninst_pay['CALC_PERC_DIFF_INSTALMENT']=np.abs(inst_pay['CALC_DIFF_INSTALMENT'])/inst_pay['AMT_INSTALMENT']\ninst_pay['CALC_PERC_DIFF_INSTALMENT'].replace(np.inf,0,inplace=True)\ninst_pay['CALC_INSTAL_PAID_LATE'] = (inst_pay['CALC_DAYS_LATE_PAYMENT'] > 0).astype(int)\ninst_pay['CALC_OVERPAID']= (inst_pay['CALC_DIFF_INSTALMENT'] < 0).astype(int)")


# In[ ]:


#DEFINING AGGREGATION RULES AND CREATING LIST OF NEW FEATURES
inst_pay_cols=[x for x in list(inst_pay.columns) if x not in ['SK_ID_CURR','SK_ID_PREV']]
inst_pay_agg={}
inst_pay_name=['SK_ID_CURR','SK_ID_PREV']
for col in inst_pay_cols:
    if 'NUM_INSTALMENT_VERSION'==col:
        inst_pay_agg[col]=[cnt_unique]#CUSTOM FUNCTION FOR COUNTING UNIQUE INSTALMENT_VERSION
        inst_pay_name.append(col+'_'+'unique')
    elif 'NUM_INSTALMENT_NUMBER'==col:
        inst_pay_agg[col]=['max','count']
        inst_pay_name.append(col+'_'+'max')
        inst_pay_name.append(col+'_'+'count')
    elif 'AMT_' in col:
        inst_pay_agg[col]=['sum','mean','max','min','var','std']
        inst_pay_name.append(col+'_'+'sum')
        inst_pay_name.append(col+'_'+'mean')
        inst_pay_name.append(col+'_'+'max')
        inst_pay_name.append(col+'_'+'min')
        inst_pay_name.append(col+'_'+'var')
        inst_pay_name.append(col+'_'+'std')
    elif 'CALC_DAYS_' in col:
        inst_pay_agg[col]=['sum']
        inst_pay_name.append(col+'_'+'sum')
    elif 'DAYS_' in col:
        inst_pay_agg[col]=['sum','max','min']
        inst_pay_name.append(col+'_'+'sum')
        inst_pay_name.append(col+'_'+'max')
        inst_pay_name.append(col+'_'+'min')
    else:
        inst_pay_agg[col]=['mean']
        inst_pay_name.append(col+'_'+'mean')


# In[ ]:


get_ipython().run_cell_magic('time', '', "#AGGREGATING DATA ON SK_ID_CURR,SK_ID_PREV USING RULES CREATED IN PREVIOUS STEP\ninst_pay_f=inst_pay.groupby(['SK_ID_CURR','SK_ID_PREV']).aggregate(inst_pay_agg)\ninst_pay_f.reset_index(inplace=True)\ninst_pay_f.columns=inst_pay_name")


# In[ ]:


#NUMBER OF MISSED INATALLMENTS
inst_pay_f['CALC_NUM_INSTALMENT_MISSED']=inst_pay_f['NUM_INSTALMENT_NUMBER_max']-inst_pay_f['NUM_INSTALMENT_NUMBER_count']


# In[ ]:


#DEFINING RULES FOR SECOND AGGREGATION ON SK_ID_CURR
inst_pay_cols=[x for x in list(inst_pay_f.columns) if x not in ['SK_ID_PREV']]
inst_pay_agg={}
inst_pay_name=['SK_ID_CURR']
for col in inst_pay_cols:
    if 'SK_ID_CURR'==col:
        inst_pay_agg[col]=['count']
        inst_pay_name.append('SK_ID_PREV_count')
    elif '_unique' in col:
        inst_pay_agg[col]=['sum']
        inst_pay_name.append(col)
    elif '_mean' in col:
        inst_pay_agg[col]=['mean']
        inst_pay_name.append(col)
    elif '_max' in col:
        inst_pay_agg[col]=['max']
        inst_pay_name.append(col)
    elif '_min' in col:
        inst_pay_agg[col]=['min']
        inst_pay_name.append(col)
    elif '_count' in col:
        inst_pay_agg[col]=['sum']
        inst_pay_name.append(col)
    else:
        inst_pay_agg[col]=['sum']
        inst_pay_name.append(col)


# In[ ]:


get_ipython().run_cell_magic('time', '', "#AGGREGATING DATA ON SK_ID_CURR\ninst_pay_f.drop(['SK_ID_PREV'],axis=1,inplace=True)\ninst_pay_fg=inst_pay_f.groupby(['SK_ID_CURR']).aggregate(inst_pay_agg)\ninst_pay_fg.reset_index(inplace=True)\ninst_pay_fg.columns=inst_pay_name")


# In[ ]:


#INSTALMENT_VERSION CHANGE
inst_pay_fg['CALC_CNT_INSTALMENT_VERSION_CHG']=inst_pay_fg['NUM_INSTALMENT_VERSION_unique']-inst_pay_fg['SK_ID_PREV_count']


# In[ ]:


#DROPING IRRELEVANT FEATURES
inst_pay_fg.drop(['DAYS_ENTRY_PAYMENT_max', 'AMT_PAYMENT_var', 'DAYS_INSTALMENT_sum'],axis=1,inplace=True)


# In[ ]:


#JOINING INSTALLMENT DATA WITH FINAL TABLE
#FINAL TABLE ALREADY CONTAINS APPLICATION, BUREAU AND PREVIOUS APPLICATION DATA
df_final=df_final.join(inst_pay_fg.set_index('SK_ID_CURR'),on='SK_ID_CURR',lsuffix='_AP', rsuffix='_INP')


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del inst_pay,inst_pay_f,inst_pay_fg
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "#READING POS_CASH_balance DATA\npos_cash=pd.read_csv(path1+'POS_CASH_balance.csv')\nprint('POS_CASH_balance set reading complete...')")


# In[ ]:


#CONVERTING MONTHS_BALANCE TO ITS ABSOLUTE VALUE
#FOR MONTHS BALANCE ABSOLUTE VALUE IS GIVING BETTER RESULTS
pos_cash['MONTHS_BALANCE']=pos_cash['MONTHS_BALANCE'].abs()


# In[ ]:


#MANUAL FEATURE ENGINEERING ON POS_CASH_BALANCE DATA
pos_cash['CALC_PERC_REMAINING_INSTAL']=pos_cash['CNT_INSTALMENT_FUTURE']/pos_cash['CNT_INSTALMENT']
pos_cash['CALC_CNT_REMAINING_INSTAL']=pos_cash['CNT_INSTALMENT']-pos_cash['CNT_INSTALMENT_FUTURE']
pos_cash['CALC_DAYS_WITHOUT_TOLERANCE']=pos_cash['SK_DPD']-pos_cash['SK_DPD_DEF']


# In[ ]:


#CONVERTING 'NAME_CONTRACT_STATUS' TO DUMMIES
pos_cash['NAME_CONTRACT_STATUS']=pos_cash['NAME_CONTRACT_STATUS'].apply(lambda x: str(x).replace(" ","_")) 
dummy=pd.get_dummies(pos_cash['NAME_CONTRACT_STATUS'],prefix='DUM_NAME_CONTRACT_STATUS')


# In[ ]:


#COMBINING DUMMIES WITH OTHER CONTINUOUS FEATURES
pos_cash_f=pd.concat([pos_cash.drop(['NAME_CONTRACT_STATUS'],axis=1),dummy],axis=1)


# In[ ]:


#DEFINING AGGREGATION RULES AND CREATING LIST OF NEW FEATURES
pos_cash_cols=[x for x in list(pos_cash_f.columns) if x not in ['SK_ID_CURR']]
pos_cash_agg={}
pos_cash_name=['SK_ID_CURR','SK_ID_PREV']
for col in pos_cash_cols:
    if 'SK_ID_PREV'==col:
        pos_cash_agg[col]=['count']
        pos_cash_name.append(col+'_'+'count')
    elif 'MONTHS_BALANCE'==col:
        pos_cash_agg[col]=['max','min','count']
        pos_cash_name.append(col+'_'+'max')
        pos_cash_name.append(col+'_'+'min')
        pos_cash_name.append(col+'_'+'count')
    elif 'DUM_' in col:
        pos_cash_agg[col]=['sum','mean','max','min']
        pos_cash_name.append(col+'_'+'sum')
        pos_cash_name.append(col+'_'+'mean')
        pos_cash_name.append(col+'_'+'max')
        pos_cash_name.append(col+'_'+'min')
    elif 'CNT_' in col:
        pos_cash_agg[col]=['max','min','sum','count']
        pos_cash_name.append(col+'_'+'max')
        pos_cash_name.append(col+'_'+'min')
        pos_cash_name.append(col+'_'+'sum')
        pos_cash_name.append(col+'_'+'count')
    else:
        pos_cash_agg[col]=['sum','mean']
        pos_cash_name.append(col+'_'+'sum')
        pos_cash_name.append(col+'_'+'mean')


# In[ ]:


get_ipython().run_cell_magic('time', '', "#AGGREGATING DATA ON SK_ID_CURR,SK_ID_PREV USING RULES CREATED IN PREVIOUS STEP\npos_cash_ff=pos_cash_f.groupby(['SK_ID_CURR','SK_ID_PREV']).aggregate(pos_cash_agg)\npos_cash_ff.reset_index(inplace=True)\npos_cash_ff.columns=pos_cash_name")


# In[ ]:


#DEFINING RULES FOR SECOND AGGREGATION ON SK_ID_CURR
pos_cash_cols=[x for x in list(pos_cash_ff.columns) if x not in ['SK_ID_CURR','SK_ID_PREV']]
pos_cash_agg={}
pos_cash_name=['SK_ID_CURR']
for col in pos_cash_cols:
    if '_sum'==col:
        pos_cash_agg[col]=['sum']
        pos_cash_name.append(col)
    elif '_mean' in col:
        pos_cash_agg[col]=['mean']
        pos_cash_name.append(col)
    elif '_max' in col:
        pos_cash_agg[col]=['max']
        pos_cash_name.append(col)
    elif '_min' in col:
        pos_cash_agg[col]=['min']
        pos_cash_name.append(col)
    elif '_count' in col:
        pos_cash_agg[col]=['sum']
        pos_cash_name.append(col)
    else:
        pos_cash_agg[col]=['sum']
        pos_cash_name.append(col)


# In[ ]:


get_ipython().run_cell_magic('time', '', "#AGGREGATING DATA ON SK_ID_CURR,SK_ID_PREV USING RULES CREATED IN PREVIOUS STEP\npos_cash_fg=pos_cash_ff.groupby(['SK_ID_CURR']).aggregate(pos_cash_agg)\npos_cash_fg.reset_index(inplace=True)\npos_cash_fg.columns=pos_cash_name")


# In[ ]:


#JOINING POS_CASH DATA WITH FINAL TABLE
#FINAL TABLE ALREADY CONTAINS APPLICATION,BUREAU,PREVIOUS APPLICATION DATA AND INSTALLMENT DATA
df_final=df_final.join(pos_cash_fg.set_index('SK_ID_CURR'),on='SK_ID_CURR',lsuffix='_AP', rsuffix='_PC')


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del pos_cash_fg,pos_cash,pos_cash_f,pos_cash_ff
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "#READING CREDIT CARD BALANCE DATA\ncc_bal=pd.read_csv(path1+'credit_card_balance.csv')\nprint('credit_card_balance set reading complete...')")


# In[ ]:


#CONVERTING MONTHS_BALANCE TO ITS ABSOLUTE VALUE
#FOR MONTHS BALANCE ABSOLUTE VALUE IS GIVING BETTER RESULTS
cc_bal['MONTHS_BALANCE']=cc_bal['MONTHS_BALANCE'].abs()


# In[ ]:


#MANUAL FEATURE ENGINEERING
cc_bal['CALC_PERC_BALANCE']=cc_bal['AMT_BALANCE']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_DRAWINGS_ATM_CURRENT']=cc_bal['AMT_DRAWINGS_ATM_CURRENT']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_DRAWINGS_CURRENT']=cc_bal['AMT_DRAWINGS_CURRENT']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_DRAWINGS_OTHER_CURRENT']=cc_bal['AMT_DRAWINGS_OTHER_CURRENT']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_DRAWINGS_POS_CURRENT']=cc_bal['AMT_DRAWINGS_POS_CURRENT']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_INST_MIN_REGULARITY']=cc_bal['AMT_INST_MIN_REGULARITY']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_PAYMENT_CURRENT']=cc_bal['AMT_PAYMENT_CURRENT']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_PAYMENT_TOTAL_CURRENT']=cc_bal['AMT_PAYMENT_TOTAL_CURRENT']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_RECEIVABLE_PRINCIPAL']=cc_bal['AMT_RECEIVABLE_PRINCIPAL']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_PERC_RECIVABLE']=cc_bal['AMT_RECIVABLE']/cc_bal['AMT_CREDIT_LIMIT_ACTUAL']
cc_bal['CALC_DAYS_WITHOUT_TOLERANCE']=cc_bal['SK_DPD']-cc_bal['SK_DPD_DEF']

CNT_DRAWING_LIST=['CNT_DRAWINGS_ATM_CURRENT','CNT_DRAWINGS_CURRENT','CNT_DRAWINGS_OTHER_CURRENT','CNT_DRAWINGS_POS_CURRENT']
cc_bal['CALC_CNT_DRAWINGS_TOTAL']=cc_bal[CNT_DRAWING_LIST].sum(axis=1)


# In[ ]:


#CONVERTING 'NAME_CONTRACT_STATUS' TO DUMMIES
cc_bal['NAME_CONTRACT_STATUS']=cc_bal['NAME_CONTRACT_STATUS'].apply(lambda x: str(x).replace(" ","_")) 
dummy=pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'],prefix='DUM_NAME_CONTRACT_STATUS')


# In[ ]:


#COMBINING DUMMIES WITH OTHER CONTINUOUS FEATURES
cc_bal_f=pd.concat([cc_bal.drop(['NAME_CONTRACT_STATUS'],axis=1),dummy],axis=1)


# In[ ]:


#DEFINING AGGREGATION RULES AND CREATING LIST OF NEW FEATURES
cc_bal_cols=[x for x in list(cc_bal_f.columns) if x not in ['SK_ID_CURR']]
cc_bal_agg={}
cc_bal_name=['SK_ID_CURR','SK_ID_PREV']
for col in cc_bal_cols:
    if 'SK_ID_PREV'==col:
        cc_bal_agg[col]=['count']
        cc_bal_name.append(col+'_'+'count')
    elif 'MONTHS_BALANCE'==col:
        cc_bal_agg[col]=['max','min','count']
        cc_bal_name.append(col+'_'+'max')
        cc_bal_name.append(col+'_'+'min')
        cc_bal_name.append(col+'_'+'count')
    elif 'AMT_' in col:
        cc_bal_agg[col]=['sum','mean','max','min','var','std']
        cc_bal_name.append(col+'_'+'sum')
        cc_bal_name.append(col+'_'+'mean')
        cc_bal_name.append(col+'_'+'max')
        cc_bal_name.append(col+'_'+'min')
        cc_bal_name.append(col+'_'+'var')
        cc_bal_name.append(col+'_'+'std')
    elif 'CNT_' in col:
        cc_bal_agg[col]=['max','min','sum','count']
        cc_bal_name.append(col+'_'+'max')
        cc_bal_name.append(col+'_'+'min')
        cc_bal_name.append(col+'_'+'sum')
        cc_bal_name.append(col+'_'+'count')
    else:
        cc_bal_agg[col]=['mean']
        cc_bal_name.append(col+'_'+'mean')


# In[ ]:


get_ipython().run_cell_magic('time', '', "#AGGREGATING DATA ON SK_ID_CURR,SK_ID_PREV USING RULES CREATED IN PREVIOUS STEP\ncc_bal_ff=cc_bal_f.groupby(['SK_ID_CURR','SK_ID_PREV']).aggregate(cc_bal_agg)\ncc_bal_ff.reset_index(inplace=True)\ncc_bal_ff.columns=cc_bal_name")


# In[ ]:


#DEFINING RULES FOR SECOND AGGREGATION ON SK_ID_CURR
cc_bal_cols=[x for x in list(cc_bal_ff.columns) if x not in ['SK_ID_CURR','SK_ID_PREV']]
cc_bal_agg={}
cc_bal_name=['SK_ID_CURR']
for col in cc_bal_cols:
    if '_sum'==col:
        cc_bal_agg[col]=['sum']
        cc_bal_name.append(col)
    elif '_var' in col:
        cc_bal_agg[col]=['mean']
        cc_bal_name.append(col)
    elif '_std' in col:
        cc_bal_agg[col]=['mean']
        cc_bal_name.append(col)
    elif '_mean' in col:
        cc_bal_agg[col]=['mean']
        cc_bal_name.append(col)
    elif '_max' in col:
        cc_bal_agg[col]=['max']
        cc_bal_name.append(col)
    elif '_min' in col:
        cc_bal_agg[col]=['min']
        cc_bal_name.append(col)
    elif '_count' in col:
        cc_bal_agg[col]=['sum']
        cc_bal_name.append(col)
    else:
        cc_bal_agg[col]=['sum']
        cc_bal_name.append(col)


# In[ ]:


get_ipython().run_cell_magic('time', '', "#AGGREGATING DATA ON SK_ID_CURR USING RULES CREATED IN PREVIOUS STEP\ncc_bal_fg=cc_bal_ff.groupby(['SK_ID_CURR']).aggregate(cc_bal_agg)\ncc_bal_fg.reset_index(inplace=True)\ncc_bal_fg.columns=cc_bal_name")


# In[ ]:


#JOINING CREDIT_CARD DATA WITH FINAL TABLE
#FINAL TABLE ALREADY CONTAINS APPLICATION,BUREAU,PREVIOUS APPLICATION DATA, INSTALLMENTS AND POS_CASH DATA
df_final=df_final.join(cc_bal_fg.set_index('SK_ID_CURR'),on='SK_ID_CURR',lsuffix='_AP', rsuffix='_CB')


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del cc_bal_fg,cc_bal,cc_bal_f,cc_bal_ff
gc.collect()


# In[ ]:


print('Final shape:',df_final.shape)


# In[ ]:


#DROPPING 'SK_ID_CURR'
df_final.drop(['SK_ID_CURR'],axis=1,inplace=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', '#IDENTIFYING HIGHLY CORRELATED FEATURES\ncorr=df_final.corr().abs()\ndrop_col=drop_corr_col(corr)\n\nprint(len(drop_col))')


# In[ ]:


print('List of highly correlated columns:\n')
drop_col


# In[ ]:


#DROPING HIGHLY CORRELATED FEATURES
df_final.drop(drop_col,axis=1,inplace=True)


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del corr
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "#ADDING FEW PURE STATISTICAL FEATURES\n#THIS HELPS IN IMPROVING SCORE\ndf_final['the_mean'] = df_final.mean(axis=1)\nprint('the_mean calculated...')\ndf_final['the_sum'] =df_final.sum(axis=1)\nprint('the_sum calculated...')\ndf_final['the_std'] = df_final.std(axis=1)\nprint('the_std calculated...')")


# In[ ]:


#CREATING FINAL X, y and test SETS
X=df_final.iloc[:len(train),:]
y=train['TARGET']
test=df_final.iloc[len(train):,:]


# In[ ]:


print('Shape of X:',X.shape,'Shape of y:',y.shape,'Shape of test:',test.shape)


# In[ ]:


#RELEASING MEMORY / GARBAGE COLLECTION
del df_final
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "#CREATING FINAL MODEL WITH STRATIFIED KFOLDS\n#FOLD COUNT 10\n#TRIED XGBClassifier, LGBMClassifier, CatBoostClassifier\n#BEST SCORE ACHIEVED BY CatBoostClassifier\n\nmodel=CatBoostClassifier(iterations=1000,\n                              learning_rate=0.05,\n                              depth=7,\n                              l2_leaf_reg=40,\n                              bootstrap_type='Bernoulli',\n                              subsample=0.7,\n                              scale_pos_weight=5,\n                              eval_metric='AUC',\n                              metric_period=50,\n                              od_type='Iter',\n                              od_wait=45,\n                              random_seed=17,\n                              allow_writing_files=False)\n\n#DATAFRAMES FOR STORING PREDICTIONS ON TRAIN DATA AS WELL AS TEST DATA\n#CAN BE USED FOR ENSEMBLE \ndf_preds=pd.DataFrame()\ndf_preds_x=pd.DataFrame()\nk=1\nsplits=10\navg_score=0\n\n#CREATING STRATIFIED FOLDS\nskf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=200)\nprint('\\nStarting KFold iterations...')\nfor train_index,test_index in skf.split(X,y):\n    df_X=X.iloc[train_index,:]\n    df_y=y.iloc[train_index]\n    val_X=X.iloc[test_index,:]\n    val_y=y.iloc[test_index]\n\n#FITTING MODEL\n    model.fit(df_X,df_y)\n\n#PREDICTING ON VALIDATION DATA\n    col_name='cat_predsx_'+str(k)\n    preds_x=pd.Series(model.predict_proba(val_X)[:,1])\n    df_preds_x[col_name]=pd.Series(model.predict_proba(X)[:,1])\n\n#CALCULATING ACCURACY\n    acc=roc_auc_score(val_y,preds_x)\n    print('Iteration:',k,'  roc_auc_score:',acc)\n    if k==1:\n        score=acc\n        model1=model\n        preds=pd.Series(model.predict_proba(test)[:,1])\n        col_name='cat_preds_'+str(k)\n        df_preds[col_name]=preds\n    else:\n        preds1=pd.Series(model.predict_proba(test)[:,1])\n        preds=preds+preds1\n        col_name='cat_preds_'+str(k)\n        df_preds[col_name]=preds1\n        if score<acc:\n            score=acc\n            model1=model\n    avg_score=avg_score+acc        \n    k=k+1\nprint('\\n Best score:',score,' Avg Score:',avg_score/splits)\n#TAKING AVERAGE OF PREDICTIONS\npreds=preds/splits")


# In[ ]:


#READING SAMPLE SUBMISSION FILE
sample=pd.read_csv(path1+'sample_submission.csv')


# In[ ]:


sample['TARGET']=preds


# In[ ]:


#CREATING SUMBISSION FILE
sample.to_csv('submission.csv',index=False)


# In[ ]:


#CREATING FILES FOR ENSEMBLE
df_preds_x.to_csv('cat_preds_x.csv',index=False)
df_preds.to_csv('cat_preds.csv',index=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', "#ENSEMBLE MODEL\n#USING LOGISTIC REGRESSION FOR NEXT LEVEL\nsample_ens=sample.copy()\n\ndf_preds_x.columns=['cat_preds_1','cat_preds_2','cat_preds_3','cat_preds_4','cat_preds_5','cat_preds_6',\n               'cat_preds_7','cat_preds_8','cat_preds_9','cat_preds_10']\n\ndf_final_ens=pd.concat([df_preds_x,df_preds],ignore_index=True)\n\ndf_final_ens['the_mean'] = df_final_ens.mean(axis=1)\nprint('the_mean calculated...')\ndf_final_ens['the_sum'] =df_final_ens.sum(axis=1)\nprint('the_sum calculated...')\ndf_final_ens['the_std'] = df_final_ens.std(axis=1)\nprint('the_std calculated...')\ndf_final_ens['the_kur'] = df_final_ens.kurtosis(axis=1)\nprint('the_kur calculated...')\n\n\nX=df_final_ens.iloc[:len(train),:]\ny=train['TARGET']\ntest=df_final_ens.iloc[len(train):,:]\n\nmodel=LogisticRegression()\n\nk=1\nsplits=10\navg_score=0\nskf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=200)\nprint('\\nStarting KFold iterations...')\nfor train_index,test_index in skf.split(X,y):\n    df_X=X.iloc[train_index,:]\n    df_y=y.iloc[train_index]\n    val_X=X.iloc[test_index,:]\n    val_y=y.iloc[test_index]\n\n    model.fit(df_X,df_y)\n\n    preds_x=pd.Series(model.predict_proba(val_X)[:,1])\n\n    acc=roc_auc_score(val_y,preds_x)\n    print('Iteration:',k,'  roc_auc_score:',acc)\n    if k==1:\n        score=acc\n        model1=model\n        preds=pd.Series(model.predict_proba(test)[:,1])\n    else:\n        preds=preds+pd.Series(model.predict_proba(test)[:,1])\n        if score<acc:\n            score=acc\n            model1=model\n    avg_score=avg_score+acc        \n    k=k+1\nprint('\\n Best score:',score,' Avg Score:',avg_score/splits)\npreds=preds/splits\n\nsample_ens['TARGET']=preds\nsample_ens.to_csv('submission_ens.csv',index=False)")

