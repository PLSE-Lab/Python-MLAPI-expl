#!/usr/bin/env python
# coding: utf-8

# <img src="https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png" alt="Count of Operation" height="800" width="800"></img>

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import KFold,StratifiedKFold
from xgboost.sklearn import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,PolynomialFeatures,MinMaxScaler,Binarizer
from sklearn.decomposition import PCA
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
gc.enable()
# Any results you write to the current directory are saved as output.


# In[ ]:


#Dataset view
path1= "../input/"
data_files=list(os.listdir(path1))
df_files=pd.DataFrame(data_files,columns=['File_Name'])
df_files['Size_in_MB']=df_files.File_Name.apply(lambda x:round(os.stat(path1+x).st_size/(1024*1024),2))
df_files


# In[ ]:


#All functions

#FUNCTION FOR PROVIDING FEATURE SUMMARY
def feature_summary(df_fa):
    print('DataFrame shape')
    print('rows:',df_fa.shape[0])
    print('cols:',df_fa.shape[1])
    col_list=['Null','Unique_Count','Data_type','Max/Min','Mean','Std','Skewness','Sample_values']
    df=pd.DataFrame(index=df_fa.columns,columns=col_list)
    df['Null']=list([len(df_fa[col][df_fa[col].isnull()]) for i,col in enumerate(df_fa.columns)])
    #df['%_Null']=list([len(df_fa[col][df_fa[col].isnull()])/df_fa.shape[0]*100 for i,col in enumerate(df_fa.columns)])
    df['Unique_Count']=list([len(df_fa[col].unique()) for i,col in enumerate(df_fa.columns)])
    df['Data_type']=list([df_fa[col].dtype for i,col in enumerate(df_fa.columns)])
    for i,col in enumerate(df_fa.columns):
        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):
            df.at[col,'Max/Min']=str(round(df_fa[col].max(),2))+'/'+str(round(df_fa[col].min(),2))
            df.at[col,'Mean']=df_fa[col].mean()
            df.at[col,'Std']=df_fa[col].std()
            df.at[col,'Skewness']=df_fa[col].skew()
        df.at[col,'Sample_values']=list(df_fa[col].unique())
           
    return(df.fillna('-'))

def drop_corr_col(df_corr):
    upper = df_corr.where(np.triu(np.ones(df_corr.shape),
                          k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.999
    to_drop = [column for column in upper.columns if any(upper[column] > 0.999)]
    return(to_drop)

def cnt_unique(df):
    return(len(df.unique()))


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Reading POS CASH balance data\npos_cash=pd.read_csv(path1+'POS_CASH_balance.csv')\nprint('POS_CASH_balance set reading complete...')")


# In[ ]:


pos_cash_fs=feature_summary(pos_cash)


# In[ ]:


pos_cash_fs


# In[ ]:


pos_cash.sort_values(['SK_ID_CURR','SK_ID_PREV']).head(30)


# In[ ]:


pos_cash['MONTHS_BALANCE']=pos_cash['MONTHS_BALANCE'].abs()


# In[ ]:


pos_cash['CALC_PERC_REMAINING_INSTAL']=pos_cash['CNT_INSTALMENT_FUTURE']/pos_cash['CNT_INSTALMENT']
pos_cash['CALC_CNT_REMAINING_INSTAL']=pos_cash['CNT_INSTALMENT']-pos_cash['CNT_INSTALMENT_FUTURE']
pos_cash['CALC_DAYS_WITHOUT_TOLERANCE']=pos_cash['SK_DPD']-pos_cash['SK_DPD_DEF']


# In[ ]:


pos_cash['NAME_CONTRACT_STATUS']=pos_cash['NAME_CONTRACT_STATUS'].apply(lambda x: str(x).replace(" ","_")) 
dummy=pd.get_dummies(pos_cash['NAME_CONTRACT_STATUS'],prefix='DUM_NAME_CONTRACT_STATUS')


# In[ ]:


dummy.head()


# In[ ]:


pos_cash_f=pd.concat([pos_cash.drop(['NAME_CONTRACT_STATUS'],axis=1),dummy],axis=1)


# In[ ]:


pos_cash_f.head()


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


pos_cash_f.shape


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


pos_cash_fg.head()


# In[ ]:


pos_cash_fg.shape


# In[ ]:


del pos_cash,pos_cash_f,pos_cash_ff
gc.collect()


# In[ ]:


train=pd.read_csv(path1+'application_train.csv',usecols=['SK_ID_CURR','TARGET'])


# In[ ]:


df_final=train.join(pos_cash_fg.set_index('SK_ID_CURR'),on='SK_ID_CURR',lsuffix='_AP', rsuffix='_POS')


# In[ ]:


df_final.shape


# In[ ]:


df_pos=df_final.drop(['SK_ID_CURR','TARGET'],axis=1)


# * Base value 0.6031624375062186

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_X,test_X,train_y,test_y=train_test_split(df_pos,train['TARGET'],random_state=200)\nmodel =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)\nmodel.fit(train_X,train_y)\nscore2=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])\nprint(score2)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#FEATURE EXCLUSION\nscore=0\nscore1=0\nscore2=0\ndrop_list=[]\ncol_list=list(df_pos.columns)\n\n\nwhile True:\n    score1=0\n    score2=0\n    for i,col in enumerate(col_list):\n        col_list.remove(col)\n        train_X,test_X,train_y,test_y=train_test_split(df_pos[col_list],train['TARGET'],random_state=200)\n        model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)\n        model.fit(train_X,train_y)\n        score2=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])\n        col_list.extend([col])\n#        dummy_1.at[i,'score']=score2\n        if score1<score2:\n            score1=score2\n            col1=col\n#        print('dropped col',col,':',score2)\n    if score<score1:\n        score=score1\n        print('dropped col',col1,':',score)\n        drop_list.extend([col1])\n        col_list.remove(col1)\n    else:\n        print('Best score achieved')\n        break\nprint(drop_list)\nprint('best score:',score)")


# * select col MONTHS_BALANCE_max : 0.5586004188463992
# * select col SK_DPD_DEF_mean : 0.5767441127822774
# * select col CNT_INSTALMENT_FUTURE_max : 0.587664251274978
# * select col MONTHS_BALANCE_min : 0.5987828728472335
# * select col CALC_PERC_REMAINING_INSTAL_sum : 0.6017536677014977
# * select col CALC_CNT_REMAINING_INSTAL_max : 0.6032922207850415
# * select col DUM_NAME_CONTRACT_STATUS_Returned_to_the_store_sum : 0.6034497206321445
# * select col CALC_DAYS_WITHOUT_TOLERANCE_mean : 0.6041772545912447
# * select col CALC_PERC_REMAINING_INSTAL_mean : 0.6047685824580172
# * select col DUM_NAME_CONTRACT_STATUS_Returned_to_the_store_mean : 0.6047816722637606
# * select col CNT_INSTALMENT_FUTURE_count : 0.6051233673114507
# * select col DUM_NAME_CONTRACT_STATUS_Active_max : 0.6051609776825202
# * select col DUM_NAME_CONTRACT_STATUS_Demand_mean : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_XNA_min : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_XNA_max : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_XNA_mean : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_XNA_sum : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Returned_to_the_store_min : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Returned_to_the_store_max : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Demand_min : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Demand_max : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Completed_min : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Canceled_min : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Canceled_max : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Canceled_mean : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Canceled_sum : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Approved_min : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Amortized_debt_min : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Amortized_debt_max : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Amortized_debt_mean : 0.6053183463120796
# * select col DUM_NAME_CONTRACT_STATUS_Amortized_debt_sum : 0.6053183463120796
# * Best score achieved
# * ['MONTHS_BALANCE_max', 'SK_DPD_DEF_mean', 'CNT_INSTALMENT_FUTURE_max', 'MONTHS_BALANCE_min', 'CALC_PERC_REMAINING_INSTAL_sum', 'CALC_CNT_REMAINING_INSTAL_max', 'DUM_NAME_CONTRACT_STATUS_Returned_to_the_store_sum', 'CALC_DAYS_WITHOUT_TOLERANCE_mean', 'CALC_PERC_REMAINING_INSTAL_mean', 'DUM_NAME_CONTRACT_STATUS_Returned_to_the_store_mean', 'CNT_INSTALMENT_FUTURE_count', 'DUM_NAME_CONTRACT_STATUS_Active_max', 'DUM_NAME_CONTRACT_STATUS_Demand_mean', 'DUM_NAME_CONTRACT_STATUS_XNA_min', 'DUM_NAME_CONTRACT_STATUS_XNA_max', 'DUM_NAME_CONTRACT_STATUS_XNA_mean', 'DUM_NAME_CONTRACT_STATUS_XNA_sum', 'DUM_NAME_CONTRACT_STATUS_Returned_to_the_store_min', 'DUM_NAME_CONTRACT_STATUS_Returned_to_the_store_max', 'DUM_NAME_CONTRACT_STATUS_Demand_min', 'DUM_NAME_CONTRACT_STATUS_Demand_max', 'DUM_NAME_CONTRACT_STATUS_Completed_min', 'DUM_NAME_CONTRACT_STATUS_Canceled_min', 'DUM_NAME_CONTRACT_STATUS_Canceled_max', 'DUM_NAME_CONTRACT_STATUS_Canceled_mean', 'DUM_NAME_CONTRACT_STATUS_Canceled_sum', 'DUM_NAME_CONTRACT_STATUS_Approved_min', 'DUM_NAME_CONTRACT_STATUS_Amortized_debt_min', 'DUM_NAME_CONTRACT_STATUS_Amortized_debt_max', 'DUM_NAME_CONTRACT_STATUS_Amortized_debt_mean', 'DUM_NAME_CONTRACT_STATUS_Amortized_debt_sum']
# * best score: 0.6053183463120796
# * CPU times: user 9h 59min 47s, sys: 2min 52s, total: 10h 2min 39s
# * Wall time: 2h 33min 46s

# In[ ]:


# %%time
# #FORWARD FEATURE SELCTION 
# score=0
# score1=0
# score2=0
# select_list=[]
# col_list=list(df_pos.columns)  
# k=0


# while True:
#     score1=0
#     score2=0
#     temp_list=select_list
#     for i,col in enumerate(col_list):
#         if k==0:
#             train_X,test_X,train_y,test_y=train_test_split(df_pos[col],train['TARGET'],random_state=200)
#             model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
#             model.fit(np.array(train_X).reshape(-1,1),train_y)
#             score2=roc_auc_score(test_y,model.predict_proba(np.array(test_X).reshape(-1,1))[:,1])
#         else:
#             temp_list.extend([col])
#             train_X,test_X,train_y,test_y=train_test_split(df_pos[temp_list],train['TARGET'],random_state=200)
#             model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
#             model.fit(train_X,train_y)
#             score2=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])
#             temp_list.remove(col)
#         if score1<=score2:
#             score1=score2
#             col1=col
# #        print('dropped col',col,':',score2)
#     k=k+1
#     if score<=score1:
#         score=score1
#         print('select col',col1,':',score)
#         select_list.extend([col1])
#         col_list.remove(col1)
#     else:
#         print('Best score achieved')
#         break
    
# print(select_list)
# print('best score:',score)

