#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data analysis libraries:
import numpy as np
import pandas as pd

# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# to display all columns:
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import numpy as np, pandas as pd, os, gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import lightgbm as lgb
import gc
from time import time
import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
warnings.simplefilter('ignore')
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np,gc # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 500)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Read train and test data with pd.read_csv():
train_id= pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
test_id = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
train_tr = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
test_tr = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")


# In[ ]:


train_id.head()


# In[ ]:


test_id.head()


# In[ ]:


train_tr.head()


# In[ ]:


test_tr.head()


# In[ ]:


train_id.info()


# In[ ]:


test_id.info()


# In[ ]:


train_tr.info()


# In[ ]:


train_tr.info()


# In[ ]:


test_tr.info()


# In[ ]:


train=pd.merge(train_tr, train_id, on = "TransactionID",how='left',left_index=True, right_index=True)
train.head()


# In[ ]:


test=pd.merge(test_tr, test_id, on = "TransactionID",how="left",left_index=True, right_index=True)
test.head()


# In[ ]:


del train_id, train_tr, test_id, test_tr


# In[ ]:


test.columns=train.columns.drop("isFraud")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n# WARNING! THIS CAN DAMAGE THE DATA \ndef reduce_mem_usage2(df):\n    """ iterate through all the columns of a dataframe and modify the data type\n        to reduce memory usage.        \n    """\n    start_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n    \n    for col in df.columns:\n        col_type = df[col].dtype\n        \n        if col_type != object:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == \'int\':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)\n        else:\n            df[col] = df[col].astype(\'category\')\n\n    end_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n    print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n    \n    return df')


# In[ ]:


#train=reduce_mem_usage2(train)
#test=reduce_mem_usage2(test)


# In[ ]:


train.isna().sum()


# In[ ]:


def make_corr(Vs):
    cols = Vs.columns
    plt.figure(figsize=(15,15))
    sns.heatmap(train[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
    #plt.title(Vs[0]+' - '+Vs[-1],fontsize=14)
    plt.show()


# In[ ]:


msno.matrix(train.loc[:,"V1":"V11"]);


# In[ ]:


make_corr(train.loc[:,"V1":"V11"])


# In[ ]:


msno.matrix(train.loc[:,"V12":"V34"]);


# In[ ]:


make_corr(train.loc[:,"V12":"V34"])


# In[ ]:


msno.matrix(train.loc[:,"V35":"V52"]);


# In[ ]:


make_corr(train.loc[:,"V35":"V52"])


# In[ ]:


msno.matrix(train.loc[:,"V53":"V74"]);


# In[ ]:


make_corr(train.loc[:,"V53":"V74"])


# In[ ]:


msno.matrix(train.loc[:,"V75":"V94"]);


# In[ ]:


make_corr(train.loc[:,"V75":"V94"])


# In[ ]:


msno.matrix(train.loc[:,"V95":"V137"]);


# In[ ]:


make_corr(train.loc[:,"V95":"V137"])


# In[ ]:


msno.matrix(train.loc[:,"V138":"V166"]);


# In[ ]:


make_corr(train.loc[:,"V138":"V166"])


# In[ ]:


msno.matrix(train.loc[:,"V167":"V216"]);


# In[ ]:


make_corr(train.loc[:,"V167":"V216"])


# In[ ]:


msno.matrix(train.loc[:,"V217":"V234"]);


# In[ ]:


make_corr(train.loc[:,"V217":"V234"])


# In[ ]:


drop_col=[]
train_colmns=train.loc[:,"V1":"V339"].columns
test_columns=test.loc[:,"V2":"V339"].columns
for col1,col2 in zip(train_colmns,test_columns):
            
        if ((train.loc[:,col1:col2].corr().loc[col2].sum()-1)>0.75) & (train[col1].isna().sum()== train[col2].isna().sum()):
            print("'"+col2+"'",', ',end='')
            drop_col.append(col2)
                


# In[ ]:


train=train.drop(drop_col,axis=1)
test=test.drop(drop_col,axis=1)
del drop_col


# In[ ]:


for col in train.columns: 
       if sum(train[col].isnull())/float(len(train.index)) > 0.90:
            print("'"+col+"'",', ',end='')
            train=train.drop(col,axis=1)
            test=test.drop(col,axis=1)


# In[ ]:


train.info()


# In[ ]:


emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 
      'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',
      'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',     
      'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 
      'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
      'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',
      'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 
      'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 
      'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
      'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
      'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',
      'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 
      'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 
      'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 
      'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 
      'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 
      'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 
      'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',
      'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}


for c in ['P_emaildomain', 'R_emaildomain']:
     train[c + '_bin'] = train[c].map(emails)
     test[c + '_bin'] = test[c].map(emails)

    


# In[ ]:


# FREQUENCY ENCODE TOGETHER
def encode_FE(df1, df2, cols):
    for col in cols:
        df = pd.concat([df1[col],df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        print(nm,', ',end='')
        
# LABEL ENCODE
def encode_LE(col,train=train,test=test,verbose=True):
    df_comb = pd.concat([train[col],test[col]],axis=0)
    df_comb,_ = df_comb.factorize(sort=True)
    nm = col
    if df_comb.max()>32000: 
        train[nm] = df_comb[:len(train)].astype('int32')
        test[nm] = df_comb[len(train):].astype('int32')
    else:
        train[nm] = df_comb[:len(train)].astype('int16')
        test[nm] = df_comb[len(train):].astype('int16')
    del df_comb; x=gc.collect()
    if verbose: print(nm,', ',end='')
        
# GROUP AGGREGATION MEAN AND STD
# https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda
def encode_AG(main_columns, uids, aggregations=['mean'], train_df=train, test_df=test, 
              fillna=True, usena=False):
    # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS
    for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column+'_'+col+'_'+agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])
                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   

                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')
                
                if fillna:
                    train_df[new_col_name].fillna(-1,inplace=True)
                    test_df[new_col_name].fillna(-1,inplace=True)
                
                print("'"+new_col_name+"'",', ',end='')
                
# COMBINE FEATURES
def encode_CB(col1,col2,df1=train,df2=test):
    nm = col1+'_'+col2
    df1[nm] = df1[col1].astype(str)+'_'+df1[col2].astype(str)
    df2[nm] = df2[col1].astype(str)+'_'+df2[col2].astype(str) 
    encode_LE(nm,verbose=False)
# GROUP AGGREGATION NUNIQUE
def encode_AG2(main_columns, uids, train_df=train, test_df=test):
    for main_column in main_columns:  
        for col in uids:
            comb = pd.concat([train_df[[col]+[main_column]],test_df[[col]+[main_column]]],axis=0)
            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
            train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')
            test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')
            print(col+'_'+main_column+'_ct, ',end='')


# In[ ]:


encode_CB('card1','addr1')


# In[ ]:


train['day'] = train.TransactionDT / (24*60*60)
train['uid'] = train.card1_addr1.astype(str)+'_'+np.floor(train.day-train.D1).astype(str)

test['day'] = test.TransactionDT / (24*60*60)
test['uid'] = test.card1_addr1.astype(str)+'_'+np.floor(test.day-test.D1).astype(str)


# In[ ]:


train[(train["card1"]==15775)&(train["addr1"]==251.0)][["TransactionID","isFraud","TransactionDT","TransactionAmt","card1","card2","card3","card4","addr1","uid"]]


# In[ ]:


na_low=[]
for col in train.loc[:,'TransactionAmt':].columns: 
       if sum(train[col].isnull())/float(len(train.index)) < 0.30:
            na_low.append(col)
            print("'"+col+"'",', ',end='')
      


# In[ ]:


na_low=['TransactionAmt' , 'ProductCD' , 'card1' , 'card2' , 'card3' , 'card4' , 'card5' , 'card6' , 'addr1' ,
        'addr2' , 'P_emaildomain' , 'C1' , 'C2' , 'C3' , 'C4' , 'C5' , 'C6' , 'C7' , 'C8' , 'C9' , 'C10' , 
        'C11' , 'C12' , 'C13' , 'C14'  , 'M6' ]


# In[ ]:


numeric=train[na_low]._get_numeric_data().columns
numeric


# In[ ]:


encode_FE(train,test,['addr1','card1','card2','card3','P_emaildomain'])


# In[ ]:


encode_CB('card1','addr1')


# In[ ]:


train['day'] = train.TransactionDT / (24*60*60)
train['uid'] = train.card1_addr1.astype(str)+'_'+np.floor(train.day-train.D1).astype(str)

test['day'] = test.TransactionDT / (24*60*60)
test['uid'] = test.card1_addr1.astype(str)+'_'+np.floor(test.day-test.D1).astype(str)


# In[ ]:


encode_FE(train,test,['uid'])
encode_AG(numeric, ['uid'],['mean',"std"], train, test, fillna=True, usena=False)


# In[ ]:


categorical_columns=test.columns.drop(test._get_numeric_data().columns)
categorical_columns=categorical_columns.drop('uid')
categorical_columns


# In[ ]:


encode_AG2(categorical_columns, ['uid'], train, test)


# In[ ]:


# FREQUENCY ENCODE: ADDR1, CARD1, CARD2, CARD3, P_EMAILDOMAIN
encode_FE(train,test,['addr1','card1','card2','card3','P_emaildomain'])
# COMBINE COLUMNS CARD1+ADDR1+P_EMAILDOMAIN
encode_CB('card1_addr1','P_emaildomain')
# FREQUENCY ENOCDE
encode_FE(train,test,['card1_addr1','card1_addr1_P_emaildomain'])
# GROUP AGGREGATE
encode_AG(['TransactionAmt','D9','D11'],['card1','card1_addr1','card1_addr1_P_emaildomain'],['mean','std'],usena=True)


# In[ ]:


del train['uid'], test['uid']


# In[ ]:


get_ipython().run_cell_magic('time', '', '# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n# WARNING! THIS CAN DAMAGE THE DATA \ndef reduce_mem_usage2(df):\n    """ iterate through all the columns of a dataframe and modify the data type\n        to reduce memory usage.        \n    """\n    start_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n    \n    for col in df.columns:\n        col_type = df[col].dtype\n        \n        if col_type != object:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == \'int\':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)\n        else:\n            df[col] = df[col].astype(\'category\')\n\n    end_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n    print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n    \n    return df')


# In[ ]:


train=reduce_mem_usage2(train)
test=reduce_mem_usage2(test)


# In[ ]:


train.shape, test.shape


# In[ ]:


categorical_columns=test.columns.drop(test._get_numeric_data().columns)
categorical_columns


# In[ ]:


from sklearn import preprocessing
for i in categorical_columns: 
    lbe=preprocessing.LabelEncoder()
    train[i]=lbe.fit_transform(train[i].astype(str))


# In[ ]:


for i in categorical_columns:    
    test[i]=lbe.fit_transform(test[i].astype(str))


# In[ ]:


train_columns=train.columns
train_columns=train_columns.drop("isFraud")
test.columns=train_columns
test.columns


# In[ ]:


train.shape, test.shape


# In[ ]:


for i in categorical_columns:
    if (test[i].max()== train[i].max())&(train[i].max()<8):
            test = pd.get_dummies(test, columns = [i])
            train=pd.get_dummies(train, columns = [i])

    


# In[ ]:


train.shape, test.shape


# feature_drop=["V6", "id_34", "V84", "V141", "id_38_0", "V111_uid_std", "V279", "V242", "V15", "V135", "V181", "V99", "V186", "V335",
#               "M9_0", "C3", "V281", "M8_1", "M5_2", "V92", "M7_2", "V226", "V132", "id_04", "M2_1", "V118_uid_mean", "V117_uid_std", 
#               "V172", "V2", "V337", "V21", "V50", "addr2_uid_mean", "id_12_1", "V293", "V194", "V123", "id_15_1", "V326", "V175", "V174",
#               "id_16_1", "V122_uid_mean", "M8_2", "V31", "M7_1", "id_38_1", "M8_0", "V319", "id_15_2", "DeviceType_1", "V118_uid_std",
#               "id_28_0", "V286", "V122_uid_std", "V95", "C3_uid_mean", "id_35_0", "DeviceType_0", "V173", "V196", "V121_uid_mean", "V287",
#               "V252", "id_28_1", "V120_uid_mean", "id_29_1", "V328", "V121_uid_std", "V108", "V247", "uid_M1_ct", "M9_2", "V8", "id_37_1",
#               "addr1_uid_std", "id_12_2", "V290", "V109", "id_29_0", "ProductCD_3", "V120_uid_std", "V117_uid_mean", "id_36_0", "id_15_0",
#               "V98", "V300", "id_16_2", "addr2_uid_std", "V334", "V101", "C3_uid_std", "id_37_0", "V284", "V288", "V115", "V138", "V104",
#               "V302", "V14_uid_std", "id_16_0", "card1_uid_std", "M6_2", "V65", "id_35_1", "V297", "V110", "card3_uid_std", "V14_uid_mean", 
#               "id_36_1", "id_15_3", "M1_1", "ProductCD_0", "V111", "V41", "id_12_0", "M1_2", "id_28_2", "V325", "V107_uid_mean", "V14", "V68", 
#               "V107_uid_std", "V116", "M3_2", "V1", "id_35_2", "id_29_2", "V114", "V65_uid_std", "M2_2", "V27_uid_std", "V27", "ProductCD_4",
#               "V240", "V65_uid_mean", "DeviceType_2", "V27_uid_mean", "V68_uid_std", "V89", "V88", "id_37_2", "id_36_2", "id_38_2", "V68_uid_mean", 
#               "M1_0", "card4_3", "uid_card6_ct", "uid_card4_ct", "V305_uid_std", "V305_uid_mean", "V305", "V241", "V122", "V121", "V120", "V118", "V117", "V107", ]

# In[ ]:


train.shape, test.shape


# In[ ]:


train_TransactionID= train["TransactionID"]
test_TransactionID=test["TransactionID"]
TransactionDT=train["TransactionDT"]
X= train.drop([ 'TransactionDT', 'TransactionID'], axis=1)
y = train['isFraud']
test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)
del train


# In[ ]:


#X=X.drop(feature_drop,axis=1)
#test=test.drop(feature_drop,axis=1)


# In[ ]:


X=X.drop("isFraud", axis=1)


# In[ ]:


X.head()


# In[ ]:


X.shape, test.shape


# In[ ]:


params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
          
         }


# In[ ]:


folds = TimeSeriesSplit(n_splits=5)

aucs = list()
feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns

training_start_time = time()
for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
    start_time = time()
    print('Training on fold {}'.format(fold + 1))
    
    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
    clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
    
    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()
    aucs.append(clf.best_score['valid_1']['auc'])
    
    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 30)
print('Training has finished.')
print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
print('Mean AUC:', np.mean(aucs))
print('-' * 30)


# In[ ]:


feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv('feature_importances_TimeFold.csv')

plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));


# In[ ]:


#clf right now is the last model, trained with 80% of data and validated with 20%
best_iter = clf.best_iteration


# In[ ]:


clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)
clf.fit(X, y)


# In[ ]:


#set the output as a dataframe and convert to csv file named submission.csv
predictions = clf.predict_proba(test)[:, 1]
output = pd.DataFrame({ "TransactionID" : test_TransactionID, "isFraud": predictions })
output.to_csv('submission_lgbm.csv', index=False)


# In[ ]:



