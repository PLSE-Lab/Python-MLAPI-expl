#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General imports

import numpy as np
import pandas as pd
import os, warnings, random

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


# In[ ]:


pd.__version__


# In[ ]:


# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
## ------------------- 


# In[ ]:


SEED = 42
seed_everything(SEED)
LOCAL_TEST = False


# In[ ]:


## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
                    
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)  
                    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
## -------------------


# In[ ]:


ls ../input/ieee-fraud-detection/


# # Data Load

# In[ ]:


print('Load Data')
train_df = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
test_df = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
test_df['isFraud'] = 0

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')


# ## Memory calculation

# In[ ]:


print('Memory usage:', round(train_df.memory_usage(deep=True).sum()/1024/1024, 2), 'MB')


# In[ ]:


print('Memory usage:', round(test_df.memory_usage(deep=True).sum()/1024/1024, 2), 'MB')


# In[ ]:


print('Memory usage:', round(train_identity.memory_usage(deep=True).sum()/1024/1024, 2), 'MB')


# In[ ]:


print('Memory usage:', round(test_identity.memory_usage(deep=True).sum()/1024/1024, 2), 'MB')


# # Reduction Step -1 

# In[ ]:


if LOCAL_TEST:
    
    for df2 in [train_df, test_df, train_identity, test_identity]:
        df = reduce_mem_usage(df2)

        for col in list(df):
            if not df[col].equals(df2[col]):
                print('Bad transformation', col)


# In[ ]:


########################### Base Minification ########################### 

train_df = reduce_mem_usage(train_df)
test_df  = reduce_mem_usage(test_df)

train_identity = reduce_mem_usage(train_identity)
test_identity  = reduce_mem_usage(test_identity)


# In[ ]:


print('Memory usage:', round(train_df.memory_usage(deep=True).sum()/1024/1024, 2), 'MB')


# # Reduction Step -2 : Type Based {transaction}

# In[ ]:


train_df.dtypes.value_counts()


# In[ ]:


def find_types(df,col_name , type_name):
    ls = []
    for col in col_name:
        if( df[col].dtype == np.dtype( type_name)):
            ls.append(col)
    return ls


# In[ ]:


encode_cols = find_types(train_df, train_df.columns ,'object' )
print(encode_cols)


# In[ ]:


for col in encode_cols:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    print(col , '-----' ,  ((temp_df[col]).unique()).shape)


# ### Frequency encoding

# In[ ]:


for col in ['ProductCD', 'card4', 'card6' ,'M4']:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    col_encoded = temp_df[col].value_counts().to_dict() 
    
    train_df[col] = train_df[col].map(col_encoded)
    test_df[col]  = test_df[col].map(col_encoded)

    print((temp_df[col]).unique())
    print(col_encoded)    


# ### Strings to ints(or floats if nan in column)

# In[ ]:


for col in ['M1','M2','M3','M5','M6','M7','M8','M9']:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    print('-----------------------' , col ,'-----------------------')
    print(temp_df[col].value_counts())


# In[ ]:


for col in ['M1','M2','M3','M5','M6','M7','M8','M9']:
    train_df[col] = train_df[col].map({'T':1, 'F':0})
    test_df[col]  = test_df[col].map({'T':1, 'F':0})


# # Reduction Step -2 : Type Based {identity}

# In[ ]:


train_identity.dtypes.value_counts()


# In[ ]:


encode_cols = find_types(train_identity, train_identity.columns ,'object' )
print(encode_cols)


# In[ ]:


for col in encode_cols:
    temp_df = pd.concat([train_identity[[col]], test_identity[[col]]])
    print(col , '-----',  ((temp_df[col]).unique()).shape)


# In[ ]:


encode_cols = list(set(encode_cols) - set(['DeviceInfo', 'id_30', 'id_31', 'id_33']))


# In[ ]:


for col in encode_cols:
    temp_df = pd.concat([train_identity[[col]], test_identity[[col]]])
    print('-----------------------' , col ,'-----------------------')
    print(temp_df[col].value_counts())


# In[ ]:


mapper = {'New':2, 'Found':1, 'Unknown':0 ,'NotFound':0}

for col in ['id_12','id_15', 'id_16','id_27','id_28','id_29']:
    train_identity[col] = train_identity[col].map(mapper)
    test_identity[col]  = test_identity[col].map(mapper)
    
encode_cols = list(set(encode_cols) - set(['id_12','id_15', 'id_16','id_27','id_28','id_29']))
encode_cols


# In[ ]:


mapper = {'T':1, 'F':0}

for col in ['id_35','id_36', 'id_37','id_38']:
    train_identity[col] = train_identity[col].map(mapper)
    test_identity[col]  = test_identity[col].map(mapper)
    
encode_cols = list(set(encode_cols) - set(['id_35','id_36', 'id_37','id_38']))
encode_cols


# In[ ]:


mapper = {'TRANSPARENT':4, 'IP_PROXY':3, 'IP_PROXY:ANONYMOUS':2, 'IP_PROXY:HIDDEN':1}

for col in ['id_23']:
    train_identity[col] = train_identity[col].map(mapper)
    test_identity[col]  = test_identity[col].map(mapper)
    
encode_cols = list(set(encode_cols) - set(['id_23']))
encode_cols


# In[ ]:


mapper = {'desktop':1, 'mobile':0}

for col in ['DeviceType']:
    train_identity[col] = train_identity[col].map(mapper)
    test_identity[col]  = test_identity[col].map(mapper)
    
encode_cols = list(set(encode_cols) - set(['DeviceType']))
encode_cols


# # NAN Condition

# In[ ]:


for df in [train_identity , test_identity ]:
    df['id_33']   = df['id_33'].fillna('0x0')
    df['id_33_0'] = df['id_33'].apply(lambda x: x.split('x')[0]).astype(int)
    df['id_33_1'] = df['id_33'].apply(lambda x: x.split('x')[1]).astype(int)
    df['id_33']   = np.where(df['id_33']=='0x0', np.nan, df['id_33'])


# In[ ]:


for col in ['id_33']:
    train_identity[col] = train_identity[col].fillna('unseen_before_label')
    test_identity[col]  = test_identity[col].fillna('unseen_before_label')
    
    le = LabelEncoder()
    le.fit(list(train_identity[col])+list(test_identity[col]))
    train_identity[col] = le.transform(train_identity[col])
    test_identity[col]  = le.transform(test_identity[col])


# In[ ]:


for df in [train_identity , test_identity ]:
    df['id_34'] = df['id_34'].fillna(':0')
    df['id_34'] = df['id_34'].apply(lambda x: x.split(':')[1]).astype(np.int8)
    df['id_34'] = np.where(df['id_34']==0, np.nan, df['id_34'])


# # Final Minification

# In[ ]:


train_df = reduce_mem_usage(train_df)
test_df  = reduce_mem_usage(test_df)

train_identity = reduce_mem_usage(train_identity)
test_identity  = reduce_mem_usage(test_identity)


# # Exporting File

# In[ ]:


train_df.to_pickle('train_transaction.pkl')
test_df.to_pickle('test_transaction.pkl')

train_identity.to_pickle('train_identity.pkl')
test_identity.to_pickle('test_identity.pkl')

