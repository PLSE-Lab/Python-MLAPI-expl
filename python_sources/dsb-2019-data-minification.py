#!/usr/bin/env python
# coding: utf-8

# # Dataset Minification

# ### The goal of this notebook is to offer a first preprocessing step so that you can manipulate this huuuuuuge dataset easily. Can save as a .pkl, which allows you to load it quickly!
# 
# * Also get data in prediction ready format
# * `For each installation_id represented in the test set, you must predict the accuracy_group of the last assessment for that installation_id.`
# 

# In[ ]:


import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

pd.set_option('display.max_colwidth', -1)


# In[ ]:


# def reduce_mem_usage(props, log=False):
#     start_mem_usg = props.memory_usage().sum() / 1024**2 
#     print("Memory usage of properties dataframe is :", round(start_mem_usg, 2), " MB")
#     NAlist = [] # Keeps track of columns that have missing values filled in. 
#     for col in props.columns:
#         if props[col].dtype != object:  # Exclude strings and timestamps
            
#             # Print current column type
#             if log: print("******************************")
#             if log: print("Column: ",col)
#             if log: print("dtype before: ",props[col].dtype)
            
#             # make variables for Int, max and min
#             IsInt = False
#             mx = props[col].max()
#             mn = props[col].min()
            
#             # Integer does not support NA, therefore, NA needs to be filled
#             if not np.isfinite(props[col]).all(): 
#                 NAlist.append(col)
#                 props[col].fillna(mn-1,inplace=True)            

#             # test if column can be converted to an integer
#             asint = props[col].fillna(0).astype(np.int64)
#             result = (props[col] - asint)
#             result = result.sum()
#             if result > -0.01 and result < 0.01:
#                 IsInt = True

            
#             # Make Integer/unsigned Integer datatypes
#             if IsInt:
#                 if mn >= 0:
#                     if mx < 255:
#                         props[col] = props[col].astype(np.uint8)
#                     elif mx < 65535:
#                         props[col] = props[col].astype(np.uint16)
#                     elif mx < 4294967295:
#                         props[col] = props[col].astype(np.uint32)
#                     else:
#                         props[col] = props[col].astype(np.uint64)
#                 else:
#                     if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
#                         props[col] = props[col].astype(np.int8)
#                     elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
#                         props[col] = props[col].astype(np.int16)
#                     elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
#                         props[col] = props[col].astype(np.int32)
#                     elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
#                         props[col] = props[col].astype(np.int64)    
            
#             # Make float datatypes 32 bit
#             else:
#                 props[col] = props[col].astype(np.float32)
            
#             # Print new column type
#             if log: print("dtype after: ",props[col].dtype)
#             if log: print("******************************")
    
#     mem_usg = props.memory_usage().sum() / 1024**2 
#     print("Memory usage is now: ", round(mem_usg, 2), " MB")
#     print("This is ",round(100 * mem_usg / start_mem_usg, 2),"% of the initial size")
#     return props


# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type != object:
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
        #else: df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# #### Loading
# 
# ###### We also remove rows from train with no labelled data.
# * https://www.kaggle.com/carlossouza/cleaning-useless-data-to-load-train-csv-faster

# In[ ]:


get_ipython().run_cell_magic('time', '', 'N_ROWS = int(1e4)\ntrain = reduce_mem_usage(pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv",\n#                     parse_dates=["timestamp"],infer_datetime_format=True\n                   )#, nrows=N_ROWS)\n                        )\ntest = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv", nrows=N_ROWS)\ntrain.head()')


# In[ ]:


train[["event_id","game_session","installation_id"]].nunique()


# In[ ]:


## parse datetime
# train.timestamp = pd.to_datetime(train.timestamp,infer_datetime_format=True)
# test.timestamp = pd.to_datetime(test.timestamp,infer_datetime_format=True)

# train.set_index("timestamp",inplace=True)


# In[ ]:


train.dtypes


# In[ ]:


start_mem_usg = train.memory_usage().sum() / 1024**2 
print("Memory usage of the train is : {:.1f} MB for now".format(start_mem_usg))
start_mem_usg = test.memory_usage().sum() / 1024**2 
print("Memory usage of the test is : {:.1f} MB for now".format(start_mem_usg))


# In[ ]:


labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
print(labels.nunique())
print(labels.shape)
print(labels.dtypes)
labels.head()


# In[ ]:


## https://www.kaggle.com/carlossouza/cleaning-useless-data-to-load-train-csv-faster

print(train.shape)
useful_installation_ids = labels['installation_id'].unique()
train = train.loc[train.installation_id.isin(labels.installation_id)]
print(train.shape)


# In[ ]:


train.head(3)


# In[ ]:


train.memory_usage().sum() / 1024**2 


# In[ ]:


# train.timestamp = pd.to_datetime(train.timestamp,infer_datetime_format=True)
# train.memory_usage().sum() / 1024**2 


# In[ ]:


train.dtypes


# ## Exploring event_data column

# ### `event_data` seems interesting. I think it is the main source of information.
# ### The data is given in json format, so we'll parse it to be able to create columns

# In[ ]:


train['event_data'] = train['event_data'].apply(lambda x: json.loads(x))
test['event_data'] = test['event_data'].apply(lambda x: json.loads(x))


# In[ ]:


event_data = train['event_data'].tolist()
unique_keys = list()
for my_json in event_data:
    unique_keys += my_json.keys()
    
unique_keys = list(set(unique_keys))
print('event_data contains {} new columns'.format(len(unique_keys)))
print('Some new columns are:', unique_keys[:5])


# In[ ]:


event_data[0:5]


# In[ ]:


for ky in tqdm(unique_keys):
    def give_me_keys(x):
        try:
            return x[ky]
        except KeyError:
            return np.nan
    train[ky] = train['event_data'].apply(give_me_keys)
    test[ky] = test['event_data'].apply(give_me_keys)
    
    
print('Train shape is:', train.shape)
print('Test shape is:', test.shape)
start_mem_usg = train.memory_usage().sum() / 1024**2 
print("Memory usage of the train dataframe is : {:.1f} MB for now".format(start_mem_usg))


# In[ ]:


# Now that we've the information contained in event_data, we can drop it
try:
    train.drop('event_data', axis=1, inplace=True)
    test.drop('event_data', axis=1, inplace=True)
except:
    pass
print(train.shape)
train.head()


# ### Remove columns which variance is very low or with too many missing values
# 
# * Modify the 2 thresholds to fit your needs

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = reduce_mem_usage(train)')


# In[ ]:


VAR_THRESH = .015 # .1
NAN_THRESH = .991 #.99

nan_dict = train.isna().mean() >= NAN_THRESH
# cols_to_drop += [k for k, v in nan_dict.items() if v]
cols_to_drop = [k for k, v in nan_dict.items() if v]

# drop twice to save on second check
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)

print("less nans, train size:", train.shape)
var_dict = train.std() <= VAR_THRESH
cols_to_drop = [k for k, v in var_dict.items() if v]

cols_to_drop = list(set(cols_to_drop))
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)

print('We dropped {} columns'.format(len(cols_to_drop)))
print('Train shape is: ', train.shape)
print('Test shape is: ', test.shape)


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


train.tail()


# ### Add labels with data for predicting
# * take last event per session and join with label. 
# 
# * get last event per game session for attaching to the label
# * we would need this for test as well

# In[ ]:


print(train.shape)
df = train.groupby(["installation_id","game_session"],as_index=False).last() # "game_session",
print("df.shape",df.shape)
df = df.merge(labels,on=["installation_id","game_session"],how="inner")
print("df.merge(labels,on=[installation_id,game_session],how=inner)", df.shape)
df.head()


# In[ ]:


# df = df.T.drop_duplicates().T ## can't drop duplicate cols, data has lists/dicts = unhashable
# df.shape


# In[ ]:


df[["event_id","game_session","installation_id"]].nunique()


# In[ ]:





# In[ ]:


df.to_csv("dsb_train_v1.csv.gz",compression="gzip")

train.to_csv("dsb_context_tr_v1.csv.gz",compression="gzip")


# In[ ]:





# * Now we will Label Encode some variable to stock them as small int (instead of objects)

# In[ ]:


# col_to_label_encode = list()
# for col in train.columns:
#     try:
#         if len(train[col].unique()) < 10:
#             col_to_label_encode.append(col)
#     except:
#         pass


# In[ ]:


# correspondance_dict = dict()

# for col in col_to_label_encode:
#     try:
#         le = LabelEncoder()
#         train[col] = le.fit_transform(train[col])
#         test[col] = le.transform(test[col])

#         keys = le.classes_
#         values = le.transform(le.classes_)
#         dictionary = dict(zip(keys, values))
#         correspondance_dict[col] = dictionary

#     except:    # the variable is not label encodable
#         pass

# correspondance_dict


# In[ ]:


# train = reduce_mem_usage(train, log=False)


# In[ ]:


# train.to_pickle('train.pkl')
# test.to_pickle('test.pkl')

