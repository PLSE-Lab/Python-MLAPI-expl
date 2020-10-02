#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import time


# ## 1. Compare list to numpy array

# In[ ]:


def create_list():
    start_time = time.time()
    
    list = [x for x in range(100000)]
    
    print(f'time required : {time.time() - start_time}')
    print(f'memory required : {list.__sizeof__()}')


# In[ ]:


def create_nparray():
    start_time = time.time()
    
    array = np.array([x for x in range(100000)])
    
    print(f'time required : {time.time() - start_time}')
    print(f'memory required : {array.__sizeof__()}')


# In[ ]:


print('list ------------')
create_list()

print('numpy.array ------------')
create_nparray()


# ## 2. Change to 32 bits

# In[ ]:


def create_64bits():
    start_time = time.time()
    
    array = np.random.rand(100000)
    
    print(f'time required : {time.time() - start_time}')
    print(f'memory required : {array.__sizeof__()}')


# In[ ]:


def create_32bits():
    start_time = time.time()
    
    array = np.random.rand(100000).astype(np.float32)
    
    print(f'time required : {time.time() - start_time}')
    print(f'memory required : {array.__sizeof__()}')


# In[ ]:


def create_16bits():
    start_time = time.time()
    
    array = np.random.rand(100000).astype(np.float16)
    
    print(f'time required : {time.time() - start_time}')
    print(f'memory required : {array.__sizeof__()}')


# In[ ]:


print('64 bits ------------')
create_64bits()

print('32 bits ------------')
create_32bits()

print('16 bits ------------')
create_16bits()


# Compare accuracy

# In[ ]:


array_accuracy = np.array([
    12345678901234567890,     # big num
    0.000000000000123456,     # small num
    12345.67890123456789      # both
])


for np_type in [np.float64, np.float32, np.float16]:
    print(f'{np_type}')
    array = array_accuracy.astype(np_type)
    
    for i in range(3):
        print(array[i])

# delete memory
del array_accuracy


# * np.float64  
#   It can handle numbers up to 16 digits.
# * np.float32  
#   It can handle numbers up to 7 digits.
# * np.float16  
#   It can handle numbers up to 4 digits.  
#   
# We need to select the digits depending on the data.  
# You can use `np.iinfo()` / `np.finfo()` method to check the maximum and minimum values of the data type.

# In[ ]:


print(np.iinfo(np.int64))
print(np.finfo(np.float16))


# The following functions are used to convert all DataFrames as a whole.

# In[ ]:


def reduce_mem_usage(df):
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
    return df


# In[ ]:


# For example

train = pd.read_csv('../input/titanic/train.csv')

print('original data ---------')
print(f'memory required : {train.__sizeof__()}')

train.info()


# In[ ]:



start_time = time.time()
train = reduce_mem_usage(train)

print('reduce_mem_usage--------')
print(f'time required : {time.time() - start_time}')
print(f'memory required : {train.__sizeof__()}')

train.info()


# ## 3. Create an intermediate file

# As an example, use data from the Titanic.  
# If you want to use the data in a separate process, you can extract it temporarily to a pickle file.

# In[ ]:


# Read data
train = pd.read_csv('../input/titanic/train.csv')
print(f'memory required : {train.__sizeof__()}')

# Create pickle file
start_time = time.time()

train.to_pickle('train_intermediate.pkl')
print(f'time required : {time.time() - start_time}')

del train


# You can also compress the data on memory.  
# You must compress and solve data, as with the creation and loading of intermediate files.

# In[ ]:


import pickle
import bz2
PROTOCOL = pickle.HIGHEST_PROTOCOL

class ZpkObj:
    def __init__(self, obj):
        self.zpk_object = bz2.compress(pickle.dumps(obj, PROTOCOL), 9)
    
    def load(self):
        return pickle.loads(bz2.decompress(self.zpk_object))


# In[ ]:


# Read data
train = pd.read_csv('../input/titanic/train.csv')

print('original data ---------')
print(f'memory required : {train.__sizeof__()}')

# Compress
start_time = time.time()
train = ZpkObj(train)

print('compressing ---------')
print(f'time required : {time.time() - start_time}')
print(f'memory required : {train.__sizeof__()}')

# Solve
start_time = time.time()
train = train.load()

print('solving ------------')
print(f'time required : {time.time() - start_time}')
print(f'memory required : {train.__sizeof__()}')

