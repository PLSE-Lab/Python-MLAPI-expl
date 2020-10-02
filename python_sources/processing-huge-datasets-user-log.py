#!/usr/bin/env python
# coding: utf-8

# This tutorial introduce the processing on a huge datasets in python. It allows you to work with a big quantity of data with your own laptop. In our example the machine has 32 cores with 17GB of Ram. About the data the file is named user_log.csv, the number of rows of the dataset is 400 Millions (6.7 GB zipped) and it correspond at the daily user logs describing listening behaviors of a user. Data collected until 2/28/2017. 
# 
# About the features:
# * msno: user id
# * date: format %Y%m%d
# * num_25: # of songs played less than 25% of the song length
# * num_50: # of songs played between 25% to 50% of the song length
# * num_75: # of songs played between 50% to 75% of of the song length
# * num_985: # of songs played between 75% to 98.5% of the song length
# * num_100: # of songs played over 98.5% of the song length
# * num_unq: # of unique songs played
# * total_secs: total seconds played
# 
# Our tutorial is composed by two parts. The first parts will be focus on the aggregation of the data, It is not possible to import all data within a dataframe at one part and then to do the aggregation. You can find several rows by users in the datasets and you are going to show how aggregate our 40 Millions of rows to have a dataset aggregated with one row by users. In the second part we are going to continue the processing but this time in order to optimize the memory usage with a few transformations.
# 

# In[1]:


#Load the required packages
import numpy as np
import pandas as pd
import time
import psutil
import multiprocessing as mp

#check the number of cores
num_cores = mp.cpu_count()
print("This kernel has ",num_cores,"cores and you can find the information regarding the memory usage:",psutil.virtual_memory())


# ![](http://)# 1. Aggregation

# The aggregation functions selected are min, max and count for the feature "date" and sum for the features "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq" and "totalc_secs". Therefore for each customers we will have the first date, the last date and the number of use of the service. Finally we will collect the number of songs played according to the length.

# In[2]:


# Writing as a function
def process_user_log(chunk):
    grouped_object = chunk.groupby(chunk.index,sort = False) # not sorting results in a minor speedup
    func = {'date':['min','max','count'],'num_25':['sum'],'num_50':['sum'], 
            'num_75':['sum'],'num_985':['sum'],
           'num_100':['sum'],'num_unq':['sum'],'total_secs':['sum']}
    answer = grouped_object.agg(func)
    return answer


# In order to aggregate our data we have to use chunksize. This option of read_csv() allow you to load massive file as small chunks in Pandas. We decide to take 10% of the total length for the chunksize. The value of chunksize will depend of your hardware that you have at disposition. But be careful it is not necessary interesting to take a small value. The time between each iterations can be too long with a small chaunksize. In order to find the best trade-off Memory usage - Time you can try different chunksize and select the best which will consume the lesser memory and which will be the faster.

# In[3]:


# Number of rows
size = 4e7 # 40 millions
reader = pd.read_csv('../input/user_logs.csv', chunksize = size, index_col=['msno'])
start_time = time.time()

for i in range(10):
    user_log_chunk = next(reader)
    if(i==0):
        result = process_user_log(user_log_chunk)
        print("Number of rows ",result.shape[0])
        print("Loop ",i,"took %s seconds" % (time.time() - start_time))
    else:
        result = result.append(process_user_log(user_log_chunk))
        print("Number of rows ",result.shape[0])
        print("Loop ",i,"took %s seconds" % (time.time() - start_time))
    del(user_log_chunk)    

# Unique users vs Number of rows after the first computation    
print(len(result))
check = result.index.unique()
print(len(check))

result.columns = ['_'.join(col).strip() for col in result.columns.values]    


# With our first function we have covered the data 40 Millions rows by 40 Millions rows but it is possible that a customer is in many subsamples. The new dataset result has 19 Millions of rows for 5 Millions of unique users. So it is necessary to compute a second time our aggragation functions. But now it is possible to do that on the whole of data beacause we have 19 Millions of rows contrary at 400 Millions at the beginning. For the second computation it is not necessary to use the chunksize.

# In[5]:


func = {'date_min':['min'],'date_max':['max'],'date_count':['count'] ,
           'num_25_sum':['sum'],'num_50_sum':['sum'],
           'num_75_sum':['sum'],'num_985_sum':['sum'],
           'num_100_sum':['sum'],'num_unq_sum':['sum'],'total_secs_sum':['sum']}
processed_user_log = result.groupby(result.index).agg(func)
print(len(processed_user_log))


# Finally with our second computation we have the whole of user (5 Millions) in the dataset processed_user_log and each row correponds at an unique user.

# In[6]:


processed_user_log.columns = processed_user_log.columns.get_level_values(0)
processed_user_log.head()


# # 2. Reduce the Memory usage

# In this part we are going ton interested in the memory usage of our data. We can see that all colums except "date_min" and "total_secs_sum" are int64. It is not always justified and it uses a lot of memory for nothing. with the function descibe we can see that only the featue "total_secs_sum" have the right type. We have changed the type for each feature to reduce the memory usage. 

# In[7]:


processed_user_log.info(), processed_user_log.describe()


# In[8]:


processed_user_log = processed_user_log.reset_index(drop = False)

# Initialize the dataframes dictonary
dict_dfs = {}

# Read the csvs into the dictonary
dict_dfs['processed_user_log'] = processed_user_log

def get_memory_usage_datafame():
    "Returns a dataframe with the memory usage of each dataframe."
    
    # Dataframe to store the memory usage
    df_memory_usage = pd.DataFrame(columns=['DataFrame','Memory MB'])

    # For each dataframe
    for key, value in dict_dfs.items():
    
        # Get the memory usage of the dataframe
        mem_usage = value.memory_usage(index=True).sum()
        mem_usage = mem_usage / 1024**2
    
        # Append the memory usage to the result dataframe
        df_memory_usage = df_memory_usage.append({'DataFrame': key, 'Memory MB': mem_usage}, ignore_index = True)
    
    # return the dataframe
    return df_memory_usage

init = get_memory_usage_datafame()

dict_dfs['processed_user_log']['date_min'] = dict_dfs['processed_user_log']['date_min'].astype(np.int32)
dict_dfs['processed_user_log']['date_max'] = dict_dfs['processed_user_log'].date_max.astype(np.int32)
dict_dfs['processed_user_log']['date_count'] = dict_dfs['processed_user_log']['date_count'].astype(np.int8)
dict_dfs['processed_user_log']['num_25_sum'] = dict_dfs['processed_user_log'].num_25_sum.astype(np.int32)
dict_dfs['processed_user_log']['num_50_sum'] = dict_dfs['processed_user_log'].num_50_sum.astype(np.int32)
dict_dfs['processed_user_log']['num_75_sum'] = dict_dfs['processed_user_log'].num_75_sum.astype(np.int32)
dict_dfs['processed_user_log']['num_985_sum'] = dict_dfs['processed_user_log'].num_985_sum.astype(np.int32)
dict_dfs['processed_user_log']['num_100_sum'] = dict_dfs['processed_user_log'].num_100_sum.astype(np.int32)
dict_dfs['processed_user_log']['num_unq_sum'] = dict_dfs['processed_user_log'].num_unq_sum.astype(np.int32)

init.join(get_memory_usage_datafame(), rsuffix = '_managed')


# With the right type for each feature we have reduced the usage by 44%. It is not negligible especially when we have a contraint on the hardware or when you need your the memory to imlplement a Machine Leaning model. It exists others methods to reduce the memory usage. You have to be careful on the type of each feature if you want to optimize the manipulation of the data.

# In[17]:


import matplotlib.pyplot as plt

data = init.join(get_memory_usage_datafame(), rsuffix = '_managed')
plt.style.use('ggplot')
data.plot(kind='bar',figsize=(10,10), title='Memory usage');


# 
