#!/usr/bin/env python
# coding: utf-8

# Title:  Power line fault detection EDA and pre processing  
# Data: https://www.kaggle.com/c/vsb-power-line-fault-detection  
# Author: [Virksaab](https://www.kaggle.com/virksaab)  
# Date:   28 December, 2018

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Paths to data and metadata

# In[ ]:


PARENT_DATA_DIR_PATH = '../input'
METADATA_TRAIN_FILE_PATH = os.path.join(PARENT_DATA_DIR_PATH, "metadata_train.csv")
TRAIN_DATA_FILE_PATH = os.path.join(PARENT_DATA_DIR_PATH, "train.parquet")


# ### Train metadata

#     Target:
#         0 : undamaged
#         1 : fault

# In[ ]:


metadata_train = pd.read_csv(METADATA_TRAIN_FILE_PATH)
print("#samples:", len(metadata_train))
metadata_train.head(15)


# Above table shows that 3 phases have same target of each signal.
# That means if one phase is damaged, others follow.
# So, each class's total signal_ids should be the multiple of 3 as shown below.

# In[ ]:


metadata_train.target.value_counts()/3


# ### Load train data

# #### BIG DATA FILE!! RAM USAGE: 7.9GB on ubuntu
# (uncomment below cell to load full data)

# In[ ]:


# traindataDF = pq.read_pandas(TRAIN_DATA_FILE_PATH).to_pandas()
# traindataDF.info()


# #### Below cell loads sample data

# In[ ]:


traindataDFsample = pq.read_pandas(TRAIN_DATA_FILE_PATH, columns=[str(i) for i in range(15)]).to_pandas()          
traindataDFsample.info()


# In[ ]:


traindataDFsample.describe()


# In[ ]:


traindataDFsample.tail(10)


# ### Plot 3 phase signals with target

# In[ ]:


traindataDFsample.iloc[:,:3].plot(title="3 phase, Target 0", figsize=(15,5));


# In[ ]:


traindataDFsample.iloc[:,3:6].plot(title="3 phase, Target 1", figsize=(15,5));


# ### Plotting 0 phase with target 0 and 1

# In[ ]:


traindataDFsample.iloc[:,0].plot(title="phase 0, Target 0", figsize=(15,5));


# In[ ]:


traindataDFsample.iloc[:,3].plot(title="phase 0, Target 1", figsize=(15,5));


# ### Group signals metadata accroding to target

# In[ ]:


target0df = metadata_train[metadata_train['target'] == 0]
target1df = metadata_train[metadata_train['target'] == 1]
print("target0data shape:", target0df.shape)
print("target1data shape:", target1df.shape)


# #### Load some target 0 (undamaged) signals and visualize

# In[ ]:


nSamples = 30


# In[ ]:


target0samplecols = [str(i) for i in list(target0df.iloc[:nSamples].signal_id)]
target0sampledata = pq.read_pandas(TRAIN_DATA_FILE_PATH, columns=target0samplecols).to_pandas()
target0sampledata.plot(title="Target 0", figsize=(15,10))


# #### Load some target 1 (fault) signals and visualize

# In[ ]:


target1samplecols = [str(i) for i in list(target1df.iloc[:nSamples].signal_id)]
target1sampledata = pq.read_pandas(TRAIN_DATA_FILE_PATH, columns=target1samplecols).to_pandas()
target1sampledata.plot(title="Target 1", figsize=(15,10))


# ### Reduce values per colm by averaging over 8
# It'll help reducing the computation without throwing any information

# In[ ]:


sample = traindataDFsample.iloc[:,3]
sample.shape


# In[ ]:


def reduce_sample(_sample, avgOver=8):
    preVal = 0
    processed_sample_list = []
    for index in range(avgOver, _sample.shape[0]+avgOver, avgOver):
        tmpdf = _sample.iloc[preVal:index]
        avgVal = tmpdf.sum()/avgOver
        processed_sample_list.append(avgVal)
        preVal = index
    return pd.Series(processed_sample_list)
processed_sample = reduce_sample(sample, 8)
processed_sample.shape


# In[ ]:


processed_sample.plot(title="reduced sample", figsize=(15,5))


# ### Apply reduction on some samples and visualize the results

# In[ ]:


# TARGET 0 (UNDAMAGED) SAMPLES
reducedtarget0sampleDF = pd.DataFrame()
for col in range(target0sampledata.shape[1]):
    tmp_pdSeries = reduce_sample(target0sampledata.iloc[:,col])
    reducedtarget0sampleDF[str(col)] = tmp_pdSeries
reducedtarget0sampleDF.shape


# In[ ]:


# TARGET 1 (FAULT) SAMPLES
reducedtarget1sampleDF = pd.DataFrame()
for col in range(target1sampledata.shape[1]):
    tmp_pdSeries = reduce_sample(target1sampledata.iloc[:,col])
    reducedtarget1sampleDF[str(col)] = tmp_pdSeries
reducedtarget1sampleDF.shape


# In[ ]:


reducedtarget0sampleDF.plot(title="Reduced target 0", figsize=(15,10))


# In[ ]:


reducedtarget1sampleDF.plot(title="Reduced target 1", figsize=(15,10))


# In[ ]:


reducedtarget1sampleDF.iloc[:,0].plot(title="Reduced target 1 single signal", figsize=(15,5))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




