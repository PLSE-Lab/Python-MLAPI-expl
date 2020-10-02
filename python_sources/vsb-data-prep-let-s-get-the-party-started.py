#!/usr/bin/env python
# coding: utf-8

# # Introduction

# I joined this competition only a **few days ago** and the first problem I faced has been the data loading. My kernels always crashed due to RAM limitations while loading "parquet" files. This notebook has a **simple aim**: summarising the competition data and **generate two sets** (train and test) **easily usable in other kernels**, mainly focused on modelling.
# 
# Thanks to [bluexleoxgreen](https://www.kaggle.com/bluexleoxgreen/simple-feature-lightgbm-baseline), especially for the idea of **splitting** the huge test set into subsets of **2K** columns.
# 
# Changes from the previous version: I deleted the part where I collected samples to put them into the exported sets. The sampling rates where too low and a lot of information about peaks and errors were lost. Here, instead, I try to add the information about amplitudes and phases of the first harmonics, with a downsampling much more "rich" than the previous one.
# 
# Running times of the kernel: **15 minutes**.
# 
# In any case, I'd highly appreciate comments or suggestions to improve my Python or the efficiency of this notebook (better slicing, multithreading or whatever).

# # Imports

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import gc
import pyarrow.parquet as pq


# In[ ]:


import os
print(os.listdir("../input"))


# # Data Loading

# In[ ]:


metadata_train = pd.read_csv("../input/metadata_train.csv")
metadata_train.info()


# In[ ]:


metadata_train.head(12)


# In[ ]:


metadata_test = pd.read_csv("../input/metadata_test.csv")
metadata_test.info()


# In[ ]:


metadata_test.head(12)


# The test set has about **3 times** the rows of the training set.

# In[ ]:


sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.info()


# In[ ]:


sample_submission.head(12)


# # Train Set Preparation (and a bit of EDA)

# Let's start loading the train.parquet file, which is actually a "second level" train set, containing information to put into the "usual" train set, metadata_train.

# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pq.read_pandas('../input/train.parquet').to_pandas()")


# In[ ]:


train.info()


# In[ ]:


train.iloc[0:7,0:10]


# The columns are the FK (signal_id). The values are voltage levels. The rows are sampling points (one every 20ms). Let's look at one signal:

# In[ ]:


x = train.index
y = train.iloc[:,0]


# In[ ]:


fig = plt.figure(figsize=(12,4))
ax1 = fig.add_axes([0,0,1,1])
ax1.plot(x,y,color='lightblue')


# In[ ]:


fig = plt.figure(figsize=(12,4))
ax1 = fig.add_axes([0,0,1,1])
ax1.set_xlim([10,300])
ax1.set_ylim([10,30])
ax1.plot(x,y,marker='o', color='orange')
ax2 = fig.add_axes([0.7,0.7,0.2,0.2])
ax2.plot(x,y, color='lightblue')


# Now, let's put our attention on trios, starting from a "good" one.

# In[ ]:


x = train.index
y0 = train.iloc[:,0]
y1 = train.iloc[:,1]
y2 = train.iloc[:,2]


# In[ ]:


fig = plt.figure(figsize=(12,4))
ax1 = fig.add_axes([0,0,1,1])
ax1.plot(x,y0,color='blue')
ax1.plot(x,y1,color='red')
ax1.plot(x,y2,color='green')


# In[ ]:


np.mean(y0)


# In[ ]:


np.min(y0)


# In[ ]:


np.max(y0)


# In[ ]:


np.std(y0)


# And now, here is one with target=1, i. e. a "bad example":

# In[ ]:


y0 = train.iloc[:,3]
y1 = train.iloc[:,4]
y2 = train.iloc[:,5]


# In[ ]:


fig = plt.figure(figsize=(12,4))
ax1 = fig.add_axes([0,0,1,1])
ax1.plot(x,y0,color='blue')
ax1.plot(x,y1,color='red')
ax1.plot(x,y2,color='green')


# I'd say that a failure is linked to peaks in two (or more phases), but let's go on. 

# In[ ]:


np.mean(y0)


# In[ ]:


np.min(y0)


# In[ ]:


np.max(y0)


# In[ ]:


np.std(y0)


# At this point we want to add some common indexes such as mean, std etc ... 

# In[ ]:


row_nr = train.shape[0]
row_nr


# In[ ]:


index_group_size=100


# In[ ]:


time_sample_idx=np.arange(0,row_nr,index_group_size)
time_sample_idx[0:10]


# In[ ]:


train_down=train.iloc[time_sample_idx,:]
train_down.iloc[:,0:10].head()


# In[ ]:


import numpy.fft as ft


# In[ ]:


def Amplitude(z):
    return np.abs(z)


# In[ ]:


def Phase(z):
    return (np.arctan2(z.imag,z.real))


# In[ ]:


df_harm=pd.DataFrame()


# In[ ]:


def find_dfa(df_source, df_dest,num_harm,base_col):
    # init
    dfa=df_dest.iloc[:,0:base_col]
    num_ap_cols = int(num_harm/2)
    for j in range(0,num_ap_cols) :
        dfa['Amp'+str(j)] = 0
        dfa['Pha'+str(j)] = 0
    dfa['ErrFun'] = 0
    dfa['ErrGen'] = 0
    # calc
    for i in range(0,len(df_source.columns)):
        dfa.loc[i]=0
        s=df_source.iloc[:,base_col+i]
        SF=ft.rfft(s)
        SF_Fundam=np.zeros(SF.size, dtype=np.complex_)
        SF_Filtered=np.zeros(SF.size, dtype=np.complex_)
        SF_Fundam[0:2]=SF[0:2]
        SF_Filtered[0:num_harm]=SF[0:num_harm]
        s_fun_rec=ft.irfft(SF_Fundam)
        s_gen_rec=ft.irfft(SF_Filtered)
        for j in range(0,num_ap_cols):
            dfa.iloc[i,base_col+2*j] = Amplitude(SF_Filtered[j])
            dfa.iloc[i,base_col+2*j+1] = Phase(SF_Filtered[j])
        dfa.iloc[i,base_col+2*num_ap_cols] = np.sqrt(np.mean((s-s_fun_rec)**2))
        dfa.iloc[i,base_col+2*num_ap_cols+1] = np.sqrt(np.mean((s-s_gen_rec)**2))
    return dfa


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_max = train.apply(np.max)\ntrain_min = train.apply(np.min)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_mean = train_down.apply(np.mean)\ntrain_std = train_down.apply(np.std)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_harm=pd.DataFrame()\nnum_harm=10\ndf_harm=find_dfa(train_down,df_harm,num_harm,0)')


# In[ ]:


df_harm.iloc[:,0:10].head()


# In[ ]:


metadata_train['mean']=train_mean.values
metadata_train['max']=train_max.values
metadata_train['min']=train_min.values
metadata_train['std']=train_std.values
for j in range(0,int(num_harm/2)) :
    metadata_train['Amp'+str(j)] = df_harm['Amp'+str(j)]
    metadata_train['Pha'+str(j)] = df_harm['Pha'+str(j)]
metadata_train['ErrFun'] = df_harm['ErrFun']
metadata_train['ErrGen'] = df_harm['ErrGen']


# In[ ]:


metadata_train.head()


# In[ ]:


df_train = metadata_train
df_train.to_csv('df_train.csv', index=False)


# # Test Set Preparation

# It's impossibile to load the test set as a whole, let's define a chunk size:

# In[ ]:


col_group_size=2000


# In addition, let's free all the avalable resources:

# In[ ]:


gc.collect()


# Now let's restart from the metadata_test:

# In[ ]:


metadata_test = pd.read_csv("../input/metadata_test.csv")


# In[ ]:


metadata_test['target']=-1
metadata_test['mean']=0
metadata_test['max']=0
metadata_test['min']=0
metadata_test['std']=0
for j in range(0,int(num_harm/2)) :
    metadata_test['Amp'+str(j)] = 0
    metadata_test['Pha'+str(j)] = 0
metadata_test['ErrFun'] = 0
metadata_test['ErrGen'] = 0


# In[ ]:


metadata_test.shape


# In[ ]:


metadata_test.head()


# Here is the function definition:

# In[ ]:


def add_info_test(metadata_df,time_sample_idx_1,col_group_size):
    col_id_start_0=np.min(metadata_test['signal_id'])
    col_id_start=col_id_start_0
    col_id_last=np.max(metadata_test['signal_id'])+1
    n_groups = int(np.round((col_id_last-col_id_start)/col_group_size))
    print('Steps = {}'.format(n_groups))
    for i in range(0,n_groups):
        col_id_stop = np.minimum(col_id_start+col_group_size,col_id_last)
        col_numbers = np.arange(col_id_start,col_id_stop)
        print('Step {s} - cols = [{a},{b})'.format(s=i,a=col_id_start,b=col_id_stop))
        print('   Adding Stats...',end="")
        col_names = [str(col_numbers[j]) for j in range(0,len(col_numbers))]
        test_i = pq.read_pandas('../input/test.parquet',columns=col_names).to_pandas()
        test_i_d1=test_i.iloc[time_sample_idx_1,:]
        test_mean_i = test_i_d1.apply(np.mean)
        test_max_i  = test_i.apply(np.max)
        test_min_i  = test_i.apply(np.min)
        test_std_i  = test_i_d1.apply(np.std)
        r_start = col_id_start - col_id_start_0
        r_stop = r_start + (col_id_stop-col_id_start)
        metadata_df.iloc[r_start:r_stop,4] = test_mean_i[0:col_id_stop-col_id_start].values
        metadata_df.iloc[r_start:r_stop,5] = test_max_i[0:col_id_stop-col_id_start].values
        metadata_df.iloc[r_start:r_stop,6] = test_min_i[0:col_id_stop-col_id_start].values
        metadata_df.iloc[r_start:r_stop,7] = test_std_i[0:col_id_stop-col_id_start].values
        print('   Adding FFT...')
        df_harm=pd.DataFrame()
        df_harm=find_dfa(test_i_d1,df_harm,10,0)
        num_ap_cols = int(num_harm/2)
        fft_base_col=8
        for j in range(0, num_ap_cols) :
            metadata_df.iloc[r_start:r_stop,fft_base_col+2*j] = df_harm.iloc[r_start:r_stop,2*j]
            metadata_df.iloc[r_start:r_stop,fft_base_col+2*j+1] = df_harm.iloc[r_start:r_stop,2*j+1]
        metadata_df.iloc[r_start:r_stop,fft_base_col+num_harm] = df_harm.iloc[r_start:r_stop, num_harm]
        metadata_df.iloc[r_start:r_stop,fft_base_col+num_harm+1] = df_harm.iloc[r_start:r_stop, num_harm+1]
        col_id_start=col_id_stop
    return (metadata_df)


# And this is its call:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'metadata_test1=add_info_test(metadata_test,time_sample_idx,col_group_size)')


# In[ ]:


metadata_test1.head()


# In[ ]:


metadata_test1.shape


# In[ ]:


metadata_test1.iloc[0:12,:]


# So we have, finally, our exportable test set:

# In[ ]:


metadata_test1.to_csv('df_test.csv', index=False)


# Now it's time to have fun with a new notebook based on these data and focused on modelling. 
# 
# Like said Pink some years ago, "[Let's get the party started!](https://www.youtube.com/watch?v=QRINgISPUWQ)"
