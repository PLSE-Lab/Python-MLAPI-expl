#!/usr/bin/env python
# coding: utf-8

# In an attempt to understand this competition better, I was reading the paper describing Alaska-I competition winner solution from [this](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/147039) discussion post. I noticed that out of many things that winner tried, one of them was splitting the dataset into three sets, namely training set (TRN), validation set (VAL), and test set (TST).

# >The training set (TRN), validation set (VAL), and test set (TST)
# contained respectively 42,500, 3,500, and 3,500 cover images (around
# 500 cover images were not used because they were corrupted or
# failed the processing pipeline). The TRN, VAL, and TST sets were
# created for each quality factor and each stego scheme in TILEdouble,
# TILEbase, and ARBITRARYbase

# They split the dataset into TRN, VAL and TST for **each quality factor and each stego scheme**. In our case this would mean splitting the dataset separately based on each quality factor of `75, 90, 95` and also based on each stego scheme of `JMiPOD, UERD, JUNIWARD`. The pseudo-code for this split would look something like this:

# ```python
# # split based on quality factors
# for quality_factor in [75, 90, 95]:
#     split_data_into(TRN, VAL, TST)
#     
# # split based on stego scheme
# for stego in ['JMiPOD', 'UERD', 'JUNIWARD']:
#     split_data_into(TRN, VAL, TST)
# 
# ```
#  
#     

# This kernel is my attempt to split the dataset based on each quality factor and stego scheme. I will call TST, VAL and TST as `train`, `valid` and `test_val` respectively and will use `sklean`'s `train_test_split` to split the dataset with a split percentage of `70`, `20` and `10` respectively. The `train-valid` split will be used for training different classifiers while the `test_val` split can be used for ensembling of those classifiers. 

# I will use the dataset from [this amazing kernel](https://www.kaggle.com/meaninglesslives/alaska2-cnn-multiclass-classifier) by @meaninglesslives as it contains information about quality factors.

# In[ ]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[ ]:


get_ipython().system('ls ../input/alaska2-image-steganalysis')


# In[ ]:


get_ipython().system('ls ../input/alaska2trainvalsplit')


# In[ ]:


split_dir = '../input/alaska2trainvalsplit'


# In[ ]:


train_split = pd.read_csv(f'{split_dir}/alaska2_train_df.csv')
train_split.head()


# In[ ]:


def transform(x):
    split = x.split('/')
    path = split[-2] + '/' + split[-1]
    return path


# I will modify the `ImageFileName` column as it contains the path of image files which is kaggle-style.

# In[ ]:


train_split['ImageFileName'] = train_split['ImageFileName'].transform(transform)
train_split.head()


# In[ ]:


train_split['Label'].value_counts()


# In[ ]:


valid_split = pd.read_csv(f'{split_dir}/alaska2_val_df.csv')
valid_split.head()


# In[ ]:


valid_split['ImageFileName'] = valid_split['ImageFileName'].transform(transform)
valid_split.head()


# In[ ]:


# combine train and valid split
df_all = pd.concat([train_split, valid_split])
df_all.head()


# In[ ]:


# sanity check that train_split + valid_split = combined 
train_split.shape[0] + valid_split.shape[0] == df_all.shape[0]


# In[ ]:


def transform(x):
    split = x.split('/')[-2]
    path = split[-2] + '/' + split[-1]
    return path


# In[ ]:


# make a column representing the stego-scheme
df_all['Stego'] = df_all['ImageFileName'].transform(lambda x: x.split('/')[-2])


# In[ ]:


df_all.head()


# In[ ]:


df_all['Stego'].value_counts()


# In[ ]:


def qf_transform(x):
    if x in [1,4,7]:
        return 75
    elif x in [2,5,8]:
        return 90
    elif x in [3,6,9]:
        return 95
    else:
        return x


# In[ ]:


# make a column representing the quality factors; for Cover images, I've set quality factor = 0
df_all['quality_factor'] = df_all['Label'].transform(qf_transform)


# In[ ]:


df_all.head()


# In[ ]:


df_all['quality_factor'].value_counts()


# In[ ]:


# save the combined df 
df_all.to_csv('df_all.csv', index=False)


# In[ ]:


# split the data based on quality factor of 75, 90 and 95
for qf in [75, 90, 95]:
    df_qf = df_all[df_all['quality_factor']==qf]
    df_qf_tr, df_qf_val_test = train_test_split(df_qf, test_size=0.3, random_state=1234, stratify=df_qf['Label'].values)
    df_qf_val, df_qf_test  = train_test_split(df_qf_val_test, test_size=0.2, random_state=1234, stratify=df_qf_val_test['Label'].values)
    print(f'Split for quality factor of {qf}...')
    #print(df_qf_tr['Label'].value_counts())
    #print(df_qf_val['Label'].value_counts())
    #print(df_qf_test['Label'].value_counts())
    print('Shape of train split: ', df_qf_tr.shape)
    print('Shape of valid split: ', df_qf_val.shape)
    print('Shape of val_test split: ', df_qf_test.shape)
    print('*'*35)
    
    #save the splits
    df_qf_tr.to_csv(f'train_split_qf_{qf}.csv', index=False)
    df_qf_val.to_csv(f'valid_split_qf_{qf}.csv', index=False)
    df_qf_test.to_csv(f'test_val_split_qf_{qf}.csv', index=False)
    


# In[ ]:


# split the data based on stego scheme of JMiPOD, UERD, JUNIWARD
for stego in ['JMiPOD', 'UERD', 'JUNIWARD']:
    df_stego = df_all[df_all['Stego']==stego]
    df_stego_tr, df_stego_val_test = train_test_split(df_stego, test_size=0.3, random_state=1234, stratify=df_stego['Label'].values)
    df_stego_val, df_stego_test  = train_test_split(df_stego_val_test, test_size=0.2, random_state=1234, stratify=df_stego_val_test['Label'].values)
    print(f'Split for Stego type {stego}...')
    #print(df_stego_tr['Label'].value_counts())
    #print(df_stego_val['Label'].value_counts())
    #print(df_stego_test['Label'].value_counts())
    print('Shape of train split: ', df_stego_tr.shape)
    print('Shape of valid split: ', df_stego_val.shape)
    print('Shape of val_test split: ', df_stego_test.shape)
    print('*'*35)
    
    #save the splits
    df_stego_tr.to_csv(f'train_split_stego_{stego}.csv', index=False)
    df_stego_val.to_csv(f'valid_split_stego_{stego}.csv', index=False)
    df_stego_test.to_csv(f'test_val_split_stego_{stego}.csv', index=False)
    


# These splits can be used to train different classifiers based on quality factor and stego-scheme using `train-valid` sets and finally combine them using `test_val` set. 
# >This kernel just shows one of the many ways in which the dataset can be split. If you other ways, please let me know me in the comments
