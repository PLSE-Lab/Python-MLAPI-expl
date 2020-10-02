#!/usr/bin/env python
# coding: utf-8

# ## PCA using built in python function

# ## import requried libraries

# In[ ]:


import os
import warnings
warnings.filterwarnings('ignore')


# ## importing libraries for handling table like data and matrices 

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load datasets

# In[ ]:


names = ['Vendors','Model_Name','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']

data = pd.read_csv('../input/relative-cpu-performance/machine.csv',names=names)
df = pd.DataFrame(data)


# In[ ]:


#to display maximun columns and rows 
pd.set_option('display.expand_frame_repr',False)


# In[ ]:


df.head()


# In[ ]:


df= df.drop(columns='Vendors',axis=1)
df= df.drop(columns='Model_Name',axis=1)
df= df.drop(columns='MYCT',axis=1)
df= df.drop(columns='ERP',axis=1)
df= df.drop(columns='PRP',axis=1)


# In[ ]:


df.head()


# In[ ]:


def minmaxdt(df):
    df['transformed_MMIN'] = df['MMIN']
    for i in range(len(df)):
        df['transformed_MMIN'][i]=np.abs((df['MMIN'][i]-np.min(df['MMIN']))/np.max(df['MMIN'])-np.min(df['MMIN']))


# In[ ]:


minmaxdt(df)
df.head()


# In[ ]:


def minmaxdt(df):
    df['transformed_MMAX'] = df['MMAX']
    for i in range(len(df)):
        df['transformed_MMAX'][i]=np.abs((df['MMAX'][i]-np.min(df['MMAX']))/np.max(df['MMAX'])-np.min(df['MMAX']))


# In[ ]:


minmaxdt(df)
df.head()


# In[ ]:


def minmaxdt(df):
    df['transformed_CACH'] = df['CACH']
    for i in range(len(df)):
        df['transformed_CACH'][i]=np.abs((df['CACH'][i]-np.min(df['CACH']))/np.max(df['CACH'])-np.min(df['CACH']))


# In[ ]:


minmaxdt(df)
df.head()


# In[ ]:


def minmaxdt(df):
    df['transformed_CHMIN'] = df['CHMIN']
    for i in range(len(df)):
        df['transformed_CHMIN'][i]=np.abs((df['CHMIN'][i]-np.min(df['CHMIN']))/np.max(df['CHMIN'])-np.min(df['CHMIN']))


# In[ ]:


minmaxdt(df)
df.head()


# In[ ]:


def minmaxdt(df):
    df['transformed_CHMAX'] = df['CHMAX']
    for i in range(len(df)):
        df['transformed_CHMAX'][i]=np.abs((df['CHMAX'][i]-np.min(df['CHMAX']))/np.max(df['CHMAX'])-np.min(df['CHMAX']))


# In[ ]:


minmaxdt(df)
df.head()


# In[ ]:


df = df.drop(columns='MMIN',axis=1)
df = df.drop(columns='MMAX',axis=1)
df = df.drop(columns='CACH',axis=1)
df = df.drop(columns='CHMIN',axis=1)
df = df.drop(columns='CHMAX',axis=1)


# In[ ]:


df.head()


# In[ ]:


cov_mat = np.cov(df.T)
cov_mat


# In[ ]:


e_vals,e_vecs=np.linalg.eig(cov_mat)
e_vals


# In[ ]:


print(e_vecs)


# In[ ]:


print(e_vecs[1])
e_vecs[1].shape


# In[ ]:


ev=e_vecs[1]


# In[ ]:


pca=np.dot(df,ev)
pca


# In[ ]:


df['PCA']=pca


# In[ ]:


df.head()


# In[ ]:


df['PCA']

