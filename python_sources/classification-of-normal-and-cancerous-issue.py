#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import gc


# In[ ]:


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
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# In[ ]:



df1 = pd.read_csv('../input/gene-expression-data/gene_expression_01.csv')


# In[ ]:


res1 = df1.transpose()


# In[ ]:


del df1
gc.collect()


# In[ ]:


res1 = reduce_mem_usage(res1)


# In[ ]:


header = res1.iloc[0]
res1 =  res1[1:]
res1.columns = header
del header
gc.collect()


# In[ ]:


df2 = pd.read_csv('../input/gene-expression-data/gene_expression_02.csv')


# In[ ]:


res2 = df2.transpose()


# In[ ]:


del df2
gc.collect()


# In[ ]:


res2 = reduce_mem_usage(res2)


# In[ ]:


header = res2.iloc[0]
res2 =  res2[1:]
res2.columns = header
del header
gc.collect()


# In[ ]:


df3 = pd.read_csv('../input/gene-expression-data/gene_expression_03.csv')


# In[ ]:


res3 = df3.transpose()
del df3
gc.collect()
res3 = reduce_mem_usage(res3)


# In[ ]:


header = res3.iloc[0]
res3 =  res3[1:]
res3.columns = header
del header
gc.collect()


# In[ ]:


df4 = pd.read_csv('../input/gene-expression-data/gene_expression_04.csv')


# In[ ]:


res4 = df4.transpose()
del df4
gc.collect()
res4 = reduce_mem_usage(res4)


# In[ ]:


header = res4.iloc[0]
res4 =  res4[1:]
res4.columns = header
del header
gc.collect()


# In[ ]:


df5 = pd.read_csv('../input/gene-expression-05-14/gene_expression_05.csv')


# In[ ]:


res5 = df5.transpose()
del df5
gc.collect()
res5 = reduce_mem_usage(res5)


# In[ ]:


header = res5.iloc[0]
res5 =  res5[1:]
res5.columns = header
del header
gc.collect()


# In[ ]:


df=pd.concat([res1,res2,res3,res4,res5])


# In[ ]:


del res1
gc.collect()
del res2
gc.collect()
del res3
gc.collect()
del res4
gc.collect()
del res5
gc.collect()


# In[ ]:


df.shape


# In[ ]:




#df6 = pd.read_csv('../input/gene-expression-data/gene_expression_06.csv')
#df7 = pd.read_csv('../input/gene-expression-data/gene_expression_07.csv')
#df8 = pd.read_csv('../input/gene-expression-data/gene_expression_08.csv')
#df9 = pd.read_csv('../input/gene-expression-data/gene_expression_09.csv')
#df10 = pd.read_csv('../input/gene-expression-data/gene_expression_10.csv')
#df11 = pd.read_csv('../input/gene-expression-data/gene_expression_11.csv')
#df12 = pd.read_csv('../input/gene-expression-data/gene_expression_12.csv')
#df13 = pd.read_csv('../input/gene-expression-data/gene_expression_13.csv')
#df14 = pd.read_csv('../input/gene-expression-05-14/gene_expression_14.csv')
#df15 = pd.read_csv('../input/gene-expression-15-20/gene_expression_15.csv')
#df16 = pd.read_csv('../input/gene-expression-15-20/gene_expression_16.csv')
#df17 = pd.read_csv('../input/gene-expression-15-20/gene_expression_17.csv')
#df18 = pd.read_csv('../input/gene-expression-15-20/gene_expression_18.csv')
#df19 = pd.read_csv('../input/gene-expression-15-20/gene_expression_19.csv')
#df20 = pd.read_csv('../input/gene-expression-15-20/gene_expression_20.csv')


# In[ ]:


import seaborn as sns
sns.countplot(df['LABEL'],label="Count")


# In[ ]:




