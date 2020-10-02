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


# In[ ]:


df = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')
df['EncodedPixels']=['' for i in range(len(df))]
df_submit = pd.read_csv("../input/unet-sai/submission.csv")
df_submit = df_submit.fillna('')


# In[ ]:


df.head()


# In[ ]:


df_submit.head()


# In[ ]:


df_submit['ImageId_ClassId'] = [(str(df_submit['ImageId'][i])+'_'+str(df_submit['ClassId'][i])) for i in range(len(df_submit))]
df_submit.head()


# In[ ]:


df_submit = df_submit[['ImageId_ClassId','EncodedPixels']]
df_submit.head()


# In[ ]:


if df.columns[0]=='ImageId_ClassId':
    df.set_index('ImageId_ClassId', inplace=True)
    df_submit.set_index('ImageId_ClassId', inplace=True)

    for name, row in df_submit.iterrows():
        df.loc[name] = row

    df.reset_index(inplace=True)

df.to_csv('submission.csv', index=False)
df.head()

