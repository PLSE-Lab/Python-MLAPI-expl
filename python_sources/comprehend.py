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


import pandas as pd
df = pd.DataFrame()
mapping = {}
source_path = "../input/aws dataset/aws dataset"


# In[ ]:


df2 = pd.read_csv("../input/aws-dataset/aws dataset/aws dataset/output-onlinetsvtools_2.csv")


# In[ ]:


df2.columns


# In[ ]:


df2.effectiveness


# In[ ]:


df['label'] = df2.effectiveness
df['review'] = df2.benefitsReview


# In[ ]:


df = df.sample(frac=1).reset_index(drop=True)


# In[ ]:


# df.to_csv('/home/pralok/Desktop/train.csv', index=False, header=False)


# In[ ]:


df.head()


# In[ ]:


df_test = pd.read_csv("../input/aws-dataset/aws dataset/aws dataset/output-onlinetsvtools.csv")


# In[ ]:


df_test.head()


# In[ ]:


test = pd.DataFrame()


# In[ ]:


test['review'] = df_test.benefitsReview


# In[ ]:


test.to_csv("test2.csv", header=False,index=False)


# In[ ]:


import json
from pprint import pprint

with open("../input/output/predictions.jsonl", "r") as f:
    f = f.readlines()


# In[ ]:


pprint(f)


# In[ ]:


predictlabels = []
for i in f:
    j= json.loads(i)["Classes"]
    predictlabels.append([j[0]["Name"]])


# In[ ]:


pprint(predictlabels)


# In[ ]:


test_input = pd.read_csv("../input/test-data/test2.csv")


# In[ ]:


test.head()


# In[ ]:


print(test_input.head())

