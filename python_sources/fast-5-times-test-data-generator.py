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
print(os.listdir("../"))

# Any results you write to the current directory are saved as output.


# ## generate 5 times test dataframe

# In[ ]:


for i in range(5):
    test_df_add = pd.read_csv("../input/sample_submission.csv")
    test_df_add["id"] = test_df_add["id"] + "_" + str(i)
    if i == 0:
        test_df = test_df_add
    else:
        test_df = pd.concat([test_df, test_df_add]).reset_index(drop=True)


# In[ ]:


df_dir = "../test/"
if not os.path.isdir(df_dir):
    os.mkdir(df_dir)


# In[ ]:


test_df.to_csv(df_dir + "sample_submission.csv")


# In[ ]:


os.listdir(df_dir)


# ## generate 5 times test dataset

# In[ ]:


import shutil
test_root = "../input/test/"
test_list = os.listdir(test_root)


# In[ ]:


print("before generate: %d" % len(os.listdir(test_root)))


# In[ ]:


img_dir = "../test/img/"
if not os.path.isdir(img_dir):
    os.mkdir(img_dir)


# In[ ]:


for i in range(5):
    for j in range(len(test_list)):
        shutil.copy(test_root + test_list[j], img_dir + test_list[j].replace(".png","") + "_" + str(i) + ".png")


# In[ ]:


print("after generate: %d" % len(os.listdir(img_dir)))


# In[ ]:


len(os.listdir(img_dir))/len(os.listdir(test_root))


# ## label copy

# In[ ]:


shutil.copy("../input/labels.csv", df_dir + "labels.csv")


# ## result

# In[ ]:


print(os.listdir("../test/"))

