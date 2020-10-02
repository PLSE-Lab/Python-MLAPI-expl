#!/usr/bin/env python
# coding: utf-8

# The purpose of this kernel is to create new train and test files that can be used for seperate algorithms in other kernels. Some of the kernels that use these files are listed here:
# 
# 
# H2O AutoML kernels:
# 
# age: https://www.kaggle.com/tunguz/trends-h2o-automl-age
# 
# domain1_var1: https://www.kaggle.com/tunguz/trends-h2o-automl-domain1-var1
# 
# domain1_var2: https://www.kaggle.com/tunguz/trends-h2o-automl-domain1-var2
# 
# domain2_var1: https://www.kaggle.com/tunguz/trends-h2o-automl-domain2-var1
# 
# domain2_var1: https://www.kaggle.com/tunguz/trends-h2o-automl-domain2-var2
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")


labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()

df.shape, test_df.shape


# In[ ]:


FNC_SCALE = 1/500

df[fnc_features] *= FNC_SCALE
test_df[fnc_features] *= FNC_SCALE


# In[ ]:


features = loading_features + fnc_features


# In[ ]:


len(features)


# In[ ]:


df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
np.save('features', features)


# In[ ]:




