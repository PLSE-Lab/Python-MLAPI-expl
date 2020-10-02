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


# # Load best score submission files from public kernels

# In[ ]:


# https://www.kaggle.com/meaninglesslives/simple-neural-net-for-time-series-classification
df_1375 = pd.read_csv("../input/submissions-plasticc/single_predictions_1.375.csv")

# https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data
df_1425 = pd.read_csv("../input/submissions-plasticc/single_predictions_1.425.csv")

# https://www.kaggle.com/mithrillion/know-your-objective
df_1431 = pd.read_csv("../input/submissions-plasticc/single_predictions_1.431.csv")

# https://www.kaggle.com/ashishpatel26/can-this-make-sense-of-the-universe-tuned
df_1685 = pd.read_csv("../input/submissions-plasticc/single_predictions_1.685.csv")

# https://www.kaggle.com/meaninglesslives/lgb-parameter-tuning
df_1686 = pd.read_csv("../input/submissions-plasticc/single_predictions_1.686.csv")


# In[ ]:


# coefs
coefs = [0.5, 0.25, 0.25]

df_blend = df_1375 * coefs[0] + df_1425 * coefs[1] + df_1431 * coefs[2]
df_blend['object_id'] = df_1375 ['object_id']

print(df_blend.shape)
df_blend.head()


# # save submission

# In[ ]:


df_blend.to_csv('blend_submission.csv', index=False)


# In[ ]:




