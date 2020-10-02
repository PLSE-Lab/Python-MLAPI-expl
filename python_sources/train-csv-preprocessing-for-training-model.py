#!/usr/bin/env python
# coding: utf-8

# ### what "any" label means?
# 
# stage_1_train.csv contains data labeled with epidural, intraparenchymal,intraventricular, subarachnoid, subdural and any.
# 
# "any" means the image includes any types of IH (intracranial hemorrhage). Therefore "any" should be sum or maximum of probability of 5 types of IH (less than 1) and it isn't needed for training our model.
# 
# Rather than that, when we train our model to predict whether an image includes IH(and what type is it), we need "normal" label which means the image doesn't includes any types of IH. 

# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


INPUT_PATH = "../input/rsna-intracranial-hemorrhage-detection/"
train_df = pd.read_csv(INPUT_PATH + "stage_1_train.csv")


# In[ ]:


label = train_df.Label.values
train_df = train_df.ID.str.rsplit("_", n=1, expand=True)
train_df.loc[:, "label"] = label


# In[ ]:


train_df = train_df.rename({0: "id", 1: "subtype"}, axis=1)
train_df.shape


# In[ ]:


train_pivot_df = pd.pivot_table(train_df, index="id", columns="subtype", values="label")


# In[ ]:


train_pivot_df.shape


# In[ ]:


train_pivot_df["normal"] = 1 - train_pivot_df["any"]


# In[ ]:


train_pivot_df.head()

