#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import feather
import pandas as pd
pd.set_option("max_columns", 300)


# In[ ]:


train_label = feather.read_dataframe("../input/creating-a-metadata-dataframe-fastai/labels.fth")
train_meta = feather.read_dataframe("../input/creating-a-metadata-dataframe-fastai/df_trn.fth")
train = train_label.merge(train_meta[["SOPInstanceUID", "PatientID", "ImagePositionPatient2"]], how="left", left_on="ID", right_on="SOPInstanceUID")


# In[ ]:


gp = train_meta.groupby("PatientID").SeriesInstanceUID.nunique().reset_index()
gp.sort_values(by="SeriesInstanceUID")


# In[ ]:


train.groupby("PatientID")[['any', 'epidural', 'intraparenchymal', 'intraventricular',
       'subarachnoid', 'subdural']].std()


# In[ ]:


train["PatientID"].value_counts()


# In[ ]:


train[train.PatientID=="ID_fff502d5"].sort_values(by="ImagePositionPatient2")


# In[ ]:




