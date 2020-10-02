#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


data = pd.read_csv(".data/sensors,smooth.csv.gz")


# In[ ]:


data


# In[ ]:


from tsfresh import extract_features
# extracted_features = extract_features(timeseries, column_id="id", column_sort="time")


# In[ ]:


features = extract_features(data.drop(columns = "AC"), column_id = "FLIGHT", column_sort = "TIME")


# In[ ]:


from tsfresh.utilities.dataframe_functions import impute


# In[ ]:


impute(features)


# In[ ]:


features.to_csv(".data/sensors,features.csv")


# In[ ]:


.data/sensors,smooth.csv.gz

