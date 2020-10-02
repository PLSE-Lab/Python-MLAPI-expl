#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In this kernel, we will store the humongous 55 Million readings in Feather format
# This format takes much lesSser time to read (approx 7 seconds)


# In[ ]:


import os
import pandas as pd
print(os.listdir("../input/"))


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Reading File\ntrain_path  = '../input/train.csv'\n\n# Set columns to most suitable type to optimize for memory usage\ntraintypes = {'fare_amount': 'float32',\n              'pickup_datetime': 'str', \n              'pickup_longitude': 'float32',\n              'pickup_latitude': 'float32',\n              'dropoff_longitude': 'float32',\n              'dropoff_latitude': 'float32',\n              'passenger_count': 'uint8'}\n\ncols = list(traintypes.keys())\n\ntrain_df = pd.read_csv(train_path, usecols=cols, dtype=traintypes)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Save into feather format, about 1.5Gb. \ntrain_df.to_feather('nyc_taxi_data_raw.feather')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# load the same dataframe next time directly, without reading the csv file again!\ntrain_df = pd.read_feather('nyc_taxi_data_raw.feather')\n\n# It took less than one tenth of time to read the file")

