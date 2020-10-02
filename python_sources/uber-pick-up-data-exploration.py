#!/usr/bin/env python
# coding: utf-8

# Exploring the basic data

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


sns.set(style="whitegrid", color_codes=True)


# In[ ]:


UBER_FOIL_FILE = '../input/Uber-Jan-Feb-FOIL.csv'
uberFOILDF = pd.read_csv(UBER_FOIL_FILE)
uberFOILDF['trips_to_active_vehicles'] = uberFOILDF['trips']/uberFOILDF['active_vehicles']
uberFOILDF.head()


# In[ ]:


uberFOILDF.info()


# In[ ]:


sns.swarmplot(x = 'dispatching_base_number', y = 'active_vehicles', data = uberFOILDF)


# In[ ]:


sns.swarmplot(x = 'dispatching_base_number', y = 'trips', data = uberFOILDF)


# In[ ]:


sns.swarmplot(x = 'dispatching_base_number', y = 'trips_to_active_vehicles', data = uberFOILDF)


# In[ ]:


UBER_RAW_DATA_APR_14_FILE = '../input/uber-raw-data-apr14.csv'
uberRawApr14DF = pd.read_csv(UBER_RAW_DATA_APR_14_FILE)
uberRawApr14DF.head()
uberRawApr14DF[uberRawApr14DF['Base'] == 'B02512']

