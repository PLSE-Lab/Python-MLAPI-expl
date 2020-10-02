#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob, re, os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# In[ ]:


DATA_DIR = "../input"


# In[ ]:


#borrowed from the1owl's notebook
dfs = { re.search('(\w*)\.csv', fn).group(1):
    pd.read_csv(fn) for fn in glob.glob(os.path.join(DATA_DIR,'*.csv'))}
for k, v in dfs.items(): locals()[k] = v


# In[ ]:


dfs.keys()


# In[ ]:


train_stores = air_visit_data.air_store_id.unique()


# In[ ]:


test_stores = sample_submission.id.str.extract("^(.*)_\d").unique()


# # Stores missing in test

# In[ ]:


missing_stores = set(train_stores) - set(test_stores) 

missing_stores


# In[ ]:


air_visit_data.head()


# In[ ]:


test_visit_data  = pd.DataFrame({"air_store_id":sample_submission.id.str.extract("^(.*)_\d"),
                                "visit_date": pd.to_datetime(sample_submission.id.str.extract("(\d+-\d+-\d+)")) }
                                ,index = sample_submission.index, )


# In[ ]:


test_visit_data.visit_date.min()


# In[ ]:


test_visit_data.visit_date.max()


# In[ ]:


air_visit_data.visit_date.max()


# In[ ]:


air_visit_data.visit_date.min()


# In[ ]:


air_visit_data.visit_date  = pd.to_datetime(air_visit_data.visit_date)


# In[ ]:


air_visit_data.dtypes


# In[ ]:


get_ipython().run_line_magic('pinfo', 'ax.legend')


# # Plot daily visitors for first 100 stores

# In[ ]:


def plot_store(store_id):
    fig, ax = plt.subplots(figsize = (20,10))
    dt2016 = air_visit_data.loc[(air_visit_data.air_store_id == store_id) & (air_visit_data.visit_date.dt.year == 2016),:]
    dt2017 = air_visit_data.loc[(air_visit_data.air_store_id == store_id) & (air_visit_data.visit_date.dt.year == 2017),:]
    ax.plot(dt2016.visit_date.dt.dayofyear, dt2016.visitors, "green")
    ax.plot(dt2017.visit_date.dt.dayofyear, dt2017.visitors, color = "red")
    ax.set(title = "store = "+ store_id)
    ax.legend(["2016", "2017"])


# In[ ]:


for s in air_visit_data.air_store_id.unique()[:100]:
    plot_store(s)
    
plt.show()


# # Missing stores

# In[ ]:


for s in missing_stores:
    plot_store(s)
    
plt.show()


# In[ ]:




