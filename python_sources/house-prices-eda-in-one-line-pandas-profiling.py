#!/usr/bin/env python
# coding: utf-8

# #### [Pandas Profiling](https://pypi.org/project/pandas-profiling/) is a simple way to perform your initial [exploratory data analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis) (EDA). It is *very* easy to use, although initially it *may* take a while to run. For that reason in this example we shall use *minimal mode* (`minimal=True`). This is a default configuration that disables expensive computations (such as correlations and dynamic binning).

# In[ ]:


import pandas as pd
from pandas_profiling import ProfileReport

# Read the training data into a pandas DataFrame
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

# Now produce a profile report:
profile = ProfileReport(train_data, minimal=True, title="Pandas Profiling Report")
# For a more complete report, with correlations and dynamic binning etc. 
# remove the minimal=True flag.


# In[ ]:


import warnings
warnings.filterwarnings("ignore") # silence an iframe warning


# In[ ]:


# now display the profile report
profile.to_notebook_iframe()


# ### Related reading:
# For more details see the [Pandas Profiling](https://pypi.org/project/pandas-profiling/) project web page.
