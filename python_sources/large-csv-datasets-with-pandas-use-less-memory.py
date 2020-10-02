#!/usr/bin/env python
# coding: utf-8

# # Tips for using pandas with large csv datasets

# ### How to get from 20.3 GB dataframe to 9.7 GB without deleting anything
# 
# The PLAsTiCC comptetion testset is nearly half-a-billion rows of csv data (453.65 million, to be exact). In the discussion I have seen people mention the more than 20 GB (in RAM) testset that we were provided with.
# The "tips" presented here may be obvious to some people, but even most of the top scorers did not use this in their public kernels.
# 
# This is useful even if you have enough RAM available and it also gives a minor speedup of calculations on the data (e.g. about 5% faster for calculating std on a column).
# 
# (Note: this kernel will not run as-is, it is just for demonstration of the methods to use on other machines)
# 

# ### Import libs

# ```
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import json
# 
# testset_file = '../input/test_set.csv'
# ```

# ### Standard way of reading in CSV with pandas

# ```
# %%time
# testset = pd.read_csv(testset_file)     # the standard way most people read in the data
# testset.info()
# ```

# ```
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 453653104 entries, 0 to 453653103
# Data columns (total 6 columns):
# object_id    int64
# mjd          float64
# passband     int64
# flux         float64
# flux_err     float64
# detected     int64
# dtypes: float64(3), int64(3)
# memory usage: 20.3 GB
# ```
# 
# As you can see, the testset occupies 20.3 GB of memory when stored in a dataframe, which is even larger than the unzipped csv file (19.3 GB).
# This will of course fail ('kernel died') in kaggle kernels, because it is larger then available memory.
# 
# Note that pandas - by default - uses 64-bit precision for all numbers (floats and ints). While this usually is okay for smaller datasets (like e.g. the training set), for very large datasets this can and should be optimized as follows.

# ### Optimized way of reading CSV with datatypes

# On the "data"  page of the competition, the actual precision and datatypes are given for each column. So we should make use of that information!

# ```
# thead = pd.read_csv(testset_file, nrows=5) # just read in a few lines to get the column headers
# dtypes = dict(zip(thead.columns.values, ['int32', 'float32', 'float64', 'float32', 'float32', 'bool']))   # datatypes as given by the data page
# del thead
# print('Datatype used for each column:\n', json.dumps(dtypes))
# testset = pd.read_csv(testset_file, dtype=dtypes)
# testset.info()
# ```

# ```
# Datatype used for each column:
#  {"object_id": "int32", "mjd": "float64", "passband": "int8", "flux": "float32", "flux_err": "float32", "detected": "bool"}
#  
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 453653104 entries, 0 to 453653103
# Data columns (total 6 columns):
# object_id    int32
# mjd          float64
# passband     int8
# flux         float32
# flux_err     float32
# detected     bool
# dtypes: bool(1), float32(2), float64(1), int32(1), int8(1)
# memory usage: 9.3 GB
# ```
# 

# So adding the `dtypes` argument to the `pd.read_csv` call makes a huge difference. The dataset is now only 9.3GB in Memory.
# 
# We can reduce it even further, if we are willing to reduce the accuracy. As we are on a "galactic timescale" here, probably timestamps with +/- 36 seconds accuracy are acceptable for the mjd colum (float64->float32), which reduces the memory footprint to 7.6 GB, almost one third of the original size. This could be done without reduction in precision or loss of information, by first subtracting the min value from all mjds to reduce the range (and neccessary number of significant digits).
# 
# If you are thinking about desparsifying the data by binning the bands into days, you could even live with integer mjds, which reduces it to 6.7 GB
# 

# ### mjd as float32
# ```
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 453653104 entries, 0 to 453653103
# Data columns (total 6 columns):
# object_id    int32
# mjd          float32
# passband     int8
# flux         float32
# flux_err     float32
# detected     bool
# dtypes: bool(1), float32(3), int32(1), int8(1)
# memory usage: 7.6 GB
# ```
# 
# ### mjd as uint16
# ```
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 453653104 entries, 0 to 453653103
# Data columns (total 6 columns):
# object_id    int32
# mjd          uint16
# passband     int8
# flux         float32
# flux_err     float32
# detected     bool
# dtypes: bool(1), float32(2), int32(1), int8(1), uint16(1)
# memory usage: 6.8 GB
# ```

# ## Storing the data
# In order to reduce your waiting time when loading the data, it makes sense to store it in a different format. Pandas offers a lot of different formats. Below is a table of the sizes on disk, in mem, read and write times using the full dataset with correct precision (9.3 GB in mem). This is of course only a quick and dirty benchmark, but should be enough to give you a feel.
# 
# The times may differ significantly on your machines, but the relative differences should stay similar. (this was measured using a GCP highmen-8 instance using the boot disk of 200GB, which is quite slow, so local NVMe storage will be much faster)
# 
# |Format|Size on disk|Read Time|Write Time|Size In Mem
# | :--- | ---: | ---: | ---: | ---:
# |csv|19.3 GB|4min 56s|  - |_ 9.3 GB
# |hdf|13.0  GB|  |_ 1min 07s | 12.7 GB
# |pickle|_ 9.3 GB|1min 28s |_ 1min 04s |_ 9.3 GB
# |pickle zip|_ 5.0 GB|2min 11s |11min 36s |_ 9.3 GB
# |feather|_ 9.0 GB|1min 17s |_ 0min 29s |_ 9.3 GB
# |parquet|_ 9.0 GB|1min 17s |_ 1min 29s | 12.7 GB|

# I have no clue why for hdf and parquet the actual in-memory size grows to 12.7 GB after writing a df of 9.3 GB to disk and then reading it back in, but this was reproducible. Maybe someone here has an explanation?

# In[ ]:





# In[ ]:




