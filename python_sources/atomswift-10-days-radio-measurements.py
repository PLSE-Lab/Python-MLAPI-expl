#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
from tzlocal import get_localzone
from datetime import datetime 


# In[ ]:


DATA_DIR = '/kaggle/input/10-days-radiation-measurements-at-living-house'


# File listing

# In[ ]:


data_files = []

for f in listdir(DATA_DIR):
    if isfile(join(DATA_DIR, f))         and f.endswith('.csv'):
        data_files.append(f)

print('Source data files:')
for f in data_files: print(f)


# Transform date format from unix time stamp to date and calculate mean doses

# In[ ]:


mean_dose = []
dates = []
for f in data_files:
    mse = pd.read_csv(join(DATA_DIR, f), sep=';')
    mse[mse.columns[0]] = pd.to_datetime(mse[mse.columns[0]], 
                                         unit='ms', 
                                         utc=True)
    mse[mse.columns[0]] = mse[mse.columns[0]]        .apply(lambda dt: dt.astimezone(get_localzone()))
    mse[mse.columns[1]] = mse[mse.columns[1]]        .apply(lambda val: float(val.replace(',', '.')))
    edge_date = datetime        .strptime(f.split('_')[1]
        .split('.csv')[0], '%d.%m.%Y')\
        .replace(hour=15, minute=0)\
        .astimezone(get_localzone())
    filtered_mse = mse[mse[mse.columns[0]] < edge_date]
    dates.append(edge_date)
    mean_dose.append(filtered_mse.iloc[:, [1]].values.mean())

dates = np.asarray(dates).reshape(len(dates), 1)
mean_dose = np.asarray(mean_dose).reshape(len(mean_dose), 1)
means = np.concatenate((dates, mean_dose), axis=1)
sorted_means = np.asarray(sorted(means, key=lambda x: x[0]))


# In[ ]:


plt.figure(figsize=(13,4))
plt.plot(sorted_means[:, 0], sorted_means[:, 1])
plt.title('Radiation doses')
plt.ylabel(mse.columns[1])
plt.xlabel('date, days')
plt.show()

