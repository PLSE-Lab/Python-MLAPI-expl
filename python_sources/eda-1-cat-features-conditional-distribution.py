#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('pylab', 'inline --no-import-all')
import pandas as pd
import seaborn as sns
import tqdm
import scipy 
import gc
import pickle
import re
import itertools
import os
import cufflinks as cf
cf.go_offline()


# Loading output data from my previous kernel (https://www.kaggle.com/daletskidenis/interactive-data-cleaning-and-preprocessing)

# In[ ]:


DATA_PATH = "../input/interactive-data-cleaning-and-preprocessing/"

with open(os.path.join(DATA_PATH, "train_clean.pkl"), 'rb') as f:
    train_data = pickle.load(f)


# Selecting category columns

# In[ ]:


def category_columns(data: pd.DataFrame):
    cats = set(data.select_dtypes(['category', 'object']).columns.tolist())
    cats = cats.difference({'MachineIdentifier', 'HasDetection'})
    return list(cats)

ALL_COLS = category_columns(train_data)


# Function to draw fraction of infected computers in each category.
# The more it has deviations from 0.5 - the more informative column is supposed to be

# In[ ]:


def graph_category_distr(data: pd.DataFrame, col):
    col_data = data[col]
    
    infected_count = sum(data.HasDetections==1) / len(data)
    balancing_multiplier = 0.5 / infected_count
    infected_counts = col_data[data.HasDetections==1].value_counts() / col_data.value_counts() * balancing_multiplier

    ic = pd.DataFrame(list(infected_counts.items()), columns=['label', 'infected'])
    ic['label'] = '(' + ic['label'].astype(str) + ')'
    
    ic.iplot(kind='bar', x='label', y='infected', title=col)

def plot_distr(col):
    graph_category_distr(train_data, col)


# In[ ]:


plot_distr('SMode')


# In[ ]:


plot_distr( 'Census_ActivationChannel')


# In[ ]:


plot_distr('Census_ProcessorManufacturerIdentifier')


# In[ ]:


plot_distr('AVProductStatesIdentifier')


# In[ ]:


plot_distr('Census_OSBranch')


# In[ ]:


plot_distr('RtpStateBitfield')


# In[ ]:


plot_distr('SkuEdition')


# In[ ]:


plot_distr('Census_DeviceFamily')


# In[ ]:


plot_distr('OsBuildLab')


# In[ ]:


plot_distr('IeVerIdentifier')


# In[ ]:


plot_distr('Processor')


# In[ ]:


plot_distr('Platform')


# In[ ]:


plot_distr('Census_FirmwareVersionIdentifier')


# In[ ]:


plot_distr('AvSigVersion')


# In[ ]:


plot_distr('OsVer')


# In[ ]:


plot_distr('SmartScreen')


# In[ ]:


plot_distr('Census_ProcessorModelIdentifier')


# In[ ]:


plot_distr('Census_FlightRing')


# In[ ]:


plot_distr('Census_OEMModelIdentifier')


# In[ ]:


plot_distr('Census_GenuineStateName')


# In[ ]:


plot_distr('CityIdentifier')


# In[ ]:


plot_distr('Wdft_RegionIdentifier')


# In[ ]:


plot_distr('CountryIdentifier')


# In[ ]:


plot_distr('OrganizationIdentifier')


# In[ ]:


plot_distr('AppVersion')


# In[ ]:


plot_distr('OsPlatformSubRelease')


# In[ ]:


plot_distr('ProductName')


# In[ ]:


plot_distr('EngineVersion')


# In[ ]:


plot_distr('Census_OSInstallTypeName')


# In[ ]:


plot_distr('Census_PrimaryDiskTypeName')


# In[ ]:


plot_distr('Census_ChassisTypeName')


# In[ ]:


plot_distr('Census_OSVersion')


# In[ ]:


plot_distr('LocaleEnglishNameIdentifier')


# In[ ]:


plot_distr('PuaMode')


# In[ ]:


plot_distr('Census_OSUILocaleIdentifier')


# In[ ]:


plot_distr('GeoNameIdentifier')


# In[ ]:


plot_distr('UacLuaenable')


# In[ ]:


plot_distr('Census_FirmwareManufacturerIdentifier')


# In[ ]:


plot_distr('Census_OEMNameIdentifier')


# In[ ]:


plot_distr('Census_OSWUAutoUpdateOptionsName')


# In[ ]:


plot_distr('Census_OSInstallLanguageIdentifier')

