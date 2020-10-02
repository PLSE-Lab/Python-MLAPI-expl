#!/usr/bin/env python
# coding: utf-8

# This particular competition is asking us to look and analyze some really big data sets. In its given form, it won't even load into pandas on the kaggle kernels. On the off chance that you don't have an extremely extravagant PC, it most likely won't load on yours either.
# 
# Important tips to handle big dataset
# 
# <b>
# <br>TIP 1 - Deleting unused variables and gc.collect()
# <br>TIP 2 - Presetting the datatypes
# <br>TIP 3 - Importing selected rows of the a file (including generating your own subsamples)
# <br>TIP 4 - Importing in batches and processing each individually
# <br>TIP 5 - Importing just selected columns
# <br>TIP 6 - Creative data processing
# <br>TIP 7 - Using Dask
# </b>
# <br>
# 
# <br>Here, I am just comparing to packages pandas and dask which one is better to load this dataset that's challange here.
# 
# we all know that dask is famous for load large datasize in fractions of second.
# 
# so let's look at some picture of both pandas and dask performance with time.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dask.dataframe as dd
import dask
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = pd.read_csv("../input/train.csv", nrows=20_00_000)\nprint("Pandas dataframe : ",train.shape)\ngc.collect()')


# In[ ]:


dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = dd.read_csv("../input/train.csv", blocksize= 256e6, dtype = dtypes)\ngc.collect()')


# ### Convert dask dataframe to pandas

# In[ ]:


train_df = train_df.compute()


# In[ ]:


train_df.shape ##see the pandas dataframe size


# you can see that what a advantage to use dask to read dataset so quickly.
# 
# <html>
# <head>
# <style>
# table, th, td {
#   border: 1px solid black;
#   border-collapse: collapse;
# }
# th, td {
#   padding: 5px;
#   text-align: left;    
# }
# </style>
# </head>
# <body>
# 
# <h4> You can check below performance</h4>
# 
# <table style="width:100%">
#   <tr>
#       <th><b>DASK</b></th>
#       <th><b>PANDAS</b></th>
#   </tr>
#   <tr>
#       <td><b>23.3 s FOR 20 LAC ROWS</b></td>
#       <td><b>380 ms FOR WHOLE DATASET</b></td>
#   </tr>
# </table>
# 
# </body>
# </html>
