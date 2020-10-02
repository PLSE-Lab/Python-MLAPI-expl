#!/usr/bin/env python
# coding: utf-8

# # Analysis of data in malware database
# This is my first kernel. My idea is just to start the understanding of the basics for Data science. My idea in this kernel is to understand what variables can be expendable based on some data analysis I collect from another kernels I read. Let it me know your thoughts about this approach. It'll be kindly appreciated.

# Let's start loading the libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os

# Any results you write to the current directory are saved as output.


# Here are the types I use. :
# - I load objects as categories. 
# - Binary values are switched to int8
# - Binary values with missing values are switched to float16 (int does not understand nan)
# - 64 bits encoding are all switched to 32, or 16 of possible

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


# Let's load the training set.

# In[ ]:


train_df = pd.read_csv('../input/train.csv', dtype=dtypes)


# In[ ]:


train_df.info()


# In[ ]:


# shape
print(train_df.shape)


# There are 83 variables

# How many columns are null?

# In[ ]:


train_df.isnull().sum()


# In[ ]:


print(train_df.info())


# In[ ]:


train_df.corr()['HasDetections']


# Histogram of HasDetections. There are several variables with positive correlation. But the greater value is: **AVProductStatesIdentifier**

# Let's do a quick snapshot of data

# In[ ]:


train_df.tail()


# Detections by Processor type

# In[ ]:


f,ax=plt.subplots(1,1,figsize=(18,8))
sns.countplot('Census_OSArchitecture',hue='HasDetections',data=train_df,ax=ax)
ax.set_title('Os Architecture:Detected vs not detected')
plt.show()


# The information in **AMD** seems more complete than in **x86**...

# Let's try when is gamer or not

# In[ ]:


f,ax=plt.subplots(1,1,figsize=(18,8))
sns.countplot('Wdft_IsGamer',hue='HasDetections',data=train_df,ax=ax)
ax.set_title('Is gamer:Detected vs not detected')
plt.show()


# First thought about this. Maybe in case of gamers' machines the information is not available? It should be skipped from the analysis?

# Violin plot between Country and OS Architecture.

# In[ ]:



#g = sns.FacetGrid(train_df, col='HasDetections')
#g.map(plt.hist, 'CountryIdentifier')

sns.violinplot(data=train_df,x="CountryIdentifier", y="Census_OSArchitecture")


# It seems balanced.

# Let's evaluate browser and detection of malware... 

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['IsBeta','HasDetections']].groupby(['IsBeta']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Detected vs Beta')
sns.countplot('IsBeta',hue='HasDetections',data=train_df,ax=ax[1])
ax[1].set_title('Browser:Detected vs Not')
plt.show()


# Conclusion, Beta status seems balanced either related to detection or not.

# Let's check Census_GenuineStateName. This variable defines it's a genuine installation or not..

# In[ ]:


train_df['Census_GenuineStateName'].value_counts().plot(kind="bar");


# Ok. Let's see the relation with the malware detection (or not).

# In[ ]:


f,ax=plt.subplots(1,1,figsize=(18,8))
sns.countplot('Census_GenuineStateName',hue='HasDetections',data=train_df,ax=ax)
ax.set_title('Is genuine or not:Detected vs not detected')
plt.show()


# First conclusion: It seems the detection is not affected or not that is genuine software or not.

# Let's evaluate the balancing on the has detections (target column)

# In[ ]:


train_df['HasDetections'].astype(int).plot.hist();


# There's balanced: Either the detected like not detected so I think this is very good. 

# In[ ]:


train_df['Platform'].value_counts().plot(kind="bar");


# Relation between platform vs total physical ram and number of processor core count

# In[ ]:


g = sns.FacetGrid(train_df, hue="HasDetections", col="Platform", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Census_TotalPhysicalRAM", "Census_ProcessorCoreCount",edgecolor="w").add_legend();


# Relations between platform and detection

# In[ ]:


f,ax=plt.subplots(1,1,figsize=(18,8))
sns.countplot('Platform',hue='HasDetections',data=train_df,ax=ax)
ax.set_title('Platform:Detections or not')
plt.show()


# Mostly are **windows10**

# In[ ]:


# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# Let's check the category columns. Checking the variables missing rate (top 20).

# In[ ]:


# Missing values statistics
missing_values = missing_values_table(train_df)
missing_values.head(20)


# * PuaMode, Census_ProcessorClass, DefaultBrowsersIdentifier, Census_IsFlightingInternal and Census_InternalBatteryType have over 70% null data.
# * Let's check some of them regarding to the target.

# Let's evaluate the top variables displayed: **PuaMode**

# In[ ]:


train_df['PuaMode'].value_counts()


# In[ ]:


sns.countplot(x='PuaMode', hue='HasDetections',data=train_df)
plt.show()


# There's some pattern here but the number is quite small.

# Let's check  **Census_ProcessorClass**.

# In[ ]:


train_df['Census_ProcessorClass'].value_counts()


# In[ ]:


sns.countplot(x='Census_ProcessorClass', hue='HasDetections',data=train_df)
plt.show()


# It seems there's a pattern here. But I think there's not enough samples...

# I think the best is remove this feature with so much nulls in the columns

# In[ ]:


del train_df['PuaMode']
del train_df['Census_ProcessorClass']
del train_df['DefaultBrowsersIdentifier']
del train_df['Census_IsFlightingInternal'] 
del train_df['Census_InternalBatteryType']


# Let's check the possible values in the category columns.

# In[ ]:


train_df.select_dtypes('category').apply(pd.Series.nunique, axis = 0)


# MachineIdentifier has almost a value per row! Let's delete this column. It seems it's a key column.

# In[ ]:


del train_df['MachineIdentifier']


# Checking the histogram.

# In[ ]:


train_df.hist(figsize=(15,20))
plt.figure()

