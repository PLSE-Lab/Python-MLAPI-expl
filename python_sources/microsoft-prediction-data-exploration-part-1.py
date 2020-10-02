#!/usr/bin/env python
# coding: utf-8

# # Microsoft Malware Prediction EDA Part 1
# 
# In this kernel, we'll load in the data and explore the data.
# 
# We have three aims:
# 
# 1. To check the variable types
# 
# 2. To see the relationship each variable has our response variable, ' HasDetections'
# 
# 3. To see how the missing values are distributed
# 
# Let's get going!

# As always, let's begin by loading in useful packages.....

# # Loading in the data

# In[ ]:


###Loading in useful packages

#for linear algebra
import numpy as np

#for data manipulation
import pandas as pd

#for plotting
import matplotlib.pyplot as pp
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#For surpressing warnings
import warnings
warnings.filterwarnings('ignore')

#For opening Zip files
import zipfile as zf


# The first step prior to loading in the data is to specify the data types for each of the variables and the size of each data type. This will save memory because Python automatically assigns 64 bits to each numeric variables,regardless of whether it actually needs it, by specifying the size of data type, we'll save memory.

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


# When , I initally ran this on Google Colab, my next step was to unzip and save the file to somewhere else. I have done this already and, as a result, I have commented out the code.

# In[ ]:


#Change the training and testing sets to pickle format

#I have commented these out because I have already completed this step and these files are avaliable on my drive now.

#pd.read_csv(zf.ZipFile("/content/drive/My Drive/Microsoft/train.zip").open("train.csv"), dtype=dtypes).to_pickle("/content/drive/My Drive/Microsoft/train.pkl")
#pd.read_csv(zf.ZipFile("/content/drive/My Drive/Microsoft/test.zip").open("test.csv"), dtype=dtypes).to_pickle("/content/drive/My Drive/Microsoft/test.pkl")


# In[ ]:


#Opening the saved pickles - which contain the data :)

train = pd.read_pickle("../input/mctrain/train.pkl")
test =pd.read_pickle("../input/mctest/test.pkl")


# # 1. Determining the types of variables
# 

# In[ ]:


#Looking at shape of training data, i.e number of rows and columns

print("Training data dimensions",train.shape)
print("Testing data dimension",test.shape)


# Okay , so we have 82 different columns variables. Training has 83 because the it has the response variable column
# 
# Let's try and distinguish the variable types

# In[ ]:


#Training variable data types
train.info()


# In[ ]:


#Make a dictionary of all the variables
all_the_vars = list(train.drop(['MachineIdentifier','HasDetections'],axis=1))

#Make a list of categorical variables
cat_vars = [i for i in all_the_vars if (train[i].dtype.name == 'category') | (train[i].dtype.name == 'object')]

#Make a list of binary variables
bin_vars = [i for i in all_the_vars if len(train[i].value_counts()) <= 2 ]

#Make a list of pure numerical variables
num_vars = [i for i in all_the_vars if (i not in cat_vars) & (i not in bin_vars)]

print('We have ',len(all_the_vars),' explanatory variables')
print('We have ',len(cat_vars),' categorical variables')
print('We have ',len(bin_vars),' binary variables')
print('We have ',len(num_vars),' numerical variables')


# We have **29 categorical variables** , **20 binary varibales**, and **33 pure numeric variables**
# 
# Notice that our target variable, **HasDetections** - which is at the bottom of the list-is an integer variable. It's values are either 0 or 1.
# 
# *Also note that  **MachineIdentifier** is the just the id for each computer. Not an explanatory variable*
# 
# 

# # 2. Seeing the relationship which the explanatory variables have with the response variable, 'HasDetections'
# 

# Let's make a bar chart of the 'HasDetections' variables, to gain an idea of how it's distributed...

# In[ ]:


#Let's examine the distribution of HasDetections
j = train['HasDetections'].value_counts()
j = j/len(train)
j.plot.bar()
pp.title('Distribution of HasDetections')
pp.xlabel('HasDetections values')
pp.ylabel('Proportion')
pp.show()


# Okay, so it looks like a pretty even split between not having malware detected and having malware detected.
# 
# If we just use historical training data, it seems that getting malware is a coin flip. Keeping this in mind, let's see how each features interacts with 'HasDetections' and if the probabilities of detecting malware for each feature are more insightful than a coin flip.

# In[ ]:


for i in cat_vars:
  length = len(train)
  j = train[i].value_counts() / length
  j = j.sort_values(ascending=False)
  j = j.iloc[:10]
  x = list(j.index)
  y = list(j.values)
  z= list()

  for j in x:
    z.append(train['HasDetections'].loc[train[i]==j].mean())
  

  fig, ax1 = pp.subplots()
  ax1.bar(x,y)
  pp.xticks(x,y)
  locs, labels = pp.xticks()
  pp.setp(labels, rotation=90)
  pp.title(i)
  pp.ylabel('Proportion')
  pp.xlabel(i + ' values')

  ax2 = ax1.twinx()
  ax2.plot(x,z,'r',linestyle='-', marker='o')
  ax1.grid(False)
  ax2.grid(False)
  pp.ylabel('P(HasDetections == 1)')

  pp.show()
  


# So my criteria for potentially useful variables or parts of variables is whether is
# | P(HasDetections ==1) - 0.5 |  > 0.04 :
# 
# 'ProductName_windowsintune', 'EngineVersion', 'AppVersion','AvSigVersion' , 'Platform_windows2016', 'Processor_x86', 'Processor_am64', 'OsVer','OsPlatformSubRelease', ' OsBuildLab', 'SkuEdition_Server', 'PuaMode_audit', 'SmartScreen_RequireAdmin', 'SmartScreen_ExisitsNotSet ', 'Census_MDC2FormFactor', 'Census_DeviceFamily_Windows.Server','Census_DeviceFamily_Windows', 'Censor_ProcessorClass_mid', 'Census_ProcessorClass_high', 'Census_PrimaryDiskTypeName_UNKNOWN', 'Census_PrimaryDiskTypeName_Unspecified', 'Census_ChassisTypeName_Other', 'Census_PowerPlatformRoleName_Slate', 'Census_PowerPlatformRoleName_AppliancePC' ,'Census_PowerPlatformRoleName_EntepriseServer', 'Census_InternalBatteryType_nimh', ##'Census_InternalBatteryType_log20'##, 'Census_OSVersion',  'Census_OSArchitecture_x86', ''Census_OSArchitecture_am64'', 'Census_OSBranch', 'Census_OSInstallTypeName', 'Census_OSWUAutoUpdateOptionsName_DownloadNotify', 'Census_GenuineStateName_UNKNOWN', 'Census_ActivationChannel' and 'Census_FlightRing'
# 

# In[ ]:


for i in bin_vars:
  length = len(train)
  j = train[i].value_counts() / length
  j = j.sort_values(ascending=False)
  x = list(j.index)
  y = list(j.values)
  z= list()

  for j in x:
    z.append(train['HasDetections'].loc[train[i]==j].mean())
  

  fig, ax1 = pp.subplots()
  ax1.bar(x,y)
  pp.title(i)
  pp.ylabel('Proportion')
  pp.xlabel(i + ' values')

  ax2 = ax1.twinx()
  ax2.plot(x,z,'r',linestyle='-', marker='o')
  ax1.grid(False)
  ax2.grid(False)
  pp.ylabel('P(HasDetections == 1)')

  pp.show()


# For binary variables,same criteria as before, the notable areas seem to be:
# 
# 'Wdft_IsGamer_1',  'Census_IsAlwaysOnAlwaysConnectedCapable_1', 'Census_IsPenCapable_1', 'Census_IsTouchEnabled_1' , 'Census_IsVirtualDevice_1', ##'Cenus_IsWIMBootEnabled_1'##, ##'Census_ThresholdOptIn_1'##, 'Census_IsFlightingInternal_1', 'Census_IsPortableOperatingSystem', 'Census_HasOpticalDiskDrive' , ##'Firewall_0'## ,'SMode_1', 'PuaMode_audit' , 'AutoSampleOptIn_1 ', 'IsProtected_0', 'IsSxsPassiveMode_1'

# In[ ]:


import gc
gc.collect()


# In[ ]:


for i in num_vars:
  length = len(train)
  j = train[i].value_counts() / length
  j = j.sort_values(ascending=False)
  x = np.array(j.index)
  y = np.array(j.values)
  z= list()

  ax = sns.kdeplot(train[i].loc[train['HasDetections']==1],label='HasDetections ==1',color='r')
  ax = sns.kdeplot(train[i].loc[train['HasDetections']==0],label='HasDetections ==0',color='b')
  pp.title(i)
  pp.ylabel('Density')
  pp.xlabel(i+' values')

  
  pp.show()


# For pure numerics, my criteria is whether there appears to at least one siginificant divergence between the lines, it's a subjective measure. the notable variables seem  to be:
# 
# 'DefaultBrowsersIdentifier' , 'AVProductStatesIdentifier', 'AVProductsInstalled' , 'CountryIdentifier', 'CityIdentifier', 'UacLuaenable', 'Censor_ProcessorModelIdentifier', 'Census_PrimaryDiskTotalCapacity' , 'Census_SystemVolumeTotalCapacity', 'Census_TotalPhysicalRAM', 'Census_InternalPrimaryDiagnonalDisplaySizeInInches' , 'Census_InternalPrimaryDisplayResolutionVertical', 'Census_FirmwareManufacturerIdentifier'
#  

# # 3. Checking Missing Values

# Let's see the breakdown of missing values in our training data
# 

# In[ ]:


#Let's creating a series which contains the proportion of missing values for each variable
mis_val = 100 * train.isnull().sum()/len(train)

#Let's view variables with missing values
mis_val[mis_val >0].sort_values(ascending=False)


# Over half of our variables have missing values :(
# 
# 
# PuaMode, Censor_ProcessorClass,DefaultBrowerIdentifier, Census_IsFlightingInternal,Census_InternalBatteryType,Census_ThresholdOptIn and Census_IsWIMBootEnabled have the greatest proportion of missing values.
# 
# Having dug further, PuaMode, Census_InternalBatteryType and SmartScreen seem potentially viable, intuitively. 
# I'm very tempted,however, to remove the other  variables  with high ratios of missing values BUT let's  examine the data further before doing this.

# # Save the files
# 
# So we've accomplished our three aims and in our next kernels, we'll deal with missing values and possibly add new variables

# In[ ]:


#Save them to separate pickles
#train.to_pickle("/content/drive/My Drive/Microsoft/train.pkl")
#test.to_pickle("/content/drive/My Drive/Microsoft/test.pkl")

