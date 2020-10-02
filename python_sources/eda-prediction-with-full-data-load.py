#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import random
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# ## Microsoft Malware Detection -> 
# 
# It is one the most challenging and huge dataset for data scientists. 
# Train Data ->  (4.08 GB)
# Test Data -> (3.54 GB)
# 
# Such a Huge Data set first will try to load only 1% of  sample data and see the nature of training data

# ## **Load Data**[](http://)

# In[ ]:


#%%time
#p = 0.1 ## 10% of the random data
#dfsample = pd.read_csv('../input/train.csv' , skiprows=lambda i: i>0 and random.random() > p)
#print(dfsample.shape)


# In[ ]:


#dfsample.head()


# ## We can see data set has 83 ->  Columns and around -> 8921483 rows with mix of Categorical and Numeric Variables -> 
# ### In order to load such a huge data set we need to manually assign data type to each variable for efficient Memory Management 

# In[ ]:


#dfsample.dtypes


# ## Let's convert data type object to category and numeric to int8, int16 and int32 for efficient memory managment. 

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


#del dfsample


# In[ ]:



numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = [c for c,v in dtypes.items() if v in numerics]
categorical_columns = [c for c,v in dtypes.items() if v not in numerics]
#df_train = pd.read_csv('../input/train.csv' , dtype= dtypes , )
#print(df_train.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', "nrows = 4000000\n#_______________________________________________________________________________\nretained_columns = numerical_columns + categorical_columns\ntrain = pd.read_csv('../input/train.csv',\n                    nrows = nrows,\n                    usecols = retained_columns,\n                    dtype = dtypes)\n#_______________________________________________________________\nretained_columns += ['MachineIdentifier']\nretained_columns.remove('HasDetections')\ntest = pd.read_csv('../input/test.csv',\n                   usecols = retained_columns,\n                   dtype = dtypes)")


# In[ ]:


train.head()


# In[ ]:


#train = reduce_mem_usage(train)
#test = reduce_mem_usage(test)


# ### Let's Join train & test Data set for efficient Data Prepocessing -> 

# In[ ]:


#df_train1 = df_train.iloc[:,:-1]


# In[ ]:


#%%time
#df_join = df_train1.append(df_test)


# In[ ]:


#df_join.head()


# In[ ]:


#df_train.describe().T 


# In[ ]:


#df_train.isna().sum()


# ## EDA 

# In[ ]:


#df_train.HasDetections.value_counts()


# In[ ]:


#df_train.corr() 


# In[ ]:


#import seaborn as sns
#sns.heatmap(df_train.corr() )


# ## Data PreProcessing 

# In[ ]:


## Removing columns having more than 50% values as null
for i in train.columns:
        #print(dfsample[i].isna().sum())
        if train[i].isna().sum() > (train.shape[0]/2):
            print(i)
            train.drop(i, inplace = True , axis =1)
print(train.shape)            


# In[ ]:


for i in train.columns:
    if train[i].dtypes in ['int8','int16','float16','float32']:
        #print(i, df_train[i].mean())
        train[i].fillna(train[i].mean(), inplace = True)
    else:
        #print(i, df_train[i].mode())
        train[i].fillna(train[i].mode(), inplace = True)


# In[ ]:


train.shape


# In[ ]:


for i in train.columns:
        if train[i].isna().sum() >0 :
            print(i)
            train.drop(i, inplace = True , axis =1)
print(train.shape)            


# In[ ]:


train.describe().T


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in train.columns:
    if train[i].dtypes not in ['int8','int16','float16','float32']:
        print(i)
        print(len(train[i].unique()))
        if len(train[i].unique()) > 50:
            train.drop(i, inplace = True , axis =1)
        else:
            train[i] = le.fit_transform(train[i])


# In[ ]:


## Let's remove that column which has high number of unique categories 


# In[ ]:


train['HasDetections'].value_counts()


# ## Modeling 

# In[ ]:


from sklearn.model_selection import train_test_split
X = train.drop(['HasDetections'], axis =1)
y = train['HasDetections']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


#del df_train
#del X
#del y


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_train, y_train)


# In[ ]:


lr.score(X_test, y_test)


# ## Ensemble - RandomForestClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, max_depth=2, min_samples_split=2, random_state=0)
rfc.fit(X_train, y_train)
print("Training Accuracy",rfc.score(X_train, y_train))
print("Test Accuracy",rfc.score(X_test, y_test))


# In[ ]:


## Ensemble - RandomForestClassifier


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=3, random_state=123).fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))


# ## Test Data -> 

# In[ ]:


test1 = test[X.columns]


# In[ ]:


for i in test1.columns:
    if test1[i].dtypes in ['int8','int16','float16','float32']:
        #print(i, df_train[i].mean())
        test1[i].fillna(test1[i].mean(), inplace = True)
    else:
        #print(i, df_train[i].mode())
        test1[i].fillna(test1[i].mode(), inplace = True)


# In[ ]:


test1['Census_GenuineStateName'].mode()
test1['Census_GenuineStateName'].fillna('IS_GENUINE', inplace = True)   
test1.Census_OSEdition.mode() 
test1['Census_OSEdition'].fillna('Professional', inplace = True)


# In[ ]:


test1.shape


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in test1.columns:
    if test1[i].dtypes not in ['int8','int16','float16','float32']:
        print(i)
        print(len(test1[i].unique()))
        if len(test1[i].unique()) > 50:
            test1.drop(i, inplace = True , axis =1)
        else:
            test1[i] = le.fit_transform(test1[i])


# In[ ]:


test_pred = clf.predict(test1)


# In[ ]:


submission = pd.DataFrame()
submission['MachineIdentifier'] = test['MachineIdentifier']
submission['HasDetections'] = test_pred
submission.to_csv('submission.csv' , index= False)

