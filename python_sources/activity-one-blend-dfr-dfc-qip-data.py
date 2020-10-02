#!/usr/bin/env python
# coding: utf-8

# ### Step Two - Blend Data
# #### Activity One - Import and Consolidate DFR, DFC, and QIP datasets 
# * Dialysis Facility Report Fiscal Year (FY) 2018
# * Dialysis Facility Compare Calendar Year (CA) 2018
# * ESRd Quality Incentive Program (QIP) Payment Year (PY) 2018 Dataset
# 
# All the three data sources contain data for year 2016.
# 
# We will use DFR as the basis for this research since it has the most comprehensive information about dialysis facilities, clinical measures, and patient populations. Since DFR does not include an data element for zip code of a facility's physical location, we will use zip code data from both DFC and QIP dataset by matching CMS Certification Number (CCN), a uinque identifier for a facility. Later we zip code to get information about socioeconomic characteristics of the community where the facilities are located.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 100)  


# DFR dataset has over 3,000 variable and 7,000 rows (facilities). The size of the dataset is close to 200 MB. We only need a fraction of the number of variables. We first set up a dictionary for a list of interested variables and their shorter names. Only variables from the list will be imported for further analysis. This will avoid importing the entire dataset.
# 
# Also we set up a dictionary of variable types for CCN, Network, and Zip code since they are categorical variable, not numerical variable even thought their values appear to be numerical.
# 
# VERY IMPORTANT:
# 
# CMS Certification Number (CCN) has six digits and Zip Code has five digits. Some CCNs and Zip Codes have leading zeros. We need to make sure the leading zeros are not truncated during the import process. THe trick is to declare it as a string type and not integer. 

# In[ ]:


dtype_dict= {'provfs': str,          # provider CCN numbers should be treated as strings
             'network': str}         # The network numbers should be treated as strings. There are 18 networks.

column_dict ={
    'provfs':'CCN',
    'chainnam':'ChainOwner',
    'network':'Network',
    'provcity': 'City',
    'provname':'Name',
    'state':'StateCode',
    'FIRST_CERTDATE':'InitCertDate',
    'owner_f':'ProfitStatus',
    'sv_Shift_start_after_5_pm':'EveningShift',
    'sv_Practices_Dialyzer_Reuse':'DialyzerReuse',
    'totstas_f':'TotalStations',
    'staffy4_f':'TotalStaff',
    'compl_cond':'ComplianceStatus',
    'allcnty4_f':'TotalPatients', 
    'medicarey4_f':'PctgMedicare',           # % Medicare
    'medpendy4_f':'PctgMedicarePending',     #  % Medicare Pending
    'nonmedy4_f':'PctgNonMedicare',          #  % Non-Medicare
    'agey4_f':'AverageAge',
    'age1y4_f':'PctgAge18',                  # % Age < 18
    'age2y4_f':'PctgAge18t64',               # % Age 18 to 64
    'age3y4_f':'PctgAge64',                  # % Age > 64
    'sexy4_f':'PctgFemale',                  # % Female
    'rac2y4_f':'PctgBlack',                  # % African American
    'eth1y4_f':'PctgHispanic',               # % Hispanic
    'srry4_f':'SRR',                         # Standardized Readmission Ratio
    'CWesarxy4_f': 'PctgESAPrescribed',      # % ESA Prescribed
    'FVhy4_f': 'PctgFluVaccine',             # % patients flu Vaccinated
    'CWavgHGBy4_f': 'AvgHemoglobin',         # Average hemoglobin levels (g/dL), of valid in range adult pt-mths, 2016
    'CWhgb1y4_f': 'PctgHemoglobin10',        # % Hemoglobin < 10 g/dL
    'CWhgb2y4_f': 'PctgHemoglobin10t11',     # % Hemoglobin 10 to 11 g/dL
    'CWhgb3y4_f': 'PctgHemoglobin11t12',     # % Hemoglobin 11 to 12 g/dL
    'CWhgb4y4_f': 'PctgHemoglobin12',        # % Hemoglobin > 12 g/dL
    'CWhdavgufry4_f': 'AvgUFR',              # Average UFR, of valid in range adult HD pt-mths, 2016'
    'CWhdufr1y4_f': 'PctgUFRLE13',           # % of adult HD pt-mths with UFR <= 13, 2016'
    'CWhdufr2y4_f': 'PctgUFRGT13',           # % of adult HD pt-mths with UFR > 13, 2016'
    'CWhdavgktvy4_f': 'AvgKtV',              # Average Kt/V, of valid in range adult HD pt-mths, 2016
    'CWhdktv1y4_f':  'PctgKtV12',            # % Kt/V < 1.2
    'CWhdktv2y4_f':  'PctgKtV12t18',         # % Kt/V 1.2 to 1.8
    'CWhdktv3y4_f': 'PctgKtV18',             # % Kt/V >= 1.8
    'CWhdktv4y4_f': 'PctgKtVOther',          # % Kt/V Missing or Out of Range
    'CWavgPy4_f':'AvgSerumPhosphorous',      # Avg serum phosphorous (mg/dL), of valid in range adult pt-mths, 2016
    'CWP5y4_f': 'PctgSerumPhosphorous70',        # % of adult pt-mths with serum phosphorous > 7.0 mg/dL, 2016
    'CWavgUnCay4_f': 'AvgCalciumUncorrected',    # Avg uncorrected calcium (mg/dL), of valid in range adult pt-mths, 2016
    'CWunCa3y4_f': 'PctgCalciumUncorrected102',  # % of adult pt-mths with uncorrected calcium > 10.2 mg/dL, 2016
    'ppavfy4_f': 'PctgFistula',              # % of patients receiving treatment w/ fistulae, 2016
    'ppcg90y4_f': 'PctgCatheterOnly90',      # % of patients with catheter only >= 90 days, 2016
}

dfr=pd.read_csv("../input/DFR-FY-2018.csv", parse_dates=True, dtype=dtype_dict, usecols=column_dict.keys())
dfr.rename(columns=column_dict, inplace=True)
print("\nThe DFR data frame has {0} rows or facilities and {1} variables or columns are selected out of over 3000.\n".format(dfr.shape[0], dfr.shape[1]))


# In[ ]:


dfr.head()   # take a look at the first five facilities


# DFR dataset contains facility name, city, and state. However, it does not contain the zip code of the facility's physical location. We need the zip code to get information about the socioeconomic characteristics of the community where the facility is located by using US Census Bureau American Community Survrey data and to determine whethere the facility is located in a rural or urban area by using Rual Urban Commuting Area (RUCA) data. These socioeconomic characteristics of a facility is important. The good new is, CMS also publishes two additional datasets related to dialysis facility quality measures and they both contain the zip code. One is dialysis facility compare (DFC) dataset. The other is ESRD Quality Incentive Program (QIP) performance score summary dataset. We will use CCN number to merge zip code from these two datasets into DFR dataset.
# 
# ##### First, let's import and merge the DFC dataset's zip code into the DFR dataframe

# In[ ]:


dtype_dict_dfc= {'Provider Number': str,       # Provider Number and Zip need to be treated as string  
                 'Zip': str}                   # or leading zeros will be dropped during the import.
column_dict_dfc={'Provider Number': 'CCN', 
                 'Zip': 'ZipCode'}             # We only need CCN and zip code column
dfc=pd.read_csv("../input/DFC-CY2018.csv", dtype=dtype_dict_dfc, usecols=column_dict_dfc.keys() )
dfc.rename(columns=column_dict_dfc, inplace=True)
print("\nThe DFC dataset has {0} rows or facilities and {1} variables or columns are selected.\n".format(dfc.shape[0], dfc.shape[1]))
dfc.info()


# In[ ]:


dfc.sample(5)    # display a random sample of five observations or facilities


# We need to find out how many facilities in DFC that are not in DFR

# In[ ]:


dfr_ccn = set(dfr['CCN'])
dfc_ccn = set(dfc["CCN"])
print("There are {0} facilities in DFC that are not in DFR".format(len(dfc_ccn - dfr_ccn)))


# Let's merge DFC into DFR so that DFR has the zip code column. This is left join since we don't want to add those DFC facilities not in DFR to the DFR dataframe.

# In[ ]:


dfr = pd.merge(dfr, dfc, on='CCN', how='left') 
dfr.shape


# Find out how many facilities are still without a zip code (There should be 57)

# In[ ]:


print("There are {0} facilities in the DFR dataframe without a zip code".format(dfr[dfr["ZipCode"].isnull()].shape[0]))


# Next, let's import and merge ESRD QIP dataset's zip code into the DFR dataframe

# In[ ]:


dtype_dict_qip= {'CMS Certification Number (CCN)': str,
                 'Zip Code': str}
column_dict_qip={'CMS Certification Number (CCN)': 'CCN', # We only need the CCN and Zip Code columns
                 'Zip Code': 'ZipCode'}
qip=pd.read_csv("../input/ESRD-QIP-PY2018.csv", dtype=dtype_dict_qip, usecols=column_dict_qip.keys())
qip.rename(columns=column_dict_qip, inplace=True)
print("\nThe QIP dataset has {0} rows or facilities and {1} variables or columns are selected.\n".format(qip.shape[0], qip.shape[1]))
qip.info()


# In[ ]:


qip.tail(5)  # take a look at the last five facilities


# We need to find out how many facilities in QIP that are not in DFR

# In[ ]:


dfr_ccn = set(dfr['CCN'])
qip_ccn = set(qip["CCN"])
print("There are {0} facilities in QIP dataframe that are not in DFR data frame".format(len(qip_ccn - dfr_ccn)))


# Let's merge QIP dataframe into the DFR dataframe so that DFR has the zip code column from the QIP dataframe. 
# This is left join since we don't want to add these 467 facilities from the QIP dataframe to the DFR dataframe.

# In[ ]:


dfr = pd.merge(dfr, qip, on="CCN", how="left")  # merge QIP dataset with DFR dataset to get zip code from QIP dataset
dfr.shape


# In[ ]:


dfr.columns    # display all columns in the DFR dataframe


# Notice, we have two zip code columns: ZipCode_x is from DFC, ZipCode_y is from QIP. 
# Let's find out how many facilities without a zip code in ZipCode_x column.

# In[ ]:


print("The new dataset has {0} facilities without zip code based on ZipCode_x column.".format(dfr[dfr['ZipCode_x'].isnull()].shape[0]))


# Let's fill the null value in ZipCode_x with zip code from ZipCode_y.

# In[ ]:


dfr['ZipCode_x'].fillna(dfr['ZipCode_y'], inplace=True)
print("The new dataset has {0} facilities without zip code".format(dfr[dfr['ZipCode_x'].isnull()].shape[0]))


# Only 6 facilities have no zip code. That is relatively small portion out of 6547 facilities.
# Let's clean this up by dropping the ZipCode_y column and rename the ZipCode_x column to ZipCode

# In[ ]:


dfr.drop("ZipCode_y", axis=1, inplace=True)
dfr.rename(columns={'ZipCode_x':'ZipCode'}, inplace=True)
dfr.shape


# In[ ]:


dfr.sample(5)  # display a random sample of five facilities


# Let's save the consolidated dataframe to a new file for further processing.

# In[ ]:


dfr.to_csv("InterimDataset.csv")


# End of Step Two - Activity One

# In[ ]:




