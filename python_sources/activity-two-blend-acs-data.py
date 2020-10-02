#!/usr/bin/env python
# coding: utf-8

# ### Step Two - Blend Data
# #### Activity Two - Obtain facility neighbourhood demographic and socioeconomic data from the American Community Survey
# 
# We use a facility's zip code to get information about demographic and socioeconomic characteristics of the community where the facilitie is located.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 50)  


# Import the interim dataset which has the zip codes of facilities physical location

# In[ ]:


dtype_dict= {'CCN': str,    
             'Network': str, 
             'ZipCode': str}

dfr=pd.read_csv("../input/activity-one-blend-dfr-dfc-qip-data/InterimDataset.csv", parse_dates=True, dtype=dtype_dict)
print("\nThe DFR data frame has {0} rows or facilities and {1} variables or columns\n".format(dfr.shape[0], dfr.shape[1]))


# In[ ]:


dfr.head()


# In[ ]:


dfr.drop(columns='Unnamed: 0', axis=1, inplace=True)  # remove this unnamed column
dfr.head()


# First, we import the DP05 dataset which has demographic information about gender, age, race, and ethnicity. 
# * HC03_VC50 = Percent of RACE - One race - Black or African American
# * HC03_VC88	= Percent of HISPANIC OR LATINO AND RACE - Total population - Hispanic or Latino (of any race)

# In[ ]:


dtype_dict_acs= {'GEO.id2': str}
column_dict_acs={'GEO.id2': 'ZipCode',
                 'HC03_VC50': 'PctgBlackACS',
                 'HC03_VC88': 'PctgHispanicACS'}
acs=pd.read_csv("../input/rmudsc/ACS_16_5YR_DP05_with_ann.csv", skiprows=[1], dtype=dtype_dict_acs, usecols=column_dict_acs.keys() )
acs.rename(columns=column_dict_acs, inplace=True)
print("\nThe DP05 ACS dataset has {0} rows or zip codes and {1} variables or columns are selected.\n".format(acs.shape[0], acs.shape[1]))
acs.info()


# In[ ]:


acs.sample(5)


# Let's merge the demographic data into DFR dataframe.

# In[ ]:


dfr = pd.merge(dfr, acs, on='ZipCode', how='left') 
dfr.shape


# Second, we import the DP03 dataset which has socioeconomic data elements.

# In[ ]:


# First, we import the DFC dataset. We only need CCN and zip code column.

dtype_dict_acs= {'GEO.id2': str}
column_dict_acs={'GEO.id2': 'ZipCode',
                 'HC03_VC07': 'UnemploymentRate',
                 'HC03_VC161': 'PctgFamilyBelowFPL'}
acs=pd.read_csv("../input/rmudsc/ACS_16_5YR_DP03_with_ann.csv", skiprows=[1], dtype=dtype_dict_acs, usecols=column_dict_acs.keys() )
acs.rename(columns=column_dict_acs, inplace=True)
print("\nThe DP03 ACS dataset has {0} rows or zip codes and {1} variables or columns are selected.\n".format(acs.shape[0], acs.shape[1]))
acs.info()


# In[ ]:


acs.sample(5)


# Let's merge the socioeconomic data into DFR dataframe.

# In[ ]:


dfr = pd.merge(dfr, acs, on='ZipCode', how='left') 
dfr.shape


# Thirdly, we import the DP02 dataset. Data element HC03_VC173 is percentage of people aged 5 and over that have a primary language other than English and speak English less than well. This is a good measure of English proficiency of the community which may be indicator of difficulty in communication that can potentially lead to readmission. 
# 

# In[ ]:


dtype_dict_acs= {'GEO.id2': str}
column_dict_acs={'GEO.id2': 'ZipCode',
                 'HC03_VC173': 'PctgPoorEnglish'
                }
acs=pd.read_csv("../input/rmudsc/ACS_16_5YR_DP02_with_ann.csv", skiprows=[1], dtype=dtype_dict_acs, usecols=column_dict_acs.keys() )
acs.rename(columns=column_dict_acs, inplace=True)
print("\nThe DP02 ACS dataset has {0} rows or zip codes and {1} variables or columns are selected.\n".format(acs.shape[0], acs.shape[1]))
acs.info()


# In[ ]:


acs.sample(5)


# In[ ]:


dfr = pd.merge(dfr, acs, on='ZipCode', how='left') 
dfr.shape


# Let save the DFR dataframe to a interim file for further processing.

# In[ ]:


dfr.to_csv("InterimDataset2.csv")


# End of Step Two - Activity Two

# In[ ]:




