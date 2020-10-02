#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# ## **The Idea**
# 
# * Analyise the data for 2009 for all the parameters which depends on grocery store decline
# * Select the target as advance or decline of grocery store from 2009 to 2014
# * Predict the data from 2014 to get the probability for next 5 years.

# ## **Data Selection**

# ## Access 
# 
# 1. Population, low access to store, 2010 - **LACCESS_POP10**
# 1. Low income & low access to store, 2010 -** LACCESS_LOWI10**
# 1. Households, no car & low access to store, 2010 - **LACCESS_HHNV10**

# In[ ]:


# Read relevent data from categry Access and Proximity to Grocery Store
Access = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='ACCESS')
Access= Access[['FIPS', 'State', 'County','LACCESS_POP10','LACCESS_LOWI10','LACCESS_HHNV10']]
Access.shape
Access.head()
Access.info()


# In[ ]:


#Create test data of 2014 for prediction
Access_test = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='ACCESS')
Access_test = Access_test[['FIPS', 'State', 'County','LACCESS_POP15','LACCESS_LOWI15','LACCESS_HHNV15']]
Access_test.head()


# ## Store
# 
# Create target first by finding difference of sum of all stores in 2009 and 2014
# 
# Selecting Features :
# 
# 1. Grocery stores, 2009 - **GROC09**
# 1. Grocery stores/1,000 pop, 2009 - **GROCPTH09**
# 1. Supercenters & club stores, 2009 - **SUPERC09**
# 1. Supercenters & club stores/1,000 pop, 2009 - **SUPERCPTH09**
# 1. Convenience stores, 2009 - **CONVS09**
# 1. Convenience stores/1,000 pop, 2009 - **CONVSPTH09**
# 1. Specialized food stores, 2009 - **SPECS09**
# 1. Specialized food stores/1,000 pop, 2009 - **SPECSPTH09**
# 1. SNAP-authorized stores, 2012 - **SNAPS12**
# 1. SNAP-authorized stores/1,000 pop, 2012 - **SNAPSPTH12**
# 

# In[ ]:


# Read relevent data from Store Sheet
Stores = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='STORES')


# In[ ]:


#find retail store decline by difference of 2009 to 2014

Stores['total_2009'] = Stores['GROC09'] + Stores['SUPERC09'] + Stores['CONVS09'] + Stores['SPECS09']
Stores['total_2014'] = Stores['GROC14'] + Stores['SUPERC14'] + Stores['CONVS14'] + Stores['SPECS14']
Stores['is_decline'] = Stores['total_2014']- Stores['total_2009']
Stores['is_store_decline'] = Stores['is_decline'].apply(lambda x : 1 if (x<0) else 0   ) 


# In[ ]:


Stores=Stores[['FIPS', 'State', 'County','GROC09','GROCPTH09','SUPERC09','SUPERCPTH09','CONVS09','CONVSPTH09','SPECS09','SPECSPTH09','SNAPS12','SNAPSPTH12','is_store_decline']]


# In[ ]:


Stores.head()


# In[ ]:


#Prepare store data for 2014
Stores_test = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='STORES')
Stores_test=Stores_test[['FIPS', 'State', 'County','GROC14','GROCPTH14','SUPERC14','SUPERCPTH14','CONVS14','CONVSPTH14','SPECS14','SPECSPTH14','SNAPS16','SNAPSPTH16']]
Stores_test.head()


# ## Health
# 
# 1. Adult diabetes rate, 2008 - **PCT_DIABETES_ADULTS08**
# 1. Adult obesity rate, 2008 - **PCT_OBESE_ADULTS08**
# 1. Recreation & fitness facilities, 2009 - **RECFAC09**

# In[ ]:


# Read relevent data from Health Sheet
health = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='HEALTH')
health=health[['FIPS', 'State', 'County','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS08','RECFAC09']]
health.head()


# In[ ]:


# Create test data for 2014
health_test = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='HEALTH')
health_test=health_test[['FIPS', 'State', 'County','PCT_DIABETES_ADULTS13','PCT_OBESE_ADULTS13','RECFAC14']]
health_test.head()


# ## Socioeconomic Characteristics
# 
# 1. Median household income, 2015 - **EDHHINC15**
# 1. Metro/nonmetro counties, 2010 - **METRO13**
# 1. Population-loss counties, 2010 - **POPLOSS10**
# 1. Poverty rate, 2015 - **POVRATE15**
# 1. Persistent-poverty counties, 2010 - **PERPOV10**

# In[ ]:


# Read relevent data from SOCIOECONOMIC Sheet
social = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='SOCIOECONOMIC')
social=social[['FIPS', 'State', 'County','MEDHHINC15','METRO13','POPLOSS10','POVRATE15','PERPOV10']]
social.head()


# In[ ]:


# Create Data for 2014
social_test = pd.read_excel('../input/food-environment-atlas-data/DataDownload.xls',sheet_name='SOCIOECONOMIC')
social_test=social_test[['FIPS', 'State', 'County','MEDHHINC15','METRO13','POPLOSS10','POVRATE15','PERPOV10']]
social_test.head()


# ## Bureau of Economic Analysis (BEA) data 
# 
# 1. Population  -  **POP_2009**
# 1. Per capita income - **Per_Cap_2009**
# 1. Personal income - **Personal_income_2009 **

# In[ ]:


BEA = pd.read_excel("../input/bureau-of-economic-analysis-data/Pop_2009.xlsx",sheet_name='2009')


# In[ ]:


BEA.head()


# In[ ]:


BEA.info()


# In[ ]:


#Read data for 2014
BEA_test = pd.read_excel('../input/us-bureau-of-economic-analysis-bea-2014-data/Pop_2014.xlsx')
BEA_test.head()


# In[ ]:


#Merging all the data

df=Stores
df=pd.merge(df, Access, on=['FIPS', 'State', 'County'])
df=pd.merge(df, health, on=['FIPS', 'State', 'County'])
df=pd.merge(df, social, on=['FIPS', 'State', 'County'])
df.head()


# In[ ]:


df1= pd.merge(df, BEA, on='FIPS',how='left')
df1.head()


# In[ ]:


df1 = df1.rename(columns={'State_x': 'State', 'County_x': 'County'})
df1.head()


# In[ ]:


df_test=Stores_test
df_test=pd.merge(df_test, Access_test, on=['FIPS', 'State', 'County'])
df_test=pd.merge(df_test, health_test, on=['FIPS', 'State', 'County'])
df_test=pd.merge(df_test, social_test, on=['FIPS', 'State', 'County'])
df_test.head()


# In[ ]:


df_test= pd.merge(df_test, BEA_test, on='FIPS',how='left')
df_test.head()


# In[ ]:


df_test = df_test.rename(columns={'State_x': 'State', 'County_x': 'County'})
df_test = df_test.drop(['County_y','State_y'],axis=1)
df_test.head()


# In[ ]:


#drop missing valued
#print('before:', df1.shape)
#df1=df1.dropna(how='any')
#print('after:', df1.shape)


# In[ ]:


df1.to_csv('counties.csv', index=False)


# In[ ]:


df_test.to_csv('test.csv',index=False)

