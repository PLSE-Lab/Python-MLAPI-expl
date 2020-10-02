#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt
plt.style.use('ggplot')
import os
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## History of Worcester MA
# 
# **1990**
# * Population: 709,705
# 
# **2000**
# * Population: 750,963
# * 282,927 households
#     * 33.60% had children under 18 living with them
#     * 52.50% were married couples living together
#     * 11.40% had a female householder with no husband present
#     * 32.20% were non-families
#     * 26.20% were individuals out of which 10.40% had someone living alone who were 65 or over
# * 192,502 families
# * 298,159 housing units, with an average density of 197 per sq mile
# * 89.61% white
# * 2.73% Black
# * 0.25% Native American
# * 2.62% Asian
# * 0.04% pacific islander
# * 2.93% Other races
# * 25.60% under the age of 18
# * 8.40% from 18-24
# * 31.10% from 24-44
# * 21.80% 45-64
# * 13% 65+
# * Median Age: 36
# * For every 100 females there were 95.50 males.
# * For every 100 females 18+ there were 92.10 males
# * Median Income for a household: 47,874
# * Median Income for a family: 58, 394
# * Male Median Income: 42,261
# * Female Median Income: 30,516
# * 9.20% of population were below the poverty line
#     * 6.80% of families
#     * 11.30% of those under the age of 18
#     * 9.50% 65+
#     
# **2010**
# * Population: 798,552
# * 303,080 Households
#     * 33.7% had children under the age of 18 living with them
#     * 50.0% were married couples
#     * 12.2% Female householder with no husband
#     * 33.2% were non-families
#     * 26.2% of all households were individuals
# * 202,602 families
# * 326,788 Housing units at an average density of 216.3 per sq mile.
# * 85.6% White
# * 4.2% Black
# * 4.0% Asian
# * 0.2% Native American
# * 3.6% Other Races
# * Median Age: 39.2
# * Median Income for Household: 64,152
# * Median Income for Family: 79,121
# * Male Median Income: 56,880
# * Female Median Income: 42,223
# * 9.5% of population were below the poverty line
#     * 6.9% families
#     * 12.1% under 18
#     * 9% 65+
#     
#    

# In[ ]:


dept_11_pov = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_poverty/ACS_16_5YR_S1701_with_ann.csv')
dept_11_edu_25 = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_education-attainment-over-25/ACS_16_5YR_B15003_with_ann.csv')
dept_11_edu = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_education-attainment/ACS_16_5YR_S1501_with_ann.csv')
dept_11_housing = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_owner-occupied-housing/ACS_16_5YR_S2502_with_ann.csv')
dept_11_race = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv')


# In[ ]:


total_population = pd.to_numeric(dept_11_race['HC01_VC03'][1:]).sum()
total_male = pd.to_numeric(dept_11_race['HC01_VC04'][1:])
total_female = pd.to_numeric(dept_11_race['HC01_VC05'][1:])

male_percent = (total_male.sum() / total_population) * 100
female_percent = (total_female.sum() / total_population) * 100


# The approximate population of Worcester county is ~ **758,919** people.
# 
# **51.67%** Female
# 
# **48.33%** Male

# In[ ]:


housing_units = pd.to_numeric(dept_11_housing['HC01_EST_VC01'][1:]).sum()


# There is a total of ~ **302,794** housing units

# In[ ]:


below_poverty_line = pd.to_numeric(dept_11_pov['HC02_EST_VC01'][1:]).sum()
percent_below_poverty_line = (below_poverty_line / total_population) * 100


# **11.84%** are below the poverty line in Worcester.

# In[ ]:


race_df = pd.DataFrame({
    'white': pd.to_numeric(dept_11_race['HC01_VC49'][1:]),
    'af_am':pd.to_numeric(dept_11_race['HC01_VC50'][1:]),
    'asian':pd.to_numeric(dept_11_race['HC01_VC56'][1:]),
    'native':pd.to_numeric(dept_11_race['HC01_VC51'][1:]),
    'hispanic': pd.to_numeric(dept_11_race['HC01_VC88'][1:]),
    'other': pd.to_numeric(dept_11_race['HC01_VC69'][1:])
}, columns=['white','af_am','asian','native', 'hispanic','other'])
fig, ax = plt.subplots(figsize=(15,7))
race_df.plot.box(ax=ax)


# In[ ]:


age_hist = pd.DataFrame({
    '< 5': pd.to_numeric(dept_11_pov['HC01_EST_VC04'][1:]),
    '5-17': pd.to_numeric(dept_11_pov['HC01_EST_VC05'][1:]),
    '18-34': pd.to_numeric(dept_11_pov['HC01_EST_VC08'][1:]),
    '35-64':pd.to_numeric(dept_11_pov['HC01_EST_VC09'][1:]),
    '65+':pd.to_numeric(dept_11_pov['HC01_EST_VC11'][1:])
}, columns=['< 5','5-17','18-34','35-64','65+'])


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
age_hist.plot.box(ax=ax)


# In[ ]:


sex_df = pd.DataFrame({
    'total': pd.to_numeric(dept_11_race['HC01_VC03'][1:]),
    'male': pd.to_numeric(dept_11_race['HC01_VC04'][1:]),
    'female': pd.to_numeric(dept_11_race['HC01_VC05'][1:])
}, columns=['total','male','female'])


# In[ ]:


locations_df = pd.DataFrame({
    'geo_id_1':dept_11_race['GEO.id'],
    'geo_id_2':dept_11_race['GEO.id2'],
    'county': dept_11_race['GEO.display-label']
    
}, columns=['geo_id_1', 'geo_id_2','county'])


# In[ ]:


age_race = pd.concat([locations_df,sex_df,age_hist, race_df ], sort=True, join='inner', axis=1)
age_race.head() # A Cleaner dataframe to work with


# In[ ]:




