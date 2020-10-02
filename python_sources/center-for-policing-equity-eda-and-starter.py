#!/usr/bin/env python
# coding: utf-8

# # Data Science for Good: Center for Policing Equity
# How do you measure Justice?
# ![https://cdn-images-1.medium.com/max/670/0*V9p8id8Jrt79OvQe.jpg](https://cdn-images-1.medium.com/max/670/0*V9p8id8Jrt79OvQe.jpg)

# ## First Look at the Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn
import matplotlib as mpl
import matplotlib.pylab as plt
plt.style.use('ggplot')
import os
print('Data Files....', os.listdir("../input/cpe-data/"))
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Variable Descriptions (aka Data Dictionary)
# "We pull a variety of files manually from the ACS website. The variables available in each of the ACS files are referenced in the data dictionary file: *ACS_variable_descriptions.csv*."
# 
# Printing only the first 20. There are a lot of different slices to this data, doubtful we will use them all.

# In[ ]:


ACSdf = pd.read_csv('../input/cpe-data/ACS_variable_descriptions.csv')
print(ACSdf.shape)
for row in ACSdf.iterrows():
    print(row[1][0], row[1][1])
    if row[0] == 20:
        break


# ## Potential Intersting variables
# - **Age and Sex of Population** 
#     - How many people live in this area? What is the distribution of genders?
# - **Race Distribution of Population**
# - **Education Atainment**
# - **Owner Occupied Housing**
# - **Poverty Level**
# - **Total Housing Units**

# # Load data from one of the areas **Dept_11-00091**

# ## First Look: Education Atainment (over 25 years old)
# The first row is label column, it must be dropped and the columns converted into numberic. This is a littly hacky but works.

# In[ ]:


dept11_education = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_education-attainment-over-25/ACS_16_5YR_B15003_with_ann.csv')
pd.to_numeric(dept11_education['HD01_VD01'].drop(0)).plot(kind='hist', figsize=(15, 5), bins=40, title='Dept 11: Population Distribution per Census Tract')


# ## Lets Get the Percentages by Census Tract, this will normalize across tracts

# In[ ]:


dept11_education = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_education-attainment-over-25/ACS_16_5YR_B15003_with_ann.csv')
dept11_education_wth_labels = dept11_education.copy()
dept11_education = dept11_education_wth_labels.drop(0)
# Make Columns numeric
for col in dept11_education.columns:
    if col[:2] == 'HD':
        dept11_education[col] = pd.to_numeric(dept11_education[col])


# In[ ]:


dept11_education_wth_labels.head()


# In[ ]:


# Make Percentages
# TODO Combine into larger bins like Elementry school, Middle School... etc
dept11_education['PCT_No_schooling'] = dept11_education['HD01_VD02'] / dept11_education['HD01_VD01']
dept11_education['PCT_Nursery_school'] = dept11_education['HD01_VD03'] / dept11_education['HD01_VD01']
dept11_education['PCT_Kindergarten'] = dept11_education['HD01_VD04'] / dept11_education['HD01_VD01']
dept11_education['PCT_1st_grade'] = dept11_education['HD02_VD05'] / dept11_education['HD01_VD01']
dept11_education['PCT_2nd_grade'] = dept11_education['HD01_VD06'] / dept11_education['HD01_VD01']
dept11_education['PCT_3nd_grade'] = dept11_education['HD01_VD07'] / dept11_education['HD01_VD01']
dept11_education['PCT_4th_grade'] = dept11_education['HD01_VD08'] / dept11_education['HD01_VD01']
dept11_education['PCT_5th_grade'] = dept11_education['HD01_VD09'] / dept11_education['HD01_VD01']
dept11_education['PCT_6th_grade'] = dept11_education['HD01_VD10'] / dept11_education['HD01_VD01']
dept11_education['PCT_7th_grade'] = dept11_education['HD01_VD11'] / dept11_education['HD01_VD01']
dept11_education['PCT_8th_grade'] = dept11_education['HD01_VD12'] / dept11_education['HD01_VD01']
dept11_education['PCT_9th_grade'] = dept11_education['HD01_VD13'] / dept11_education['HD01_VD01']
dept11_education['PCT_10th_grade'] = dept11_education['HD01_VD14'] / dept11_education['HD01_VD01']
dept11_education['PCT_11th_grade'] = dept11_education['HD01_VD15'] / dept11_education['HD01_VD01']
dept11_education['PCT_12th_grade'] = dept11_education['HD01_VD16'] / dept11_education['HD01_VD01']
dept11_education['PCT_high_school_diploma'] = dept11_education['HD01_VD17'] / dept11_education['HD01_VD01']
dept11_education['PCT_GED or alternative credential'] = dept11_education['HD01_VD18'] / dept11_education['HD01_VD01']
dept11_education['PCT_college_less than 1 year'] = dept11_education['HD01_VD19'] / dept11_education['HD01_VD01']
dept11_education['PCT_college 1 or more years'] = dept11_education['HD01_VD20'] / dept11_education['HD01_VD01']
dept11_education['PCT_Associates degree'] = dept11_education['HD01_VD21'] / dept11_education['HD01_VD01']
dept11_education['PCT_Bachelors degree'] = dept11_education['HD01_VD22'] / dept11_education['HD01_VD01']
dept11_education['PCT_Masters degree'] = dept11_education['HD01_VD23'] / dept11_education['HD01_VD01']
dept11_education['PCT_Professional school degree'] = dept11_education['HD01_VD24'] / dept11_education['HD01_VD01']
dept11_education['PCT_Doctorate degree'] = dept11_education['HD01_VD25'] / dept11_education['HD01_VD01']


# In[ ]:


# Plot Distributions.
# TODO think of a more interesting way to display this data. group into levels. Cumulative?
for col in dept11_education.columns:
    color_loc = 0
    if col[:3] == 'PCT':
        dept11_education[col].plot(kind='hist',
                                   figsize=(15, 2),
                                   bins=int(round(dept11_education[col].max()*100)),
                                   title=col,
                                   xlim=(0,1))
        plt.show()


# ## Plot Distribution in stacked bar charts

# In[ ]:


dept11_education['PCT_Some_Elementary'] = dept11_education['PCT_Kindergarten'] +                                           dept11_education['PCT_1st_grade'] +                                           dept11_education['PCT_2nd_grade'] +                                           dept11_education['PCT_3nd_grade'] +                                           dept11_education['PCT_4th_grade'] +                                           dept11_education['PCT_5th_grade']

dept11_education['PCT_Some_Middle_School'] = dept11_education['PCT_7th_grade'] +                                           dept11_education['PCT_8th_grade'] +                                           dept11_education['PCT_9th_grade'] +                                           dept11_education['PCT_10th_grade'] +                                           dept11_education['PCT_11th_grade']
dept11_education['PCT_High_School'] = dept11_education['PCT_12th_grade']
dept11_education['PCT_Some_College'] = dept11_education['PCT_college_less than 1 year'] + dept11_education['PCT_college 1 or more years']


# In[ ]:


dept11_education


# In[ ]:


dept11_education.set_index('GEO.display-label')[['PCT_No_schooling','PCT_Some_Elementary',
                  'PCT_Some_Middle_School',
                  'PCT_High_School',
                  'PCT_Some_College',
                  'PCT_Associates degree',
                  'PCT_Bachelors degree',
                  'PCT_Masters degree',
                  'PCT_Professional school degree',
                  'PCT_Doctorate degree']].plot(kind='barh', stacked=True, figsize=(15,45))


# ## Look at the next dataset **Owner Occupied Housing**

# In[ ]:


dept11_housing_metadata = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_owner-occupied-housing/ACS_16_5YR_S2502_metadata.csv')
dept11_housing = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_owner-occupied-housing/ACS_16_5YR_S2502_with_ann.csv')


# ## Lots of fields..... overlap with other datasets

# In[ ]:


dept11_housing_metadata


# ## Plot distribution of Housing Units by Tract

# In[ ]:


dept11_housing['HC01_EST_VC01'][0]


# In[ ]:


pd.to_numeric(dept11_housing['HC01_EST_VC01'].drop(0)).plot(kind='hist',
                                                            bins=25,
                                                            title='Occupied housing units; Estimate; Occupied housing units',
                                                            figsize=(15 ,5),
                                                            color='g')


# 

# In[ ]:


dep11_poverty_metadata = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_poverty/ACS_16_5YR_S1701_metadata.csv')
dep11_poverty = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_poverty/ACS_16_5YR_S1701_with_ann.csv')


# In[ ]:


dep11_poverty_metadata.head()


# In[ ]:


dep11_poverty_metadata.head()


# In[ ]:


pd.to_numeric(dep11_poverty['HC01_EST_VC01'].drop(0)).plot(kind='hist',
                                                            bins=25,
                                                            title=dep11_poverty['HC01_EST_VC01'][0],
                                                            figsize=(15 ,5),
                                                            color='y')


# ## Percentage in poverty

# In[ ]:


pd.to_numeric(dep11_poverty['HC03_EST_VC01'].replace('-',0).drop(0)).plot(kind='hist',
                                                            bins=25,
                                                            title=dep11_poverty['HC03_EST_VC01'][0],
                                                            figsize=(15 ,5),
                                                            color='m')


# ## Whoa, what's that one region with really high poverty rate?

# In[ ]:


pd.to_numeric(dep11_poverty['HC03_EST_VC01'].replace('-',0).drop(0)).max()


# In[ ]:


dep11_poverty.loc[dep11_poverty['HC03_EST_VC01'] == '58.3']['GEO.display-label'].values[0]


# # Worcester County, Massachusetts
# ![https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Map_of_Massachusetts_highlighting_Worcester_County.svg/200px-Map_of_Massachusetts_highlighting_Worcester_County.svg.png](https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Map_of_Massachusetts_highlighting_Worcester_County.svg/200px-Map_of_Massachusetts_highlighting_Worcester_County.svg.png)
# 
# From Wikipedia:
# ```The median income for a household in the county was $64,152 and the median income for a family was $79,121. Males had a median income of $56,880 versus $42,223 for females. The per capita income for the county was $30,557. About 6.9% of families and 9.5% of the population were below the poverty line, including 12.1% of those under age 18 and 9.0% of those age 65 or over.[15]
# ```
# 
# Something is wrong because they can't possibly be at 58.3 Percent
# 
# ## TODO Figure this out

# In[ ]:




