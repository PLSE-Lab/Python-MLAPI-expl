#!/usr/bin/env python
# coding: utf-8

# ## Overview
# The Center for Policing Equity (CPE) is research scientists, race and equity experts, data virtuosos, and community trainers working together to build more fair and just systems. Data and science are our tools; law enforcement and communities are our partners. Our aim is to bridge the divide created by communication problems, suffering and generational mistrust, and forge a path towards public safety, community trust, and racial equity.
# 
# Police departments across the United States have joined our National Justice Database, the first and largest collection of standardized police behavioral data. In exchange for unprecedented access to their records (such as use of force incidents, vehicle stops, pedestrian stops, calls for service, and crime data), our scientists use advanced analytics to diagnose disparities in policing, shed light on police behavior, and provide actionable recommendations. Our highly-detailed custom reports help police departments improve public safety, restore trust, and do their work in a way that aligns with their own values.

# ## Data Science for Good : Problem Statement
# 
# How do you measure justice? And how do you solve the problem of racism in policing? We look for factors that drive racial disparities in policing by analyzing census and police department deployment data. The ultimate goal is to inform police agencies where they can make improvements by identifying deployment areas where racial disparities exist and are not explainable by crime rates and poverty levels. The biggest challenge is automating the combination of police data, census-level data, and other socioeconomic factors.

# #### We are trying to find the insights from given data. We are first playing with some EDA part of data

# *  Loading important library for projects

# In[ ]:


import numpy as np 
import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os 


# In[ ]:


Departments = [f for f in os.listdir("../input/cpe-data/") if f.startswith("Dept")]
print(Departments)


# In[ ]:


Departments


# ### Reading the Data from Files

# ## Now let us load the File For Dept_11-00091
# #### Loading the Poverty File

# In[ ]:


Path="../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_poverty/"


# In[ ]:


ACS_Poverty= pd.read_csv(Path + "ACS_16_5YR_S1701_with_ann.csv")


# In[ ]:


ACS_Poverty.head()


# In[ ]:





# In[ ]:





# In[ ]:




