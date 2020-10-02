#!/usr/bin/env python
# coding: utf-8

# **Objective** <br /> Classify the breast cancer patients who could survive more than 5 years or less after undergoing surgery

# In[ ]:


#import useful packages for EDA (Exploratory Data Analysis)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Reading Haberman's Survival data set file (.csv)
#read_csv() is used to read csv file
#using proper path as defined i.e '../input/haberman.csv'
path='../input/haberman.csv'
sample_data=pd.read_csv(path)


# * *As we know  not a proper naming has given to columns , reading data from a particular column will be  difficult*

# In[ ]:


sample_data.columns=['AGE','SURGERY_YEAR','NODES_DETECTED','STATUS']
print(sample_data.head())
print(sample_data.tail())


# **Observation** <br />
#  *Describing 'STATUS' column * <br />
#              1 = the patient survived 5 years or longer  <br />
#              2 = the patient died within 5 year

# In[ ]:


print(sample_data.shape[0])
print(sample_data.shape[1])
#value_counts() is used to count distinct values
print(sample_data['STATUS'].value_counts())


# **Observation** <br />
# * This data set contains 305 data points and 4 features
# * Out of 305 breast cancer patients , 224 patients has survived more than 5 years and 81 less than this
# * This indicates 73.44 % of patients had a comparatively successful surgery

# In[ ]:


sns.set_style('whitegrid')
sns.FacetGrid(sample_data,hue='STATUS',size=5).map(sns.distplot,'AGE').add_legend()
plt.show()


# In[ ]:


sns.set_style('whitegrid')
sns.FacetGrid(sample_data,hue='STATUS',size=5).map(sns.distplot,'SURGERY_YEAR').add_legend()
plt.show()


# In[ ]:


sns.set_style('whitegrid')
sns.FacetGrid(sample_data,hue='STATUS',size=5).map(sns.distplot,'NODES_DETECTED').add_legend()
plt.show()

