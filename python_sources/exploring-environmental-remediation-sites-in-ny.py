#!/usr/bin/env python
# coding: utf-8

# # Exploring Environmental Remediation Sites in New York
# 
# In this notebook, I'll use the environment remediation sites data open sourced for New York State. The aim is to explore the various sites and the contaminants that are causing great harm.

# # Import libraries
# 
# `Pandas` is a great library to work with data and has many built in methods to import and explore the dataset in the notebook. I'll also import `matplotlib` and `geopandas` to work with location based data.

# In[ ]:


import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# Let's first explore all the files that are available.

# In[ ]:


get_ipython().system('ls ../input/')


# There are 4 files in total. 
# * The file `NYSDEC_EnvironmentalRemediationSites_DataDictionary.pdf` contains description of each column.
# * The file `NYSDEC_EnvironmentalRemediationSites_Overview.pdf` provides an overview of the about background.
# * The file `environmental-remediation-sites.csv` contains the data about all the sites.
# * The file `socrata_metadata.json` includes the metadata information.

# # Dataset

# In[ ]:


dataset = pd.read_csv('../input/environmental-remediation-sites.csv')
dataset.info()


# The following information can be drawn:
# * There are 70,324 sites
# * Each site information has 42 different data columns
# * Some of the columns have null values and information is missing such as `Address 2` and `Waste Name`
# * The entries in each column are of type integer, floats and objects

# Let's first explore how the information is distributed in the available dataset.

# In[ ]:


dataset.columns


# Each record includes information about the site, the program, wastes disposed and more. Each record also includes a lot of information in the form of addresses, ZIP codes and more.

# # Program Types
# 
# There are a total of 5 different program types in the dataset:
# * HW - State Superfund Program
# * BCP - Brownfield Cleanup Program
# * VCP - Voluntary Cleanup Program
# * ERP - Environmental Restoration Program
# * RCRA - Hazardous Waste Management Program

# In[ ]:


program_types = dataset.groupby('Program Type')['Program Number'].count()
plt.figure(figsize = (12, 8))
plt.title("Various Program Types in New York")
plt.ylabel("Count")
plt.xlabel("Program Type")
sns.barplot(x = program_types.index, y = program_types.values)


# The maximum number of sites are **State Superfund Program** consisting of more than 50,000 sites. The least common type are **Hazardous Waste Management Program**.

# # Site Class
# 
# The class/status of each site is identified using an alpha-numeric code as described below:
# * 02 - The disposal of hazardous waste represents a significant threat to the environment or to health
# * 03 - Contamination does not presently constitute a significant threat the environment or to health
# * 04 - The site has been properly closed but that requires continued site management
# * 05 - No further action required 
# * **A - Active
# * C - Completed
# * P - Sites where preliminary information indicates that a site may have contamination
# * PR - Sites that are, or have been, subject to the requirements of the RCRA
# * N - No further action

# In[ ]:


site_classes = dataset.groupby('Site Class')['Program Number'].count()
plt.figure(figsize = (12, 8))
plt.title("Status/Class of each remediation site in New York")
plt.ylabel("Count")
plt.xlabel("Site Class/Status")
sns.barplot(x = site_classes.index, y = site_classes.values)


# From the plot/data, it is difficult to comprehend how the class and status of a site may correlate with one another. However, it does appear that **many sites have been Completed**.

# # Project Completion Date
# 
# Each site is associated with a date that it is projected to be closed or has been closed. I'll explore the sites that have been closed and analyse how they were disributed across a timeline.**

# In[ ]:


completed_sites = dataset[dataset['Site Class'] == 'C ']
completed_sites['Project Completion Date'] = pd.to_datetime(completed_sites['Project Completion Date']).dt.strftime("%Y")
completed_sites = completed_sites.groupby('Project Completion Date')['Program Number'].count()
plt.figure(figsize = (12, 8))
plt.title("Completion dates for various remediation sites in New York")
plt.ylabel("Count")
plt.xlabel("Date")
plt.xticks(rotation = 90)
sns.lineplot(x = completed_sites.index, y = completed_sites.values)


# As we can see from the line plot above, very few sites closed from 1985 - 2005, however, many the number grew significantly after that. The maximum sites closed in the year 2015.
# 
# There could have been less closes as there were less number of sites back in the period of 1985 - 2005 which grew as the waste production increased with human population outburst.

# # Contaminants
# 
# Rather than exploring the waste names, it's better that we know which site has what contaminant. This may allow for similar solutions to be applied to sites that have same contaminants.

# In[ ]:


len(dataset['Contaminants'].dropna().unique())


# There are over 237 different contaminants that have to be dealt with.

# In[ ]:


contaminants = dataset.groupby('Contaminants')['Program Number'].count().sort_values(ascending = False)
contaminants.head(10)


# **Lead** is the most common contaminant across all sites with a whopping 2500+ sites with it.

# In[ ]:


contaminants.tail(10)


# The least common contaminants are **pickle liquor**, **mineral/white spirits** and **calcium carbonate**.

# # Control Type 
# 
# Defines the type of control - Institutional or Engineering.

# In[ ]:


control_types = dataset.groupby('Control Type')['Program Number'].count()
labels = [index + ": {:.2f}%".format(count/dataset['Control Type'].count()*100) for index, count in zip(control_types.index, control_types.values)]
plt.figure(figsize = (12, 8))
plt.title("Control types for various remediation sites in New York")
plt.pie(x = control_types.values, labels = labels)


# **Deed Restriction** is the most common type of control type with appoximately 45% of the total dataset.

# # Conclusion
# 
# From the analysis, we were able to draw a number of conclusions:
# 1. The most common Program Type is State Superfund Program
# 2. Maximum number of project sites closed in the year 2015
# 3. The most common contaminant is Lead and the least common are pickle liquor, mineral/white spirits and calcium carbonate
# 4. The most common control type os deed restriction
# 
# As a next step, we can use the location information about the sites and plot their locations on a map.
