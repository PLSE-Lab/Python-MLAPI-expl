#!/usr/bin/env python
# coding: utf-8

# ## Analysis of Kiva Loans
# The [Kiva.org](https://www.kaggle.com/kiva) dataset is the first in [Kaggle's Data Science for Good Initiative](http://blog.kaggle.com/2017/11/16/introducing-data-science-for-good-events-on-kaggle/). Kiva has provided a dataset of loans issued over the last two years, and participants are invited to use this data as well as source external public datasets to help Kiva build models for assessing borrower welfare levels.
# 
# **Objective:** Pair Kiva's data with additional data sources to estimate the welfare level of borrowers in specific regions, based on shared economic and demographic characteristics.
# 
# **Datasets Used:** Besides the Kiva Loans dataset, we use:
# 1. Multi-dimenional Poverty Measures dataset from OPHI - https://www.kaggle.com/ophi/mpi
# 2. Additional Kiva Snapshot - https://www.kaggle.com/gaborfodor/additional-kiva-snapshot

# ## Summary of Findings
# 
# ### Data Quality
# * There is a lot of missing data, especially in the file on poverty indicator (MPI). In fact the non-null values for country and region in the Kiva MPI dataset are the same as in the OPHI MPI dataset. It seems that the Kiva MPI dataset was derived from OPHI.
# * Only 8% of the loan rows could be joined with the MPI data. To have more meaningful analysis, we would need to combine the Kiva Loans data with other socio-economic indicators.
# 
# 
# ### Global Trends
# * Higher the MPI of a region, lower is the number of loans given and the median loan amount
# 
# 

# In[1]:


# basic setup common to all analysis

import os
import numpy as np
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

get_ipython().run_line_magic('matplotlib', 'inline')
color = sns.color_palette()
py.init_notebook_mode(connected=True)

INPUT_DIR = '../input/' # location for Kaggle data files
print(os.listdir(INPUT_DIR))


# In[2]:


KIVA_DIR = INPUT_DIR + 'data-science-for-good-kiva-crowdfunding/'
MPI_DIR = INPUT_DIR + 'mpi/'

print(os.listdir(KIVA_DIR))


# ## What are the files in the Kiva dataset
# As we can see from the output of `os.listdir(KIVA_DIR)` there are four files in the Kiva dataset:
# * kiva_loans.csv
# * loan_themes_by_region.csv
# * kiva_mpi_region_locations.csv
# * loan_theme_ids.csv

# In[3]:


# read the data - may take some time
kiva_loans = pd.read_csv(KIVA_DIR + "kiva_loans.csv")
kiva_mpi_locations = pd.read_csv(KIVA_DIR + "kiva_mpi_region_locations.csv")
loan_theme_ids = pd.read_csv(KIVA_DIR + "loan_theme_ids.csv")
loan_themes_by_region = pd.read_csv(KIVA_DIR + "loan_themes_by_region.csv")

# find out the shape of the data
print("kiva_loans:",kiva_loans.shape)
print("kiva_mpi_locations:",kiva_mpi_locations.shape)
print("loan_theme_ids:",loan_theme_ids.shape)
print("loan_themes_by_region",loan_themes_by_region.shape)


# ## What does the dataset contain
# Based on the shapes above we can see that the main data is in the kiva_loans  and loan_themes files. Let's see what each of these files contains by drawing some random samples.
# 
# ### Loans

# In[4]:


kiva_loans.sample(10)


# ### MPI Locations

# In[5]:


kiva_mpi_locations.sample(10)


# MPI refers to [Multi-dimensional Poverty Index](https://en.wikipedia.org/wiki/Multidimensional_Poverty_Index) . It is calculated as:
# 
# *MPI = H X A*
# 
# Where:
# * H: Percentage of people who are MPI poor (incidence of poverty)
# * A: Average intensity of MPI poverty across the poor (%)
# 
# So the higher the MPI the higher the poverty in the region.

# ### Loan Themes

# In[6]:


loan_theme_ids.sample(10)


# ### Loan Themes by Region

# In[7]:


loan_themes_by_region.sample(10)


# ## What data is missing
# Real-world data is rarely complete. To find out missing missing data in each dataset, let's create a function which takes a dataset and finds out percentage of null data by column

# In[8]:


def missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print ("Missing data in Loans")
missing_data(kiva_loans)


# In[9]:


print ("Missing data in MPI Locations")
missing_data(kiva_mpi_locations)


# In[10]:


print ("Missing data in Loan Themes")
missing_data(loan_theme_ids)


# In[11]:


print ("Missing data in Loan Themes by Region")
missing_data(loan_themes_by_region)


# That's a lot of missing data, especially in MPI locations. Our goal is to estimate welfare levels of borrowers by region. So we need clean data for region and some indicator of poverty/welfare. So let's eliminate rows with missing data for country and region from Loans, MPI-locations and Loan-Themes by region.

# In[12]:


loans = kiva_loans.dropna(subset = ['country','region'])
mpi = kiva_mpi_locations.dropna(subset = ['country','region'])
loan_themes_region = loan_themes_by_region.dropna(subset = ['country','region'])

# see numnber of rows dropped:
print("kiva_loans:", kiva_loans.shape)
print("after removal:", loans.shape)

print("kiva_mpi_locations:", kiva_mpi_locations.shape)
print("after removal:", mpi.shape)

print("loan_themes_by_region",loan_themes_by_region.shape)
print("after removal:", loan_themes_region.shape)


# We are ok on the loans dataset (91% rows remain), but the MPI lcoations dataset has been reduced to almost 1/3rd. Let's see if the MPI data from OPHI does any better.

# ## Using Multi-Dimensional Poverty Measures from OPHI
# Let's see the files in the MPI dataset from OPHI:

# In[13]:


print(os.listdir(MPI_DIR))


# In[14]:


mpi_national = pd.read_csv(MPI_DIR + "MPI_national.csv")
mpi_subnational = pd.read_csv(MPI_DIR + "MPI_subnational.csv")

# find out the shape of the data
print("mpi national:",mpi_national.shape)
print("mpi subnational:",mpi_subnational.shape)


# *The number of rows in the OPHI MPI dataset is exactly equal to the non-null rows from the Kiva MPI dataset*
# 
# Let's see if there are any missing values in the OPHI dataset.

# In[15]:


missing_data(mpi_subnational)


# Let's merge the OPHI MPI dataset with that of Kiva

# In[16]:


# renaming some columns to make it consistent with the Kiva MPI dataset
mpi_subnational.rename(columns={'World region': 'world_region', 
                                'Sub-national region': 'region',
                                'Intensity of deprivation Regional': 'deprivation_intensity',
                                'Headcount Ratio Regional': 'headcount_ratio',
                                'Country': 'country'}, 
                       inplace=True)

mpi = pd.merge(mpi, mpi_subnational)
mpi.shape


# In[17]:


mpi_subnational.sample(10)


# ## Poverty by Region
# Let's build some intuition on what the  data tells us about poverty in the world

# In[18]:


plt.figure(figsize=(16,9))
sns.barplot(x=mpi.world_region.value_counts().values,
            y=mpi.world_region.value_counts().index)
plt.title("Poverty by World-Region")


# In[19]:


plt.figure(figsize=(9,16))
sns.barplot(x=mpi.country.value_counts().values,
            y=mpi.country.value_counts().index)
plt.title("Poverty by country")


# ## Purpose of Loans
# We examine why people take loans

# In[20]:


def plot_loan_purpose(df, title):
    plt.figure(figsize=(8, 8)) 
    sns.barplot(x=df.values[::-1],
                y=df.index[::-1])
    plt.title(title)

loan_sectors = loans['sector'].value_counts()[:20]
plot_loan_purpose(loan_sectors, 'Loan by Sector')


# In[21]:


loan_activity = loans['activity'].value_counts()[:20]
plot_loan_purpose(loan_activity, 'Loan by Activity')


# In[22]:


loan_use = loans['use'].value_counts()[:20]
plot_loan_purpose(loan_use, 'Loans by Use')


# ## Analysing Loans by Poverty Levels
# Let's see whether we get meaningful result once we combine the MPI data with Loans. We will do this using a [left-join](http://chris.friedline.net/2015-12-15-rutgers/lessons/python2/04-merging-data.html):

# In[23]:


loans_mpi = pd.merge(loans, mpi, how='left')
loans_mpi.count()


# Out of the 614k rows in Loans, only 51k rows (8.3%) have MPI information. We definitely need a different poverty indicator dataset. But for now we will continue with this dataset. First let's remove all the rows without MPI data and then we will start analyzing the data.

# In[24]:


df = loans_mpi.dropna(subset=['MPI'])
df.sample(10)


# ## Data Analysis
# 
# ### MPI vs. Number of Loans
# 

# In[25]:


def reg_plot(x, y, title):
    plt.figure(figsize=(16,9))
    sns.regplot(x, y, fit_reg=True)
    plt.title(title)
    plt.show()

dlc = df.groupby(['country','region','MPI'])['loan_amount'].count().reset_index(name='loan_count')
reg_plot(dlc.MPI, dlc['loan_count'], 'MPI vs. Loan Count')


#    There's an outlier here. Let's get rid of that

# In[26]:


dlc.loc[dlc['loan_count'] == dlc['loan_count'].max()]


# In[27]:


dlc = dlc.loc[dlc['loan_count'] < dlc['loan_count'].max()]
reg_plot(dlc.MPI, dlc['loan_count'], 'MPI vs. Loan Count')


# *Higher the MPI, lower is the number of loans given to the region.*
# 
# Let's see the trend for the mean loan amount.

# In[28]:


dlm = df.groupby(['country','region','MPI'])['loan_amount'].median().reset_index(name='median_loan_amount')
reg_plot(dlm.MPI, dlm['median_loan_amount'], 'MPI vs. Median Loan Amount')


# Again there's an outlier. Let's get rid of that and re-plot the graph

# In[29]:


dlm.loc[dlm['median_loan_amount'] == dlm['median_loan_amount'].max()]


# In[30]:


dlm = dlm.loc[dlm['median_loan_amount'] < dlm['median_loan_amount'].max()]
reg_plot(dlm.MPI, dlm['median_loan_amount'], 'MPI vs. Median Loan Amount')


# *Higher the MPI, lower is the median loan amount.*
# 
# **To be continued ....**
