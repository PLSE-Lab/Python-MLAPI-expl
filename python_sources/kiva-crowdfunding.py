#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# Kiva.org is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. Kiva lenders have provided over $1 billion dollars in loans to over 2 million people.
# Our objective is to pair Kiva's data with additional data sources to estimate the welfare level of borrowers in specific regions, based on shared economic and demographic characteristics. 

# # Data analysis

# ### Import Libraries

# In[1]:


# Data processing
import numpy as np 
import pandas as pd

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Initial setup
get_ipython().run_line_magic('matplotlib', 'inline')
color = sns.color_palette()
sns.set_style('dark')


# ### Read data

# In[3]:


# kiva crowd funding data
kiva_loans = pd.read_csv("../input/kiva_loans.csv")
kiva_mpi_region_locations = pd.read_csv("../input/kiva_mpi_region_locations.csv")
loan_theme_ids = pd.read_csv("../input/loan_theme_ids.csv")
loan_themes_by_region = pd.read_csv("../input/loan_themes_by_region.csv")


# ### Loan Amount distribution

# In[4]:


sns.set(rc={"figure.figsize":(10, 5)})
sns.distplot(kiva_loans[kiva_loans['loan_amount'] < 5000].loan_amount, bins=[x for x in range(0, 5100, 200)], kde=False, color='c', label='loan_frequency')
plt.legend()
plt.show()


# ### Loan distribution by country

# In[5]:


sns.set(rc={"figure.figsize":(15, 8)})
sns.countplot(y="country", data=kiva_loans, order=kiva_loans.country.value_counts().iloc[:20].index)
plt.title("Distribution of kiva loans by country")
plt.ylabel('')
plt.show()


# ### Loan distribution by sector

# In[6]:


sns.set(rc={"figure.figsize": (15, 8)})
sns.countplot(y="sector", data=kiva_loans, order=kiva_loans.sector.value_counts().iloc[:20].index)
plt.title("Distribution of loans by Sector")
plt.ylabel("")
plt.show()


# ### Loan amount distribution by country

# In[ ]:


sns.set(rc={"figure.figsize": (15, 10)})
sns.boxplot(x='loan_amount', y='country', data=kiva_loans, order=kiva_loans.country.value_counts().iloc[:10].index)
plt.title("Distribution of loan amount by country")
plt.ylabel("")
plt.show()


# ### Loan amount distribution by sector

# In[ ]:


sns.set(rc={"figure.figsize": (15, 10)})
sns.boxplot(x='loan_amount', y='sector', data=kiva_loans, order=kiva_loans.sector.value_counts().iloc[:10].index)
plt.title("Distribution of loan amount by sector")
plt.ylabel("")
plt.show()


# In[ ]:




