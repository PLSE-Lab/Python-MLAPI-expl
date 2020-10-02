#!/usr/bin/env python
# coding: utf-8

# # Visualize BigQuery public data with Pandas and Matplotlib.

# In[2]:


# Import data visualization libraries.
import pandas
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# We use this helper function to show how much quota a query consumed.
def print_usage(job):
    print('Processed {} bytes, {} MB billed (cache_hit={})'.format(
        job.total_bytes_processed, job.total_bytes_billed / (1024 * 1024), job.cache_hit))

# Connect to BigQuery.
from google.cloud import bigquery
client = bigquery.Client()


# ## USA Names
# 
# Let's take a look at the USA Names dataset to visualize some time series data. How has the uniqueness of names changed over time?

# In[4]:


sql = """
SELECT year, gender, COUNT(DISTINCT name) / SUM(number) as names_per_person
FROM `bigquery-public-data.usa_names.usa_1910_current`
GROUP BY gender, year
"""
job = client.query(sql)  # Run the query.
df = job.to_dataframe()  # Wait for the query to finish, and create a Pandas DataFrame.
print_usage(job)  # How much quota did this query consume?


# A good first step at visualizing data for exploration is to look at it in a table. Since there could be many rows, use the `head()` method to get a small preview of the data.

# In[5]:


df.head()


# Sometimes it's more natural to process data in Python than it is in SQL. Keep in mind that you have limited CPU and memory available to the Python kernel, so this only works for small-ish (on the order of gigabytes) DataFrames.

# In[6]:


pivot = pandas.pivot_table(
    df, values='names_per_person', index='year', columns='gender')
pivot.head()


# The `plot()` method displays a line graph by default. This is usually quite useful for visualizing time series.

# In[7]:


# Plot name "uniqueness" (number of distinct names per person) over time.
pivot.plot(fontsize=20, figsize=(15,7))


# ### Visualizations for communication
# 
# The Pandas plot method is handy for data exploration, but it doesn't offer as much control over the display as the underlying matplotlib library. When creating visualizations for communication, use the `matplotlib.pyplot` module for fine-grained control over label placement and other visual properties.

# In[26]:


# Set the font size
# See: https://stackoverflow.com/a/3900167/101923
font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 20}
plt.rc('font', **font)

# Create the plot, and add labels.
plt.figure(figsize=(15, 7))
plt.plot(pivot.index, pivot['F'], label='Female name uniqueness')
plt.plot(pivot.index, pivot['M'], label='Male name uniqueness')
plt.ylabel('Names per Person')
plt.xlabel('Year')
plt.title('US Names Uniqueness')
plt.legend()


# ## IRS 990 Data
# 
# Distributions are especially important to visualize before you create a model. For example, you don't want to fit a Gaussian distribution to data that are very skewed.

# In[35]:


sql = """
SELECT grsincfndrsng AS fundraising
FROM `bigquery-public-data.irs_990.irs_990_2016`
WHERE grsincfndrsng > 0
"""
job = client.query(sql)
df = job.to_dataframe()
print_usage(job)


# In[36]:


df.plot.hist(bins=50, fontsize=20, figsize=(15,7))


# It's clear from the previous visualization that this data is highly skewed. We can use BigQuery to transform the data when the table is too big to process in-memory.

# In[30]:


sql = """
SELECT LOG10(grsincfndrsng) AS log_fundraising
FROM `bigquery-public-data.irs_990.irs_990_2016`
WHERE grsincfndrsng > 0
"""
job = client.query(sql)
df = job.to_dataframe()
print_usage(job)


# In[31]:


df.plot.hist(bins=50, fontsize=20, figsize=(15,7))


# For multi-dimensional data, as scatter plot can provide additional insights about the distribution and relations between the dimensions.

# In[32]:


sql = """
SELECT
  LOG10(payrolltx) AS log_payrolltx,
  LOG10(noemplyeesw3cnt) AS log_employees,
  LOG10(officexpns) AS log_officeexpns,
  LOG10(legalfees) AS log_legalfees
FROM `bigquery-public-data.irs_990.irs_990_2016`
WHERE noemplyeesw3cnt > 0
  AND payrolltx > 0
  AND officexpns > 0
  AND legalfees > 0
"""
job = client.query(sql)
df = job.to_dataframe()
print_usage(job)


# In[33]:


df.plot.scatter(
    x='log_employees',
    y='log_payrolltx',
    fontsize=20,
    figsize=(15,7))


# To compare among all combinations of many dimensions, use a scatter matrix.

# In[34]:


import pandas.plotting
pandas.plotting.scatter_matrix(df, alpha=0.2, figsize=(15, 15))


# In[ ]:




