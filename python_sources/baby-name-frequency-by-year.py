#!/usr/bin/env python
# coding: utf-8

# This project is mostly for practicing SQL/BigQuery.
# 
# Insperation: How common would it be to name ones offspring 'Hunter'? Could a name such as 'Hunter' be considered unisex?

# In[ ]:



import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from google.cloud import bigquery
import bq_helper
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")

client = bigquery.Client()


# To begin, I'm copying the Data set to a csv file, which I will put to the side for later use.

# Below is the code to copy the entire database into a dataframe/CSV file. Since this notebook is for practicing SQL I wont be referencing this other than to double check my work.

# In[ ]:


query = """SELECT year, gender, name, sum(number) as number FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""
names = usa_names.query_to_pandas_safe(query)
names.to_csv("usa_names.csv")


# In[ ]:


names.sample(10)


# In[ ]:


dataset_ref = client.dataset("usa_names", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)


# In[ ]:


tables = list(client.list_tables(dataset))

for table in tables:  
    print(table.table_id)


# In[ ]:


client.list_rows(table, max_results=5).to_dataframe()


# In[ ]:


##Now for a more specific query to get the exact info I'm looking for

query = """SELECT year, gender, name, sum(number) as number FROM `bigquery-public-data.usa_names.usa_1910_current`  WHERE name = 'Hunter'  GROUP BY year, gender, name"""
hunter = usa_names.query_to_pandas_safe(query)
hunter.sample(10)


# For the cell below I have two commented out print statements, these were originally in so that I could figure out a mistake I made (I had failed to include part of matplotlib) and I chose to leave them in for later reference

# In[ ]:


##Separating the data set into a female and male set
hunter_f = hunter[hunter.gender.isin(['F'])]
hunter_m = hunter[hunter.gender.isin(['M'])]

#print(hunter_m.head())
#print(hunter.isnull)


# In[ ]:


plt.figure(figsize=(24,12))
plt.title('Frequency of the name Hunter over the years')


sns.lineplot(y=hunter_m['number'], x=hunter_m['year'], label="Male")
sns.lineplot(y=hunter_f['number'], x=hunter_f['year'], label='Female')


#plt.xlabel("Date")
#plt.ylabel("Value($)")
#sns.set_style("white")


# As you can see above, the name 'Hunter' wasn't common until around the 1980s, and was not present as a female name until a bit later.
# One could easily copy and substitute this code for other possible names.
# 
# Next, to answer the question "would this name be considered normal?", I will compare the frequency of this name to the mean name frequency. For the sake of succinctness I will be limiting the data to the last 20 years.

# In[ ]:


safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query = """
        SELECT year, gender, name, sum(number) as number
        FROM `bigquery-public-data.usa_names.usa_1910_current`
        WHERE year >= 1998 and year <=2019
        GROUP BY year, gender, name
        ORDER BY year

        """

names_volume = client.query(query, job_config=safe_config)
names_volume = names_volume.to_dataframe()
names_volume.sample(10)
##While the .head() function is typically used to check if a data set looks correctly configured i tend to use sample so that the data i'm looking at is always called different


# In[ ]:


mean_data =  names_volume.groupby('year')['number'].mean()
display(mean_data)
names_volume.describe()


# Looking at the average population with each name listed per year one could say that anyone with a name of 300 or more would be considered common. However, we have neither checked for, nor adjusted for outliers (which the describe function shows us are common). So lets do a simple kde plot to see (a bar or violin graph would also be a good choice)

# In[ ]:


plt.figure(figsize=(36,12))

sns.kdeplot(data=names_volume["number"], shade=True)


# In[ ]:


plt.figure(figsize=(36,12))

sns.violinplot(x=names_volume["number"], color='green')


# So between the kde plot and the violin plot we can see that it is fairly common to have a name that is not fairly common.
# 
# The last thing I would like to do is chart how the name 'Hunter' compares to the 25%, mean, and 75% mark for the data set on a simple line graph (should be done in the next couple of days)...
# 
# 
