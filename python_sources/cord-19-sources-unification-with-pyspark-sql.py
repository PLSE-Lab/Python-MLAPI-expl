#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">COVID-19</font></center></h1>
# 
# <h2><center><font size="5">CORD-19 Sources unification with pyspark SQL</font></center></h2>
# 
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>1. Introduction</a> 
# - <a href='#2'>2. Metadata</a>
# - <a href='#3'>3. Data preprocessing and unification</a>
# - <a href='#4'>4. Conclusion</a>
#     
#     
# # <a id='1'>1. Introduction</a>
# 
# The objective of this kernel is simple: to produce a single, complete and clean covid.csv files containing all papers from the four different sources. The output size will be about 370 MB.
# 
# If you, just like me, love digging into the data, you do not need to go through this kernel. Just start a new kernels and import the cleaned file:
# 
#     pd.read_csv("/kaggle/input/single-complete-flattened-text-based-dataset/clean_covid.csv")
#     
# I will use pyspark SQL to first load all JSON data into a single spark Dataframe. Subsequently, by using query selectors, I will filter out only the column with only relevant text data. This will be the selected column (new columns may be added when time goes by):
# 
# Disclaimer: this is a working in progress. Please, share your valuable opinions in the comments box!

# # <a id='2'>2. Metadata</a>
# 
# Here, we load the data and transform three columns to permits to easily work with columns in future.
# 
# - `source_x` becomes `source`
# - `Microsoft Academic Paper ID` becomes `mic_id`
# - `WHO #Covidence` becomes `who_covidence`
# 
# Let's visualize it:

# In[ ]:


get_ipython().system('pip install pyspark')

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

import glob
import json

from pathlib import Path

root_path = Path('/kaggle/input/CORD-19-research-challenge/2020-03-13')
metadata_path = root_path / Path('all_sources_metadata_2020-03-13.csv')

metadata = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})

metadata.rename(columns={'source_x': 'source', 'Microsoft Academic Paper ID': 'mic_id', 'WHO #Covidence': 'who_covidence'}, inplace=True)

print("There are ", len(metadata), " sources in the metadata file.")

metadata.head(2)


# The dataset `metadata` contains 14 columns. Just by reading the different names and looking at the values we can already have a good understanding of the underline data.

# # <a id='3'>3. Data preprocessing and unification</a>

# In[ ]:


all_json = glob.glob( str(root_path) + '/**/*.json', recursive=True)
print("There are ", len(all_json), "sources files.")


# In[ ]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = (
    SparkSession.builder.appName("covid")
    .master("local[*]")
    .config("spark.driver.memory", "16g")
    .config("spark.executor.memory", "16g")
    .config("spark.driver.maxResultSize", "4g")
    .getOrCreate()
)

data = spark.read.json(all_json, multiLine=True)
data.createOrReplaceTempView("data")

#data.printSchema()


# In[ ]:


# Select text columns
covid_sql = spark.sql(
        """
        SELECT
            metadata.title AS title,
            abstract.text AS abstract, 
            body_text.text AS body_text,
            back_matter.text AS back_matter,
            paper_id
        FROM data
        """)

# Convert it to pandas and join all texts
covid_pd = covid_sql.toPandas()
covid_pd['abstract'] = covid_pd['abstract'].str.join(' ')
covid_pd['body_text'] = covid_pd['body_text'].str.join(' ')
covid_pd['back_matter'] = covid_pd['back_matter'].str.join(' ')

covid_pd.head()
covid_pd.to_csv('clean_covid.csv', index=False)


# # <a id='4'>4. Conclusion</a>
# 
# In the near future, I will also add into the database the authors as well as the bibliography columns. Feel free to leave a comment if you have questions or advices.
# 
# Thank you for reading, and ... stay healthy!
# 
