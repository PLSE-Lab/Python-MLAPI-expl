#!/usr/bin/env python
# coding: utf-8

# I'm still learning all the in-and-outs of ML and creating/sharing notebooks. All constructive feedback would be greatly appreciated.
# 
# # Import the data

# In[ ]:


import numpy as np
import pandas as pd

data_filepath = '../input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv'
df = pd.read_csv(data_filepath)
df.head()


# Analyse the dataset using pandas profiling

# # Initial data analysis
# ## Pandas profiling

# In[ ]:


import pandas_profiling
pandas_profile = pandas_profiling.ProfileReport(df, progress_bar=False)
pandas_profile.to_widgets()


# ## Pandas profiling results
# ### Categorial data
# Although many columns seem to be categorical, we can see they still show a high cardinality. For example "Key skills" has 26909 distinct values for 30000 data entries.
# Looking at the data this seems to have two main reasons:
# 1. Categories are not standardized
# 2. Multi-selection of categories is group together as comma-seperated into the same column
# 
# ### Missing values
# We have missing values for several columns. Usually around 2% per column, with a couple of outliers. "Key Skills" with 4.2% and "Role Category" with 7.7% are most notable.
# 
# ### Correlations/Interactions
# Due to the nature of the data (non-numerical), Pandas profiling doesn't perform any correlation analysis
# 
# # Specific column analysis

# In[ ]:


# Let's look at the freshness of the data using the crawl timestamp

import plotly.express as px
df['Crawl Timestamp_dt'] = pd.to_datetime(df['Crawl Timestamp']) #Convert to Pandas DateTime
px.box(df, y="Crawl Timestamp_dt", points="all", hover_data=["Uniq Id"])


# Crawling seems to have occured during two clusters: July 4 2019-July 8 2019 and Aug 4 2019-Aug 7 2019

# In[ ]:


# Let's now look at categorical columns such as Functional Area

func_area_df = (df['Functional Area'].str.split(' , ', expand=True) #Multiple categories are combined in the column as comma-separated so we first need to split this
     .stack() # We then stack them again as they'd otherwise be represented as separate columns
     .value_counts() # As last transformation we count all the values
    )

# We only want to show the top values, and combine the smaller values together
top_func_area_df = pd.concat([func_area_df[:20], pd.Series(func_area_df[20:].sum(), index=["Others"])])


fig = px.pie(top_func_area_df, values=top_func_area_df.values, names=top_func_area_df.index)
fig.update_traces(textposition='inside', textinfo='percent+label')


# In[ ]:


# Let's do the same with Industry and Key Skills

industry_df = (df['Industry'].str.split(', ', expand=True) #Multiple categories are combined in the column as comma-separated so we first need to split this
     .stack() # We then stack them again as they'd otherwise be represented as separate columns
     .value_counts() # As last transformation we count all the values
    )

# We only want to show the top values, and combine the smaller values together
top_industry_df = pd.concat([industry_df[:30], pd.Series(industry_df[30:].sum(), index=["Others"])])
px.bar(top_industry_df)


# We notice there is a large amount of IT and Software jobs

# In[ ]:


# Let's do the same with Industry and Key Skills

key_skills_df = (df['Key Skills'].str.split('\| ', expand=True) #Multiple categories are combined in the column as comma-separated so we first need to split this
     .stack() # We then stack them again as they'd otherwise be represented as separate columns
     .value_counts() # As last transformation we count all the values
    )

# We only want to show the top values, and combine the smaller values together
top_key_skills_df = pd.concat([key_skills_df[:30], pd.Series(key_skills_df[30:].sum(), index=["Others"])])
fig = px.bar(top_key_skills_df)
fig.update_layout(yaxis_type="log")


# We notice a large amount of different skills required, without any real outlier.
# As already noticed when analyzing Industry, we notice a large amount of programming skills in the top

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc = WordCloud(max_words=30, background_color='white', width = 2400, height = 800, min_font_size = 10)

plt.imshow(wc.generate_from_frequencies(df['Role'].astype(str).value_counts()))


# # Suggested further work
# A more detailed look at job experience and job salary would be useful, especially converting the string value to a range and analyzing this would be useful
