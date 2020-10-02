#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will look at which names are equally favored by both genders.

# In[1]:


import numpy  as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from bq_helper import BigQueryHelper
from wordcloud import WordCloud, STOPWORDS

matplotlib.rcParams["figure.figsize"] = (20, 7)


# In[2]:


db = BigQueryHelper(
    active_project="bigquery-public-data",
    dataset_name="usa_names"
)


# ## Calculating our "unisex index"
# 
# First, we retrieve the data from BigQuery.

# In[3]:


query_string = """
SELECT
    name,
    year,
    gender,
    sum(number) AS count
FROM
    `bigquery-public-data.usa_names.usa_1910_current`
GROUP BY
    name,
    year,
    gender
"""

counts = db.query_to_pandas_safe(query_string)
counts.head()


# Then, we reshape it to have the name and year as keys.

# In[4]:


counts_by_gender = pd.pivot_table(
    counts,
    index=["name", "year"],
    columns=["gender"],
    values="count",
    aggfunc=np.sum,
    fill_value=0
)
counts_by_gender["total"] = counts_by_gender["F"] + counts_by_gender["M"]
counts_by_gender.head()


# Lastly, we compute the proportions for each gender and more importantly `AbsDiff`. The absolute difference between the male and female proportions is the value we will be using to determine if a name is unisex. Its values range from  0 (equal split between genders ) to 1 (name is used by only one gender).

# In[5]:


proportions = pd.DataFrame(index=counts_by_gender.index)

proportions["Total"]   = counts_by_gender["total"]
proportions["F"] = counts_by_gender["F"] / counts_by_gender["total"]
proportions["M"] = counts_by_gender["M"] / counts_by_gender["total"]
proportions["AbsDiff"] = (proportions["F"] - proportions["M"]).abs()

proportions.head()


# Here's an example of a popular unisex name.

# In[6]:


popular_names = proportions["Total"] > 10000
pretty_unisex = proportions["AbsDiff"] < 0.2
proportions[popular_names & pretty_unisex]


# ## Popular Unisex names
# 
# We say a name is popular in a year if there were more than 500 applicants for it. We consider a name to be unisex if the gender proportions differ by less than 30 percentage points.

# In[23]:


unisex = proportions[
    (proportions["Total"]   > 500) &
    (proportions["AbsDiff"] < 0.3)
].copy().reset_index()

unisex = unisex.groupby("name").sum()
unisex["Total"].head()


# In[25]:


wordcloud = WordCloud(
    max_font_size=50, 
    stopwords=STOPWORDS,
    background_color='black',
    collocations=False,
    width=600,
    height=300,
)

image = wordcloud.generate_from_frequencies(unisex["Total"].to_dict())

plt.figure(figsize=(25, 10))
plt.title("Wordcloud for Popular Unisex Names", fontsize=35)
plt.imshow(image)
plt.axis('off')
plt.show()


# That's funny. Who names their child "Infant"?
#  ..or "Unknown" 

# ## More Work
# 
# Now that we've seen what popular unisex names, there's still more I'd like to explore. Are there names that became more accepted as unisex as the years go by? If there are, was it a traditionally female name that males also now have? Or the other way around? What could have influenced this change?

# In[26]:


proportions.to_csv("usa_names_gender_proportions.csv")

