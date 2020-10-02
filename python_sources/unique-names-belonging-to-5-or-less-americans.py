#!/usr/bin/env python
# coding: utf-8

# When I saw the [USA Names Dataset](https://www.kaggle.com/datagov/usa-names) I immediate wondered: how many Americans have such a unique name that the number of people with that name can be counted on one hand (5 or less people with that name) throughout the history of the social security recordings? This kernel explores that question.
# 
# But first a tangential song that I couldn't help but add in here as a link...

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo('22QYriWAF-U', width=800, height=450)


# In[ ]:


import numpy as np
import pandas as pd
import os
import bq_helper
from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS

init_notebook_mode(connected=True)

from bq_helper import BigQueryHelper
usa = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="usa_names")
bq_assistant = BigQueryHelper("bigquery-public-data", "usa_names")
#bq_assistant.list_tables()


# In[ ]:


# Review the table schema
bq_assistant.table_schema("usa_1910_current")


# In[ ]:


# Take a quick peak at the data
bq_assistant.head("usa_1910_current", num_rows=10)


# Below in the WITH query I sum the number of occurrences for a name throughout the entire history of the US Social Security system. Then I only keep those with 5 or less occurrences in the WHERE clause of the main SELECT.

# In[ ]:


query1 = """
WITH temp1 AS
    (
    SELECT
     names1.name,
     SUM(names1.number) AS total_occurrences
    FROM `bigquery-public-data.usa_names.usa_1910_current` AS names1
    GROUP BY
     names1.name
    )
SELECT
  names2.name,
  names2.gender,
  names2.state,
  names2.year,
  names2.number
FROM       temp1
INNER JOIN `bigquery-public-data.usa_names.usa_1910_current` AS names2
 ON temp1.name = names2.name
WHERE
 temp1.total_occurrences <= 5
ORDER BY
 names2.name
;
"""
#usa.estimate_query_size(query1)

response1 = usa.query_to_pandas_safe(query1, max_gb_scanned=0.5)


# In[ ]:


response1.head(10)


# In[ ]:


response1.tail(10)


# Now some questions to get answered...
# 
# **1)** Which state has the highest number of these rare names?

# In[ ]:


# aggregate by state
unique_names_states = response1.groupby('state', as_index=False).agg({"number": "sum"})
unique_names_states = unique_names_states.sort_values(by='number', ascending=False)

# visualize on go chart that allows for user scroll over to interact with it
x = unique_names_states.state
y = unique_names_states.number
trace1 = go.Bar(x=x, y=y, opacity=0.75, name="state sum")
layout = dict(height=400, title='Number of people by state with very unique names', legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);


# **2)** Are these rare names more prevalent for men or women?

# In[ ]:


# aggregate by gender
unique_names_gender = response1.groupby('gender', as_index=False).agg({"number": "sum"})
unique_names_gender = unique_names_gender.sort_values(by='number', ascending=False)

# visualize on go chart that allows for user scroll over to interact with it
x = unique_names_gender.gender
y = unique_names_gender.number
trace1 = go.Bar(x=x, y=y, opacity=0.75, name="gender sum")
layout = dict(height=400, title='Very unique American names by gender', legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);


# **3)** Has the occurrence of rare names changed over time?

# In[ ]:


# aggregate by year
unique_names_year = response1.groupby('year', as_index=False).agg({"number": "sum"})
unique_names_year = unique_names_year.sort_values(by='number', ascending=True)

# visualize on go chart that allows for user scroll over to interact with it
x = unique_names_year.year
y = unique_names_year.number
trace1 = go.Bar(x=x, y=y, opacity=0.75, name="year sum")
layout = dict(height=400, title='Very unique American names by year', legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);


# A quick word cloud for fun just to get a better look at a sample of the very unique names.

# In[ ]:


#mask = np.array(Image.open('../input/word-cloud-masks/image.png'))

txt = " ".join(response1['name'].sample(n=300).dropna().unique())
wc = WordCloud(max_words=300, 
               #mask=mask,
               stopwords=STOPWORDS,
               max_font_size=12,
               min_font_size=6,
               colormap='copper', 
               background_color='White').generate(txt)
plt.figure(figsize=(16,18))
plt.imshow(wc)
plt.axis('off')
plt.title('');


# **A few things I've noticed in this analysis:**
# - The drop in very unique names around the time of World War II and the decade thereafter is interesting.
# - The notable difference in number of unique names between males and females was higher than I would have expected.
# - It seems that perhaps the Social Security administration always represents "5" as the lowest number because I was unable to find anything less than in any of the data.
