#!/usr/bin/env python
# coding: utf-8

# # How Important is the App Label Category?
# 
# It seems plausible that app categories hold some information about the gender and age group of mobile users. 
# 
# There are 145 unique categories in the training set and the distributions vary signficantly between men and women by age group. 
# 
# For simplicity, I limited the categories to the top 20 most frequently seen in the training data. 
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
from bokeh.charts import Histogram, output_notebook, show
output_notebook()

data_train = pd.read_csv("../input/gender_age_train.csv")
events = pd.read_csv("../input/events.csv")
app_events = pd.read_csv("../input/app_events.csv")
app_labels = pd.read_csv("../input/app_labels.csv")
label_categories = pd.read_csv("../input/label_categories.csv")
print("Loaded!")


# In[ ]:


# Combine the data into a single dataframe
def combine(data):
    df = data.merge(events, how='left', on='device_id')
    df = df.merge(app_events, how='left', on='event_id')
    df = df.merge(app_labels, how='left', on='app_id')
    df = df.merge(label_categories, how='left', on='label_id')
    return df

df = combine(data_train)


# In[ ]:


# Number of unique categories
len(df.category.unique())


# In[ ]:


# Get the frequency counts for categories
count = df.category.value_counts()
count.head(20)


# In[ ]:


# Print histograms for random 20 categories (within the top 70)
categories = df.category.value_counts()[:70].sample(20, random_state=0)


for idx, cat in categories.iteritems():
    cat_data = df[df.category == idx]
    p = Histogram(cat_data, values='age', color='gender', bins=12,
              title="Age and Gender Distribution - Category: "+idx, legend='top_right'
              )
    show(p)

