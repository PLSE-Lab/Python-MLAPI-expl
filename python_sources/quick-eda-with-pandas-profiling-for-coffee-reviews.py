#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis of Coffee reviews
# 
# In order to understand the data from Coffee reviews provided by [Coffee Quality Institute](https://database.coffeeinstitute.org/), let's do a quick Exploratory Data Analysis (EDA) using 
# [pandas_profiling](https://github.com/pandas-profiling/pandas-profiling), which can perform an EDA out-of-the-box! Check this out in the following.

# ## Importing dependencies

# In[ ]:


import numpy as np
import pandas as pd
import pandas_profiling as pp


# ## Loading pre-cleaned data

# * For Arabica coffee (our larger dataset)

# In[ ]:


df_coffee_reviews_arabica = pd.read_csv("../input/coffee-quality-database-from-cqi/arabica_data_cleaned.csv")

df_coffee_reviews_arabica.head(10)


# * For Robusta coffee (unfortunately, we have a small number of reviews for this one)

# In[ ]:


df_coffee_reviews_robusta = pd.read_csv("../input/coffee-quality-database-from-cqi/robusta_data_cleaned.csv")

df_coffee_reviews_robusta.head(10)


# The columns `Unnamed` looks like a duplicated DataFrame index. Let's drop it!

# In[ ]:


df_coffee_reviews_arabica.drop(columns=[df_coffee_reviews_arabica.columns[0]], inplace=True)
df_coffee_reviews_robusta.drop(columns=[df_coffee_reviews_robusta.columns[0]], inplace=True)


# Have DataFrames same columns?

# In[ ]:


df_coffee_reviews_arabica.columns


# In[ ]:


df_coffee_reviews_robusta.columns


# In[ ]:


df_coffee_reviews_arabica.columns == df_coffee_reviews_robusta.columns


# Apparently the columns are not the same for DataFrames. Let's check the difference:

# In[ ]:


df_coffee_reviews_arabica.columns.difference(df_coffee_reviews_robusta.columns)


# However, the difference between datasets seems to have the same information, but with different names. Let's now change the names for Robusta dataset.

# In[ ]:


df_coffee_reviews_robusta.rename(
    columns={
        "Salt...Acid": "Acidity",
        "Fragrance...Aroma": "Aroma",
        "Bitter...Sweet": "Sweetness",
        "Uniform.Cup": "Uniformity",
        "Mouthfeel": "Body"
    },
    inplace=True
)


# Most of the above renamings are clear, except for `Mouthfeel -> Body`. Without a prior knowledge of the field, this equivalence would be hard to get. I took it from [here](https://espressocoffeeguide.com/all-about-coffee-2/coffee-flavor/body/).
# 
# Now let's check again to see if we have the same columns in both DataFrames:

# In[ ]:


df_coffee_reviews_arabica.columns.difference(df_coffee_reviews_robusta.columns)


# Now DataFrames are ready to be merged!

# ## Exploring datasets with Pandas profiling

# In[ ]:


pp.ProfileReport(df_coffee_reviews_arabica)


# In[ ]:


pp.ProfileReport(df_coffee_reviews_robusta)


# ## Merging datasets
# 
# Let's now merge both datasets in a unique DataFrame.

# In[ ]:


df_coffee_reviews = pd.concat(
    [df_coffee_reviews_arabica, df_coffee_reviews_robusta],
    ignore_index=True,
    sort=False
)

df_coffee_reviews.head(10)


# In[ ]:


pp.ProfileReport(df_coffee_reviews)


# From this merged dataset, we can proceed with further further analysis or export it to a unique `csv` file. Actually this is how `merged_data_cleaned.csv` was generated.
# 
# ## Final remarks
# 
# If you take a look on the correlation plots, you probably will note that some numerical features seems to be highly correlated. Well, then... what we could get from these relations? This is your time to shine!
