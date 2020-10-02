#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Installing Side Table
get_ipython().system('pip install sidetable')


# #### Loading Libraries

# In[ ]:


import numpy as np
import pandas as pd
import sidetable
import ast
from sklearn.preprocessing import MultiLabelBinarizer


# #### Loading Data

# In[ ]:


path = "../input/flipkart-products/flipkart_com-ecommerce_sample.csv"


# In[ ]:


df = pd.read_csv(f"{path}")
df.head()


# In[ ]:


df["product_category_tree"].apply(lambda x: ast.literal_eval(x)).apply(lambda x: len(x[0].split(">>"))).argmax()


# In[ ]:


df["product_category_tree"].iloc[1482]


# In[ ]:


category_levels = (df["product_category_tree"]
     .apply(lambda x: ast.literal_eval(x))
     .apply(lambda x: x[0].replace("\\","").replace("'","").split(" >> "))
     .apply(pd.Series)
     .rename(columns= lambda x: "level_"+str(x)))


# In[ ]:


df = pd.concat([category_levels, df], axis=1)


# In[ ]:


df.head()


# Now, we have categories and product names as separate columns to play with pandas extension side table.

# In[ ]:


print(f"Total Number of top level categories {df['level_0'].nunique()}")


# Top 10 categories out of 265 total categories

# In[ ]:


df["level_0"].value_counts()[:10]


# In[ ]:


df["level_0"].value_counts(normalize=True)[:10]


# In[ ]:


df.stb.freq(["level_0"])[:10]


# Returns a dataframe of frequencies with percentage, cumulative count and cumulative percentage calculated.

# In[ ]:


df.stb.freq(["level_0"],thresh=0.6, other_label="other_categories")


# When we want to look at the categories that contribures for the top 60% which may required to help us focus on it, we can use "thresh" option which filters based on threshold ratio provided with additional option to rename the records outside the threshold using "other_label"

# In[ ]:


df.stb.freq(["level_0"], thresh=0.6) # without the other label


# By default, records outside the threshold are grouped as "Others"

# We can obtain the categories WITH RATINGS which contributes to top 60% 

# In[ ]:


df[df["overall_rating"] != "No rating available"].stb.freq(["level_0"], thresh=0.6, other_label="other_rated_categories")


# Percentage of contribution of Jewellery and Footwear categories are high but they didn't recieve ratings from the customer which means they are not yet sold much even though they have many different product varities.

# Another useful feature of sidetable is that we can generate this frequency datafraame in terms of other column values

# In[ ]:


df[df["overall_rating"] != "No rating available"].stb.freq(["level_0"], value = "discounted_price", thresh=0.6)


# In terms of current selling price (i.e discounted price) Computers, Clothing and Jewellery are accumulating more money than other categories like Watches, Footwear,etc.

# We can request for grand-totals and sub-totals by calling subtotal on the dataframe

# In[ ]:


(df[df["level_0"].isin(["Computers", "Clothing", "Jewellery"])]
    .groupby(["level_0", "level_1"]).agg({"discounted_price": ["sum"]}).stb.subtotal())


# Other important features is reporting "missing" values out of the box

# In[ ]:


df.stb.missing()


# Also, `%` can be suffixed for the percentange columns by styling

# In[ ]:


df.stb.freq(["brand"], thresh=0.10, style=True)

