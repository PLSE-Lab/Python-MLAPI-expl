#!/usr/bin/env python
# coding: utf-8

# ![auto-1291491_1280.jpg](attachment:auto-1291491_1280.jpg)

# # US CAR Sales
# ### This dataset contains details of cars that are listed for sale in the United States. Lets observe some relationship between different features in this set using Seaborn and Matplotlib.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/usa-cers-dataset/USA_cars_datasets.csv")
df = df.drop('Unnamed: 0', 1)
df.head()


# In[ ]:


df.shape


# ### Features in the Dataset

# In[ ]:


df.columns


# ### Vehicle Status
# * Clean Vehicles are ones that have not suffered any critical damage or loss.
# * Salvage Vehicles are vehicles that have been damaged and/or deemed a total loss by an insurance company.

# In[ ]:


fig, ax = plt.subplots()
sns.set_palette("ocean")
sns.countplot(x = "title_status", data = df, ax = ax)
ax.set_xlabel("Vehicle Status")
ax.set_ylabel("Count")
plt.show()


# ### Sale Price and Mileage Distributions

# In[ ]:


sns.set_palette("muted")
fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, figsize=(16,11))
sns.set_style("darkgrid")
sns.distplot(df['price'],ax = ax0)
sns.distplot(df["mileage"],hist = True, rug = True, ax= ax1)
ax0.set_xlabel("Sale Price($)")
ax1.set_xlabel("Distance Travelled (Miles)")


# ### Sale Price Distribution Based on Vehicle Registration Year

# In[ ]:


custom_palette = ["blue", "green", "orange","red","yellow", "purple"]
sns.set_palette(custom_palette)
fig, ax = plt.subplots(figsize = (16,11))
sns.scatterplot(x = "year", y = "price", hue = "title_status", data = df, ax = ax)
ax.set_xlabel("Vehicle Registration Year")
ax.set_ylabel("Sale Price($)")
plt.show()


#  ### Mileage Distribution Based on Sale Price and Registration Year

# In[ ]:


fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, figsize=(16,9))
sns.set_style("darkgrid")
sns.scatterplot(x = "price", y = "mileage", data = df, ax =ax0)
sns.scatterplot(x = "year", y = "mileage", data = df, ax= ax1)
ax0.set_xlabel("Sale Price($)")
ax1.set_xlabel("Vehicle Registration Year")
ax0.set_ylabel("Distance Travelled (Miles)")
ax1.set_ylabel("Distance Travelled (Miles)")
plt.show()


# ### Price Distribution of Top Five Colors
# #### White, Black, Gray, Silver and Red coloured cars account for 84.43% of cars in the dataset.

# In[ ]:


top5_colors = list(df.color.value_counts()[0:5].index)
top5_colors


# In[ ]:


df_top5_color = df[df["color"].isin(top5_colors)]
fig, ax = plt.subplots(figsize = (16,11))
sns.boxplot(x = "color", y = "price",data = df_top5_color,palette = "inferno", ax = ax)
ax.set_xlabel("Color")
ax.set_ylabel("Sale Price($)")
plt.show()


# ### Price Distribution of Brands with more than Ten Cars for Sale

# In[ ]:


df["brand"].value_counts()[df["brand"].value_counts() >= 10].index


# In[ ]:


over10_brands = df["brand"].value_counts()[df["brand"].value_counts() >= 10].index
df_over10_cars_per_brand = df[df["brand"].isin(over10_brands)]
fig, ax = plt.subplots(figsize = (16,16))
sns.swarmplot(data = df_over10_cars_per_brand, x = "price", y = "brand", ax = ax)
ax.set_xlabel("Sale Price($)")
ax.set_ylabel("Brand")


# ### Price Distribution of States with more than 100 Cars for Sale

# In[ ]:


over100_states = df["state"].value_counts()[df["state"].value_counts() >= 100].index
df_over100_cars_per_state = df[df["state"].isin(over100_states)]
fig, ax = plt.subplots(figsize = (16,16))
sns.lvplot(data=df_over100_cars_per_state, y="state",
x="price", ax=ax)
ax.set_xlabel("Sale Price($)")
ax.set_ylabel("State")

