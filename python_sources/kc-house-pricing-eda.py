#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")
from mpl_toolkits.basemap import Basemap # For geographical map


# In[ ]:


df = pd.read_csv("../input/kc_house_data.csv")


# In[ ]:


df.head(1)


# # Findout the geographical location where exactly that country and state is by just looking at Longitude and Lattitude in dataset
# So it's in North-West America!!!

# In[ ]:


map = Basemap(lat_0=47.0, lon_0=-121)
plt.figure(figsize=[20,10])
map.drawcoastlines(linewidth=.5,color="g")
map.drawcountries(linewidth=.5)
map.drawcounties()
# map.drawstates(color="r")
map.fillcontinents(color="green",alpha=.1)
plt.scatter(df.long,df.lat,alpha=.6,color="r")
plt.show()


# # Lets drop those columns which is irrelavant for further analysis

# In[ ]:


df.drop(["id"],axis=1,inplace=True)


# # Find out correlation between columns

# In[ ]:


corr = df.corr()


# In[ ]:


plt.figure(figsize=(16,8))
sns.heatmap(corr,annot=True,cmap="RdBu")


# # Draw only lower triangle for correlation map

# In[ ]:


plt.figure(figsize=(16,8))
mask = np.zeros_like(corr,dtype=np.bool)

# Create a msk to draw only lower diagonal corr map
mask[np.triu_indices_from(mask)] = True
sns.set_style(style="white")
sns.heatmap(corr,annot=True,cmap="RdBu",mask=mask)


# # Highlight only those values which are highly correlated (.5 and above)

# In[ ]:


# corr[corr>=.5]
plt.figure(figsize=(16,8))
mask = np.zeros_like(corr[corr>=.5],dtype=np.bool)

# Create a msk to draw only lower diagonal corr map
mask[np.triu_indices_from(mask)] = True
sns.set_style(style="white")
sns.heatmap(corr[corr>=.5],annot=True,mask=mask,cbar=False)


# # Regression plot for highly correlated columns
# Due to so many plots I have commented. Please uncomment if you want to see the plots

# In[ ]:


# plt.figure(figsize=(16,8))
# for idx in high_corr.index:
#     corr_idx = high_corr.index[high_corr[idx].notna()]
#     for c_i in corr_idx:
#         if c_i != idx:
#             sns.scatterplot(df[c_i],df[idx],alpha=.2,color="m")
#             plt.show()


# In[ ]:


# plt.figure(figsize=(16,8))
# for idx in high_corr.index:
#     corr_idx = high_corr.index[high_corr[idx].notna()]
#     for c_i in corr_idx:
#         if c_i != idx:
#             sns.regplot(df[c_i],df[idx],color="g")
#             plt.show()


# # House pricing over time period

# In[ ]:


date_df = df.sort_values(by="date")
plt.figure(figsize=(16,8))
sns.scatterplot(date_df.date,date_df.price,hue=date_df.floors,alpha=.9,size=date_df.grade,palette="winter_r")
plt.xticks([])
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
# sns.scatterplot(df.zipcode,df.price,hue=date_df.floors,alpha=.9,size=date_df.grade,palette="winter_r")
# sns.boxplot(df.zipcode,df.price)
sns.swarmplot(df.zipcode,df.price)
plt.xticks([])
plt.show()


# # Price based on house condition

# In[ ]:


plt.figure(figsize=(16,8))
sns.scatterplot(df.condition,df.price)
plt.xticks([])
plt.show()


# # House price based on year it was built

# In[ ]:


plt.figure(figsize=(16,8))
yr_price = df.sort_values(by="yr_built")
sns.lineplot(yr_price.yr_built.sort_values(),yr_price.price)
plt.xticks([])
plt.show()


# # Find out which zipcode (area) has maximum house price

# In[ ]:


plt.figure(figsize=(16,8))
zip_price = df.sort_values(by="price",ascending=False)
# zip_price
sns.barplot(x=zip_price.zipcode,y=zip_price.price,palette="cool")
plt.xticks(rotation=90)
plt.show()


# # Living area Vs house price

# In[ ]:


plt.figure(figsize=(16,8))
high_corr = corr[corr>=.5]
sns.set_style(style="darkgrid")
sns.scatterplot(df.sqft_living,df.price,hue=df.yr_renovated)


# # Date Vs Bedroom

# In[ ]:


df.head(1)


# In[ ]:


plt.figure(figsize=(16,4))
date_bedroom = df.sort_values(by="date")
# date_bedroom
sns.scatterplot(date_bedroom.date,date_bedroom.bedrooms,hue=df.bathrooms,size=df.bedrooms,palette="winter")
# plt.xticks(rotation=90)
plt.show()


# # See the price if house is in waterfront

# In[ ]:


sns.barplot(df.waterfront,df.price)


# # See the price if house is in view

# In[ ]:


sns.barplot(df.view,df.price)


# # See the price if house is in different conditions

# In[ ]:


sns.barplot(df.condition,df.price)


# # See the price if house has different garde
# So higher the grade then higher the price

# In[ ]:


plt.figure(figsize=(16,4))
sns.barplot(df.grade,df.price)


# # Count the supply of sqft_living flats

# In[ ]:


plt.figure(figsize=(16,4))
sns.countplot(df.sqft_living[:200],palette="winter_r")
plt.xticks([])
plt.show()

