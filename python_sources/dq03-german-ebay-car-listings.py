#!/usr/bin/env python
# coding: utf-8

# ## German eBay cars sales data exploration

# Dataset of used cars from eBay Kleinanzeigen, a classifieds section of the German eBay website. Original source: https://www.kaggle.com/orgesleka/used-cars-database/data

# In[ ]:


import pandas as pd
import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


autos=pd.read_csv('/kaggle/input/used-cars-database-50000-data-points/autos.csv',encoding='Latin-1')


# In[ ]:


autos.head(5)


# In[ ]:


autos.info()


# Cols with nulls values to be treated: vehicleType,gearbox,model,fuelType,notRepairedDamage

# In[ ]:


columns_names = autos.columns


# In[ ]:


columns_names


# In[ ]:


columns_names = autos.columns
columns_names_converted=[]

## helper function to conver columns names

import re
def convert_2_snakecase (name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

## applyting the function
for i in columns_names:
    columns_names_converted.append(convert_2_snakecase(i))

## applyting transformation
autos.columns=columns_names_converted


# In[ ]:


autos


# In[ ]:


autos.describe(include='all')


# Columns like "offer_type","seller","abtest","gearbox","not_repaired_damage" can be dropped, they have only 2-3 unique values each. "Price","odometr," should be stripped and converted to numeric values.
# 
# It looks like the num_photos column has 0 for every column. We'll drop this column, plus the other two we noted as mostly one value.
# 

# In[ ]:


autos["nr_of_pictures"].value_counts()


# In[ ]:


autos = autos.drop(["nr_of_pictures", "seller", "offer_type"], axis=1)


# ### Cleaning text=> numeric transformation

# In[ ]:


autos['price'].unique()


# In[ ]:


autos['price']=autos['price'].str.replace("$","").str.replace(",","").astype(int)


# In[ ]:


autos['price'].head(10)


# In[ ]:


autos['odometer'].unique()


# In[ ]:


autos['odometer']=autos['odometer'].str.replace("km","").str.replace(",","").astype(int)


# In[ ]:


autos['odometer'].unique()


# In[ ]:


autos.rename(columns={'odometer':'odometer_km'},inplace=True)


# In[ ]:


autos


# ### Cleaning text=> numeric outliers

# Price column research

# In[ ]:


print(autos['price'].describe())
autos['price'].value_counts().sort_index(ascending=True)


# There're some obvious outliers and should be removed. Zero-dollars cars are definitely a mistake, while 1$ could be some starting price taking into an account the data was taken from eBay.

# In[ ]:


autos = autos[autos["price"].between(1,351000)]
autos['price'].describe()


# As we see we significantly improved general stat for the Price column, dropping it down just from mean 9,800 to mean 5,900

# In[ ]:


autos['odometer_km'].describe()
autos['odometer_km'].value_counts().sort_index(ascending=False)


# ### Cleaning data text => date values

# In[ ]:


autos[['date_crawled','date_created','last_seen']][0:5]


# In[ ]:


autos['date_crawled'].value_counts(normalize=True, dropna=False).head(10)


# In[ ]:


autos['year_of_registration'].describe()


# There're some wrong values for 'year_of_registration' that need to be dropped off.

# In[ ]:


print("year_of_registration BEFORE cleanup:")
print(autos['year_of_registration'].describe())
autos = autos[autos["year_of_registration"].between(1900,2016)]
print("year_of_registration AFTER cleanup:")
print(autos['year_of_registration'].describe())


# As we see now stat is quite meaningful, it's changed significantly from unrealistic min-max vals.

# In[ ]:


autos['year_of_registration'].value_counts(normalize=True).head(10)


# It appears that most of the vehicles were first registered in the past 20 years.
# 

# ### Exploring Brand column

# In[ ]:


autos['brand'].unique()


# In[ ]:


autos['brand'].value_counts().index


# ### Most expensive cars by brands in the listings

# In[ ]:


brands_list=autos['brand'].value_counts().index
brands_mean_price={}

for i in brands_list:
    mean_price = autos['price'][autos['brand']==i].mean()
    brands_mean_price[i]=int(mean_price)
    
    

import operator
brands_mean_price_sorted = sorted(brands_mean_price.items(), key=operator.itemgetter(1),reverse=True)
    
brands_mean_price_sorted[:15]


# ### Mean mileage calculation by brand 

# In[ ]:


brands_list=autos['brand'].value_counts().index
brands_mean_mileage={}


for i in brands_list:
    mean_mileage = autos['odometer_km'][autos['brand']==i].mean()
    brands_mean_mileage[i]=int(mean_mileage)
    
brands_mean_mileage    


# In[ ]:


bmp_series = pd.Series(brands_mean_price)

b_mileage_series = pd.Series(brands_mean_mileage)

#print(bmp_series)

df = pd.DataFrame(bmp_series, columns=['mean_price'])
df['mean_mileage']=b_mileage_series

df


# The range of car mileages does not vary as much as the prices do by brand, instead all falling within 10% for the top brands. There is a slight trend to the more expensive vehicles having higher mileage, with the less expensive vehicles having lower mileage.
# 
