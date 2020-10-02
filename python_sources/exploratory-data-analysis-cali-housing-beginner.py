#!/usr/bin/env python
# coding: utf-8

# # Problem :
# 
# #### Preforming exploratory data analysis on the `callifornia housing` data set.
# 
# # Aim :
# * Visualizing statistical relationship from features in the californai housing dataset to again insights on what house feature contributes to it's price.
# * Use machine learning to model an application that would be able to predict housing prices when given a set of parameters (based of this data set).
# 
# # Data :
# #### The data is gotten from the [kaggle website](https://www.kaggle.com/c/californiahousing)

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
import tarfile
import urllib
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')


# In[ ]:


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
# HOUSING_PATH = os.path.join("datasets", "housing")
# HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL):
#     os.makedirs(housing_path, exist_ok=True)
#     tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url)
    housing_tgz = tarfile.open('housing.tgz')
    housing_tgz.extractall('housing.tgz')
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
df = load_housing_data()


# #### Looking at the first five data samples of the data set

# In[ ]:


df.head()


# #### Let's look at some statistical attributes of the dataset

# In[ ]:


df.describe()


# In[ ]:


df.info()


# #### 1 Before we start visualizing the data, we would like to add some additional features. Looking at the data set we could a two additional columns which would represent ` Number of rooms in each houshold` and `Number of bedrooms in each houshold`.
# 
# #### 2 Since countable items do not come in decimal points we would change the data type of the newly created features to type `int`. But before we do that, from the statistical description of the dataset we would notice the `total_bedrooms` count is 20433 from a total count of about 20640 which is about `207` points `missing/not computed` during data collection.
# 

# In[ ]:


df['bedrooms'] = round(df['total_bedrooms'] / df['households'])
df['rooms'] = round(df['total_rooms'] / df['households'])
df['bedrooms'].fillna(np.median, inplace=True, axis=0)
df['total_bedrooms'].fillna(np.median, inplace=True, axis=0)


# In[ ]:


df


# In[ ]:


df['total_bedrooms'] = pd.to_numeric(df['total_bedrooms'], errors='coerce')
df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')


# ### Extra feature engineering:
# For convinence the median_house name is changed to price, and the median_income would be up scaled to it original values.

# 

# In[ ]:


df['price'] = df['median_house_value']
df.drop('median_house_value', axis=1,inplace=True) 


# In[ ]:


df['median_income'] = df['median_income'] * 10000


# # Visualization

# In[ ]:


df.head()


# #### Let's plot a scatter plot of the `median_income` and the `price`

# In[ ]:


sns.relplot(x='median_income', y='price', data=df, height=8, )
plt.show()


# #### From the above graph it indicates that families with a `median_income` below the #8000 mark purchase houses with prices lower than $400000.

# #### Let's plot a scatter plot of the `median_income` and the `price` based on `median age`

# In[ ]:


sns.relplot(x='median_income', y='price', data=df, 
            hue='housing_median_age', height=8, palette='jet')
plt.show()


# ####  The scatter plot shows that age has little influence of income distribution.

# #### Let's check out the wealth and `population` distribution

# In[ ]:


sns.relplot(x='latitude', y='longitude', data=df, 
           hue='population', palette='jet', height=8)
plt.show()


# ####  The plot shows that californai is sparesly populated, with only one point having a rather heigh `population` cluster. We would further look at how the prices of house a scatter based on location.

# In[ ]:


sns.relplot(x='latitude', y='longitude', data=df, 
            hue='price', height=8, palette='jet')
plt.xlabel('latitude',fontsize=16)
plt.ylabel('longitude', fontsize=16)
plt.show()


# #### California's wealthy class seems to reside along the costal region on the state. We would then like to plot the relationship between the location and the `median_income` distribution

# In[ ]:


sns.relplot(x='latitude',y='longitude', data=df,
           hue='price' ,size='median_income', palette='jet', height=10)
plt.show()


# #### Not many houses earn more than $100000 as their `median_income`

# In[ ]:


df


# ####  Does the number of `rooms` have any relationship with the price of the house. let's plot a line graph to check.

# In[ ]:


sns.relplot('rooms', 'price', data=df, 
            kind='line', height=6, sort=True)
plt.show()


# In[ ]:


rooms = df[df['rooms'] <= 20]
df.pivot_table(rooms, 'rooms').iloc[:20,[7]]


# ####  From above the more rooms the less the price of the house except for houses with more than 10 rooms, `Note` : Houses with astronomically high number of rooms should be regarded as this may be hotels or caused by mislabelling of data.
# #### houses with `rooms` in the range of 0-20 cost the most.

# In[ ]:


sns.relplot('bedrooms', 'price', data=df,
           height=6, kind='line', sort=True)
plt.show()


# In[ ]:


rooms = df[df['bedrooms'] <= 20]
df.pivot_table(rooms, 'bedrooms').iloc[:12, [6]]


# ####  From above the more bedrooms the less the price of the house except for houses with more than 10 bedrooms, `Note`: Houses wit astronomically high number of bedrooms should be regarded as this may be hotels or caused by mislabelling of data.

# In[ ]:


sns.relplot('ocean_proximity', 'population', data=df, 
            kind='line', sort=True,
            height=6, palette='jet')
plt.show()


# ####  Population distribution of California's per-region, less have island houses, most live around `<1H ocean` region 

# In[ ]:


sns.relplot('ocean_proximity', 'median_income', data=df,
           kind='line', sort=True, height=6, palette='jet')
plt.show()


# #### `Island` dwellers have the lowest `median_income` rates, while `<1H ocean` have a very high `median_income` rate.

# In[ ]:


sns.relplot('ocean_proximity', 'price', data=df,
           kind='line', sort=True, height=6, palette='jet')
plt.show()


# #### `Island`houses have the highest `price` rates, while `Inland` house `price` are of very low  rate. 

# In[ ]:


sns.relplot('ocean_proximity', 'rooms', data=df,
           kind='line', sort=True, height=6, palette='jet')
plt.show()


# #### `Inland` houses have the highest number of `rooms`.

# In[ ]:


sns.relplot('ocean_proximity', 'housing_median_age', data=df,
           kind='line', sort=True, height=6, palette='jet')
plt.show()


# #### `Island`houses have the highest `housing median age`, while `Inland` houses have a `housing_median_age` of `25`. 

# In[ ]:


sns.relplot('ocean_proximity', 'bedrooms', data=df,
           kind='line', sort=True, height=6, palette='jet')
plt.show()


# #### `Island`houses have the highest number of `bedrooms`, while `<1H Ocean` and ` Near Bay` have only `1` room. 

# #### This is part of my #100daysofmlcode learning process.
