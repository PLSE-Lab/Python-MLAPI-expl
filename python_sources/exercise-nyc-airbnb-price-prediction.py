#!/usr/bin/env python
# coding: utf-8

# ## Notes
# 
# These codes are originated from below kernels:
# 
# * https://www.kaggle.com/dgomonov/data-exploration-on-nyc-airbnb
# * https://www.kaggle.com/mpanfil/nyc-airbnb-data-science-ml-project
# 
# Based on these, original code will be added more and more.

# ## 1. Preparing Data

# ### Import packages

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### Load data

# In[ ]:


nyc_airbnb = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

nyc_airbnb.head()


# **Checking dataset size**
# 
# The dataset has 48,895 records and 16 features

# In[ ]:


print("The number of records(examples): {}".format(nyc_airbnb.shape[0]))
print("The number of columns(features): {}".format(nyc_airbnb.shape[1]))


# ### Observe data

# **Checking column dtype**
# 
# The columns which have numerical values are as follows:
# 
# `latitude`, `longitude`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`

# In[ ]:


# checking type of every column in the dataset
nyc_airbnb.dtypes


# **Checking null values**
# 
# The folling columns have null values:
# 
# `name`, `host_name`, `last_review`, `reviews_per_month`

# In[ ]:


print("Null values in NYC Airbnb 2019 dataset:")
# checking total missing values in each column in the dataset
nyc_airbnb.isnull().sum()


# Especially, there are many missing reviews (`last_review`, `reviews_per_month`).
# 
# Numbers of missing values of them are same (10052), which needs to be inspected. There are possibility that missing values are in the same row.
# 
# In fact, we can see that review related column are not significant and not difficult to handle.
# 
# * **last_review**: The date last review was uploaded. If there were no reivews for the, date will simply missing. This column seems to be insignificant thus could be dropped in data cleansing time.
# 
# * **reviews_per_month**: This column means review rate for month. We can easily replace NaN values with 0.0. 
# 

# **Inspecting name column null values**

# In[ ]:


# https://numpy.org/doc/1.18/reference/generated/numpy.where.html
# numpy.where: Return elements chosen from x or y depending on condition.
null_names = pd.DataFrame(np.where(nyc_airbnb['name'].isnull())).transpose()
null_host_names = pd.DataFrame(np.where(nyc_airbnb['host_name'].isnull())).transpose()

concat_null_names = pd.concat([null_names, null_host_names], axis=1, ignore_index=True)
concat_null_names.columns = ['Null rows in name column', 'Null rows in host_name column']
concat_null_names


# **Missing data visualization with `missingno`**
# 
# - `msno.bar` is a simple visualization of nullity by column.
# 
# - Bar chart shows total records and bar itself representsnon-missing values, thus we could see the how much the nullity of certain column by space.
# 
# - Each bar has label which is non-missing values on top of the chart.

# In[ ]:


import missingno as msno
missing_value_columns = nyc_airbnb.columns[nyc_airbnb.isnull().any()].tolist()
print("Missing value columns: {}".format(missing_value_columns))
msno.bar(nyc_airbnb[missing_value_columns], figsize=(15,8), color='#2A3A7E', 
         fontsize=15, labels=True)  # Can switch to a logarithmic scale by specifying log=True


# In[ ]:


msno.matrix(nyc_airbnb[missing_value_columns], width_ratios=(10, 1),
            figsize=(20, 8), color=(0, 0, 0), fontsize=12, sparkline=True, labels=True)


# ### Wrangling and cleaning data

# **Dropping columns**

# The column `last_review` seems to be insignificant for exploration and prediction.

# In[ ]:


nyc_airbnb.drop(['last_review'], axis=1, inplace=True)


# Also, `host_name` column contains actual person names, which might be sensible data.

# In[ ]:


nyc_airbnb.drop(['host_name'], axis=1, inplace=True)


# In[ ]:


nyc_airbnb.head(5)


# **Handling NaN values**

# After dropping two columns(`host_name`, `last_review`), dataset has still null values. Replacing all NaN values with zero seems to make sense.

# In[ ]:


nyc_airbnb['name'].fillna(value=0, inplace=True)


# In[ ]:


nyc_airbnb['reviews_per_month'].fillna(value=0, inplace=True)


# In[ ]:


nyc_airbnb.isnull().sum()


# ## Exploring data

# ### Checking categorical values
# 
# As we see above, we could find out three categorical columns, which are `neighbourhood_group`, `neighbourhood` and `room_type`.
# 
# `neighbourhood_group` and `neighbourhood` columns have informative area data.
# 
# The data could be utilized for analyzing data based on `neighbourhood`. Also, checking unique values in categorical values could help understanding of data.

# In[ ]:


# check unique category values
# find out which neighbourhood_group exist in dataset
print('Neighbourhood_group: {}'.format(nyc_airbnb['neighbourhood_group'].unique()))


# The dataset has five `neighbourhood_gruop` categories, which are `Brooklyn`, `Manhattan`, `Queens`, `Staten Island`, `Bronx`.

# In[ ]:


# check unique category values
# find out which neighbourhood exist in dataset
nyc_airbnb['neighbourhood'].unique()


# In[ ]:


nyc_airbnb['neighbourhood'].nunique()


# `neighbourhood` column has 221 categories.

# In[ ]:


nyc_airbnb['room_type'].unique()


# The `room_type` are `Private room`, `Entire home/apt` and `Shared room`.

# ### Top reviewed

# In[ ]:


nyc_airbnb.sort_values(by='number_of_reviews', ascending=False).head(5)


# In[ ]:


nyc_airbnb.sort_values(by='reviews_per_month', ascending=False).head(5)


# ## Visualization

# ### 1. Counts of airbnb in neighbourhood group
# 

# In[ ]:


plt.figure(figsize=(15, 8))
plt.title('Counts of airbnb in neighbourhood group', fontsize=15)
sns.countplot(x='neighbourhood_group', data=nyc_airbnb, 
              order=nyc_airbnb['neighbourhood_group'].value_counts().index,
              palette='BuGn_r')


# There seems to be that most of airbnb in NYC are in `Manhattan` and `Brooklyn`

# ### 2. Counts of airbnb in neighbourhood group with room type

# In[ ]:


plt.figure(figsize=(15, 8))
plt.title('Counts of airbnb in neighbourhood group with room type', fontsize=15)
sns.countplot(x='neighbourhood_group', data=nyc_airbnb, hue='room_type',
              palette="Set2")


# ### 3. Top neighbourhoods with  room type

# In[ ]:


top_neigh = nyc_airbnb['neighbourhood'].value_counts().reset_index().head(10)  # Top 10
top_neigh = top_neigh['index'].tolist()  # get top 10 neighbourhood names

plt.figure(figsize=(15, 8))
plt.title('Top neighbourhoods with room type', fontsize=15)
viz = sns.countplot(x='neighbourhood', data=nyc_airbnb.loc[nyc_airbnb['neighbourhood'].isin(top_neigh)],
              hue='room_type', palette='GnBu_d')
viz.set_xticklabels(viz.get_xticklabels(), rotation=45)


# ### 4. Price distribution by negihbourhood group

# In[ ]:


# check wholde dataset price stats
nyc_airbnb['price'].describe()


# In[ ]:


plt.figure(figsize=(15, 8))
sns.distplot(nyc_airbnb['price'])


# In[ ]:


nyc_airbnb['price'].quantile(.98)


# In[ ]:


plt.figure(figsize=(15, 8))
sns.distplot(nyc_airbnb[nyc_airbnb['price'] < 550]['price'])


# In[ ]:


plt.figure(figsize=(15, 8))
plt.title('Density and distribution of prices for each neighbourhood group', fontsize=15)
sns.violinplot(x='neighbourhood_group', y='price', 
               data=nyc_airbnb[nyc_airbnb['price'] < 550], palette='Set3')


# Let's examine detail stats by neighbourhood group for better understanding.

# In[ ]:


# Brooklyn
sub_1_brooklyn = nyc_airbnb.loc[nyc_airbnb['neighbourhood_group'] == 'Brooklyn']
price_sub_1 = sub_1_brooklyn[['price']]

# Manhattan
sub_2_manhattan = nyc_airbnb.loc[nyc_airbnb['neighbourhood_group'] == 'Manhattan']
price_sub_2 = sub_2_manhattan[['price']]

# Queeens
sub_3_queens = nyc_airbnb.loc[nyc_airbnb['neighbourhood_group'] == 'Queens']
price_sub_3 = sub_3_queens[['price']]

# Staten Island
sub_4_staten = nyc_airbnb.loc[nyc_airbnb['neighbourhood_group'] == 'Staten Island']
price_sub_4 = sub_4_staten[['price']]

# Bronx
sub_5_bronx = nyc_airbnb.loc[nyc_airbnb['neighbourhood_group'] == 'Bronx']
price_sub_5 = sub_5_bronx[['price']]

price_list_by_group = [price_sub_1, price_sub_2, price_sub_3, price_sub_4, price_sub_5]


# Integrates all individual stats into dataframe.

# In[ ]:


integ_price_stats_list = []
neigh_groups = nyc_airbnb['neighbourhood_group'].unique().tolist()

for price_group, group_name in zip(price_list_by_group, neigh_groups):
  stats = price_group.describe()  # count / mean / std / 25% / 50% / 75% / max
  stats = stats.iloc[1:]  # mean / std / 25% / 50% / 75% / max
  stats.reset_index(inplace=True)
  stats.rename(columns={'index': 'Stats', 'price': group_name}, inplace=True)
  stats.set_index('Stats', inplace=True)
  integ_price_stats_list.append(stats)

price_stats_df = pd.concat(integ_price_stats_list, axis=1)
price_stats_df


# In[ ]:


cmap = sns.cubehelix_palette(as_cmap=True)

wo_extreme = nyc_airbnb[nyc_airbnb['price'] < 550]

f, ax = plt.subplots()
f.set_size_inches(20, 10)
points = ax.scatter(wo_extreme['latitude'], wo_extreme['longitude'], 
                    c=wo_extreme['price'], cmap=cmap)
f.colorbar(points)


# ### Host location distribution

# In[ ]:


plt.figure(figsize=(15, 8))
sns.scatterplot(data=nyc_airbnb, x='longitude', y='latitude', 
                hue='neighbourhood_group', palette='Set3')


# ## Predictions

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[ ]:


nyc_airbnb.drop(['name', 'id'], inplace=True, axis=1)


# In[ ]:


# encodes categorical values
le = LabelEncoder()

le.fit(nyc_airbnb['neighbourhood_group'])
nyc_airbnb['neighbourhood_group'] = le.transform(nyc_airbnb['neighbourhood_group'])

le.fit(nyc_airbnb['neighbourhood'])
nyc_airbnb['neighbourhood'] = le.transform(nyc_airbnb['neighbourhood'])

le.fit(nyc_airbnb['room_type'])
nyc_airbnb['room_type'] = le.transform(nyc_airbnb['room_type'])

nyc_airbnb.head(5)


# In[ ]:


# records with price zero are sorted on top
nyc_airbnb.sort_values('price', ascending=True, inplace=True)


# In[ ]:


nyc_airbnb = nyc_airbnb[11:-6]


# In[ ]:


lm = LinearRegression()


# In[ ]:


X = nyc_airbnb.drop(['price', 'longitude'], inplace=False, axis=1)
y = nyc_airbnb['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

lm.fit(X_train, y_train)


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


# Evaluated metrics

mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
r2 = metrics.r2_score(y_test, predictions)

print('MAE (Mean Absolute Error): %s' %mae)
print('MSE (Mean Squared Error): %s' %mse)
print('RMSE (Root mean squared error): %s' %rmse)
print('R2 score: %s' %r2)


# In[ ]:


# Avtual vs predicted values

error = pd.DataFrame({'Actual Values': np.array(y_test).flatten(), 'Predicted Values': predictions.flatten()})
error.head(10)

