#!/usr/bin/env python
# coding: utf-8

# # Airbnb Prices Berlin

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[ ]:


style.use("fivethirtyeight")


# ## Load data

# In[ ]:


listings = pd.read_csv("/kaggle/input/berlin-airbnb-data/listings_summary.csv")


# In[ ]:


listings.shape


# As there are so many variables, let's select the ones that can be useful to predict price.

# In[ ]:


# define the columns to keep
column_names=['id', 
       
       'host_id', 
       'host_total_listings_count',  

       'neighbourhood_group_cleansed',  
       'latitude', 'longitude',
              
       'property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 
       'amenities',      
              
       'price',  'security_deposit', 'extra_people',
       'cleaning_fee',
        'minimum_nights',
       'maximum_nights', 
       
       'number_of_reviews', 
       'cancellation_policy'
             ]


# In[ ]:


data = listings[column_names]
print(data.shape)


# ## Data cleaning and preparation

# In[ ]:


data.info()


# First, let's change the format of the price columns to numeric.

# In[ ]:


data[['price', 'cleaning_fee', 'extra_people','security_deposit']].head(2)


# In[ ]:


## clean price columns

prices=['price', 'cleaning_fee', 'extra_people', 'security_deposit']

for p in prices:
    data[p]=pd.Series([str(s).replace('$', '').replace(',', '') for s in data[p]]).astype(float)
    


# In[ ]:


data[['price', 'cleaning_fee', 'extra_people', 'security_deposit']].head(2)


# As we have NAs for deposit and cleaning fee, let's fill these fields with 0 (as we can assume that if they are not mentioned there is no charge).

# In[ ]:


data.security_deposit.fillna(0, inplace=True)
data.cleaning_fee.fillna(0,inplace=True)


# There are still a few missing values which can be dropped:

# In[ ]:


#missing values
data.isnull().sum()


# In[ ]:


data.dropna(inplace=True)


# In[ ]:


data.shape


# ## Exploration

# In[ ]:


data["price"].describe()
# appears we have some extreme outliers


# For price, it appears that there are some very large outliers (8600 per night seems excessive for an Airbnb). Let's clean this field after looking at the properties included in the dataset.

# In[ ]:


plt.figure(figsize=(10,2))
ax=sns.boxplot(data["price"],color="darkblue")
ax.set(xlim=(0, 500))


# ### __Property type__

# In[ ]:


plt.figure(figsize=(30,5))
sns.boxplot(x="property_type",y="price",data=data)


# As expected, the prices are very dispersed, and it seems that Hotel prices represent most of the high prices. Looking at the property types, we have 90% apartments, and for the other categories very few observations. I will keep only apartments for modeling to try to build a good model with this data.

# In[ ]:


data = data[(data['property_type'] == 'Apartment')]
data.drop(['property_type'], axis=1, inplace=True)


# Let's remove the price outliers now:

# In[ ]:


len(data[data["price"]>200])


# In[ ]:


len(data[data["price"]>200])
data=data[data["price"]<200]

len(data[data["price"]==0])
data=data[data["price"]>0] 


# ### Map of apartments

# In[ ]:


viz1=data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(10,7), 
        c="price", cmap="gist_heat_r", colorbar=True, sharex=False)

berlin_centre = (52, 13)


# In[ ]:


sns.distplot(data["price"],kde=False)


# Transforming the price to a log scale improves its distribution only slightly, so I decide to just remove the outliers on the right tail:

# In[ ]:


sns.distplot(np.log1p(data["price"]),kde=False)


# In[ ]:


data=data[data["price"]<160]


# In[ ]:


data.shape


# ### Neighbourhoods

# In[ ]:


plt.figure(figsize=(10,5))
data['neighbourhood_group_cleansed'].value_counts().sort_values().plot(kind='barh', color='darkblue')
plt.title('Number of Accommodations per District');


# In[ ]:


plt.figure(figsize=(20,5))

chart=sns.boxplot(x="neighbourhood_group_cleansed", y="price", data=data, palette="viridis")
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=18)
plt.title('Neighbourhood price distribution')
chart


# ### Bedrooms

# In[ ]:


sns.boxplot(x="bedrooms", y="price", data=data,palette="summer")


# In[ ]:


plt.figure(figsize=(5,5))
sns.heatmap(data.groupby(['neighbourhood_group_cleansed', 'bedrooms']).price.median().unstack(), 
            cmap='Blues', annot=True, fmt=".0f")
plt.title("Median Prices by Neighbourhood and Bedrooms")
plt.show()


# ### Cancellation policy

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x="cancellation_policy", y="price", data=data, palette='viridis')


# In[ ]:


data["cancellation_policy"].value_counts()


# In[ ]:


data = data[(data['cancellation_policy'] == 'flexible') |
            (data['cancellation_policy'] == 'moderate') |
            (data['cancellation_policy'] == 'strict_14_with_grace_period')]


# ### Bathrooms

# In[ ]:


plt.figure(figsize=(10,3))
sns.boxplot(x="bathrooms",y="price",data=data)


# In[ ]:


# cleaning the outliers
data["bathrooms"].value_counts()
data=data[data["bathrooms"]<=2]


# ### Amenities

# In[ ]:


from collections import Counter


# In[ ]:


amenities = Counter()
data['amenities'].str.strip('{}')               .str.replace('"', '')               .str.lstrip('\"')               .str.rstrip('\"')               .str.split(',')               .apply(amenities.update)

amenities.most_common(10)


# In[ ]:


# create a new dataframe for plotting
sub_df = pd.DataFrame(amenities.most_common(25), columns=['amenity', 'count'])


# In[ ]:


# plot the top amenities
plt.figure(figsize=(10,5))
sub_df.sort_values(by=['count'], ascending=True).plot(kind='barh', x='amenity', y='count',  
                                                      figsize=(10,7), legend=False, color="darkblue",
                                                      title='Amenities')
plt.xlabel('Count');


# After some exploration, these amenities seem to lead to different prices, so let's create binary variables out of them.

# In[ ]:


data['Laptop_friendly_workspace'] = data['amenities'].str.contains('Laptop friendly workspace')
data['TV'] = data['amenities'].str.contains('TV')
data['Hot water']=data['amenities'].str.contains('Hot water')
data['Family_friendly'] = data['amenities'].str.contains('Family/kid friendly')
data['Hair_dryer'] = data['amenities'].str.contains('Hair_dryer')
data['Smoking_allowed'] = data['amenities'].str.contains('Smoking allowed')


# In[ ]:


data.drop(['amenities'], axis=1, inplace=True)


# ### Distance to centre

# The last feature to create is the distance to city centre, to see if this can be a predictor of price in this case.

# In[ ]:


import geopy.distance

def calc_distance(lat,long):
    berlin_centre = (52.5, 13.4)
    apartment=(lat,long)
    
    return geopy.distance.distance(berlin_centre,apartment).km 


# In[ ]:


data['distance'] = data.apply(lambda x: calc_distance(x.latitude, x.longitude), axis=1)


# ## Modeling

# In[ ]:


data.columns


# In[ ]:


data.drop(['latitude', 'longitude', 'id','host_id'
          ], axis=1, inplace=True)


# In[ ]:


# encode categorical columns

def encode_columns(column, data):
    
    data = pd.concat([data,pd.get_dummies(data[column],prefix=column)],axis=1)
    data.drop(column, axis=1, inplace=True)
    
    return data


# In[ ]:


categorical_columns = [ 
       'neighbourhood_group_cleansed', 'room_type', 
       'cancellation_policy'
]
    
for col in categorical_columns:
    data=encode_columns(col,data)


# In[ ]:


data.head(2)


# In[ ]:


# make binary columns the right type 

data['Laptop_friendly_workspace'] = data['Laptop_friendly_workspace'].astype(int)
data['TV'] = data['TV'].astype(int)
data['Hot water'] = data['Hot water'].astype(int)
data['Family_friendly'] = data['Family_friendly'].astype(int)
data['Hair_dryer'] = data['Hair_dryer'].astype(int)
data['Smoking_allowed'] = data['Smoking_allowed'].astype(int)


# In[ ]:


data.isna().sum()


# In[ ]:


data.dropna(inplace=True)


# In[ ]:


data.info()


# In[ ]:


data.shape


# ### Split data for modeling

# In[ ]:


y = data[["price"]]


# In[ ]:


X = data.drop(["price"], axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)


# In[ ]:


# scale data so that numerical variables have the same scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


randf = RandomForestRegressor()

randf.fit(X_train, y_train)
randf_prediction=randf.predict(X_test)


# In[ ]:


round(np.sqrt(mean_squared_error(y_test,randf_prediction)),2)


# In[ ]:


round(r2_score(y_test,randf_prediction),2)


# ### Cross-validation

# In[ ]:


import sklearn.metrics
from sklearn.model_selection import cross_validate


# In[ ]:


CV_randf=cross_validate(randf,X_train,y_train,scoring=["neg_mean_squared_error"],
                        cv=5)


# In[ ]:


print("The mean RMSE in the Cross-Validation is: {:.2f}%".
      format((np.sqrt(abs(np.mean(CV_randf["test_neg_mean_squared_error"]))))))


# In[ ]:


import xgboost as xgb


# In[ ]:


booster = xgb.XGBRegressor(colsample_bytree=0.6, gamma=0.2, learning_rate=0.1, 
                           max_depth=6, n_estimators=100, random_state=4)


booster.fit(X_train, y_train)

y_pred_train = booster.predict(X_train)
y_pred_test = booster.predict(X_test)


# In[ ]:


RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"RMSE: {round(RMSE, 2)}")


# In[ ]:


r2 = r2_score(y_test, y_pred_test)
r2
print(f"R2: {round(r2, 2)}")


# Using the xgboost improves the result slightly, but as we can see it's hard to predict accurately on the data due to the high variance in price. Our prediction is still much smaller than the sd so we are better off, and we explain about 55% of the price variance with the model, which can be used to recommend a price to someone wanting to post their apartment on Airbnb.
# 
# Let's see which features predict price the most (as selected by the xgboost model):

# In[ ]:


feat_importances = pd.Series(booster.feature_importances_, index=X.columns)
feat_importances.nlargest(15).sort_values().plot(kind='barh', color='darkblue', figsize=(10,5))
plt.xlabel('Relative Feature Importance with XGBoost')


# 
