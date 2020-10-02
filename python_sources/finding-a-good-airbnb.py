#!/usr/bin/env python
# coding: utf-8

# THE PURPOSE OF THIS NOTEBOOK IS TO EXPLORE INSIGHTS HIDDEN IN THE DATA AND LATER MAKE PREDICTIONS AND COMPARE ACCURACIES

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
data.shape


# In[ ]:


data.head()


# I was actually curious if price had any relation with number of reviews. Which is un true in this case

# In[ ]:


import seaborn as sns
sns.lmplot( x="price", y="number_of_reviews", data=data, fit_reg=False, hue='room_type', legend=True)
 


# In[ ]:


# No space
sns.jointplot(x=data["price"], y=data["number_of_reviews"], kind='kde', color="grey", space=0)


# Brooklyn and Manhattan comprises of highest airbnb listings in New York!

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(14,8))
sns.countplot(data.sort_values('neighbourhood_group').neighbourhood_group,palette='Set3',alpha=0.8)
plt.title('Neighbourhood wise Airbnb listings in NYC')
plt.xlabel('Neighbours')
plt.ylabel('Registered Airbnbs')
plt.show()


# In[ ]:


plt.figure(figsize=(14,8))
sns.countplot(data.sort_values('room_type').room_type,palette='Set1',alpha=0.8)
plt.title('Room Types Airbnb listings in NYC')
plt.xlabel('Room Types')
plt.ylabel('Registered Airbnbs')
plt.show()


# Shared rooms are much less compared to Entire homes or rooms listing. Shared room is also a niche of Hostels and demand seems to be low. 

# In[ ]:


manhattan = data.loc[data['neighbourhood_group'] == "Manhattan"]
brooklyn = data.loc[data['neighbourhood_group'] == "Brooklyn"]

plt.stem(manhattan["room_type"], manhattan["price"], use_line_collection = True)
plt.ylim(0, 11000)


# In[ ]:


plt.stem(brooklyn["room_type"], brooklyn["price"], use_line_collection = True)
plt.ylim(0, 11000)


# Manhattan and Brooklyn seems to have equal distribution amongst room type for all listings in New York
# 

# In[ ]:


manhattan.describe()


# In[ ]:


brooklyn.describe()


# In[ ]:


import matplotlib.pyplot as plt
x = sns.pairplot(data)
x


# In[ ]:


plt.figure(figsize=(14,8))
sns.distplot(data[data.neighbourhood_group=='Manhattan'].price,color='black',hist=False,label='Manhattan')
sns.distplot(data[data.neighbourhood_group=='Brooklyn'].price,color='green',hist=False,label='Brooklyn')
sns.distplot(data[data.neighbourhood_group=='Queens'].price,color='blue',hist=False,label='Queens')
sns.distplot(data[data.neighbourhood_group=='Staten Island'].price,color='maroon',hist=False,label='Staten Island')
sns.distplot(data[data.neighbourhood_group=='Long Island'].price,color='lavender',hist=False,label='Long Island')
plt.title('Price Distribution for different Boroughs for less than 3000')
plt.xlim(0,3000)
plt.show()


# We can see Manhatten's curve is skewed showing its prices are relatively higher

# 
# 
# plt.figure(figsize=(14,8))
# sns.distplot(data[data.neighbourhood_group=='Manhattan'].reviews_per_month,color='black',hist=False,label='Manhattan')
# sns.distplot(data[data.neighbourhood_group=='Brooklyn'].reviews_per_month,color='green',hist=False,label='Brooklyn')
# sns.distplot(data[data.neighbourhood_group=='Queens'].reviews_per_month,color='blue',hist=False,label='Queens')
# sns.distplot(data[data.neighbourhood_group=='Staten Island'].reviews_per_month,color='maroon',hist=False,label='Staten Island')
# sns.distplot(data[data.neighbourhood_group=='Long Island'].reviews_per_month,color='lavender',hist=False,label='Long Island')
# plt.title('Reviews per Month for different Boroughs')
# plt.xlim(0,10)
# plt.show()

# Queens and Brooklyn listings gets higher number of reviews!

# In[ ]:


fig, ax = plt.subplots(figsize=(14,14))


img=plt.imread('../input/new-york-city-airbnb-open-data/New_York_City_.png', 0)
coordenates_to_extent = [-74.258, -73.7, 40.49, 40.92]
ax.imshow(img, zorder=0, extent=coordenates_to_extent)

scatter_map = sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group',s=20, ax=ax, data=data)
ax.grid(True)
plt.legend(title='Neighbourhood Groups')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(14, 14))

cmap = plt.get_cmap('jet') # ColorMaps
c = data.price           # color, sequence, or sequence of color, optional
alpha = 0.5                # The alpha blending value, between 0 (transparent) and 1 (opaque).
label = "data"
price_heatmap = ax.scatter(data.longitude, data.latitude, label=label, c=c, 
                            cmap=cmap, alpha=0.4)

plt.title("Heatmap by Price $")
plt.colorbar(price_heatmap)
plt.grid(True)
plt.show()


# Lets draw the same but for price under $500

# In[ ]:


data_2 = data[data.price < 500]

fig, ax = plt.subplots(figsize=(14, 14))

cmap = plt.get_cmap('jet') # ColorMaps
c = data_2.price           # color, sequence, or sequence of color, optional
alpha = 0.5                # The alpha blending value, between 0 (transparent) and 1 (opaque).
label = "data_2"
price_heatmap = ax.scatter(data_2.longitude, data_2.latitude, label=label, c=c, 
                            cmap=cmap, alpha=0.4)

plt.title("Heatmap by Price $")
plt.colorbar(price_heatmap)
plt.grid(True)
plt.show()


# We can see most of the listings lie within the range of 50 to 150 as we see the map!

# Firstly Deploying Linear Regression

# In[ ]:


data = data.drop(['name','id','host_name','last_review'],axis=1,inplace=True)
data = data['reviews_per_month']=data['reviews_per_month'].replace(np.nan, 0)


# In[ ]:


import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor


# In[ ]:



# Fit label encoder
le = preprocessing.LabelEncoder()                                            
le.fit(data_3['neighbourhood_group'])
data_3['neighbourhood_group']=le.transform(data_3['neighbourhood_group'])   

 # Transform labels to normalized encoding.
le = preprocessing.LabelEncoder()
le.fit(data_3['neighbourhood'])
data_3['neighbourhood']=le.transform(data_3['neighbourhood'])

le = preprocessing.LabelEncoder()
le.fit(data_3['room_type'])
data_3['room_type']=le.transform(data_3['room_type'])

data_3.sort_values(by='price',ascending=True,inplace=True)

