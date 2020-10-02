#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import seaborn as sns


# Let's begin by inputting our `business.json` data from the Yelp Dataset.

# In[ ]:


businesses = pd.read_json('../input/business.json', lines=True)


# That's a lot of data! Now, we don't want to preform analysis on cities that have sparse Yelp data, so which city has the most representation in this dataset?

# In[ ]:


businesses.groupby('city').count().sort_values('state', ascending=False)['state'].head(20)


# In[ ]:


# Constructing Las Vegas polygon (so that we can include nearby cities of North LV, Henderson, etc...)
polygon = pd.DataFrame(data=np.array([[36.320491, -115.384039], [35.965241, -115.364835],                                       [35.948735, -114.919536], [36.320152, -114.945260]]), columns=['lat', 'lon'])
polygon


# In[ ]:


def in_vegas(x, y):
    """Whether a longitude-latitude (x, y) pair is in Las Vegas polygon:
    36.320491, -115.384039 upper left
    36.320152, -114.945260 upper right
    35.965241, -115.364835 lower left
    35.948735, -114.919536 lower right
    
    ADAPTED FROM http://alienryderflex.com/polygon/"""
    
    j = len(polygon.index) - 1
    lon = polygon['lon'].values # lon = x
    lat = polygon['lat'].values # lat = y
    odd_nodes = False
    for i in range(j + 1):
        if (lat[i] < y and lat[j] >= y) or (lat[j] < y and lat[i] >= y):
            if (lon[i] + (y - lat[i]) / (lat[j] - lat[i]) * (lon[j] - lon[i]) < x):
                odd_nodes = not odd_nodes
        j = i        
    return odd_nodes


# In[ ]:


vegas = businesses
vegas['in'] = vegas.apply(lambda row : in_vegas(row['longitude'], row['latitude']), axis=1) &     vegas.apply(lambda row : in_vegas(row['longitude'], row['latitude']), axis=1)
vegas = vegas[vegas['in'] == True].drop(columns=['in'])


# In[ ]:


len(vegas.index)


# In[ ]:


vegas.groupby('city').count().sort_values('state', ascending=False)['state'].head(10)


# In[ ]:


def businesses_plot(t):
    plt.scatter(t['longitude'], t['latitude'], s=2, alpha=0.2)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Businesses in Las Vegas area')
    
plt.figure(figsize=(10, 10))
businesses_plot(vegas)


# In[ ]:


vegas = vegas[vegas['review_count'] > 20]


# In[ ]:


len(vegas.index)


# In[ ]:


vegas['num_stars'] = vegas['stars'].apply(round)
plt.figure(figsize=(10, 10))
sns.scatterplot(x='longitude', y='latitude', hue='stars', data=vegas, palette='RdBu', alpha=0.5, s=20)
plt.title('Businesses with 20+ Yelp Reviews Plotted Based on Aggregated Number of Stars')
plt.legend();


# In[ ]:


mcdonalds = vegas[vegas['name'] == 'McDonald\'s'].sort_values('stars', ascending=False)
plt.figure(figsize=(10, 10))
sns.scatterplot(x='longitude', y='latitude', data=vegas, palette='Blues', s=10, alpha=0.5)
sns.scatterplot(x='longitude', y='latitude', hue='stars', data=mcdonalds, palette='Reds_r', s=90)
plt.title('McDonald\'s Locations in Las Vegas, NV from Yelp Reviews')
plt.legend(title='Yelp Rating for McDonald\'s Locations', loc=1);


# In[ ]:


# vegas.to_csv(r'vegas_business_data.csv')
# mcdonalds.to_csv(r'mcdonalds_yelp_reviews.csv')


# In[ ]:


# Getting list of business_ids for McDonald's locations in Las Vegas, NV
# list(mcdonalds['business_id'])


# In[ ]:


plt.figure(figsize=(10, 10))
sns.scatterplot(x='longitude', y='latitude', hue='stars', data=vegas, palette='RdBu', size=0.2, alpha=0.8)
sns.scatterplot(x='longitude', y='latitude', hue='stars', data=mcdonalds, palette='RdBu', s=200)
plt.legend().remove()
plt.xlim(-115.190249, -115.127894)
plt.ylim(36.096868, 36.130218)


# In[ ]:


from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


# In[ ]:


X = np.array(vegas[['longitude', 'latitude']]) * 1000000
y = np.array(vegas[['stars']]).flatten() * 10
y


# In[ ]:


h = 900  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
n_neighbors = 15

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(xx / 1000000, yy / 1000000, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0] / 1000000, X[:, 1] / 1000000, c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
#     plt.xlim(xx.min() / 1000000, xx.max() / 1000000)
#     plt.ylim(yy.min() / 1000000, yy.max() / 1000000)
    plt.scatter(mcdonalds['longitude'], mcdonalds['latitude'], s=750, c='red', marker='*')
    plt.xlim(-115.190249, -115.127894)
    plt.ylim(36.096868, 36.130218)
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('longitude')
    plt.ylabel('latitude')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




