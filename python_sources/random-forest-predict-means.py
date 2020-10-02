#!/usr/bin/env python
# coding: utf-8

# The purpose of this kernel is to compare linear regression algorithm and random forest regression algorithm accuracy on the dataset.
# 
# Firstly, let's import needed modules:

# In[2]:


import numpy as np
import pandas as pd


# # Load data
# 
# Let's first look at the data fields:

# In[3]:


df_income = pd.read_csv('../input/us-household-income-stats-geo-locations/kaggle_income.csv', encoding='ISO-8859-1')
df_income.info()


# And the data column means:

# In[4]:


df_income.describe()


# Clean data

# In[5]:


df_income[df_income['sum_w']>50000]


# # Understand data
# 
# ## Histogram
# 
# We firstly draw histogram for Mean and sum_w columns:

# In[6]:


df_income.hist(column='Mean')
df_income.hist(column='sum_w')


# As I found there are very few sum_w greater than 2000, we remove all the rows with huge sum_w:

# In[7]:


df_income = df_income[df_income['sum_w'] < 2000]
df_income.hist(column='sum_w')


# ## Means by state
# 
# To better understand the data, we firstly show it on map. As it's data about USA, we use Basemap to draw USA map. Now firstly, we draw the data density on map:

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

fig1 = plt.figure(figsize=(16,8))

m = Basemap(
    llcrnrlon=-119,
    llcrnrlat=22,
    urcrnrlon=-64,
    urcrnrlat=49,
    projection='lcc',
    lat_1=33,
    lat_2=45,
    lon_0=-95
)
m.drawmapboundary(zorder=0)
m.fillcontinents(color='#ffffff', lake_color='aqua', zorder=1)
m.drawcountries(linewidth=1.5)
m.drawcoastlines()
m.drawstates()

x, y = m(
    np.array(df_income['Lon']),
    np.array(df_income['Lat'])
) # load data into map
ax = fig1.add_subplot(111)
ax.scatter(
    x,
    y,
    3,
    marker='o',
    color='k',
    zorder=1.5
) # draw dots on map

plt.title('Data Distribution')


# We can see that most of the data are of east cost. Now we show the heat map:

# In[9]:


fig2 = plt.figure(figsize=(16,8))

m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
m.drawmapboundary(zorder=0)
m.fillcontinents(color='#ffffff', lake_color='aqua', zorder=1)
m.drawcountries(linewidth=1.5)
m.drawcoastlines()
m.drawstates()

#normalize sum_w
df_income['sum_w_n'] = 20 * (1 + (df_income['sum_w'] - df_income['sum_w'].mean()) / df_income['sum_w'].std())

ax = fig2.add_subplot(111)
scatter1 = ax.scatter(
    x,
    y,
    3,
#     s=np.array(df_income['sum_w_n']), # dot size
    c=np.array(df_income['Mean']), # You can change here to ALand or AWater
    cmap=plt.cm.YlOrRd,
    marker='o',
    zorder=1.5
) # draw dots on map

plt.colorbar(scatter1)
plt.title('Heat map')


# Now we begin to draw state map based on state means:

# In[10]:


from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon

fig3 = plt.figure(figsize=(16,8))

m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, projection='lcc', lat_1=33, lat_2=45, lon_0=-95)

shp_info = m.readshapefile('../input/usa-map-shape/st99_d00','states',drawbounds=True)
colors={}
statenames=[]
cmap = plt.cm.YlOrRd
vmin = 48000
vmax = 90000

series_state = df_income.groupby(['State_Name'])['Mean'].mean()

for shapedict in m.states_info:
    statename = shapedict['NAME']
    if statename not in ['District of Columbia','Puerto Rico']:
        pop = series_state[statename]
        colors[statename] = cmap(np.sqrt((pop - vmin) / (vmax - vmin)))[:3]
    statenames.append(statename)

ax = plt.gca()
for nshape, seg in enumerate(m.states):
    if statenames[nshape] not in ['District of Columbia','Puerto Rico']:
        color = rgb2hex(colors[statenames[nshape]])
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
        
plt.colorbar(ax.imshow(np.arange(90000, 48000, -1).reshape(4200, 10), cmap='YlOrRd'))
plt.title('Heat map by State')


# # Split data
# 
# Firstly, we need to split data into training data and test data:

# In[11]:


from sklearn.model_selection import train_test_split

# Convert string to NaN
df_income['Area_Code_Num'] = pd.to_numeric(df_income['Area_Code'], errors='coerce')
# Convert NaN to 0
df_income['Area_Code_Num'].fillna(0, inplace=True)
# Convert string to number
dummies_Type = pd.get_dummies(df_income['Type'], prefix= 'Type')
dummies_Primary = pd.get_dummies(df_income['Primary'], prefix= 'Primary')
# Add number columns
df_income_new = pd.concat([df_income, dummies_Type, dummies_Primary], axis=1)
# Drop string columns
df_income_new.drop(['id', 'State_Name', 'State_ab', 'County', 'City', 'Place', 'Type', 'Primary', 'Median', 'Stdev', 'Area_Code', 'Mean'], axis=1, inplace=True)
# Split data into training data and cross validation data
X = df_income_new
y = df_income[['Mean']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)
X_train.head(10)


# # Linear Regression
# 
# Let's first try the Linear Regression. Before running linear regression, we need to normalize data.

# In[12]:


from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler

mapper = DataFrameMapper([(X_train.columns, StandardScaler())])
scaled_features = mapper.fit_transform(X_train.copy())
X_train_scaled = pd.DataFrame(scaled_features, index=X_train.index, columns=X_train.columns)
scaled_features_test = mapper.fit_transform(X_test.copy())
X_test_scaled = pd.DataFrame(scaled_features_test, index=X_test.index, columns=X_test.columns)
X_train_scaled.head(10)


# Then we train the data, and use the train result to predict test data.

# In[13]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print('Explained variance score: %.2f' % explained_variance_score(y_test, y_pred))
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))


# Apparently, the result is not so good,  we got `0.05` score here. Linear regression is not suit for this dataset.
# 
# # Random Forest
# 
# Now we change to random forest.

# In[14]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

model = RandomForestRegressor(random_state=0, n_jobs=-1)
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)

print('Explained variance score: %.2f' % explained_variance_score(y_test, y_pred))
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))


# We got a score of `0.47`. Seems much better now, but still not good enough.
# 
# # K Nearest Neighbors
# 
# Try for K-Neighbors:

# In[15]:


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=18)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('Explained variance score: %.2f' % explained_variance_score(y_test, y_pred))
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))


# I tried the nearest neighbour algorithm, but still did not get a better result.
# 
# 
# I am new learner of machine learning. And don't know how to improve this score better, if you have any good idea, please teach me. Thanks!

# # Nearest Point
# 
# Get the nearest point by using KDTree.

# In[25]:


from scipy import spatial

pt = [35, -100] # Put random point here 
tree = spatial.KDTree(df_income[['Lat', 'Lon']])
dist, ind = tree.query(pt, k=4)
df_income.loc[ind]


# In[26]:


from sklearn.neighbors import KDTree

tree = KDTree(df_income[['Lat', 'Lon']])
dist, ind = tree.query([pt], k=4)
df_income.loc[ind[0]]


# In[27]:


fig1 = plt.figure(figsize=(16,8))

m = Basemap(
    llcrnrlon=-119,
    llcrnrlat=22,
    urcrnrlon=-64,
    urcrnrlat=49,
    projection='lcc',
    lat_1=33,
    lat_2=45,
    lon_0=-95
)
m.drawmapboundary(zorder=0)
m.fillcontinents(color='#ffffff', lake_color='aqua', zorder=1)
m.drawcountries(linewidth=1.5)
m.drawcoastlines()
m.drawstates()

array_x = np.array(df_income.loc[ind[0]]['Lon'])
array_x = np.append(array_x, -100)
array_y = np.array(df_income.loc[ind[0]]['Lat'])
array_y = np.append(array_y, 35)
x, y = m(
    array_x,
    array_y
) # load data into map
ax = fig1.add_subplot(111)
ax.scatter(
    x,
    y,
    30,
    marker='o',
    color='k',
    zorder=1.5
) # draw dots on map

plt.title('Data Distribution')


# This is strange, seems very far from the calculated result. Does someone know why?

# In[19]:


from math import cos, asin, sqrt

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a))

def closest(data, v):
    return min(data, key=lambda p: distance(v['lat'],v['lon'],p['lat'],p['lon']))

tempDataList = [{'lat': 39.7612992, 'lon': -86.1519681}, 
                {'lat': 39.762241,  'lon': -86.158436 }, 
                {'lat': 39.7622292, 'lon': -86.1578917}]

v = {'lat': 39.7622290, 'lon': -86.1519750}
print(closest(tempDataList, v))

