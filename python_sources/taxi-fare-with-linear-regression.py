#!/usr/bin/env python
# coding: utf-8

# # About This Notebook
# This is my 3rd notebook in the world of Kaggle. Throughtout this notebook, I will try to keep things as simple as possible. I would be happy to take any query you might have after you finish exploring this kernel. If you find this notebook useful, you can check out my other two kernels thta you might find useful as well.

# # Outlines
# * [1. Problem Description and Objective](#1)
# * [2. Importing Packages and Collecting Data](#2)
# * [3. Missing Value Treatment](#3)
# * [4. Univariate Analysis](#4)
# * [5. Outliers Treatment](#5)
# * [6. Feature Engineering](#6)
# * [7. Bivariate Analysis](#7)
# * [8. Location Visualization](#8)
# * [9. Model Building](#9)
# * [10. End Note](#10)

# # 1.Problem Description and Objective <a id="1"></a>
# In this challange, we're asked to predict the amount of fare for a taxi ride in New York City given the pickup and dropoff locations, number of passengers in a ride, and the pickup data time for a ride. So fare amount is our target variable and rest of the variables are our predictor variables. Hence, its a supervised regression problem.
# 
# # 2.Importing Packages and Collecting Data <a id="2"></a>
# Since the train data contains about 55 million observations, its not feasible to deal with all of them. Hence, we would only use 5 million instances off 55 million instances to analyse and train our models later on.

# In[ ]:


# Import basic required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
plt.style.use('bmh')
sns.set_style({'axes.grid':False}) 

# Advanced visualization modules(datashader)
import datashader as ds
import datashader.transfer_functions as tf
from datashader.colors import viridis, inferno

# Folium visualization for geographical map
import folium as flm
from folium.plugins import HeatMap


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Downcasting data types to reduce momory consumption\ndtypes = {}\nfor key in ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']:\n    dtypes[key] = 'float32'\nfor key in ['passenger_count']:\n    dtypes[key] = 'uint8'\n    \n# Read in train and test data (5 million rows)\ntrain = pd.read_csv('../input/train.csv', nrows = 5_000_000, dtype = dtypes).drop('key', axis = 1)\ntest = pd.read_csv('../input/test.csv', dtype = dtypes)")


# In[ ]:


# Now check out the data types 
print('Dtypes after downcasting except pickup_datetime:')
display(train.dtypes)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# 'pickup_datetime' should be in datetime format. Let's convert it\n# Don't forget to set 'infer_datetime_format=True'. Otherwise it takes forever :)\ntrain['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], infer_datetime_format=True)\ntest['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], infer_datetime_format=True)")


# In[ ]:


# Current memory usage(in MB) by columns after conversion
print('Memory usage(in MB) by variables after conversion:')
display(np.round(train.memory_usage(deep = True)/1024**2, 4))


# In[ ]:


# Look at the data we're going to deal with
print('Preview train data:')
display(train.head())
print('Preview test data:')
display(test.head())


# # 3.Missing Value Treatment <a id="3"></a>

# In[ ]:


# Missing values in train data
print('Missing values in train data:')
display(train.isna().sum())

# Missing values in test data
print('Missing values in test data:')
display(test.isna().sum())


# Since we're dealing with 5 millions observations, dropping those 36 missing observations might not affect the model. However, there is no missing values in test data.

# In[ ]:


# Drop missing observations from train data.
train.dropna(how = 'any', axis = 0, inplace = True)
# Shape of the df after dropping missing rows
print('Shape of the df after dropping missing rows:{}'.format(train.shape))


# # 4.Univariate Analysis <a id="4"></a>
# Univariate analysis separately explores the distribution of each variable in a data set. It looks at the range of values, as well as the central tendency of the values. Univariate data analysis does not look at relationships between various variables (like bivariate and multivariate analysis) rather it summarises each variable on its own. Methods to perform univariate analysis will depend on whether the variable is categorical or numerical. For numerical variable, we would explore its shape of distribution (distribution can either be symmetric or skewed) using histogram and density plots. For categorical variables, we would use bar plots to visualize the absolute and proportional frequency distribution. Knowing the distribution of the feature values becomes important when you use machine learning methods that assume a particular type of it, most often Gaussian. **Let's starts off with our target variable:**

# ## 4.1 fare_amount

# In[ ]:


# Distrubution of target variable with skewness
fig, ax = plt.subplots(figsize = (14,6))
sns.distplot(train.fare_amount, bins = 200, color = 'firebrick', ax = ax)
ax.set_title('Distribution of fare_amount (skewness: {:0.5})'.format(train.fare_amount.skew()))
ax.set_ylabel('realtive frequency')
plt.show()


# Clearly our target variable is right skewed with a skewness over 4.5 that indicates the presence of outliers. Outliers will be in the following section.
# ## 4.2 passenger_count.
# Let's see the class distribution of passenger_count

# In[ ]:


# Class distribution of passenger_count
fig, ax = plt.subplots(figsize = (14,6))
class_dist = train.passenger_count.value_counts()
class_dist.plot(kind = 'bar', ax = ax)
ax.set_title('Class distribution of passenger_count')
ax.set_ylabel('absolute frequency')
plt.show()


# The class distribution is imbalanced since some classes outnumber some other classes. Most of the passengers like to travel alone. One important takeaway from the plot is that some  claasses(like 208, 129, and 51) turn out to be outliers. Since a taxi is not big enough to carry 208, 129 or 51 passengers, these instances would be removed in the outliers treatment section.

# # 5.Outliers Treatment <a id="5"></a>

# ## 5.1 fare_amount

# In[ ]:


# Look at the abnormalities using descritive stats
train.fare_amount.describe()


# fare_amount shouldn't be negative. So let's drop those negative intances from fare_amount.

# In[ ]:


# Drop fare_amount less than 0.
neg_fare = train.loc[train.fare_amount<0, :].index
train.drop(neg_fare, axis = 0, inplace = True)

# Rerun the descriptive stats
train.fare_amount.describe()


# Well! We're not done yet! Based on [this discussion](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/discussion/63319) on [real-world taxi fares in New York City](http://nymag.com/nymetro/urban/features/taxi/n_20286/), I'm also going to remove any fares less than $2.50 that appears to be the minimum fare, so any values in the training set less than this amount must be errors in data collection or entry.
# 
# I'll also remove any fares greater than $100. I'll justify this based on the limited number of fares outside this bounds, but it might be possible that including these values helps the model! I'd encourage you to try different values and see which works best.

# In[ ]:


# Drop rows greater than 100 and lesser than 2.5
fares_to_drop = train.loc[(train.fare_amount>100) | (train.fare_amount<2.5), :].index
train.drop(fares_to_drop, axis = 0, inplace = True)
print('Shape of train data after dropping outliers from fare_amount:{}'.format(train.shape))


# ## 5.2 latitudes and longitudes
# For the latitude and longitude columns, we can use statistics as well as our intuition for removing outliers. Here I'll find the 2.5% and 97.5% percentile values in each column and keep only the values close to that range. I would encourage to try out other methods like IQR and Z-score to see which works better.

# In[ ]:


# Check the 2.5 and 97.5 percentile of los nad lats
def percentile(variable):
    two_and_half = variable.quantile(0.25)
    ninty_seven_half = variable.quantile(0.975)
    print('2.5 and 97.5 percentile of {} is respectively: {:0.2f}, and {:0.2f}'.format(variable.name, two_and_half, ninty_seven_half))
    
percentile(train.pickup_latitude)
percentile(train.dropoff_latitude)
percentile(train.pickup_longitude)
percentile(train.dropoff_longitude) 


# Based on these values, we would remove outliers from lons and lats.

# In[ ]:


# For lats, our range is 40 to 42 degrees(with 40 and 42)
train = train.loc[train.pickup_latitude.between(left = 40, right = 42), :]
train = train.loc[train.dropoff_latitude.between(left = 40, right = 42), :]

# For lons, our range is -75 to -72 degrees(with 40 and 42)
train = train.loc[train.pickup_longitude.between(left = -75, right = -72), :]
train = train.loc[train.dropoff_longitude.between(left = -75, right = -72), :]
print('Shape of train data after after dropping outliers from lats and lons: {}'.format(train.shape))


# ## 5.3 passenger_count

# In[ ]:


# Check out the descriptive stats first
train.passenger_count.describe()


# Maximum number of passengers is 208. I don't think a taxi (or even a bus) can carry 208 passengers. This must be an error. Let's drop it. Also from univariate section, we are gonna drop passenger count of value 129, 51, 9, and 7.

# In[ ]:


# Drop passenger_count of 208 and 129, 9, and 7.
passenger_count_to_drop = train.loc[(train.passenger_count==208) | (train.passenger_count==129) | (train.passenger_count==9) | (train.passenger_count==7)].index
train.drop(passenger_count_to_drop, axis = 0, inplace = True)
print('Shape of train data after dropping outliers from passenger_count:{}'.format(train.shape))

# Let's check again the passenger_count
display(train.passenger_count.describe())


# Now the maximum no. of passengers is 6 which makes sense.

# # 6.Feature Engineering <a id="6"></a>
# In this section we would try to calculate the travelling distance based on latitudes and longitudes. Also we would like to extract pickup hour, day, date, month, and year from 'pickup_datetime' to see if they influnce the fare_amount. To do so we will merge train and test data.

# In[ ]:


# Merged train and test data across rows
merged = pd.concat([train,test], axis = 0, sort=False)


# ## 6.1 great_circle_distance
# The great-circle distance or orthodromic distance is the shortest distance between two points on the surface of a sphere, measured along the surface of the sphere (as opposed to a straight line through the sphere's interior). The distance between two points in Euclidean space is the length of a straight line between them, but on the sphere there are no straight lines.
# We can calculate the distance travelled by a passenger given the pickup and dropoff latitudes and longitudes using haversine formula **assuming the earth is a perfect sphere rather than an ellipsoid**. For more on great circle distance and  haversine [see](https://en.wikipedia.org/wiki/Great-circle_distance)

# In[ ]:


# Calculate great circle distance using haversine formula
def great_circle_distance(lon1,lat1,lon2,lat2):
    R = 6371000 # Approximate mean radius of earth (in m)
    
    # Convert decimal degrees to ridians
    lon1,lat1,lon2,lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Distance of lons and lats in radians
    dis_lon = lon2 - lon1
    dis_lat = lat2 - lat1
    
    # Haversine implementation
    a = np.sin(dis_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dis_lon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dis_m = R*c # Distance in meters
    dis_km = dis_m/1000 # Distance in km
    return dis_km

# Create a column named greate_circle_distance
merged['great_circle_distance'] = great_circle_distance(merged.pickup_longitude, merged.pickup_latitude, merged.dropoff_longitude, merged.dropoff_latitude)


# ## 6.2 euclidean_distance
# The Euclidean distance or Euclidean metric is the "ordinary" straight-line distance between two points in Euclidean space. **It assumes the surface to be flat (rather than spherical or ellipsoidal)**. Hence it should not be as accurate as great circle or [vincenty distance](https://en.wikipedia.org/wiki/Vincenty%27s_formulae) (that assumes the earth to be ellipsoid). But we would take that to see if there is any correlation between this distance and fare_amount. Since euclidean distance is for cartesian plane, we need to convert our longitudes and latitudes into cartesian coordinates. See [here](https://en.wikipedia.org/wiki/Euclidean_distance) for more on euclidean distance. **If a = (x1,y1) and b = (x2,y2), then the euclidean distance between a and b is given by np.sqrt(x1-x2)^2+(y1-y2)^2)**

# In[ ]:


# Convert lons and lats into cartesian coordinates. Assume the earth as sphere not ellipsoid
R = 6371000 # Approximate mean radius of earth (in m)
 # lons and lats must be in radians
lon1,lat1,lon2,lat2 = map(np.radians, [merged.pickup_longitude, merged.pickup_latitude, merged.dropoff_longitude, merged.dropoff_latitude])
merged['pickup_x'] = R*np.cos(lon1)*np.cos(lat1)
merged['pickup_y'] = R*np.sin(lon1)*np.cos(lat1)
merged['dropoff_x'] = R*np.cos(lon2)*np.cos(lat2)
merged['dropoff_y'] = R*np.sin(lon2)*np.cos(lat2)

# Now calculate the euclidean distance
x1 = merged['pickup_x']
y1 = merged['pickup_y']
x2 = merged['dropoff_x']
y2 = merged['dropoff_y']
merged['euclidean_distance'] = (np.sqrt(( x1 - x2)**2 + ( y1 - y2)**2))/1000 # in km


# ## 6.3 manhattan_distance
# Manhattan distance is the sum of the absolute values of the differences of the coordinates. **If a = (x1,y1) and b = (x2,y2), then the manhattan distance between a and b is given by |x1 - x2| + |y1 - y2|**. For more on manhattan distance [see](https://en.wikipedia.org/wiki/Taxicab_geometry)

# In[ ]:


# Calculate manhattan distance from x and y coordinates
merged['manhattan_distance'] = (np.abs(x1 - x2) + np.abs(y1 - y2))/1000 # in km


# ## 6.4 abs_lon_diff, and abs_lat_diff
# Now we also want to create two variables by taking absolute differences of longitudes and latitudes to see how they perform as predictors for fare_amount.

# In[ ]:


# Create two variables taking absolute differences of lons and lats
merged['abs_lon_diff'] = np.abs(merged.pickup_longitude - merged.dropoff_longitude)
merged['abs_lat_diff'] = np.abs(merged.pickup_latitude - merged.dropoff_latitude)


# ## 6.5 pickup_hour, day, date, month, and year
# Let's extract pickup_hour, pickup_day, pickup_date, pickup_month, and pickup_year from 'pickup_datetime' to check if there is any association between 'fare_amount', and them. 

# In[ ]:


# Extract pickup_hour, day, date, month, and year from pickup_datetime.
merged['pickup_hour'] = merged.pickup_datetime.dt.hour
merged['pickup_date'] =  merged.pickup_datetime.dt.day
merged['pickup_day_of_week'] =  merged.pickup_datetime.dt.dayofweek
merged['pickup_month'] =  merged.pickup_datetime.dt.month
merged['pickup_year'] =  merged.pickup_datetime.dt.year


# In[ ]:


# Let's see the current dtypes and total memory consumption by variables in MB
print('Current Data Types:')
display(merged.dtypes)
print('\n Total memory consumption in MB: {}'.format(np.sum(merged.memory_usage(deep = True)/1024**2)))


# We will convert pickup_hour, pickup_date, pickup_day_of_week, pickup_month to uint8. All the distances  will be converted to float32 dtypes. We would also delete key, fare_amount (due to merging), and pickup_datetime and see how much memory we can save.

# In[ ]:


# Drop variables 
merged.drop(['key', 'pickup_datetime'], axis = 1, inplace = True)


# In[ ]:


# Downcasting variables
merged.loc[:, ['pickup_hour', 'pickup_date', 'pickup_day_of_week', 'pickup_month']] = merged.loc[:, ['pickup_hour', 'pickup_date', 'pickup_day_of_week', 'pickup_month']].astype(np.uint8)
merged.loc[:, ['great_circle_distance', 'euclidean_distance', 'manhattan_distance']] = merged.loc[:, ['great_circle_distance', 'euclidean_distance', 'manhattan_distance']].astype(np.float32)
merged.loc[:, ['pickup_year']] = merged.loc[:, ['pickup_year']].astype('int16')

# Check total memory consumption after downcasting
print('Total memory consumption after downcasting in MB: {}'.format(np.sum(merged.memory_usage(deep = True)/1024**2)))


# In[ ]:


# Let's separate train and test data again
train_df = merged.iloc[0:4892576, :]
test_df = merged.iloc[4892576:, :] 
test_df.drop('fare_amount', axis = 1, inplace = True) # Due to concatenation


# # 7.Bivariate Analysis <a id="7"></a>
# Being the most important part, bivariate analysis tries to find the relationship between two variables. We will look for correlation or association between our predictor and target variables. Bivariate analysis is performed for any combination of categorical and numerical variables. The combination can be: Numerical & Numerical, Numerical & Categorical and Categorical & Categorical. Different methods are used to tackle these combinations during analysis process. If the combination is continuous numerical vs continuous numerical, we would use regression plot to observe the correlation. One the other hand, for categorical vs continuous numerical combination, we would observe the association with box plots, bar plots and pivot tables.

# In[ ]:


# Let's see which variables have the strongest and weakest correlation with fare_amount
corr = train_df.corr().sort_values(by='fare_amount', ascending=False)
fig, ax = plt.subplots(figsize = (20,12))
sns.heatmap(corr, annot = True, cmap ='BrBG', ax = ax, fmt='.2f', linewidths = 0.05, annot_kws = {'size': 17})
ax.tick_params(labelsize = 15)
ax.set_title('Correlation with fare_amount', fontsize = 22)
plt.show()


# So it turns out the distance variables (as expected) has the strongest positive correlation with fare_amount followed by abs_lon_diff, and abs_lat_diff. On the other hand pickup_latitude and dropoff latitude have the strongest negative correlation among the variables. Some variables have correlation zero or almost zero.
# 
# ## 7.1 continuous numerical vs continuous numerical variable
# Here all the distance variables, abs_lon_diff, abs_lat_diff, pickup_x, pickup_longitude, dropoff_x, dropoff_longitude, pickup_y, dropoff_y, dropoff_latitude, pickup_latitude are continuous numerical variables. Hence we would use regression plot to observe the correlation between these variables and fare_amount. To speed up the process, we will take only 10000 instances off 4.5 million instances (scatter plot is very slow for large samples) to plot regression plot between our predictor and target variables.

# In[ ]:


# Plot subplots of regression plots
continuous_var = train_df.iloc[0:10000, :].select_dtypes(include = ['float32', 'float64']).drop('fare_amount', axis = 1)
fig, axes = plt.subplots(7,2, figsize = (40,80))
for ax, column in zip(axes.flatten(), continuous_var.columns):
    x = continuous_var[column]
    y = train_df.fare_amount.iloc[0:10000]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    sns.regplot(x = continuous_var[column], y = y, ax = ax, line_kws={'label':'r: {}\np: {}'.format(r_value,p_value)})
    ax.set_title('{} vs fare_amount'.format(column), fontsize = 36)
    fig.suptitle('Regression Plots', fontsize = 45)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 22)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 22)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.legend(loc = 'best', fontsize = 32)
fig.delaxes(ax = axes[6,1])
fig.tight_layout(rect = [0, 0.03, 1, 0.97])


# We can see the some correlation correlation values are reduced. This is due to downsampling (i.e., we only use 10k samples off over 4.5 million samples to plot regression plots). A p value of 0 indicates that we're 100% confident that there is a statistically significant correlation between target variables and fare_amount.

# #### Create a function to plot regression plot
# def regression_plot(predictor, target, xlabel, title):
#     fig = sns.jointplot(x = predictor, y = target, kind = 'reg')
#     fig.fig.set_size_inches(12,7)
#     ax = plt.gca()
#     ax.set_xlabel(xlabel, size = 15)
#     ax.set_ylabel('fare_amount ($usd)', size = 15)
#     ax.set_title(title, size = 18, fontweight='bold')
#     plt.tight_layout()

# ## 7.2 categorical vs continuous numerical variable
# Let's dig deeper to explore the relationship between categorical variables and fare_amount. A box plot can reveal if there is any association between predictor categorical variables and target variable. If any catgorical variabes are highly associated with fare_amount, mean fare_amount should be different across different groups of those categorical variables. We can see this pattern using pivot table. Again we will only use 10000 instances to speed up the rendering.

# In[ ]:


# Extract categorical variable first
cat_var = train_df.iloc[0:10000, :].select_dtypes(include = ['uint8'])
cat_var = pd.concat([cat_var, train_df.pickup_year.iloc[0:10000]], axis = 1)

# A box plot to visualize the association between fare_amount and categorical variables
fig, axes = plt.subplots(3,2,figsize = (20,25))
for ax, column in zip(axes.flatten(), cat_var.columns):
    sns.boxplot(x = cat_var[column], y = train_df.fare_amount.iloc[0:10000], ax = ax)
    ax.set_title('{} vs fare_amount'.format(column), fontsize = 22)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 16)
    ax.set_xlabel(column, fontsize = 18)
    ax.set_ylabel('fare_amount', fontsize = 18)
    fig.suptitle('Association with categorical variables', fontsize = 26)
fig.tight_layout(rect = [0, 0.03, 1, 0.97])


# None of the catagorical variables seem to be highly associated with fare_amount except pickup_year. Since pickup_year is highly associated with fare_amount, mean fare_amount should be different across the classes (groups) of pickup_year. We can visualize this pattern using pivot table.

# In[ ]:


# Let's group mean fare_amount by pickup_year to see if there is a pattern.
pivot_year = pd.pivot_table(train_df, values = 'fare_amount', index = 'pickup_year', aggfunc = ['mean'])
print('Mean fare_amount across the classes of pickup_year: \n{}'.format(pivot_year))
# or train.fare_amount.groupby([train.pickup_year]).mean()


# In[ ]:


# A bar plot would be more helpful to visualize this pattern
fig, ax = plt.subplots(figsize = (15,5))
pivot_year.plot(kind = 'bar', legend = False, color = 'firebrick', ax = ax)
ax.set(title = 'pickup_year vs mean fare_amount', ylabel= 'mean fare_amount')
plt.show()


# Well! There is infact a pattern. There is an increasing trend in mean fare from yeay 2009 to 2015 (this can be expected). Since fare_amount is positively correlated to pickup_year (upward trend), you might need to pay more in the coming year onwards.

# # 8.Locaion Visualization <a id="8"></a>
# Since we're dealing with location data, we may want to visualize the pickup and dropoff locations. We can also plot the map of NYC and then plot data points on it to see the concentrations of most pickups and dropoffs. As thre are over 4.5 million observations, its quite impossible to plot longitudes vs latitudes (or location) using conventional plotting packages. Hence we will use 'datashader' to plot longitudes vs latitudes (i.e., locations) using all the 4.5 million instances.
# 
# ## 8.1 pickup_locations

# In[ ]:


# x_range and y_range for pickup_locations
print('x_range and y_range for pickup_locations:')
print(train_df.pickup_longitude.min(), train_df.pickup_longitude.max())
print(train_df.pickup_latitude.min(), train_df.pickup_latitude.max())

# x_range and y_range for dropoff_locations
print('\nx_range and y_range for dropoff_locations:')
print(train_df.dropoff_longitude.min(), train_df.dropoff_longitude.max())
print(train_df.dropoff_latitude.min(), train_df.dropoff_latitude.max())


# So we've x_range (-74.99828, -72.06699) and y_range (40.850838, 41.998108). However, most of the rides occur within the x_range (-74.05, -73.7) and y_range (40.6, 40.85). So for nice visualation, we'll skip the outliers to plot both pickup and dropoff locations. And we would also use the same range to plot dropoff locations like pickup locations.

# In[ ]:


# Create a function to plot longitudes vs latitudes of rides
def plot_location(lon,lat, c_map):
    # Initial datashader visualization configuration
    pickup_range = dropoff_range = x_range, y_range = ((-74.05, -73.7), (40.6, 40.85))
    # Initiate canvas and create grid
    cvs = ds.Canvas(plot_width = 1080, plot_height = 600, x_range = x_range, y_range = y_range)
    agg = cvs.points(train, lon, lat)
    # Create image map with custom color map
    img = tf.shade(agg, cmap = c_map, how = 'eq_hist')
    return tf.set_background(img, 'black')


# In[ ]:


# Show image map of pickup locations with viridis color map
plot_location('pickup_longitude', 'pickup_latitude', viridis)


# In[ ]:


# Show image map of pickup locations with inferno color map
plot_location('pickup_longitude', 'pickup_latitude', inferno)


# We can see some patterns especially on the main streen of NYC. That means more rides start from the main streen than any oher streets. However,  we're only visualizing 4.5 million points. If we would use more data, the trend would be more evident. Now it would be more useful if we could visualize NYC map and then plot those pickup points. This would give us much concise information about the pickup locations on the streets of NYC. For this, we will use package 'folium'. However, for faster rendering (and some limitations of folium), we would plot only 20000 points in folium map. We will also use heatmap instead of scatter points to capture the pattern better.

# In[ ]:


# Create a function to plot folium heatmap
def plot_map(lat, lon):
    # Lat and lon of nyc to plot the map of nyc
    map_nyc = flm.Map(location = [40.7141667, -74.0063889], zoom_start = 12, tiles = "Stamen Toner")
    # creates a marker for nyc
    flm.Marker(location = [40.7141667, -74.0063889], icon = flm.Icon(color = 'red'), popup='NYC').add_to(map_nyc)
    # Plot heatmap of 20000 lats and lons points
    lat_lon = train.loc[0:20000, [lat, lon]].values
    HeatMap(lat_lon, radius = 10).add_to(map_nyc)
    #map_nyc.save('HeatMap.html')
    return map_nyc


# In[ ]:


# Plot street map of NYC and then plot heatmap of pickup locations on it.
plot_map('pickup_latitude', 'pickup_longitude')


# That's beautiful, isn't it? Look like some pickups are from water. That's might be errors (or outliers). Again this heatmap shows most pickups are from the main street of NYC and a few of them are from 'John F Kenedy Int'l Airport' (zoom in to see it).
# 
# ## 8.2 dropoff_locations
# Let's visualize dropoff locations with datashader and then plot heatmap of lats and lons of dropoffs in the streets of NYC with folium.

# In[ ]:


# Show image map of pickup locations with inferno color map
plot_location('dropoff_longitude', 'dropoff_latitude', inferno)


# In[ ]:


# viridis color map is even better to capture patterns
plot_location('dropoff_longitude', 'dropoff_latitude', viridis)


# Again like pickups, most of the drops were to the main street of NYC than any other streets. Let's create heatmap of dropoff lats and lons on the map of NYC.

# In[ ]:


# Plot street map of NYC and then plot heatmap of dropoffs lats and lons on it.
plot_map('dropoff_latitude', 'dropoff_longitude')


# Again, like pickups, dropoffs are mostly concentrated on the main street of NYC and just a few to 'John F Kenedy Int'l Airport' (zoom in to see it).

# # 9.Model Building <a id="9"></a>
# To keep things as simple as possible, we would only try linear regression. I would encourage you to try no-linear models (like random forest, gradient boosting etc).

# In[ ]:


# Get the data ready for training and predicting
y_train = train_df.fare_amount
X_train = train_df.drop(['fare_amount'], axis = 1)
X_test = test_df


# In[ ]:


# Train and predict using linear regression
from sklearn.linear_model import LinearRegression

# Instantiate linear regression object
linear_reg = LinearRegression()

# Train with the objt
linear_reg.fit(X_train, y_train)

# Make prediction
y_pred = linear_reg.predict(X_test)


# In[ ]:


# Create csv file for submission
submission = pd.DataFrame()
submission['key'] = test.key
submission['fare_amount'] = y_pred
submission.to_csv('sub_with_linear_reg', index = False)


# # 10.End Note <a id="10"></a>
# **This submission scored 4.26176 on leaderboard.** Yes that's not a great score! But considering the fact that we have only used linear regression, its not a bad score either. May be trying out random forest, gradient boosting or xgboost might improve the score. Any suggestion or query is welcomed. Last but not the least, some upvotes would be appreciated upon finding this kernel useful.
