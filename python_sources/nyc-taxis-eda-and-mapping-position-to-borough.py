#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook, I will use some census tract information to get a mapping between the pickup 
# and dropoff locations and associated borough/county. The geography around the New York City area
# means that there are generally large jumps in travel times when going between boroughs and between New York and New Jersey. Features such as the George Washington Bridge, the Triboro Bridge, the Queens Midtown Tunnel, Holland and Lincoln Tunnels provide chokepoints where traffic jams are very common. Including this information may help improve models without needing to train something like a neural net to find where these common traffic problem areas are.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv',index_col=0)
test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv',index_col=0)


# ## Pickup Locations
# 
# To display the locations as a histogram, we actually want a logarithmic color scale because the data is heavily weighted toward taxi trips in Manhattan below about 96th St. The densest spot for pickups is in Midtown.
# 
# There is also a decent density of pickups along the waterfronts across from Manhattan.
# 
# The largest clusters outside Manhattan seem to be LaGuardia and JFK Airports and the highways leading up to the airports.

# In[ ]:


latmin = 40.48
lonmin = -74.28
latmax = 40.93
lonmax = -73.65
ratio = np.cos(40.7 * np.pi/180) * (lonmax-lonmin) /(latmax-latmin)
from matplotlib.colors import LogNorm
fig = plt.figure(1, figsize=(8,ratio*8) )
hist = plt.hist2d(train.pickup_longitude,train.pickup_latitude,bins=199,range=[[lonmin,lonmax],[latmin,latmax]],norm=LogNorm())
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Pickup Locations')
plt.colorbar(label='Number')
plt.show()


# ## Dropoff Locations
# 
# The dropoff locations appear to do a better job covering the whole city but are still very Manhattan-centric.
# We also see a number of dropoffs at Newark Airport (close to (-74.2,40.7)) and in Jersey City and Hoboken.

# In[ ]:


fig = plt.figure(1, figsize=(8,ratio*8) )
hist = plt.hist2d(train.dropoff_longitude,train.dropoff_latitude,bins=199,range=[[lonmin,lonmax],[latmin,latmax]],norm=LogNorm())
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Dropoff Locations')
plt.colorbar(label='Number')
plt.show()


# ## Getting Census Tract Information
# 
# I recently uploaded a dataset with some census demographic and economic information
# and also some data to map from coordinates to census tract.
# 
# For now, I'll just add in the census tract information. Tracts are a good way to package the data
# since they often follow political and physical boundaries. In New York, this will allow us
# to figure out things such as which trips require crossing a river.
# These types physical boundaries will potentially cause significant increases in trip time.

# In[ ]:


def get_census_data():
    blocks = pd.read_csv('../input/new-york-city-census-data/census_block_loc.csv')
    #blocks = blocks[blocks.County.isin(['Bronx','Kings','New York','Queens','Richmond'])]
    census = pd.read_csv('../input/new-york-city-census-data/nyc_census_tracts.csv',index_col=0)
    blocks['Tract'] = blocks.BlockCode // 10000
    blocks = blocks.merge(census,how='left',left_on='Tract',right_index=True)
    #blocks = blocks.dropna(subset=['Borough'],axis=0)
    return blocks,census

def convert_to_2d(lats,lons,values):
    latmin = 40.48
    lonmin = -74.28
    latmax = 40.93
    lonmax = -73.65
    lon_vals = np.mgrid[lonmin:lonmax:200j]
    lat_vals = np.mgrid[latmin:latmax:200j]
    map_values = np.zeros([200,200],'l')
    dlat = lat_vals[1] - lat_vals[0]
    dlon = lon_vals[1] - lon_vals[0]
    for lat,lon,value in zip(lats,lons,values):
        lat_idx = int(np.rint((lat - latmin) / dlat))
        lon_idx = int(np.rint((lon-lonmin) / dlon ))        
        if not np.isnan(value):
            map_values[lon_idx,lat_idx] = value
    return lat_vals,lon_vals,map_values

blocks,census = get_census_data()
blocks_tmp = blocks[blocks.County_x.isin(['Bronx','Kings','New York','Queens','Richmond'])]
map_lats, map_lons,map_tracts_nyc = convert_to_2d(blocks_tmp.Latitude,blocks_tmp.Longitude,blocks_tmp.Tract)
map_lats, map_lons,map_tracts = convert_to_2d(blocks.Latitude,blocks.Longitude,blocks.Tract)


# ## Coordinates to Census Tract Mapping
# 
# Now I'll define a function to map from latitude and longitude to census tract.
# Anything defined outside the area right near New York will be given tract number 0 so that 
# we can easily find those trips.

# In[ ]:


def get_tract(lat,lon):
    latmin = 40.48
    lonmin = -74.28
    latmax = 40.93
    lonmax = -73.65
    dlat = (latmax-latmin) / 199
    dlon = (lonmax-lonmin) / 199
    if (latmin<lat<latmax) and (lonmin<lon<lonmax):
        lat_idx = int(np.rint((lat - latmin) / dlat))
        lon_idx = int(np.rint((lon-lonmin) / dlon )) 
        return map_tracts[lon_idx,lat_idx]
    return 0


# In[ ]:


train.info()


# In[ ]:


train['pu_tracts'] = np.array([get_tract(lat,lon) for lat,lon in zip(train.pickup_latitude,train.pickup_longitude)])
train['do_tracts'] = np.array([get_tract(lat,lon) for lat,lon in zip(train.dropoff_latitude,train.dropoff_longitude)])

test['pu_tracts'] = np.array([get_tract(lat,lon) for lat,lon in zip(test.pickup_latitude,test.pickup_longitude)])
test['do_tracts'] = np.array([get_tract(lat,lon) for lat,lon in zip(test.dropoff_latitude,test.dropoff_longitude)])


# ## What tracts have the most pickups and dropoffs?
# 
# Now that we have added in the tract information, we can see pickup and dropoff stats for individual tracts.

# In[ ]:


pickups = train['pu_tracts'].value_counts()
dropoffs = train['do_tracts'].value_counts()


# Here, I plot out the tracts with the 5 largest numbers of pickups. The color scale just has everything in order but the absolute scale is meaningless.

# In[ ]:


top_tracts = [x for x in pickups.index.values[0:5]]
top_tracts.reverse()
values = 0.250*(1-np.isin(map_tracts_nyc,top_tracts+[0]))

for i in range(len(top_tracts)):
    values += (i+7)*(map_tracts_nyc==top_tracts[i])/11

fig = plt.figure(1,figsize=[7,7])
im = plt.imshow(values.T,origin='lower',cmap='jet')
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Most Common Pickup Points in NYC Limits')
plt.colorbar(im,fraction=0.045, pad=0.04)
plt.show()


# It turns out that we could have probably guessed the biggest pickup points. In descending order, it looks like they are:
# 
#   1. LaGuardia Airport
#   2. Penn Station
#   3. JFK Airport
#   4. Central Park
#   5. Columbus Circle
#   
# The airports and Penn Station are transportation hubs, so it would make sense for them to have many pickups. They all have large numbers of people with luggage that might be inconvenient on public transportation.
# 
# Central Park also has many pickups but it likely only has so many because it takes up a large area in the middle of Manhattan. Other tracts will just be much smaller.
# 
# Next, we can look at dropoffs.

# In[ ]:


top_tracts = [x for x in dropoffs.index.values[0:5]]
top_tracts.reverse()
values = 0.250*(1-np.isin(map_tracts_nyc,top_tracts+[0]))

for i in range(len(top_tracts)):
    values += (i+7)*(map_tracts_nyc==top_tracts[i])/11

fig = plt.figure(1,figsize=[7,7])
im = plt.imshow(values.T,origin='lower',cmap='jet')
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Most Common Dropoff Points in NYC Limits')
plt.colorbar(im,fraction=0.045, pad=0.04)
plt.show()


# The top dropoff points are all in Manhattan. Penn Station and Central Park are the top two. The other three appear to be a piece of the West Side including part of Chelsea, a section of Midtown east of Penn Station (it might be close to Grand Central), and finally a small bit of Midtown between Times Square and Columbus Circle.
# 
# ## Densest Pickup/Dropoff Points
# 
# With our tract mapping, we also can calculate an approximate area for each tract to then find the densest places where taxi pickups and dropoffs are done. This will get more accurate as a finer grid of points is used.
# 
# This is not perfect, however, since many tracts also include a large amount of water.

# In[ ]:


areas = blocks.Tract.value_counts()


# In[ ]:


pu_area_norm = pickups
do_area_norm = dropoffs

pu_area_norm = pd.concat([pu_area_norm,areas],join='inner',axis=1)
do_area_norm = pd.concat([do_area_norm,areas],join='inner',axis=1)

pu_area_norm['areas'] = pu_area_norm.pu_tracts/pu_area_norm.Tract
do_area_norm['areas'] = do_area_norm.do_tracts/do_area_norm.Tract

pu_areas = pu_area_norm.areas.sort_values(ascending=False)
do_areas = do_area_norm.areas.sort_values(ascending=False)


# In[ ]:


top_tracts = [x for x in pu_areas.index.values[0:10]]
top_tracts.reverse()
values = 0.250*(1-np.isin(map_tracts_nyc,top_tracts+[0]))

for i in range(len(top_tracts)):
    values += (i+7)*(map_tracts_nyc==top_tracts[i])/16
    

fig = plt.figure(1,figsize=[7,7])
im = plt.imshow(values.T,origin='lower',cmap='jet')
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Densest Pickup Points in NYC Limits')
plt.colorbar(im,fraction=0.045, pad=0.04)
plt.show()


# In[ ]:


top_tracts = [x for x in do_areas.index.values[0:10]]
top_tracts.reverse()
values = 0.250*(1-np.isin(map_tracts_nyc,top_tracts+[0]))

for i in range(len(top_tracts)):
    values += (i+7)*(map_tracts_nyc==top_tracts[i])/16

fig = plt.figure(1,figsize=[7,7])
im = plt.imshow(values.T,origin='lower',cmap='jet')
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Densest Dropoff Points in NYC Limits')
plt.colorbar(im,fraction=0.045, pad=0.04)
plt.show()


# For both pickups and dropoffs, the densest spots for taxi activity are around Midtown.
# 
# Midtown is also the biggest business district and home to many tourist attractions, so this is probably what we might guess if we didn't have any data.

# ## Borough/County Mapping
# 
# Near New York City, the boundaries between many of the areas are rivers. There are not many crossings across the Hudson and East Rivers, so bridges and tunnels create chokepoints that might have large effects on travel times.

# In[ ]:


train['PU_Location'] =  train.pu_tracts//1000000
fips_codes = {36061:'Manhattan',36081:'Queens',36047:'Brooklyn',
              36005:'Bronx',36085:'Staten Island',36059:'Nassau',36119:'Westchester',
              34017:'NJ',34013:'NJ',34003:'NJ',34039:'NJ',
              34031:'NJ',34023:'NJ',34025:'NJ',0:'Unknown'
             }
train.PU_Location = train.PU_Location.map(fips_codes)

test['PU_Location'] = test.pu_tracts//1000000
test.PU_Location = test.PU_Location.map(fips_codes)

train.PU_Location.value_counts()


# We see that in this dataset, taxi pickups are mainly in Manhattan, with Queens second. The Queens pickups are mostly at the airports, which explains why there is such a large difference between Queens and Brooklyn.
# 
# Below, we see similar stats for dropoffs, but with Brooklyn and Queens almost even. There are far fewer dropoffs in the Bronx compared to Queens and Brooklyn. Some of this may be because much of the Bronx is closer to Manhattan, so public transportation options are more attractive. 
# 
# There are also several thousand dropoffs in New Jersey. Our histograms showed that these are concentrated at Newark Airport, Jersey City, and Hoboken.

# In[ ]:


train['DO_Location'] =  train.do_tracts//1000000
train.DO_Location = train.DO_Location.map(fips_codes)
test['DO_Location'] =  test.do_tracts//1000000
test.DO_Location = test.DO_Location.map(fips_codes)
train.DO_Location.value_counts()


# ## Travel Times for Travel Between Different Counties

# In[ ]:


pd.set_option('display.max_rows', 200)
print(train.groupby(['PU_Location','DO_Location'])['trip_duration'].describe()[['count','25%','50%','75%']])
pd.reset_option('display.max_rows')


# # A Very Simple Model
# 
# From the leaderboard, we see that just taking the mean duration gives us a score of 0.892.
# 
# Can we beat that using just borough/county mappings?
# 
# We can easily implement this by converting the mapping into integers and then training a decision tree.
# I'll do this using sklearn with a DecisionTreeRegressor with the default parameters.
# 
# This would be a very simple model, but it should give us somewhat better results. Instead of taking the
# overall mean, we would be getting the mean for each combination (except for maybe combining combinations that have
# a very small number of samples).
# 
# To check the performance, I will use a 10-fold cross validation.

# In[ ]:


loc_map = {"Manhattan":0,"Queens":1,"Brooklyn":2,"Bronx":3,
           "NJ":4,"Unknown":5,"Staten Island":6,"Nassau":7,
           "Westchester":8}

train.PU_Location = train.PU_Location.map(loc_map)
train.DO_Location = train.DO_Location.map(loc_map)

test.PU_Location = test.PU_Location.map(loc_map)
test.DO_Location = test.DO_Location.map(loc_map)


# In[ ]:


targets = np.log(train.trip_duration+1)
Xall = train[['PU_Location',"DO_Location"]]

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
nsplits = 10
tree_model = DecisionTreeRegressor()
kf = KFold(n_splits = nsplits,random_state=999)

err_train_ave = 0 
err_test_ave = 0
err_train_std = 0
err_test_std = 0
counter = 0
for train_index, test_index in kf.split(Xall):
    X_train, X_test = Xall.iloc[train_index], Xall.iloc[test_index]
    y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]
    tree_model.fit(X_train,y_train)
    pred_train = tree_model.predict(X_train)
    pred_test = tree_model.predict(X_test)
    err_train = np.sqrt(mean_squared_error(y_train,pred_train))
    err_test = np.sqrt(mean_squared_error(y_test,pred_test))
    err_train_ave += err_train
    err_test_ave += err_test
    err_train_std += err_train*err_train
    err_test_std += err_test * err_test
    print('%i Train Err: %f, Validation Err: %f'%(counter,err_train,err_test))
    counter+=1

err_train_ave /= nsplits
err_test_ave /= nsplits
err_train_std /= nsplits
err_test_std /= nsplits
err_train_std = np.sqrt((err_train_std - err_train_ave*err_train_ave)*nsplits / (nsplits-1))
err_test_std = np.sqrt((err_test_std - err_test_ave*err_test_ave)*nsplits / (nsplits-1))

print("\nTrain Ave: %f Std. Dev: %f" %(err_train_ave,err_train_std))
print("Validation Ave: %f Std. Dev: %f"%(err_test_ave,err_test_std))


# ## Results
# 
# We see here that we get an average score of 0.707 for both the train and validation sets.
# This tells us that our model is not overtraining. This is what we expect since we are ignoring most of the 
# information in the data. The samples are heavily Manhattan-centric, so while this improves upon the 
# mean-value model, it is still not very good. To keep improving this, we will want to improve 
# our model and also do some feature engineering.
# 
# ## Additional features
# 
# Obviously, we will want to do something directly with the latitude and longitude points. For simplicity, rather
# than include both the start and end points, we may want to instead include the start position,
# the distance, and the direction.
# 
# The time should also matter a lot, so we will want to engineer some features out of the start time. The time 
# of day and day of weeek (or combined into a time of week) should be powerful features. There will be large
# differences between commute times and non-commute times as well as weekday/weekend differences.

# # A Simple Linear Model
# 
# We can start by building a model for linear regression. I'll do this by getting the distance travelled in each trip.

# In[ ]:


train.head()


# In[ ]:


train['dlon'] = (train.dropoff_longitude-train.pickup_longitude) * np.pi/180 *                np.cos((train.dropoff_latitude+train.pickup_latitude) * 0.5 * np.pi/180)
train['dlat'] = (train.dropoff_latitude-train.pickup_latitude) * np.pi/180
Re = 6371 # Earth radius in km

train['dist'] = Re*np.hypot(train.dlon,train.dlat)
train['pu_do_code'] = train.PU_Location + 10 * train.DO_Location

test['dlon'] = (test.dropoff_longitude-test.pickup_longitude) * np.pi/180 *                np.cos((test.dropoff_latitude+test.pickup_latitude) * 0.5 * np.pi/180)
test['dlat'] = (test.dropoff_latitude-test.pickup_latitude) * np.pi/180
test['dist'] = Re*np.hypot(test.dlon,test.dlat)
# Encode Pickup/Dropoff location into a single number
test['pu_do_code'] = test.PU_Location + 10 * test.DO_Location


train.head()


# ## Histogram of Distances
# 
# We actually see that there are several peaks at larger distances seen here, probably corresponding to the airports.

# In[ ]:


plt.hist(train.dist,bins=100,range=[0,50])
plt.yscale('log')
plt.show()


# The metric that we're using to score this competition is the squared difference between the logarithm of the duration and the logarithm of the prediction. Since we generally expect time to be proportional to distance, here we'll expect that log(time) is proportional to log(distance). Thus, we'll want the logarithm in our fit.
# 
# I'll also add several other terms to allow for the fact that the scaling may not be perfect at very short or very long distances due to things like traffic lights, highway access times, etc. Here, I'm using log(d), sqrt(d), d, d^1.5, and d^2.
# 
# It actually turns out that for the linear model, the location encoding doesn't really add anything, so I've removed it, but the code is still there. I've done it this way to avoid pitfalls of blindly running a one hot encoding function, where we might get different columns in the test and training sets.

# In[ ]:


train['ldist'] = np.log(train.dist + 0.01)
train['d2'] = train.dist*train.dist
train['d1_2'] = np.sqrt(train.dist)
train['d3_2'] = train.dist * train.d1_2

test['ldist'] = np.log(test.dist + 0.01)
test['d2'] = test.dist*test.dist
test['d1_2'] = np.sqrt(test.dist)
test['d3_2'] = test.dist * test.d1_2

X_all = train[['dist','ldist','d2','d1_2','d3_2']]
X_test = test[['dist','ldist','d2','d1_2','d3_2']]

#for i in range(9):
#    for j in range(9):
#        code = 1.0*(train['pu_do_code'] == (10*i+j))
#        if code.sum() > 10:
#            X_train['PU_DO_%i%i'%(i,j)] = code

y_all = np.log(train['trip_duration']+1)


# In[ ]:


X_all.head()


# ## Model Validation
# 
# I have a huge number of samples and only 5 features, so I'm just going to use a default linear regression model with no regularization. I haven't checked this but it is highly likely that regularization will basically never do anything useful here.
# 
# I'm also not normalizing the feature set, which might hurt the fit process a bit but it seems like things converge pretty readily.
# 
# For validation, I'm using a 10-fold cross validation.

# In[ ]:


from sklearn.linear_model import LinearRegression

nsplits = 10
lin_model = LinearRegression()
kf = KFold(n_splits = nsplits,random_state=999)
err_train_ave = 0 
err_test_ave = 0
err_train_std = 0
err_test_std = 0
counter = 0
for train_index, test_index in kf.split(X_all):
    X_train, X_val = X_all.iloc[train_index], X_all.iloc[test_index]
    y_train, y_val = y_all.iloc[train_index], y_all.iloc[test_index]
    lin_model.fit(X_train,y_train)
    pred_train = lin_model.predict(X_train)
    pred_test = lin_model.predict(X_val)
    err_train = np.sqrt(mean_squared_error(y_train,pred_train))
    err_test = np.sqrt(mean_squared_error(y_val,pred_test))
    err_train_ave += err_train
    err_test_ave += err_test
    err_train_std += err_train*err_train
    err_test_std += err_test * err_test
    print('%i Train Err: %f, Validation Err: %f'%(counter,err_train,err_test))
    counter+=1
err_train_ave /= nsplits
err_test_ave /= nsplits
err_train_std /= nsplits
err_test_std /= nsplits
err_train_std = np.sqrt((err_train_std - err_train_ave*err_train_ave)*nsplits / (nsplits-1))
err_test_std = np.sqrt((err_test_std - err_test_ave*err_test_ave)*nsplits / (nsplits-1))

print("\nTrain Ave: %f Std. Dev: %f" %(err_train_ave,err_train_std))
print("Validation Ave: %f Std. Dev: %f"%(err_test_ave,err_test_std))


# ## Running on the Test Set
# 
# Now I'll fit the model to the full training set and then calculate the predicted values for the test set.
# 
# I'll expect a score of around 0.515, which isn't bad for such a simple model given a complicated data set. I haven't even looked at timing information yet, so there are going to be many ways to improve.

# In[ ]:


lin_model.fit(X_all,y_all)
#print(lin_model.coef_)
#print(lin_model.intercept_)
#print(X_test[X_test.isnull().any(axis=1)])
log_test_pred = lin_model.predict(X_test)

test_pred = np.exp(log_test_pred) - 1
#print(test_pred)
test['pred'] = [ np.max(x,0) for x in test_pred ]
X_out = test[['pred']]
X_out.columns.values[0] = 'trip_duration'
#print(X_out)
#X_out['trip_duration'] = test_pred
X_out.to_csv('LinearModel.csv')

