#!/usr/bin/env python
# coding: utf-8

# ## Because I dig food.
# 
# I thought this might be an interesting data set to dig into. Typically when I go into things like this, I usually let the data take me to a question rather than just throw out questions.
# 
# So lets see where we go.

# In[77]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import os
from collections import Counter
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import geopandas as gpd
from shapely.geometry import Point

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv('../input/FAO.csv',  encoding='ISO-8859-1')
print(df.shape)
df.head()


# Thre is quite a bit of information in here. Lets start with seeing how many countries we have in here and a little comparison.

# In[3]:


countries = Counter(df.Area)
print("Number of Countries : {}".format(len(countries)))
fig, ax=plt.subplots(1,1, figsize=(18,8));
df.Area.hist(ax=ax, xrot=60);


# 
# It looks like we have 174 different countries in the data but not all have the same number of instances.

# In[4]:


## Getting the variations in food and feed
Counter(df.Element)


# ## FOOD AND FEED
# 
# There are two different elements in the data. Lets take a look at how they trend.

# In[5]:


## food feed mean
ff_mean = df.groupby('Element').mean()
ff_med = df.groupby('Element').median()


# In[6]:


## drop unused columns
ffmean_by_date = ff_mean.drop(['Area Code','Item Code','Element Code','latitude','longitude'],axis=1).T
ffmed_by_date = ff_med.drop(['Area Code','Item Code','Element Code','latitude','longitude'],axis=1).T


# In[7]:


## re-index the years to date time
nidx = []
for i in ffmean_by_date.index:
    nidx.append(pd.to_datetime(i[1:]))
ffmean_by_date.index = nidx
ffmed_by_date.index = nidx


# In[8]:


fig, ax=plt.subplots(1,2,figsize=(16,8))
ffmed_by_date.plot(ax=ax[0]);
ax[0].set_ylabel('Median of Total (1000s of Tonnes)');
ffmean_by_date.plot(ax=ax[1]);
ax[1].set_ylabel('Mean of Total (1000s of Tonnes)');


# So it looks like over the years, the mean and median of total tons of food and feed have increased which is what we would expect to see given population increase and demands. Interestingly, the median is significantly smaller than the mean here.
# Perhaps this means fewer countries are supplying most of the food / feed? Lets keep going in this direction.

# ## BY COUNTRY
# 
# I want to look at this two different ways so the first thing we will do is group by the AREA and find the sum of all Food/Feed Items.
# 
# Then we will take a look at the MEAN over the years.

# In[9]:


## sum the items by count and drop unneccesary columns
sum_ff_bycnt = df.groupby('Area').sum().drop(['Item Code','Element Code','Area Code','latitude','longitude'],axis=1)
sum_ff_bycnt.head()


# In[10]:


## get the mean for the latitudes and longitudes for later for each area (should be pretty similar)
mean_lat_lon_bycnt = df.groupby('Area').mean()[['latitude','longitude']]
mean_lat_lon_bycnt.head()


# So now we have the summed food/feed of all items per country and a separate frame for the average latitude and longitude.

# In[11]:


## Take the mean of the sums year over year
year_item_mean = sum_ff_bycnt.mean(axis=1)
print(year_item_mean.sort_values(ascending=False).head(5))
print(year_item_mean.sort_values(ascending=False).tail(5))


# In[12]:


fig, ax=plt.subplots(1,1,figsize=(14,8))
year_item_mean.sort_values(ascending=False).iloc[:30].plot(kind='bar', ax=ax, rot=90);
plt.ylabel('1000s of Tonnes (Food & Feed)');


# Once again, we are working off the year over year mean of the sums of all the items food+feed for each country.
# In my opinion, I think that is the most sound way to aggregate the data and looking at the first 30,  it still highlights the distribution of the counts of tons in total food / feed. China Mainland nearly doubles the US in this regard, and population information aside, that is staggering. 
# 
# What do the mean trends for these 5 countries look like over the years?

# In[13]:


## get top 5 from index
cnt = year_item_mean.sort_values(ascending=False).index[:5].values
# print(cnt)
top5 = sum_ff_bycnt.T
top5.index = nidx
fig, ax=plt.subplots(1,1,figsize=(14,8))
for c in cnt:
    top5[c].plot(ax=ax, legend=True);
plt.title('Top 5 Countries - Sum of all Items Year Over Year');
plt.ylabel('Food+Feed : 1000s Tonnes');


# Generally, it appears the top five continue to grow and this chart really highlights China's growth that jumped in the 1990s. 
# 
# What about the bottom 5?

# In[14]:


## get bottom 5 from index
cnt = year_item_mean.sort_values(ascending=False).index[-5:].values
# print(cnt)
bot5 = sum_ff_bycnt.T
bot5.index = nidx
fig, ax=plt.subplots(1,1,figsize=(14,8))
for c in cnt:
    bot5[c].plot(ax=ax, legend=True);
plt.title('Bottom 5 Countries - Sum of all Items Year Over Year');
plt.ylabel('Food+Feed : 1000s Tonnes');


# Even the bottom five countries show growth which would be consistent with over all population growth.
# 
# Lets pull those lats and lons back in the data set and overlay feed+food with a map.

# In[15]:


## pul the mean lats and longs into the sums DF
sum_ff_bycnt['lat'] = mean_lat_lon_bycnt['latitude']
sum_ff_bycnt['lon'] = mean_lat_lon_bycnt['longitude']

## using panadas geometry
geometry = [Point(xy) for xy in zip(sum_ff_bycnt.lon, sum_ff_bycnt.lat)]
crs = {'init': 'epsg:4326'}
gitems = gpd.GeoDataFrame(sum_ff_bycnt, crs=crs, geometry=geometry)

## world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
## Just a quick and dirty /100 normalization to make the plot below more feasible.
sum_ff_bycnt['Year_Mean'] = year_item_mean/100 #np.min(year_item_mean)
sum_ff_bycnt.head()


# In[16]:


fig, ax=plt.subplots(1,1,figsize=(18,9))
base = world.plot(color='white', edgecolor='black',ax=ax);
# gitems.plot(ax=ax)
src = ax.scatter(sum_ff_bycnt.lon, sum_ff_bycnt.lat, marker='o', 
           s = sum_ff_bycnt.Year_Mean, c=sum_ff_bycnt.Year_Mean,
           cmap=plt.get_cmap('jet'), alpha=0.4)
plt.colorbar(src);
plt.title('Year over Year Sum of Average Food+Feed  :  1000s of Tonnes / 100');


# There are probably some other cools ways to interpret this data but I like this geo map above. 
# 
# It really highlights the top 5 countries with respect to the others well.
# 
# There are some strange spots - such as the few over the Pacific Ocean to the middle left on the map.
# 
# I wonder where those are stemming from?
# 

# In[17]:


weird_points = sum_ff_bycnt[ ((sum_ff_bycnt['lat'] < 0) & (sum_ff_bycnt['lat'] > -25)) & (sum_ff_bycnt['lon'] < -130) ]
weird_points


# Ooooh okay. Yeah, a quick confirmation with google maps (which Im not showing here) helps us out.

# ## Different Items
# 
# I wonder what the top items are?
# 
# We can basically use the same code above to see the top items as well.

# In[18]:


items = Counter(df.Item)
print('Number of Items : {}'.format(len(items)), '\n', items)


# In[19]:


## sum the items by count and drop unneccesary columns
sum_items = df.groupby('Item').sum().drop(['Item Code','Element Code','Area Code','latitude','longitude'],axis=1)
sum_items.head()


# In[20]:


year_item_mean2 = sum_items.mean(axis=1)
print(year_item_mean2.sort_values(ascending=False).head(5))
print(year_item_mean2.sort_values(ascending=False).tail(5))


# We know from above that not every country produces every time, so it would be beneficial to find a way to creatively normalize these sums.

# In[21]:


fig, ax=plt.subplots(1,1,figsize=(14,8))
year_item_mean2.sort_values(ascending=False).iloc[:30].plot(kind='bar', ax=ax, rot=90);
plt.title('Mean Year over Year of Total Items - Non Normalized');


# Cereals seem strange to me but Im not sure what exactly that entails.
# 
# Sometimes you dig into data and see what you would kind of expect. Regardless of how cereals are classified, they are definitely made from processed grains so I guess its no surprise that would be so large for food and feed.

# In[22]:


## get top 5 from index
cnt2 = year_item_mean2.sort_values(ascending=False).index[:5].values
# print(cnt)
top5i = sum_items.T
top5i.index = nidx
fig, ax=plt.subplots(1,1,figsize=(14,8))
for c in cnt2:
    top5i[c].plot(ax=ax, legend=True);
plt.title('Top 5 Items - Sum of all Items Year Over Year');
plt.ylabel('Number of Items');


# The early 90's saw an interesting jump in both cereals and milk here. Nothing comes to memory and a brief google search didnt tell me much on the history of why that might be the case.

# ## Modeling
# 
# Lets take a pause from the analysis here and pick something we have looked at. 
# 
# Lets take a look at the food and feed production of cereal in the U.S. and China.

# In[23]:


us_ff = df[df.Area == 'United States of America'][df.Item == 'Cereals - Excluding Beer']
us_ff.head()
# print(us_ff.columns)


# In[24]:


## convert to a time series
us_ff_ts = us_ff.drop(['Area Abbreviation', 'Area Code', 'Area', 'Item Code', 'Item',
                       'Element Code', 'Element','Unit', 'latitude', 'longitude', ], axis=1).T
us_ff_ts.index = nidx
us_ff_ts.columns = ['Feed','Food']

cm_ff = df[df.Area == 'China, mainland'][df.Item == 'Cereals - Excluding Beer']
cm_ff_ts = cm_ff.drop(['Area Abbreviation', 'Area Code', 'Area', 'Item Code', 'Item',
                       'Element Code', 'Element','Unit', 'latitude', 'longitude', ], axis=1).T
cm_ff_ts.index = nidx
cm_ff_ts.columns = ['Feed','Food']

us_ff_ts.head()


# In[25]:


fig, ax=plt.subplots(1,2,figsize=(18,8), sharey=True)
us_ff_ts.plot(style=['b-','r-'],ax=ax[0]);
cm_ff_ts.plot(style=['b-','r-'],ax=ax[1]);
plt.ylabel('1000s Tonnes');
ax[0].set_title('US Food and Feed - Cereals');
ax[1].set_title('China Food and Feed - Cereals');


# Now that is interesting! Quiet a different dynamic for both countries.
# 
# For the U.S. we see the uptick in Food in the 80's but it doesnt compare to the amount in Feed.
# 
# For China, we see an incredible jump in both Food and Feed due (assumingly) to the population boom.
# 
# A real point of interest here is the exponential swing in the 2010's for China in Feed perhaps from them entering the Global Trade market further as costs of business are low there.

# ## Modeling
# 
# Lets take a look at a simple SARIMAX model for China since it is more interesting than the US.

# In[78]:


import statsmodels.api as sma
import statsmodels.graphics as smg
import sklearn.metrics as skm


# In[27]:


## First take a look at the log level of the items to 
## see what possible serial correlation exists
_ = smg.tsaplots.plot_acf(np.log(cm_ff_ts.Feed))
_ = smg.tsaplots.plot_pacf(np.log(cm_ff_ts.Feed))


# For China and Feed it looks like we have serial correlation out to 1 lag that we can incorporate into our model.

# In[55]:


## modeling the first difference at the log level
mod = sma.tsa.SARIMAX(endog=np.log(cm_ff_ts.Feed), order = (1,1,0), simple_differencing=False)
fit = mod.fit()
print(fit.summary())


# In[56]:


_ = fit.plot_diagnostics(figsize=(12,8))


# If we taje a quick look at the diagnostics of our model and data we find the following (moving from L to R, T to B):
# - The standardized residual appears to be mostly white noise.
# - The distribution appears fairly normal relative to sampling.
# - Their might be some outliers in the data based on the Normal Q-Q plot.
# - There is no further evidence of serial correlation in the residual.

# In[73]:


def pred_forc(fit):
    ## make sure simple_differencing in SARIMAX = False
    pred = fit.predict()
    pred.iloc[0] = np.log(cm_ff_ts.Feed.iloc[0]) #fill initial point since pred makes it 0
    pred_lvl = np.exp(pred)
    return pred_lvl


# In[81]:


pred_lvl = pred_forc(fit)
fig, ax=plt.subplots(1,1,figsize=(12,8))
pred_lvl.plot(ax=ax, legend=True, label='Prediction');
cm_ff_ts.Feed.plot(ax=ax, legend=True);
plt.title('China Mainland - Feed');
## Average Percent Error 
ape = (cm_ff_ts.Feed.values-pred_lvl.values).sum()/cm_ff_ts.Feed.values.sum()
plt.xlabel('Average Percent Error = {:.3f}%'.format(ape*100));


# Using an First Difference, AR 1 model we can predict the Feed with a low Average Percent Error.
# 
# I should dig into creating a forecast as well.

# In[84]:


# fit.forecast(steps=20).plot()


# In[ ]:




