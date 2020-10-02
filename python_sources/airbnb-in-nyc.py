#!/usr/bin/env python
# coding: utf-8

# # Analyzing New York City Airbnb data
# Here we will use [open data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) 
# to analyze some metrics based on price and location of Airbnb listings in New York City,
# ultimately estimating the gross monthly income as a function of a listing's price per night.

# First import the necessary libraries and load in the datasets we'll be using.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

# plotting
from matplotlib import pyplot as plt
import seaborn as sns

# interactive mapping
import geopandas as gpd
import folium 

# load in datasets
# airbnb data
listings=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
# to create interactive maps
neighborhoods=gpd.read_file('/kaggle/input/new-york-shapefile-16/cb_2016_36_tract_500k.shp')


# Let's take a look at what's available in these dataframes.

# In[ ]:


listings.head()


# In[ ]:


neighborhoods.head()


# Before we do anything, let's remove possible duplicates from `listings`.

# In[ ]:


# drop any possible duplicates
listings.drop_duplicates(inplace=True)


# Let's make one dataframe that contains the airbnb listings and the geospatial data.

# In[ ]:


# use the latitude and longitude info in the airbnb dataframe to copy it to a GeoDataFrame, 
# then merge it with the neighborhoods geodataframe
geolistings = gpd.GeoDataFrame(listings, geometry=gpd.points_from_xy(listings.longitude, listings.latitude))
geolistings.crs = neighborhoods.crs
listings=gpd.sjoin(neighborhoods,geolistings)


# Cool, now we have a dataframe that has all airbnb and geo listings! 
# 
# Before we map anything, let's get a sense of the data.

# In[ ]:


listings.describe()


# Wow! There is a listing at $10k per night! Let's look at a histogram to see how many listings are that expensive.

# In[ ]:


plt.yscale('log')
priceHist = listings.price.hist(bins=20)
plt.xlabel('Price (USD)')
plt.ylabel('Number of NYC listings')


# These are certainly a small minority in the distribution. 
# 
# In this analysis let's only use listings with a price that at least 100 listings have. 
# From this histo it looks like this is somewhere near $1000, but let's zoom in to get an exact amount.

# In[ ]:


plt.yscale('log')
priceHist = listings.price.hist(bins=20,range=(500,1500))
plt.xlabel('Price (USD)')
plt.ylabel('Number of NYC listings')


# Okay, to avoid fluctuations, let's set the cutoff cleanly at $700. 
# 
# Our final price distribution:

# In[ ]:


plt.yscale('log')
priceHist = listings.price.hist(bins=20,range=(0,700))
plt.xlabel('Price (USD)')
plt.ylabel('Number of NYC listings')


# In[ ]:


listings = listings.loc[listings.price<700]
listings=listings.reset_index()


# We also saw with `listings.describe()` that there was a listing with a minimum night requirement of 1250! 
# Let's check that out as well.

# In[ ]:


plt.yscale('log')
priceHist = listings.minimum_nights.hist(bins=20)
plt.xlabel('Minimum nights')
plt.ylabel('Number of NYC listings')


# Again, there are clearly some outliers. 
# Let's apply the same requirement here: Keep only listings with a minimum night requirement below the 100 listing cutoff.

# In[ ]:


plt.yscale('log')
priceHist = listings.minimum_nights.hist(bins=20,range=(0,90))
plt.xlabel('Minimum nights')
plt.ylabel('Number of NYC listings')


# Let's apply a cut of 30 minimum nights.

# In[ ]:


listings = listings.loc[listings.minimum_nights<30]
listings=listings.reset_index()


# Some more logistics of the dataset:

# In[ ]:


listings.info()


# Let's look at which boroughs these listings are in.

# In[ ]:


listings.groupby([listings.neighbourhood_group]).id.count().plot(kind='bar')
plt.ylabel('Number of listings')


# This can also be seen from a heat map:

# In[ ]:


# from https://www.kaggle.com/alexisbcook/interactive-maps
from IPython.display import IFrame

def embed_map(m, file_name):
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')


# In[ ]:


heatmapnyc = folium.Map(location=[40.7, -74],tiles='cartodbpositron', zoom_start=10) 
from folium.plugins import HeatMap
HeatMap(data=listings[['latitude', 'longitude']], radius=8).add_to(heatmapnyc)
embed_map(heatmapnyc, "heatmapnyc.html")


# Perhaps expectedly, most of the listings are in Manhattan and Brooklyn.
# 
# Let's see which room types are most popular in each borough.

# In[ ]:


listings.groupby([listings.neighbourhood_group,listings.room_type]).id.count().unstack().plot(kind='bar',stacked=True)
plt.ylabel('Number of listings')


# Let's see the distribution in prices split by borough.

# In[ ]:


bk = listings.loc[(listings.neighbourhood_group=='Brooklyn')].price.hist(bins=20,label='Brooklyn',histtype='step',linewidth=2)
mh = listings.loc[(listings.neighbourhood_group=='Manhattan')].price.hist(bins=20,label='Manhattan',histtype='step',linewidth=2)
qu = listings.loc[(listings.neighbourhood_group=='Queens')].price.hist(bins=20,label='Queens',histtype='step',linewidth=2)
si = listings.loc[(listings.neighbourhood_group=='Staten Island')].price.hist(bins=20,label='Staten Island',histtype='step',linewidth=2)
bx = listings.loc[(listings.neighbourhood_group=='Bronx')].price.hist(bins=20,label='Bronx',histtype='step',linewidth=2)
plt.legend()
plt.xlabel('Price (USD)')
plt.ylabel('Number of listings')


# This is of course dominated by the boroughs with the most listings,
# making it hard to see the distributions for those with fewer entries. 
# Let's normalize these histos.

# In[ ]:


bk = listings.loc[(listings.neighbourhood_group=='Brooklyn')].price.hist(bins=20,label='Brooklyn',density=True,histtype='step',linewidth=2)
mh = listings.loc[(listings.neighbourhood_group=='Manhattan')].price.hist(bins=20,label='Manhattan',density=True,histtype='step',linewidth=2)
qu = listings.loc[(listings.neighbourhood_group=='Queens')].price.hist(bins=20,label='Queens',density=True,histtype='step',linewidth=2)
si = listings.loc[(listings.neighbourhood_group=='Staten Island')].price.hist(bins=20,label='Staten Island',density=True,histtype='step',linewidth=2)
bx = listings.loc[(listings.neighbourhood_group=='Bronx')].price.hist(bins=20,label='Bronx',density=True,histtype='step',linewidth=2)
plt.legend()
plt.xlabel('Price (USD)')
plt.ylabel('Number of listings')


# Here we can see that Manhattan has the most expensive listings, and the Bronx has the least.
# 
# Try a cleaner way of viewing the same info.

# In[ ]:


sns.catplot(x="neighbourhood_group", y="price",
            kind="box", data=listings)


# And now the same clean quality, but with some more of the shape information that we had in the histograms:

# In[ ]:


sns.catplot(x="neighbourhood_group", y="price",
            kind="violin", data=listings,cut=0)


# Here is even clearer that Manhattan is pricier than the other boroughs.
# 
# How much of this is driven by different rental types?

# In[ ]:


sns.catplot(x="neighbourhood_group", y="price",hue='room_type',
            kind="violin", data=listings,cut=0,figsize=(50,50))


# So the Manhattan listings are more expensive than the others for all room types,
# but the mean price for a private or shared Manhattan room is less expensive than the entire space in a different borough.
# 
# Let's see how the mean price looks on a map.

# In[ ]:


mapnyc = folium.Map(location=[40.7, -74],tiles='cartodbpositron', zoom_start=10)
meanPrice=listings.groupby('NAME').price.mean()

folium.Choropleth(geo_data=listings.set_index("NAME").__geo_interface__, 
           data=meanPrice,
           key_on="feature.id", 
           fill_color='YlGnBu', 
           line_color='none',
           legend_name='Mean Airbnb price (USD)'
          ).add_to(mapnyc)

# Display the map (commented for now)
# embed_map(mapnyc, 'mapnyc.html')


# So Manhattan near and south of Central Park clearly have the highest density of high prices.
# 
# We could split this up to get a more specific map by adding selections to the `meanPrice` calculation, i.e.:
# `meanPrice=listings.loc[(listings.room_type=='Private room')].groupby('NAME').price.mean()`

# I would like to try to answer the question: 
# 
# > What is the gross monthly income of each listing?
# 
# This could be used for example to predict the monthly income as a function of price, for each room type in each neighborhood,
# which would be useful if I had a new listing and wanted to know the optimal price at which to list.
# 
# The gross monthly income can be estimated roughly by:
# 
# income = (number of nights booked/month)\*price/night
# 
# (Note this is not the net monthly income, since it ignores the costs associated with owning and running the listing.)
# 
# We have price/night in the dataframe. 
# What information do we have to help us estimate how many nights the listing is booked?

# In[ ]:


listings.dtypes


# According to discussion on the dataset page, unfortunately,
# `availability_365` is the number of nights the listing is available before any nights have been booked. 
# So this doesn't help calculate the number of nights booked.
# 
# Additionally, `number_of_reviews` is heavily dependent on how recently the space was listed, so this is not so insightful.
# 
# Let's use `reviews_per_month`. 
# We can scale this by `minimum_nights` to account for stays that are required to span multiple nights.
# 
# Since some fraction of guests do not leave reviews, 
# and since some fraction of stays will be longer than the minimum number of nights,
# we will use this metric with the understanding that it gives a lower estimate of the gross income.

# In[ ]:


# first, remove the listings with nan reviews_per_month (nonzero as seen from `listings.info` above)
listings.dropna(subset=['reviews_per_month'],inplace=True)
listings.reset_index(drop=True,inplace=True)


# In[ ]:


listings['reviewsPerMonthScaled']=listings.reviews_per_month*listings.minimum_nights


# Let's take a first look at the scaled reviews per month as a function of price.

# In[ ]:


sns.relplot(x="price",
            y="reviewsPerMonthScaled",
            data=listings,
            hue=listings.neighbourhood_group,
            kind='line',
            ci=None,
         )


# This is quite noisy! Let's round the prices to the nearest 10, to be able to see the plot a bit more clearly.

# In[ ]:


# make a coarsePrice column which bins in multiples of 10.
listings['coarsePrice']=(listings.price/10).round()*10 # divide then multiply because round function is designed for decimals


# In[ ]:


listings[['price','coarsePrice']].head(10)


# In[ ]:


sns.relplot(x="coarsePrice",
            y="reviewsPerMonthScaled",
            data=listings,
            hue=listings.neighbourhood_group,
            kind='line',
            ci=None,
         )


# For prices below around $400, there is no strong correlation, perhaps a positive slope for Manhattan.
# For high prices, although the statistics run out quickly, there seems to be a negative slope.
# 
# Also check split by room type, on which price is heavily dependent.

# In[ ]:


# since I'm quickly checking 3 distributions, make each one a bit smaller with `height`
sns.relplot(x="coarsePrice",
            y="reviewsPerMonthScaled",
            data=listings.loc[(listings.room_type=="Entire home/apt")],
            hue=listings.neighbourhood_group,
            kind='line',
            ci=None,
            height=3
         )
sns.relplot(x="coarsePrice",
            y="reviewsPerMonthScaled",
            data=listings.loc[(listings.room_type=="Private room")],
            hue=listings.neighbourhood_group,
            kind='line',
            ci=None,
            height=3
         )
sns.relplot(x="coarsePrice",
            y="reviewsPerMonthScaled",
            data=listings.loc[(listings.room_type=="Shared room")],
            hue=listings.neighbourhood_group,
            kind='line',
            ci=None,
            height=3
         )


# Now let's look at the estimated gross monthly income as a function of price.

# In[ ]:


listings['estMonthlyIncome']=listings.price*listings.reviewsPerMonthScaled

sns.relplot(x="coarsePrice",
            y='estMonthlyIncome',
            data=listings,
            hue=listings.neighbourhood_group,
            kind='line',
            ci=None
         )


# At low prices, the lines seem to be similar for each borough! 
# 
# Although statistical fluctuations dominate at higher prices, making it harder to quantify trends,
# it's clear that at some point there is a discontinuity, where 
# the slope of income versus price decreases, and in some cases becomes negative.
# This may be due to the fact that fewer guests are likely to spend the high prices,
# so these spaces are often unoccupied.
# 

# So we totally run out of stats for the shared rooms, 
# but in general it seems the estimated gross monthly income as a function of price 
# is not so dependent on the borough in which the listing resides.
# 
# This suggests, if you're looking to buy property with the sole purpose of listing it on Airbnb, 
# you'd do better to buy in an area with more affordable real estate.

# Let's also compare room types in each borough. 
# We'll draw on separate axes so that the plots don't get too busy.
# Also, only check Manhattan and Brooklyn which have the best statistics.

# In[ ]:


sns.relplot(x="coarsePrice",
            y='estMonthlyIncome',
            data=listings[(listings.price<400) & (listings.neighbourhood_group=='Manhattan') & ((listings.room_type=='Entire home/apt') | (listings.room_type=='Private room'))],
            hue=listings.neighbourhood_group,
            style=listings.room_type,
            kind='line',
            ci=None,
            height=3
         )
sns.relplot(x="coarsePrice",
            y='estMonthlyIncome',
            data=listings[(listings.price<400) & (listings.neighbourhood_group=='Brooklyn') & ((listings.room_type=='Entire home/apt') | (listings.room_type=='Private room'))],
            hue=listings.neighbourhood_group,
            style=listings.room_type,
            kind='line',
            ci=None,
            height=3
         )


# As may have been expected, the discontinuity which marks the "turning point" of profitability
# occurs at lower prices for private rooms than it does for entire homes.
# It also happens at a lower price for Brooklyn than it does for Manhattan.
# However before this discontinuity, the lines are near-equal for both room types.

# Now, let's use scikit-learn to fit a line to the listings that are entire homes with prices less than $400 
# (where we know the data is a line and independent of borough).

# In[ ]:


# Import necessary scikit-learn packages
from sklearn import linear_model
from sklearn.metrics import r2_score

# Split the data into training/testing sets
from sklearn.model_selection import train_test_split

# to fit a polynomial
from sklearn.preprocessing import PolynomialFeatures

# First create a function since we'll be repeating these few lines several times
def fitPredictPlot__incomeVsPrice(data, model='linear'):

    X,Y=data[['price']],data.estMonthlyIncome
    
    if model=='poly': # if fitting a 2nd order polynomial
    # https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions
        poly = PolynomialFeatures(degree=2)
        X=poly.fit_transform(data[['price']])
    
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

    # Create linear regression object
    fitter=linear_model.LinearRegression()
    
    # Train the model using the training sets
    fitter.fit(X_train,y_train)
    y_pred = fitter.predict(X_test)
    
    # Variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    
    # Plot the output
    sns.relplot(x="coarsePrice",
                y='estMonthlyIncome',
                data=data,
                hue=listings.neighbourhood_group,
                style=listings.room_type,
                kind='line',
                ci=None,
                height=5
             )
    if model=='linear':
        xaxis=X_test
    elif model=='poly':
        xaxis=X_test[:,1]
        
    plt.scatter(xaxis, y_pred, color='blue', linewidth=3)

    # return the regression object
    return fitter


# In[ ]:


fitPredictPlot__incomeVsPrice(listings.loc[(listings.room_type=='Entire home/apt') & (listings.price<400)])


# This does rather well to predict the estimated gross monthly income as a function of price! 
# 
# We can use more complicated models to predict the non-linearities at higher prices,
# or to model income as a function of location as well as price.
# This would be especially useful for a homeowner trying to determine the optimal listing price for a new listing.

# In[ ]:


# now try a 2nd order polynomial
# first do it just for manhattan private rooms to see how it goes

fitPredictPlot__incomeVsPrice(listings.loc[(listings.room_type=='Private room') & (listings.neighbourhood_group=='Manhattan')],
                              model='poly')


# While the low statistics make it hard to see, the shape of this curve seems to model the data quite well.
# 
# Let's see how it looks for all boroughs and room types.
# 
# **We can now also predict the optimal price for a listing given its room type and borough!**

# In[ ]:


import itertools
for borough,roomType in itertools.product(['Manhattan','Queens','Bronx','Staten Island','Brooklyn'],['Entire home/apt','Private room','Shared room']):
    fitter=fitPredictPlot__incomeVsPrice(listings.loc[(listings.room_type==roomType) & (listings.neighbourhood_group==borough)],
                                         model='poly')

    # print ('parameters: ',regr.coef_)
    # location of max/min for f(x)=ax^2+bx+c is -b/(2a)
    priceMax=-fitter.coef_[1]/(2*fitter.coef_[2])
    print ('The optimal price for this %s listing in %s is $%d' %(roomType,borough,priceMax))


# Okay there are obviously some cases to understand, some of which may be improved with more stats, or with removal of more outliers. But overall this seems to be a good starting place to provide price recommendations for Airbnb hosts!
# 

# Now think about it in a more traditional data science way: treat the price as the target and provide predictions based on all other features in the dataset

# In[ ]:


# First drop categorial data and features that I've added
listings.dtypes


# In[ ]:


# also drop borough information, this can be picked up via latitude and longitude
listings_numer=listings.drop(columns=['index','STATEFP','COUNTYFP','TRACTCE','AFFGEOID','GEOID','NAME','LSAD','ALAND','AWATER','geometry','index_right','id','name','host_id','host_name','neighbourhood_group','neighbourhood','estMonthlyIncome','coarsePrice','reviewsPerMonthScaled','last_review'])


# In[ ]:


for roomType in ['Entire home/apt','Private room','Shared room']:
    thisListings = listings_numer.loc[(listings_numer.room_type==roomType)].drop(columns=['room_type'])
    all_X = thisListings.drop(columns=['price'])
    all_y = thisListings['price']
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2)

    fitter=linear_model.LinearRegression()
    fitter.fit(X_train,y_train)
    y_pred = fitter.predict(X_test)
    
    # Variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))

    d = pd.DataFrame({'actual %s' %roomType: y_test, 'predicted %s' %roomType: y_pred})
    sns.relplot(x="predicted %s" %roomType,
                y="actual %s" %roomType,
                data=d,
                color="red")
    
    plt.plot(y_pred, y_pred, color='blue', linewidth=3)


# There is a large spread, especially for the entire home category, which lead to small R^2 values.
# But this seems to be a good starting place for providing recommended listing prices to Airbnb hosts! 
# 
# Some ways this could be improved include more granular geometry information (proximity to points of interest, crime rates, etc) and more granular timing information (perhaps the listings could be more expensive on New Years Eve or in the summer months when the demand for listings is higher). 

# In[ ]:


corr = listings.corr()
sns.heatmap(corr)


# In[ ]:




