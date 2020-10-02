#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## An Exploratory Data Analysis (EDA) on a Airbnb listings dataset 

# In[ ]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.style.use('bmh')
import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:.6f}'.format
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 12, 'font.family': 'Verdana'})


# In[ ]:


df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# After importing the dataset we check out the first 10 records, in order to get acquainted with the structure of the dataset.

# In[ ]:


df.head(10)


# In[ ]:


print('Number of records -->', df.shape[0])
print('Number of features -->', df.shape[1])


# In[ ]:


print('Features type:')
df.dtypes


# In[ ]:


print('Number of null values for each feature:')
df.isnull().sum()


# A recap of the data structure:

# In[ ]:


df.info()


# In this dataset we will utilize only a subset of features: for example we don't care about the gender of host, because we are not going to study how the listings are distributed on male and female (i'm not saying that it's wrong to mantain that feature, but that it's useless for our aims), so we are going to eliminate the <em>host name</em> feature. Or for example we don't care about how the listings are named (it may be interesting to analize even this feature but i guess it is less important than other aspects of the dataset) so we are going to eliminate also the <em>name</em> feature.<br>
# We eliminate also the features with too many null values.
# At the end, the feature we are interested in examining are:<br><br>
# <em>id</em> ---> it identifies the listing: so id values are unique<br>
# <em>host_id</em> ---> it identifies the host id: an host id can be associated with more than a listing, so this feature's values are not unique in the dataset <br>
# <em>neighbourhood_group</em> ---> for geolocation analysis <br>
# <em>neighbourhood</em> ---> for a more precise geolocation analysis <br>
# <em>latitude</em> ---> for the exact geolocation <br>
# <em>longitude</em> ---> for the exact geolocation <br>
# <em>room_type</em> ---> Very likely different types have different distribution daily prices so it's important to do a distinction <br>
# <em>price</em> ---> price for a night<br>
# <em>minimum_nights</em> ---> minimum nights to book for a single booking <br>
# <em>number_of_reviews</em> ---> number of reviews for a listing <br>

# <b>Our purpose is to know how Airbnb prices vary in New York, how they vary between the room types, and which areas of the City have the largest number of listings.</b>

# In[ ]:


df1 = df[['id', 'host_id', 'neighbourhood_group', 'neighbourhood',           'latitude', 'longitude', 'room_type','price','minimum_nights','number_of_reviews']]


# First 10 records of dataset used for our purposes.

# In[ ]:


df1.head(10)


# The skimmed dataset ha 48895 records and 10 features.

# In[ ]:


df1.shape


# Which are the unique neighbourhood groups and the unique room types included in the dataset?

# In[ ]:


print('Neighbourhood group:', pd.unique(df1.neighbourhood_group), '\n', 'Room type:',pd.unique(df1.room_type))


# Let's plot the scatterplot of the listings of our dataset:

# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))
sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', ax = ax, s = 20, alpha = 0.2, data=df1);
plt.title('Scatterplot evidencing Airbnb listing density in New York');


# The areas with thickened circles are the areas where the number of Airbnb listings is large.

# Below (on the left) we have a confirmation of these different listing densities in New York.
# Instead on the right we distinguish between the different listings categories.

# In[ ]:


groupedbyZone = df1.groupby('neighbourhood_group')
fig, ax = plt.subplots(1,2, figsize = (14,6))
sns.countplot(df1['neighbourhood_group'], ax = ax[0], linewidth=1, edgecolor='w')
sns.countplot(df1['neighbourhood_group'], hue = df1['room_type'], ax = ax[1], linewidth=1, edgecolor='w')
ax[0].set_xlabel('Borough', labelpad = 10);
ax[1].set_xlabel('Borough', labelpad = 10);
ax[0].set_ylabel('Listings number');
ax[1].set_ylabel('Listings number');
plt.tight_layout();


# In Airbnb the host can specify a minimum number of nights for that listing.
# Which are the average minimum number of nights per listing for the various boroughes and room type? You can see below.

# In[ ]:


sns.catplot('neighbourhood_group', 'minimum_nights', hue = 'room_type', data = df1, 
            kind = 'bar', ci = None, linewidth=1, edgecolor='w', height=8.27, aspect=11.7/8.27)
plt.xlabel('Borough', fontsize = 15, labelpad = 15)
plt.xticks(fontsize = 13)
plt.ylabel('Average minimum nights per listing',fontsize = 17, labelpad = 14);


# And which are the number of reviews for the various boroughes and room types?

# In[ ]:


sns.catplot('neighbourhood_group', y = 'number_of_reviews', hue = 'room_type',  kind = 'bar', 
            ci = None, data = df1, linewidth=1, edgecolor='w', height=8.27, aspect=11.7/8.27)
plt.xlabel('Borough', fontsize = 15, labelpad = 15)
plt.xticks(fontsize = 13)
plt.ylabel('Average number of reviews per listing', fontsize = 17, labelpad = 14);


# <br><br> 
# ### Now check out the principal prices statistics of these Airbnb listings in all New York.

# In[ ]:


print(df1.price.describe(), '\n')
print('--> 98th Price percentile:',np.percentile(df1.price, 98), '$')


# The maximum price is \\$ 10000 and it represents a clear outlier given the fact that the 98th percentile is $ 550.<br> 
# <u>The outlier analysis is a wide topic and we will dive into that in the second part of our study.</u><br> 
# For now we just study the 98th percentile truncated distribution of the prices.

# So let's calculate the 98th percentile for each borough.

# In[ ]:


price98thPerc = pd.pivot_table(df1, values = ['price'], index = ['neighbourhood_group'],                                 aggfunc = lambda x: int(np.percentile(x, 98)))
price98thPerc.rename(columns = {'price' : '98th price percentile'}, inplace = True)
#price98thPerc.iloc[:,0] = price98thPerc.iloc[:,0].map('$ {}'.format)
price98thPerc


# Let's continue our analysis deleting the outlier prices: for each borough we esclude the prices greater than the corresponding 98th percentile.
# Indeed, in this exploratory analysis the outliers don't represent our subject of study and they don't invalidate the logic of our analysis.

# In[ ]:


df1_merged = pd.merge(df1, price98thPerc, left_on ='neighbourhood_group', right_on = price98thPerc.index)
df1_noPriceOutliers = df1_merged[df1_merged['price'] < df1_merged['98th price percentile']]

numberOutliers = df1.shape[0] - df1_noPriceOutliers.shape[0]
print('In all New York there are {} listing extreme prices.'. format(numberOutliers))
print('But they represents only the {} % of total listing prices.'.format(round(numberOutliers / df1.shape[0], 3)))


# Now we can plot the scatterplot evidencing the different listing prices in New York without take care of extreme prices. In this way the scatterplot is more informative about the difference of prices relative to different parts of the City. You can see that below.

# In[ ]:


plt.figure(figsize=(14,10))
ax = plt.gca()
df1_noPriceOutliers.plot(kind='scatter', x='longitude', y='latitude', c='price', ax=ax, cmap=plt.get_cmap('RdBu'), colorbar=True, alpha=0.7);


# New York has 5 boroughes but how many neighbourhoods are present in this dataset?

# In[ ]:


print('There are {} neighbourhoods present in this dataset.'.format(len(df1.neighbourhood.unique())))


# Which are the ten neighbourhoods with the most Airbnb listings?

# In[ ]:


intop10  = df1[df1.neighbourhood.isin(list(df1.neighbourhood.value_counts(ascending = False).head(10).index))]
topten = intop10.neighbourhood.value_counts(ascending = False).to_frame()
topten.rename(columns = {'neighbourhood': 'number of listings'}, inplace = True)
topten


# In[ ]:


print('Fraction and Cumulative fraction of top 10 neighbourhood over total listings:')
neighweight = pd.DataFrame([intop10.neighbourhood.value_counts()*100 / df.neighbourhood.value_counts().sum(), 
             np.cumsum(intop10.neighbourhood.value_counts()*100 / df.neighbourhood.value_counts().sum())],\
                index = ['% over total listings in New York','cumulative % over total listing in New York'])
neighweight = neighweight.T
#neighweight.rename(columns = {neighweight.columns[0]:'% over total listings', neighweight.columns[1]: 'cumulative %'})
neighweight.name = 'Top 10 Neighbourhood'
neighweight = neighweight.applymap('{:.1f}'.format)
neighweight


# So, together, these 10 neighbourhood cover almost the 50% of all New York Airbnb listings.
# Where they are located? Below the top 10 most listing-populated neighbourhood are scattered in blue color.

# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))
sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', ax = ax, s=20, alpha = 0.4, data=df1);
sns.scatterplot(x='longitude', y='latitude', ax = ax, s=20,                 color = 'b', label = 'Top 10 neighbourhood \nfor Airbnb listings', alpha = 0.8, data = intop10);
ax.legend(loc = 'upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1);
ax.set_title('Airbnb listings density in New York\n Top 10 dense neighbourhood in Blue');


# Where is located Williamsburg, the most listing-populated neighbourhood? Let's see...

# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))

sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', ax = ax, s=20, alpha = 0.4, data=df1);
sns.scatterplot(x='longitude', y='latitude', ax = ax, s=20, color = 'b',                 alpha = 0.8, data = intop10[intop10.neighbourhood == 'Williamsburg'], label = 'Williamsburg');
ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1);
ax.set_title('Airbnb listings density in New York\n Williamsburg is the most dense neighbourhood');


# Ok. Let's start now the second part of this analysis where we are going to focus about prices.
# Check out the distribution of listing prices in all New York (until the 98th percentile).

# In[ ]:


plt.figure(figsize = (8,8))
df1.price.plot(kind = 'hist', bins = 700, linewidth=1, edgecolor='white')
plt.axis([0,600,0,7000]);
plt.xlabel('Price', labelpad = 10);
plt.ylabel('Number of listings per bin-price', labelpad = 15);
plt.title('Airbnb daily price distribution in New York without outliers', pad = 15);


# Let's do a brief digression: this graph is not so enlightening because it doesn't distinguish between the different boroughes and the different types of room.
# Nevertheless it tells us one important thing: even without the outliers, the empirical price distribution of New York Airbnb listings is right-skewed.
# So, as measure of central tendency i prefer to use the median (more robust), and not the mean.
# Therefore as measure of dispersion i prefere the interquartile range (IQR) and not the standard deviation (the standard deviation includes the mean in the formula, so if i don't use the mean as location metric i have to avoid the standard deviation).
# Maybe it's better to transform the price distribution to the log-price distribution: indeed the log transformation is appropriate when the empirical distribution can't take negative values and it is right-skewed. Moreover, if this latter transformation is very similar to a normal distribution (it can be verified with a QQ-plot), hence the price distribution follows a log-normal distribution.
# But the point is: when we read an airbnb listing we reason in price or in log-prices? In prices of course, so for the moment we keep the original price-distribution. The point is: do we need to know which is the theoretical distribution (it cannot even exist sometimes) that is closest to the empirical price distribution? For the purpose of this analysis i don't think so.
# Indeed, an exploratory data analysis (EDA) with graphical means and tables, even with no precise associations with rigid theoretical models or distributions can be sufficient (and clearer) for certain goals.
# Sometimes we may need to model in a more formal way, sometimes EDA can be sufficient to grasp what we want to know.
# 
# 
# That said, let's plot the empirical price distribution distinguishing by borough:

# In[ ]:


g = df1_noPriceOutliers.groupby(['neighbourhood_group'])
import warnings
warnings.filterwarnings('ignore')
fig, ax = plt.subplots(figsize = (16,11))
for _ , (k, group) in enumerate(g):
    #ax[i].set_title(k)
    group.price.hist(normed = False, ax = ax, bins = 40, label = k, alpha = 0.5, linewidth=1, edgecolor='white')
    ax.legend();
ax.set_title('Price Histogram for borough', fontsize = 16, pad = 18);
ax.set_xlabel('Price', fontsize = 15, labelpad = 12)
ax.set_ylabel('Frequency in absolute value', fontsize = 15, labelpad = 12);


# As you've noticed in this not normalized histogram, in Manhattan and Brookling not only price-distribution is more right-skewed (listings are more expensive) but there are also more listings, as you already know examining the first part of this analysis.
# Because of last consideration, it's better to zoom the histogram for the Bronx and Staten Island boroughes:

# In[ ]:


fig, ax = plt.subplots(figsize = (14,10))
import warnings
warnings.filterwarnings('ignore')
for _ , (k, group) in enumerate(g):
    if k in ['Bronx', 'Staten Island']:
        group.price.hist(normed = False, ax = ax, bins = 40, label = k, alpha = 0.5, linewidth=1, edgecolor='white')
        ax.legend();
ax.set_title('Price Histogram for Bronx and Staten Island', fontsize = 16, pad = 18);
ax.set_xlabel('Price', fontsize = 15, labelpad = 12)
ax.set_ylabel('Frequency in absolute value', fontsize = 15, labelpad = 12);


# Ok, we now are going to differentiate price distribution not only by borough, but also by room-type.

# In[ ]:


colors = ['red','tan','blue','green','lime']
fig, ax = plt.subplots(3, 1, figsize = (18,18))
doublegrouped = df1_noPriceOutliers.groupby(['room_type','neighbourhood_group'])
for i, (name, combo) in enumerate(doublegrouped):
    if i <= 4:
        combo.price.plot(kind = 'hist', ax = ax[0], bins = 40, 
                         label = name, alpha = 0.5, linewidth=1, edgecolor='white');
        ax[0].legend()
        ax[0].set_title('Entire home / apt')
    elif 5 <= i <= 9:
        combo.price.plot(kind = 'hist', ax = ax[1], bins = 40, label = name, alpha = 0.5, linewidth=1, edgecolor='white');
        ax[1].legend()
        ax[1].set_title('Private room')
    else:
        combo.price.plot(kind = 'hist', ax = ax[2], bins = 40, label = name, alpha = 0.5, linewidth=1, edgecolor='white');
        ax[2].legend()
        ax[2].set_title('Shared room')
for i in range(3):
    ax[i].set_ylabel('Frequency in absolute value', fontsize = 15, labelpad = 14)
plt.suptitle('Price histograms by room type and borough', fontsize = 20);


# Let's zoom below for the Bronx and Staten Island boroughes:

# In[ ]:


fig, ax = plt.subplots(3, 1, figsize = (16,16))
doublegrouped = df1_noPriceOutliers.groupby(['room_type','neighbourhood_group'])
for i, (name, combo) in enumerate(doublegrouped):
    if i <= 4 and name[1] in ['Bronx', 'Staten Island']:
        combo.price.plot(kind = 'hist', ax = ax[0], bins = 40, 
                         label = name, alpha = 0.5, linewidth=1, edgecolor='white');
        ax[0].legend()
        ax[0].set_title('Entire home / apt')
    elif 5 <= i <= 9 and name[1] in ['Bronx', 'Staten Island']:
        combo.price.plot(kind = 'hist', ax = ax[1], bins = 40, label = name, alpha = 0.5, linewidth=1, edgecolor='white');
        ax[1].legend()
        ax[1].set_title('Private room')
    elif i > 9 and name[1] in ['Bronx', 'Staten Island']:
        combo.price.plot(kind = 'hist', ax = ax[2], bins = 40, label = name, alpha = 0.5, linewidth=1, edgecolor='white');
        ax[2].legend()
        ax[2].set_title('Shared room')
for k in ax:
    k.set_ylabel('Listings number by bin-price', labelpad = 12)
plt.suptitle('Price histogram by room type for Bronx and Staten Island', fontsize = 15);


# I think that a recap of the last histograms with a pivot table is useful:

# In[ ]:


from scipy.stats import iqr #let's import the interquartile range function from scipy


# In[ ]:


by_room_type = pd.pivot_table(df1, values = ['price'], index = ['room_type','neighbourhood_group'], aggfunc = {"price":[np.median, np.count_nonzero, iqr]})
subtables = []
for row in by_room_type.index.levels[0]:
    subtables.append(by_room_type.loc[[row]].sort_values(by = ('price','median'), ascending = False))
by_room_type = pd.concat(t for t in subtables)

by_room_type[('price','median')] = by_room_type[('price','median')].map('$ {:.0f}'.format)
by_room_type[('price','iqr')] = by_room_type[('price','iqr')].map('$ {:.0f}'.format)
by_room_type[('price','count_nonzero')] = by_room_type[('price','count_nonzero')].map(int)

by_room_type.columns.set_levels(['number listings','IQR','median price'],level=1,inplace=True)
by_room_type.columns = by_room_type.columns.droplevel(0)
by_room_type = by_room_type[['median price', 'IQR','number listings']] # change the column order
by_room_type


# A final consideration for this first part of the study: we can continue our analysis and investigate fully even the most strange relations and patterns, but the risk is to lose the whole picture. Informations must be assimilated by the analyst and adding more and more informations can be worthy only if they bring a sufficient value factor.
# Otherwise is better to truncate the information extracting process and focus the attention on the most relevant informations.
