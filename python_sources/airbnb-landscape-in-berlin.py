#!/usr/bin/env python
# coding: utf-8

# # So you want to start an Aibnb in Berlin, Germany? 
# 
# To start with, it is nice to get an idea about what kind of properties drive the market and what their features are. What makes a for a profitable vacation rental in this famous city? 
# 
# This is an exploratory analysis of the airbnb market in Berlin, Germnay as of Noevember, 2018. The goal of this analysis is to get an idea about what drives the rental market of airbnb given the specs of the properties.
# 
# 
# # About the data set
# This is a detailed Berlin listings data set sourced from the Inside Airbnb website by Britta Bettendorf, in order to understand the rental landscape and try to recommend a price for a newbie entering the market. The dataset is named listings.csv.gz and was scraped on November 07th 2018.

# In[ ]:


# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)
get_ipython().run_line_magic('matplotlib', 'inline')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# A function to label catergordical variables
def label_encoding(df, col_name):
    df[col_name] = df[col_name].astype('category')
    df[col_name+"_cat"] = df[col_name].cat.codes
    return


# # Get the raw dataset

# In[ ]:


#  Get data
data = pd.read_csv("/kaggle/input/berlin-airbnb-data/listings_summary.csv")
print("data : " + str(data.shape))


# In[ ]:


data.head(2)


# In[ ]:


data.columns


# In[ ]:


data.dtypes


# # Pre-processing

# In[ ]:


# Check for duplicates
idsUnique = len(set(data.id))
idsTotal = data.shape[0]
idsDupli = idsTotal - idsUnique
print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")


# ### As a start, select the subset of features in an Airbnb that would matter in people using it or not.
# 
# - During the initial pre-processing, found out that squre-foot, weekly_price and monthly_price were mostly NULL. So those were removed from the final analysis.
# - security deposit and cleaning fee were also dropped

# In[ ]:


# Select features 
data_select = data[['zipcode','latitude', 'longitude',
       'property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities',
       'price','instant_bookable',
       'is_business_travel_ready', 'cancellation_policy']]
data_select.describe().transpose()


# In[ ]:


# Drop all rows with NA's In this analysis I am not going to worry about filling up the missing data 
# because there is enough data to work with.
data_select = data_select.dropna()
data_select.isnull().sum()


# ### Encode categorical variables

# In[ ]:


# Give unique labels to categorical variables
label_encoding(data_select, "property_type")
label_encoding(data_select, "room_type")
label_encoding(data_select, "bed_type")
label_encoding(data_select, "cancellation_policy")


# ### Pricing Distrubtion

# In[ ]:


# Convert price to float and drop $ symbol
data_select['price'] = data_select.price.str.replace('$', '').str.replace(',', '').astype(float)

n, bins, patches = plt.hist(data_select['price'], 50, density=True, facecolor='g', alpha=0.75, log = True)
plt.title("Pricing Distribution")
plt.xlabel("Price $")
plt.ylabel("Counts (log)")


# ### Apply a cut on the 'price' to remove outliers

# In[ ]:


# Slice the price data to study the outliers
price_low = data_select[data_select.price<300]
#price_low.describe().transpose()
n, bins, patches = plt.hist(price_low['price'], 50, density=True, facecolor='g', alpha=0.75, log = False)
plt.title("Price < $300")


# ####  The pricing distribution have a range 0-9000. Most of the data are concentrated in the price range < $300. I am going to use the 300 as a reasonable upper cut that represents the majority of the data in this data sample.

# In[ ]:


print("% Data loss from the price cut : " + str((data_select.shape[0] - price_low.shape[0])*100/data_select.shape[0]))


# # Final data set used for the analysis

# ### Summary of the data

# In[ ]:


price_low.describe().transpose()


# ### Basic property features

# In[ ]:


from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
words = price_low[['property_type']].copy()
words['property_type'] = words['property_type'].apply(lambda x: x.strip())
counts = words.property_type.value_counts()

wordcloud = WordCloud(background_color='white',
                          stopwords=stopwords,
                          max_words=50,
                          max_font_size=100
).generate_from_frequencies(counts)

# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


# In[ ]:


fig, axs = plt.subplots(2, 2, figsize = (15,15))
plt.subplot(2, 2, 1)
price_low.property_type.value_counts()[:].plot(kind='barh',
                                                 width=1.0,align="center", log = True)
plt.title("Property Type")

plt.subplot(2, 2, 2)
price_low.room_type.value_counts()[:].plot(kind='barh',
                                           width=1.0,align="center", log = True)
plt.title("Room Type")

plt.subplot(2, 2, 3)
price_low.bed_type.value_counts()[:].plot(kind='barh',
                                                 width=1.0,align="center", log = True)
plt.title("Bed Type")

plt.subplot(2, 2, 4)
price_low.cancellation_policy.value_counts()[:].plot(kind='barh',
                                                 width=1.0,align="center", log = True)
plt.title("Canellation Policy")


# In[ ]:


results = Counter()
price_low['amenities'].str.strip('{}')               .str.replace('"', '')               .str.lstrip('\"')               .str.rstrip('\"')               .str.split(',')               .apply(results.update)

#results.most_common(30)
# create a new dataframe
am_df = pd.DataFrame(results.most_common(30), 
                     columns=['amenity', 'count'])

# plot the Top 20
am_df.sort_values(by=['count'], ascending=True).plot(kind='barh', x='amenity', y='count',  
                                                      figsize=(10,7), legend=False,
                                                      width=1.0,align="center",
                                                      title='Amenities')
plt.xlabel('Count');


# # Exploratory Data Analysis

# ### Correlations

# In[ ]:


# Correlations
corr = price_low.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ## Location effect on Price

# In[ ]:


price_low.plot(kind="scatter", x="longitude", y="latitude", 
               alpha=0.4, figsize=(10,7), c="price", 
               cmap="YlGn", colorbar=True, sharex=False);


# ## Price breakdown for property types

# In[ ]:


# Group the data frame by property type and extract a number of stats from each group
price_low.groupby(['property_type']).agg({ 
    # find the min, max, and sum of the price column
    'price': ['min', 'max', 'mean', 'count']})


# In[ ]:


# Group the data frame by property type and extract a number of stats from each group
price_low.groupby(['room_type']).agg({ 
    # find the min, max, and sum of the price column
    'price': ['min', 'max', 'mean']})


# ## Basic Insights
# 
# Based on this sample, the Airbnb market in Berlin is mostly dominated by Apartments with the average price of an apartment being close to ~$55. Most of them are either full rentals or private rooms consisting of real beds. 
# There are lot of amenities offered out of which wifi, kitchen and heating are dominant with essentials (not sure what that is exactly) and washer following closely.
# 
# Most of the properties are located close to the city center with the prices diriving up as you get close to the center.
# 
# 
# 
# 
