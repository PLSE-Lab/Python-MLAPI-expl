#!/usr/bin/env python
# coding: utf-8

# ### Data Science using OSEMN
# 
# The first main step is to obtain all needed informations and import all needed libraries. Here I included the geojson data file from King County and extracted the needed data to help me visuzlize housing data. Next comes the handling of data which is either missing or just not clean. By going through the data and addressing the bits and peices ie NaN etc, helps me understand the data and hopfully can derive some statistics and visualizations. this follows a classifications and scaling of features. Finishing with the interpretation of what I can find.
# 
# 1. Obtaining data
# 2. Scrubbing data
# 3. Exploring data
# 4. Modeling data 
# 5. Interpreting results
# 

# ### Import all needed libraries 

# In[ ]:


import os
import math
import random

import pandas as pd
import numpy as np
import json

from scipy import stats
from scipy import linalg

import statsmodels.api as sm
import statsmodels.stats.stattools as sms
from statsmodels.formula.api import ols

from sklearn import metrics
from sklearn import linear_model
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import missingno as msno

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
# set style
sns.set_style('whitegrid')
# overriding font size and line width
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# map visualization
import folium
from folium.plugins import HeatMap

# don't print matching warnings
import warnings
warnings.filterwarnings('ignore') 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print(os.listdir("../input"))


# In[ ]:


# local functions which are in a seperate python file
#
def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    """
    Perform a forward-backward feature selection based on p-value from statsmodels.api.OLS

    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features

    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """

    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)

        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True

            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()

        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        # null if pvalues is empty
        worst_pval = pvalues.max()

        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)

            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break

    return included


def display_heatmap(data):
    """
    Display a heatmap from a given dataset

    :param data: dataset
    :return: g (graph to display)
    """

    # Set the style of the visualization
    # sns.set(style = "white")
    sns.set_style("white")

    # Create a covariance matrix
    corr = data.corr()

    # Generate a mask the size of our covariance matrix
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = None

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 12))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, sep=20, n=9, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(corr, cmap=cmap, mask=mask, square=True)

    return g


def display_jointplot(data, columns):
    """
    Display seaborn jointplot on given dataset and feature list

    :param data: dataset
    :param columns: feature list
    :return: g
    """

    sns.set_style('whitegrid')

    for column in columns:
        g = sns.jointplot(x=column, y="price", data=data, dropna=True,
                          kind='reg', joint_kws={'line_kws': {'color': 'red'}})

    return g


def display_plot(data, vars, target, plot_type='box'):
    """
    Generates a seaborn boxplot (default) or scatterplot

    :param data: dataset
    :param vars: feature list
    :param target: feature name
    :param plot_type: box (default), scatter, rel
    :return: g
    """

    # pick one dimension
    ncol = 3
    # make sure enough subplots
    nrow = math.floor((len(vars) + ncol - 1) / ncol)
    # create the axes
    fig, axarr = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, 20))

    # go over a linear list of data
    for i in range(len(vars)):
        # compute an appropriate index (1d or 2d)
        ix = np.unravel_index(i, axarr.shape)

        feature_name = vars[i]

        if plot_type == 'box':
            g = sns.boxplot(y=feature_name, x=target, data=data, width=0.8,
                            orient='h', showmeans=True, fliersize=3, ax=axarr[ix])

        # elif plot_type == 'scatter':
        else:
            g = sns.scatterplot(x=feature_name, y=target, data=data, ax=axarr[ix])

        # else:
        #     col_name = vars[i]
        #     g = sns.relplot(x=feature_name, y=target, hue=target, col=col_name,
        #                     size=target, sizes=(5, 500), col_wrap=3, data=data)

    return g


def map_feature_by_zipcode(zipcode_data, col):
    """
    Generates a folium map of Seattle
    :param zipcode_data: zipcode dataset
    :param col: feature to display
    :return: m
    """

    # read updated geo data
    king_geo = "cleaned_geodata.json"

    # Initialize Folium Map with Seattle latitude and longitude
    m = folium.Map(location=[47.35, -121.9], zoom_start=9,
                   detect_retina=True, control_scale=False)
    # tiles='stamentoner')

    # Create choropleth map
    m.choropleth(
        geo_data=king_geo,
        name='choropleth',
        data=zipcode_data,
        # col: feature of interest
        columns=['zipcode', col],
        key_on='feature.properties.ZIPCODE',
        fill_color='OrRd',
        fill_opacity=0.9,
        line_opacity=0.2,
        legend_name='house ' + col
    )

    folium.LayerControl().add_to(m)

    # Save map based on feature of interest
    m.save(col + '.html')

    return m


def measure_strength(data, feature_list, target):
    """
    Calculate a Pearson correlation coefficient and the p-value to test for non-correlation.

    :param data: dataset
    :param feature_list: feature list
    :param target: feature name
    :return:
    """

    print("Pearson correlation coefficient R and p-value \n\n")

    for k, v in enumerate(feature_list):
        r, p = stats.pearsonr(data[v], data[target])
        print("{0} <=> {1}\t\tR = {2} \t\t p = {3}".format(target, v, r, p))


def heatmap_features_by_loc(data, feature):
    """
    Generates a heatmap based on lat, long and a feature

    :param data: dataset
    :param feature: feature name
    :return:
    """
    max_value = data[feature].max()

    lat = np.array(data.lat, dtype=pd.Series)
    lon = np.array(data.long, dtype=pd.Series)
    mag = np.array(data[feature], dtype=pd.Series) / max_value

    d = np.dstack((lat, lon, mag))[0]
    heatmap_data = [i for i in d.tolist()]

    hmap = folium.Map(location=[47.55, -122.0], zoom_start=10, tiles='stamentoner')

    hm_wide = HeatMap(heatmap_data,
                      min_opacity=0.7,
                      max_val=max_value,
                      radius=1, blur=1,
                      max_zoom=1,
                      )

    hmap.add_child(hm_wide)

    return hmap


# In[ ]:


# We can import the above function from a seperate python file:
#
# import function_filename as f
#
# you can check out the the documentation for the rest of the autoreaload modes
# by apending a question mark to %autoreload, like this:
# %autoreload?
#
# %load_ext autoreload
# %autoreload 2


# #### Description of what can be found in the dataset
# 
# + **ida** notation for a house
# + **date** Date house was sold
# + **price** Price is prediction target
# + **bedrooms** Number of Bedrooms/House
# + **bathrooms** Number of bathrooms/bedrooms
# + **sqft_living** square footage of the home
# + **sqft_lot** square footage of the lot
# + **floors** Total floors (levels) in house
# + **waterfront** House which has a view to a waterfront
# + **view** Has been viewed
# + **condition** How good the condition is ( Overall )
# + **grade** overall grade given to the housing unit, based on King County grading system (see below)
# + **sqft_above** square footage of house apart from basement
# + **sqft_basement** square footage of the basement
# + **yr_built** Built Year
# + **yr_renovated** Year when house was renovated
# + **zipcode** zip
# + **lat** Latitude coordinate
# + **long** Longitude coordinate
# + **sqft_living15** Living room area in 2015 (implies-- some renovations) This might or might not have affected the lot size area
# + **sqft_lot15** lotSize area in 2015 (implies-- some renovations)
# 
# http://www5.kingcounty.gov/sdc/FGDCDocs/resbldg_extr_faq.htm
# ##### BLDGGRADE
# Buildling grade (Source: King County Assessments)
# + Value - Definition
# + 0     - Unknown
# + 1     - Cabin
# + 2     - Substandard
# + 3     - Poor
# + 4     - Low
# + 5     - Fair
# + 6     - Low Average
# + 7     - Average
# + 8     - Good
# + 9     - Better
# + 10    - Very Good
# + 11    - Excellent
# + 12    - Luxury
# + 13    - Mansion
# + 20    - Exceptional Properties
# 

# # Questions:
# 
# 
# 1. Is location of a house by zipcode/neighborhood an indicator for the house price? 
# 2. Do have zipcodes (neighborhoods) with the higher housing density an effect on selling price?
# 3. Does grade, condition and renovation of a house reflect in the price?

# ## Obtaining Data

# In[ ]:


# read data and read date correctly
#
dataset = pd.read_csv("../input/kc_house_data.csv", parse_dates = ['date'])


# ### Collecting basic informations about the data set

# In[ ]:


dataset.shape


# In[ ]:


dataset.dtypes


# ___
# 
# # Scrubbing Data
# 
# 
# ## Cleaning Data

# In[ ]:


# Display all missing data
#
msno.matrix(dataset);


# In[ ]:


# Handling Null values for view
#
dataset.view.fillna(0, inplace=True)


# In[ ]:


# Handling yr_renovated
# - create new column 'is_renovated' and 'yr_since_renovation'
# - if sqft_living15 > sqft_living set renovated
# - drop yr_renovated
#
import datetime
cur_year = datetime.datetime.now().year

def calc_years(row):
    return cur_year - row['yr_renovated'] if row['yr_renovated'] > 0 else 0

def set_renovated(row):
    return 1 if row['yr_since_renovation'] > 0 or row['sqft_living'] != row['sqft_living15'] else 0

# Set yr_renovated to int
dataset.yr_renovated.fillna(0, inplace = True)
# now I can convert yr_renovated to int
dataset.yr_renovated = dataset.yr_renovated.astype('int64')

dataset['yr_since_renovation'] = dataset.apply(calc_years, axis = 1)

# Create category 'is_renovated'
dataset['is_renovated'] = dataset.apply(set_renovated, axis=1)
# Binning
bins = [0., 1950., 1980., 1990., 2000., 2015.]
names = ['never', 'before 1980', '1980-1989', '1990-1999', '2000-2015']
dataset['yr_renov_bins'] = pd.cut(dataset['yr_renovated'], bins, labels=names, right=False)
dataset.yr_renov_bins.fillna('never', inplace=True)

dataset.drop(columns=['yr_renovated'], inplace=True)


# In[ ]:


print(cur_year)


# In[ ]:


dataset.yr_built.shape


# In[ ]:


dataset.yr_built.value_counts()


# In[ ]:


# While are at it, lets convert yr_built to house_age and drop yr_built
#
dataset['house_age'] = cur_year - dataset.yr_built
# dataset.drop(columns=['yr_built'], inplace=True)


# In[ ]:


dataset.house_age.value_counts()


# In[ ]:


# To answer this question, it's best to build a new variable (feature engineering) ...
dataset['yr_built_cat'] = dataset['house_age'].apply(lambda x: ('old' if x >= 50 else 'middle-aged') if x >= 15 else 'modern')

# ... and turn it into a category
dataset['yr_built_cat'] = pd.Categorical(dataset['yr_built_cat'], categories = ['old', 'middle-aged', 'modern'])
dataset.head(2)


# In[ ]:


dataset.yr_built_cat.value_counts()

msno.matrix(dataset)
# In[ ]:


# What is the percential of NaN in waterfront?
#
print(dataset.waterfront.isnull().sum() / dataset.shape[0])


# In[ ]:


# Because the percential is about 10% we set the NaN values to zero
#
dataset.waterfront.fillna(0, inplace=True)

# Waterfront - Level Up:
# We could try to determine by lat/long if a house is at the waterfront or not by 
# implementing k-nearest neighbor 
# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
# In[ ]:


msno.matrix(dataset);


# In[ ]:


dataset.shape


# ### Cleaning basement feature

# In[ ]:


# Handling sqft_basement
#
def calc_basement(row):
    """
    Calculate basement sqft based on difference sqft_living and sqft_above
    Deals at the same time with the '?' string
    
    :param row: feature (column)
    :return: value (sqft)
    """
    return row['sqft_living'] - row['sqft_above'] if row['sqft_above'] < row['sqft_living']  else 0

dataset.sqft_basement = dataset.apply(calc_basement, axis = 1)


# In[ ]:


# sort dataset by date and reset index (Do I have a good reason for it? No.)
#
dataset = dataset.sort_values(by = ['date'])
dataset = dataset.reset_index(drop=True)


# ### Get the big picture
# Correlation Matrix
#
dataset.corr()abs(dataset.corr()) > 0.7
# In[ ]:


display_heatmap(dataset);


# Initial observation (based on the darker colours/higher and lower values):
#     * In general there are very few strong correlations (around +/-0.7 and beyond)
#     * price correlates to sqft_living/15 and grade
#     * grade correlates with sqft_above
#     * house_age correlates with bathrooms, floors, grade, sqft_above
#     * we can consolidate sqft_living, sqft_living15 and sqft_above

# ___
# 
# ### Get a gerneral overview via scatter plot

# In[ ]:


dataset.head()


# In[ ]:


dataset.columns


# In[ ]:


dataset['zipcode'] = dataset['zipcode'].astype(int)


# In[ ]:


cols = ['bedrooms', 'bathrooms', 'sqft_above', 'sqft_basement', 'sqft_living15', 
        'sqft_lot15', 'yr_since_renovation', 'house_age', 'zipcode']


# In[ ]:


ncol = 3 # pick one dimension
nrow = math.floor((len(cols)+ ncol-1) / ncol) # make sure enough subplots
fig, axarr = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, 20)) # create the axes

for i in range(len(cols)): # go over a linear list of data
    ix = np.unravel_index(i, axarr.shape) # compute an appropriate index (1d or 2d)

    name = cols[i]
    dataset.plot(kind='scatter', x=name, y='price', ax=axarr[ix], label=name) 

plt.tight_layout()
plt.show();
# plt.savefig('pics/scatter_plot_1.png', dpi = 320)


# **Notes:** 
#     * Cross referencing on trulia, there are houses with a high sqft_living15 as well a price at $24,000,000
#     * And as it turns out the number of bathrooms can be 8 or even 9, and it looks like that it might have some effect on the price
#     * But 30+ bedrooms is an outlier, they as well looks like can have some effect on the price 
#     * How does age (yr_built) and sqft_living coerlate?
#     * It looks like that zipcodes are coerlated to price

# ## Investigating some of the outliers in numbers

# In[ ]:


dataset.sqft_lot15.value_counts(bins=10, sort=False)


# In[ ]:


dataset.sqft_living15.value_counts(bins=10, sort=False)


# In[ ]:


dataset.bedrooms.value_counts(bins=10, sort=False)


# In[ ]:


dataset.price.value_counts(bins=10, sort=False)


# Regarding to [trulia](https://www.trulia.com/for_sale/53033_c/price;d_sort/) most of the so called outliers seen in the plot above seem legit (as of 11/02/2018).
# 

# ---
# ### Investigating Continuos Variables in relationship to price

# In[ ]:


# 'house_age', 'sqft_basement', 'sqft_above', 'sqft_living15',  'sqft_lot15', 'yr_since_renovation'
#
continous_vars = ['sqft_living15',  'sqft_lot15', 'house_age', 'yr_since_renovation']


# In[ ]:


display_jointplot(dataset, continous_vars)


# In[ ]:


measure_strength(dataset, continous_vars, 'price')


# ### Investigate Discrete Variables
# 

# In[ ]:


discrete_vars = ['grade', 'condition', 'view', 'floors', 'bedrooms', 'bathrooms']

f.display_jointplot(dataset, discrete_vars)
# In[ ]:


# Display box-and-whisker plot
#
display_plot(dataset, discrete_vars, 'price')


# In[ ]:


measure_strength(dataset, discrete_vars, 'price')


# ___
# 
# # Answer #1
# 
# ### Visualize house prices and density by zipcode
# 
# Due to missing data we can't run the cells below.
# 

# In[ ]:


# # Set zipcode type to string (folium)
# dataset['zipcode'] = dataset['zipcode'].astype('str')

# # get the mean value across all data points
# zipcode_data = dataset.groupby('zipcode').aggregate(np.mean)
# zipcode_data.reset_index(inplace = True)


# In[ ]:


# # count number of houses grouped by zipcode
# #
# dataset['count'] = 1
# t = dataset.groupby('zipcode').sum()
# t.reset_index(inplace = True)
# t = t[['zipcode', 'count']]
# zipcode_data = pd.merge(zipcode_data, t, on='zipcode')

# # drop count from org dataset
# dataset.drop(['count'], axis = 1, inplace = True)


# In[ ]:


# # Get geo data file path
# geo_data_file = os.path.join('data', '../input/king_county_wa_zipcode_area.geojson')

# # load GeoJSON
# with open(geo_data_file, 'r') as jsonFile:
#     geo_data = json.load(jsonFile)
    
# tmp = geo_data

# # remove ZIP codes not in geo data
# geozips = []
# for i in range(len(tmp['features'])):
#     if tmp['features'][i]['properties']['ZIPCODE'] in list(zipcode_data['zipcode'].unique()):
#         geozips.append(tmp['features'][i])

# # creating new JSON object
# new_json = dict.fromkeys(['type','features'])
# new_json['type'] = 'FeatureCollection'
# new_json['features'] = geozips

# # save uodated JSON object
# open("../input/cleaned_geodata.json", "w").write(json.dumps(new_json, sort_keys=True, indent=4, separators=(',', ': ')))


# In[ ]:


# map_feature_by_zipcode(zipcode_data, 'count')


# In[ ]:


# map_feature_by_zipcode(zipcode_data, 'price')


# In[ ]:


# Get the top 5 zipcode by price
#
# zipcode_data.nlargest(5, 'price')['zipcode']


# ###### Observation:
#     * The most pricey zipcode 98039 seems to be also one of the less densly populated zipcode.
#     * The housing density is focused around Seattle 

# # Answer #2
# 
# Location, location, location. Waterfront properties are by far the most expensive once.

# In[ ]:


# Initialize Folium Map with Seattle latitude and longitude

# from folium.plugins import HeatMap

# max_val = dataset.price.max()

# lat = np.array(dataset.lat, dtype=pd.Series)
# lon = np.array(dataset.long, dtype=pd.Series)
# mag = np.array(dataset.price, dtype=pd.Series)

# d = np.dstack((lat, lon, mag))[0]
# heatmap_data = [i for i in d.tolist()]

# m = folium.Map(location=[47.35, -121.9], zoom_start=9, detect_retina=True, control_scale=False)
# HeatMap(heatmap_data, radius=1, blur=1).add_to(m)
# m


# In[ ]:


dataset.plot(kind="scatter", x="long", y="lat", figsize=(16, 8), c="price", 
             cmap="gist_heat_r", colorbar=True, sharex=False);
plt.show();


# Simpler representation, but in this cae more effective. And brings the point across.

# ---
# # Answering question  #3
# 
# ### Relational plots
# Lets visualize the relationship of price and sqft_living15 by grade and condition to hopefuly get a deeper inside
sns.catplot(x="is_renovated", y="price", data=dataset, height=4, aspect=2)
plt.title('\nIs Renovated vs. Price\n', fontweight='bold')
plt.xlabel('Is Renovated')
plt.ylabel('Price');
# In[ ]:


sns.relplot(x="sqft_living15", y="price", hue="price", col="is_renovated", 
            size="price", sizes=(5, 500), col_wrap=3, data=dataset);


# It lloks like that renovation will affect the price

# In[ ]:


dataset.is_renovated.value_counts()

dataset.info()
# In[ ]:


# get statistics for houses which are renovated
df_is_renovated = dataset[dataset['is_renovated'] == 1.0]

subset = ['price', 'bedrooms', 'floors', 'sqft_living15', 'sqft_lot15']
is_renovated_descriptives = round(df_is_renovated[subset].describe(), 2)
is_renovated_descriptives


# In[ ]:


df_not_renovated = dataset[dataset['is_renovated'] == 0.0]

subset = ['price', 'bedrooms', 'floors', 'sqft_living15', 'sqft_lot15']
not_renovated_descriptives = round(df_not_renovated[subset].describe(), 2)
not_renovated_descriptives


# In[ ]:


is_renovated_descriptives.price.median()


# In[ ]:


not_renovated_descriptives.price.median()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x='is_renovated', y='price', hue='is_renovated', data=dataset, palette="PuBu_r")

# add title, legend and informative axis labels
ax.set_title('\nMedian Prices depending on if House is renovated\n', fontsize=14, fontweight='bold')
ax.set(ylabel='Price', xlabel='Is Renovated')
ax.legend(loc=2);


# In[ ]:


dataset['price'][dataset.is_renovated.max()] - dataset['price'][dataset.is_renovated.min()]


# Looks like that renovating can pay off by about $120,000, but there is no garantie for it.
# 
# ---

# In[ ]:


sns.relplot(x="sqft_living15", y="price", hue="price", col="condition",
            size="price", sizes=(5, 500), col_wrap=3, data=dataset);


# In[ ]:


# plot this dataframe with seaborn
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x='condition', y='price', hue='yr_built_cat', data=dataset, palette="PuBu_r")

# add title, legend and informative axis labels
ax.set_title('\nMedian Prices depending on Condition and Age of Houses\n', fontsize=14, fontweight='bold')
ax.set(ylabel='Price', xlabel='Condition')
ax.legend(loc=2);


# Houses which lay in the 3-5 catergory of condition (especially condition 4 for modern homes) seem to have higher price than older homes.
# 
# ---

# In[ ]:


sns.relplot(x="sqft_living15", y="price", hue="price", col="grade", 
            size="price", sizes=(5, 500), col_wrap=3, data=dataset);


# In[ ]:


# plot this dataframe with seaborn
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x='grade', y='price', hue='yr_built_cat', data=dataset, palette="PuBu_r")

# add title, legend and informative axis labels
ax.set_title('\nMedian Prices depending on Condition and Age of Houses\n', fontsize=14, fontweight='bold')
ax.set(ylabel='Price', xlabel='Grade')
ax.legend(loc=2);


# Grade reflects in the price more in older houses, especially houses older than 50 years.
# 
# **Conclusion:**
# - Whether you renovate or not is a matter of the outcome you desire. But a simple home improvement seems to help with the selling price.
# - The condition your house is in is important, especially you want to make sure you are in the category 3-5.
# - The grade given for your house reflects in the price on older houses and therefore especially important.

# ___
# 
# ### Categorize Data
# 
# We need to create dummy vars for our categorical variables. **One-hot encoding** shall do the trick. 

# In[ ]:


dataset['condition'] = dataset['condition'].astype('category', ordered = True)
dataset['waterfront'] = dataset['waterfront'].astype('category', ordered = True)
dataset['is_renovated'] = dataset['is_renovated'].astype('category', ordered = False)
dataset['view'] = dataset['view'].astype('category', ordered = False)

# Create category 'has_basement'
dataset['has_basement'] = dataset.sqft_basement.apply(lambda x: 1 if x > 0 else 0)
dataset['has_basement'] = dataset.has_basement.astype('category', ordered = False)


# In[ ]:


# Set dummies (we may want to add zipcode as well)
cat_columns = ['floors', 'view', 'condition', 'waterfront', 'is_renovated', 'has_basement']

for col in cat_columns:
    dummies = pd.get_dummies(dataset[col])
    dummies = dummies.add_prefix("{}_".format(col))
    
    dataset.drop(col, axis=1, inplace=True)
    dataset = dataset.join(dummies)


# In[ ]:


# replace the '.' in the column name
for col in dataset.columns:
    if col.find('.') != -1: 
        dataset.rename(columns={col: col.replace('.', '_')}, inplace=True)


# ### Dropping features
# 
# Finally we shall drop eature we are still carrying around but for sure are not needed.

# In[ ]:


# dropping id and date
dataset.drop(['id', 'date', 'lat', 'long'], axis = 1, inplace = True)


# In[ ]:


dataset.head()


# In[ ]:


dataset.describe()


# ## Normalize dataset

# In[ ]:


# Using MinMax
#
minmax_df = dataset[['house_age', 'yr_since_renovation', 'zipcode']]

scaler = preprocessing.MinMaxScaler()
minmax_scaled_df = scaler.fit_transform(minmax_df)
minmax_scaled_df = pd.DataFrame(minmax_scaled_df, columns=['house_age', 'yr_since_renovation', 'zipcode'])


# In[ ]:


# Using Robust for price and sqft
#
robust_df = dataset[['price', 'sqft_above', 'sqft_living15', 'sqft_lot15']]

scaler = preprocessing.RobustScaler()
robust_scaled_df = scaler.fit_transform(robust_df)
robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['price', 'sqft_above', 'sqft_living15', 'sqft_lot15'])


# ## Concat normalized data and selected feature into new dataframe

# In[ ]:


dataset_ols = pd.concat([dataset[['grade', 'bedrooms', 'bathrooms', 'condition_3', 'condition_4', 
                                  'condition_5']], minmax_scaled_df, robust_scaled_df], axis=1)


# In[ ]:


dataset_ols.head()


# ___
# 
# ## Regression Model

# In[ ]:


ols_results = []
if len(ols_results) != 1:
    ols_results = [['ind_var', 'r_squared', 'intercept', 'slope', 'p-value', 'normality (JB)']]


# In[ ]:


features = ['grade', 'bedrooms', 'bathrooms', 'house_age', 'yr_since_renovation', 'sqft_above',
            'sqft_living15', 'sqft_lot15', 'zipcode', 'condition_3', 'condition_4', 'condition_5']


# In[ ]:


def run_ols_regression(store_results, data, target, feature, show_plots=False):
    """
    Run ols model, prints model summary, displays plot_regress_exog and qqplot
    
    :param data: dataset
    :param target: target feature name
    :param feature: feature name
    :return:
    """
    
    formula = target + '~' + feature
    model = ols(formula=formula, data=data).fit()

    df = pd.DataFrame({feature: [data[feature].min(), data[feature].max()]})
    pred = model.predict(df)

    if show_plots:
        print('Regression Analysis and Diagnostics for formula: ', formula)
        print('\n')

        fig = plt.figure(figsize=(16, 8))
        fig = sm.graphics.plot_regress_exog(model, feature, fig=fig)
        plt.show();

        residuals = model.resid
        fig = sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
        fig.show();
    
    # append all information to results
    store_results.append([feature, model.rsquared, model.params[0], model.params[0],
                        model.pvalues[1], sms.jarque_bera(model.resid)[0]])


# ### Check out each feature by itself

# In[ ]:


for feature in features:
    run_ols_regression(ols_results, dataset_ols, 'price', feature)


# In[ ]:


pd.DataFrame(ols_results)


# ### Go through the selection process

# In[ ]:


y = dataset_ols['price']
X = dataset_ols.drop(['price'], axis=1)


# In[ ]:


result = stepwise_selection(X, y, verbose = True)
print('resulting features:')
print(result)


# ### Do your regression model

# In[ ]:


pred = '+'.join(features)
formula = 'price~' + pred


# In[ ]:


model = ols(formula=formula, data=dataset_ols).fit()
model.summary()


# We shall drop 'yr_since_renovation', 'sqft_above', 'bedrooms', 'condition_3', 'condition_4' 
# from the feature list

# ## Regression Model Validation

# In[ ]:


y = dataset_ols['price']
X = dataset_ols.drop(['price', 'yr_since_renovation', 'sqft_above', 'condition_3', 'condition_4'], axis=1)


# In[ ]:


X_int = sm.add_constant(X)
model = sm.OLS(y, X_int).fit()
model.summary()


# ---
# Iterating over the feature seem not to make any improvements on our R-squared value.
# 
# ##### Furthermore:
# - The regression output shows that most variables are statistically significant with **p-values** less than 0.05. 
# - With regards to the **coefficients**, most variables are positively correlated with the price, only (lower) grades, a renovation status years back and the building year are negatively correlated.
# 
# #### Final thoughts:
# The grade of a house has more impact on the price if a house is older than 15 years and even more so if older than 50 years. That said the grade reflects to some extend the size of a house (cabin vs mansoin) but mostly the condition the house is in (poor, average all the way to luxury). The condition of a house has some effect on the price if it is in the range 3 - 5 on all houses no matter the age. One can make the argument that the conidition of a house is coralated wheather or not it is renovated. Renovation has a effect on the price of the house of a median average of $120,000. It is not known what kind of investment one has to make in order to gain such return.
# 

# ## Sanity Check using sklearn

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test))


# In[ ]:


# Fitting the model to the training data
linreg = LinearRegression().fit(X_train, y_train)

# Calc preditors on the train and test set
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)


# In[ ]:


# Calc residuals
train_residuals = y_hat_train - y_train
test_residuals = y_hat_test - y_test


# In[ ]:


# Calc MSE (Mean Squared Error)
train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squarred Error:', train_mse)
print('Test Mean Squarred Error:', test_mse)


# ### Residual Histogram

# In[ ]:


fig = plt.figure(figsize=(16, 8))
sns.distplot(y_test - y_hat_test, bins=100);
# sns.distplot(test_residuals, bins=50)


# ## Cross Validation
num = 100
train_err = []
test_err = []

for i in range(num):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    linreg.fit(X_train, y_train)

    y_hat_train = linreg.predict(X_train)
    y_hat_test = linreg.predict(X_test)
    
    train_err.append(mean_squared_error(y_train, y_hat_train))
    test_err.append(mean_squared_error(y_test, y_hat_test))
    
plt.scatter(list(range(num)), train_err, label='Training Error')
plt.scatter(list(range(num)), test_err, label='Testing Error')
plt.legend();
# In[ ]:


train_error = []
test_error = []

for t in range(5, 95):
    train_temp = []
    test_temp = []
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t/100)
        linreg.fit(X_train, y_train)

        y_hat_train = linreg.predict(X_train)
        y_hat_test = linreg.predict(X_test)

        train_temp.append(mean_squared_error(y_train, y_hat_train))
        test_temp.append(mean_squared_error(y_test, y_hat_test))
    
    # save average train/test errors
    train_error.append(np.mean(train_temp))
    test_error.append(np.mean(test_temp))

fig = plt.figure(figsize=(16, 12))
plt.scatter(range(5, 95), train_error, label='training error')
plt.scatter(range(5, 95), test_error, label='testing error')

plt.legend()
plt.show()


# In[ ]:


from sklearn.model_selection import cross_val_score

cv_5_results  = np.mean(cross_val_score(linreg, X, y, cv=5, scoring="neg_mean_squared_error"))
cv_10_results = np.mean(cross_val_score(linreg, X, y, cv=10, scoring="neg_mean_squared_error"))
cv_20_results = np.mean(cross_val_score(linreg, X, y, cv=20, scoring="neg_mean_squared_error"))


# In[ ]:


print(cv_5_results, cv_10_results, cv_20_results)


# ### Regression Evaluation

# In[ ]:


print('Measure of the quality of an estimator - values closer to zero are better\n\n')
print('MAE: ', metrics.mean_absolute_error(y_test, y_hat_test))
print('MSE: ', metrics.mean_squared_error(y_test, y_hat_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_hat_test)))


# ---
# ## Answers
# 1. Zip Code (neighborhood) can be an indicator for house prices 
#     (see the top 5 zip codes 98039, 98004, 98040, 98112, 98102).
# 2. Housing density in condery is less an indicator for the house price.
# 3. Regards grade and condition of the house I believe that with the data given we have too little informations and therefore is inconclusive.
