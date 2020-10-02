#!/usr/bin/env python
# coding: utf-8

# # The Data-Set and Project Task
# 
# The data-set contains information on used cars ads which were publushed on Craigslist website.
# 
# Craigslist is the world's largest collection of used vehicles for sale. The data was scraped as to include every used vehicle entry within the United States on posted on Craigslist.
# 
# This data is scraped every few months, it contains most all relevant information that Craigslist provides on car sales including columns like price, condition, manufacturer, latitude/longitude and more.
# 
# In This project I will try to create a data pipeline process which will be able to predict the offered sale price of used cars sale offers posts, or, in other words, to predict a used car market value, by its attributes.

# # **Preliminaries**

# ## Importing

# In[ ]:


import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split as split

from sklearn.metrics import mean_squared_error as mse

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree


# In[ ]:


pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# ## Reading the Data

# In[ ]:


# For Kaggle Env run...

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

cars = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')
cars.head()


# In[ ]:


# cars = pd.read_csv('vehicles.csv')
# cars.head()


# ## Initial Data Inspection

# ### General information and NaN values detection

# In[ ]:


print(cars.shape, '\n')
print(cars.info())


# ### NaN values detection

# In[ ]:


cars.isna().sum()


# #### Initial Findings:
# 1. 'county' column contains only NaN values and should be removed.
# 2. 'url', 'region_url' and 'image_url' columns, contains only URL adressess with no value for analysis or for prediction and should be removed.
# 3. 'id' column offers no additional inforamtion whereas 'vin' column serves as a unique identifier, and should be removed.
# 4. 'description' column contains vehicle description that can not offer any contribution for regression prediction (perhaps can assit with NLP?) and should be removed.
# 5. 'condition', 'cylinders', 'vin', 'drive', 'type' and 'paint_color' columns contins more than 150,000 NaN values each, and should be handeled if kept.

# ### Duplicates Detection
# 
# Let us try to locate duplicates rows by the 'vin' column, for its values should represnt a unique identifier

# In[ ]:


print('Number of duplicated rows according to "vin" column:', cars.vin.duplicated().sum())
print('Number of duplicated rows according to all columns:', cars.duplicated().sum())


# That is strange, duplicated() according to 'vin' column returns 358081 rows as duplicated. It will be a good idea to verify by the 'year' column and maybe by other columns... 

# In[ ]:


print('Number of duplicated rows according to "vin" and "year" columns:', cars.duplicated(subset=['year', 'vin']).sum())
print('Number of duplicated rows according to "vin" and "id" columns:', cars.duplicated(subset=['id', 'vin']).sum())
print('Number of duplicated rows according to "vin", "id" and "year" columns:', cars.duplicated(subset=['id', 'year', 'vin']).sum())
print('Number of duplicated rows according to "id" column:', cars.id.duplicated().sum())


# In[ ]:


print('vin column nunique:', cars.vin.nunique(), '\n')
print('vin column value_counts:', cars.vin.value_counts())


# In[ ]:


print('id column nunique:', cars.id.nunique(), '\n')
print('id column value_counts:', cars.id.value_counts())


# #### More Findings
# 
# 1.   'vin' column contains garbage data, can not be used as a unique identifier and should be removed.
# 2.   'id' column can serve for duplicate detection, but whereas no duplicated found, 'id' column can also be removed, while the index can serve as a unique identifier.

# ### Inspecting Data-Set data-types

# In[ ]:


cars.dtypes.value_counts()


# In[ ]:


print('Numeric features:', '\n', cars.select_dtypes(exclude=object).columns, '\n')
print('Object features:', '\n', cars.select_dtypes(include=object).columns)


# #### More Findings
# 
# 1.   Only 'price' and 'odometer' columns contains actual numeric data. 'county' is all NaN values, 'year' is realy categorical, 'id' is just an indetifier, and 'lat' and 'long' contains geographical data.
# 2.   'url' columns contains no data, 'cylinders' should probably be conversed to a numeric feature.

# ### Inspecting 'price' (target) column values

# In[ ]:


print('Number of Unique values in the price (target) feature:', cars.price.nunique(), '\n')


# In[ ]:


cars.price.describe()


# In[ ]:


cars.price.value_counts()


# #### The most common value, by far, is zero (0), which does not represent an actual value of 0, but sellers refrained from disclosing their asking proce. Price, beeing the target feature can not have such concealed price values and thay should all be removes from the data-set. Furthrt research parsuant to the zero priced rows deletion, revealed many 1 USD priced rows, which should also be removed. Domain exploration shows that the minimum price for a functional used car would be around 250 USD, hence dropping rows of cars priced less than that should serve as a good treshhold.

# In[ ]:


cars.price.loc[cars['price'] > 120000]


# In[ ]:


cars.price.loc[cars['price'] > 120000].value_counts()


# #### Domain research shows that the average price for a recent model of a sedan car type is about 15,000 USD and the most expensive SUV car type is about 26,000 USD on average. A used Ferrary sells for less than 100,000 USD, Porsche for 56,000 USD, and a Nercedes-Benz for 30,000 USD. Only extremely special cars such as Lamborghini (185k) or McLaren (179k) sells for mor than 120,000 USD, Therefore, it will be safe to consider any sell price above 120k as outliars. That conclusion is supported by the fact that many of those prices are obvious garbage such as the '123456' and '123456789' strings shown above, and those 10 figures numbers. [U.S Used Cars Price Trends]: (https://www.cargurus.com/Cars/price-trends/)

# # Initial Cleaning the Data Set

# ### Deleting 'URL', 'county', 'vin' and 'id' columns

# In[ ]:


def del_by_label(df, label_list, axis):
    return df.drop(labels=label_list, axis=axis)


# In[ ]:


print(cars.shape)
cars = del_by_label(cars, ['url', 'region_url', 'image_url', 'county', 'vin', 'id'], 1)
cars.shape


# ### Deleting '0', '1' and more than '120k' USD priced rows

# In[ ]:


print(cars.shape)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'price'] < 120].index), 0)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'price'] > 120000].index), 0)
cars.shape


# # Splitting the data-set to features and target, train and test, sub-sets

# In[ ]:


# X_cars = cars.drop(labels='price', axis=1)
# y_cars = cars.price


# In[ ]:


# X_cars_train, X_cars_test, y_cars_train, y_cars_test = split(X_cars, y_cars, random_state=1)


# In[ ]:


# print('X_cars_train shape:', X_cars_train.shape)
# print('X_cars_test shape:', X_cars_test.shape)
# print('y_cars_train shape:', y_cars_train.shape)
# print('y_cars_testn shape:', y_cars_test.shape)


# # Exploratory Data Analysis

# In[ ]:


cars.head(10)


# In[ ]:


cars.shape


# ## Price (Target) feature ditribution

# In[ ]:


print('Number of Unique values in the price (target) feature:', cars.price.nunique(), '\n')


# In[ ]:


cars.price.describe()


# In[ ]:


cars.hist(column='price', grid=True, bins=80, range=(0, 80000), figsize=(10, 4))


# In[ ]:


print('Number of values above 50,000:', cars.price.loc[cars['price'] > 50000].count())
print('Number of values above 60,000:', cars.price.loc[cars['price'] > 60000].count())
print('Number of values above 70,000:', cars.price.loc[cars['price'] > 70000].count())
print('Number of values above 80,000:', cars.price.loc[cars['price'] > 80000].count())


# In[ ]:


sns.violinplot(x=cars.price)


# ### Conclusions:
# 1. The vast majority of 'price' values is found between 100 USD to 30,000 USD.
# 2. Range of 'price' column values is probably wide enough to require a Log transform.
# 3. Whereas number of observations priced over 60,000 USD is relatively small, some quality check should be done for those raws to make sure it can be safly feed into a model.

# ## Used Cars Locations

# In[ ]:


fig, ax = plt.subplots(figsize=(22, 5))
ax.set_title('Number of Used Cars Ads per State')
sns.barplot(x=cars.state.value_counts().index, y=cars.state.value_counts().values, ax=ax)


# In[ ]:


print("Number of states included on 'state' column:", cars.state.nunique())


# ### Does used cars median price vary between states?

# In[ ]:


fig, ax = plt.subplots(figsize=(22, 5))
ax.set_title('Median Marked price per State')
sns.barplot(x=cars.state.value_counts().index, y=cars.groupby('state')['price'].median(), ax=ax)


# ### Is 'region' and 'lat' and 'long' columns stores an informative added value?

# In[ ]:


price_per_location = {'state': cars.state, 'median_price_per_state': cars.groupby('state')['price'].transform('median'),
                      'region': cars.region, 'median_price_per_region': cars.groupby(['state', 'region'])['price'].transform('median'),
                      'lat': cars.lat, 'long': cars.long, 'median_price_per_lat_and_long': cars.groupby(['state', 'lat', 'long'])['price'].transform('median')
                     }

price_per_location = pd.DataFrame(price_per_location)
price_per_location


# In[ ]:


grouped_price_per_location = (price_per_location.groupby(['state', 'region', 'lat', 'long'])
                              ['median_price_per_state', 'median_price_per_region', 'median_price_per_lat_and_long'].mean())
grouped_price_per_location


# ### It seems that median price within each region contained in each of the states is different than the median price of the state.
# #### Let's view New-York state and Califorania state regions, for instance.

# In[ ]:


grouped_price_per_location.loc['ny', :]


# In[ ]:


grouped_price_per_location.loc['ca', :]


# ### How many pairs of each unique 'lat' and 'long' pairs are there?

# In[ ]:


lat_long_unique_count = grouped_price_per_location.groupby(['lat', 'long'])['median_price_per_lat_and_long'].count()
lat_long_unique_count


# In[ ]:


lat_long_unique_pairs_instances = lat_long_unique_count.value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(14, 4))
ax.set_title("'lat' and 'long' Unique Pairs Count")
ax.set_xlabel('Number of Unique Pairs Instances')
ax.set_ylabel('Unique Pairs')
sns.barplot(x=lat_long_unique_pairs_instances.index, y=lat_long_unique_pairs_instances, ax=ax)


# In[ ]:


print("Number of unique values in 'state' column:", cars.state.nunique())
print("Number of unique values in 'region' column:", cars.region.nunique())


# ### Conclusions:
# 1. 'region' and 'lat' and 'long' columns does holds excess information over 'state' column.
# 2. The vast majority of 'lat' and 'long' pairs (62,271 of 71,189, 88%) has only 1 observation, Only 10 pairs (out of 71,189, or 0.014%) has more than 30 observations, and therefore those pairs are statistically unusable. It is most likely because the 'lat' and 'long' pairs are spesific to each used car location (or specific to each PC from which the used car ad was posted...).
# 3. Each state has few regions contained in it, so as the region column has the state column information aggregated in it.
# 4. There are 8 times more unique values on the 'region' column than the 'state' column.

# ## Used Cars Year of Manufacture and Odometer

# In[ ]:


print("Number of NaN values in 'year' column:", cars.year.isnull().sum())


# In[ ]:


cars.year.value_counts().to_frame().T


# In[ ]:


fig, ax = plt.subplots(figsize=(24, 4))
ax.set_title("Median Price per Year")
plt.xticks(rotation=90)
sns.barplot(x=cars.year, y=cars.groupby('year')['price'].transform('median'), ax=ax)


# ### From Domain research it became clear that used cars older than 30 (used cars manufactured prior to 1990) are not usually traded on cars second market, and it is  evidently demonstrated from the graph, whereas median price is trending down as expected from 2020 cars down to a minimum median price on 1988 cars, a trend that reflects the true age range of on the cars second market, while older cars prices from that point down, are (fluctuately) trended up, a trend which is most likely deriving from the added value of those older cars "classic" or "antique" attributes, and does not reflect their value as a utility used cars. (The anomalies of the years such as 1909 and 1945 originates from not enough used cars from those years and from cahnges of the fluctuate up-trend)

# ### The conclusion is that a used car prediction model should be splitted to two, a model for utilty used cars, manufactured from 1988 and up, and a model for older cars, which most of their value stems from their classical and preservatin traits. Notwithstanding, The data-set does not contain any features relating specifically to old used cars, hence building a predictive model for their prices is not plausible. That leaves only the utility used cars predictive model.

# In[ ]:


print("Number of NaN values in 'odometer' column:", cars.odometer.isnull().sum())


# In[ ]:


print('Number of cars with odometer value equal to zero:', cars.odometer[cars.odometer == 0].count())


# In[ ]:


print('Number of cars with odometer value equal to zero and condition value "new":',cars.loc[((cars.odometer == 0) & (cars.condition == 'new')), :].shape[0])


# In[ ]:


print('Number of cars with age value other than zero and condition value "new":', cars.loc[((cars.year != 2020) & (cars.condition == 'new')), :].shape[0])


# In[ ]:


print('Number of cars with age value other than zero and odometer value is zero:', cars.loc[((cars.odometer == 0) & (cars.year != 2020)), :].shape[0])


# In[ ]:


print('Number of cars with age value other than zero and odometer value is zero:', cars.loc[((cars.odometer == 0) & (cars.year < 1988)), :].shape[0])


# #### There are instances of cars older than 2020, with condition othe than 'new', and odometer value of '0'. Some of those instances are older than 1988, which might suggest 'classic' car where odometer value is not importent, but the other such instances are probably bed data which can confuse the model.

# In[ ]:


cars.hist(column='odometer', grid=True, bins=100, range=(0, 400000), figsize=(10, 4))


# In[ ]:


print('Number of cars with odometer value above 400,000:', cars.odometer[cars.odometer > 400000].count())


# In[ ]:


print('Number of cars with odometer value less than 100:', cars.odometer[cars.odometer < 10].count())


# In[ ]:


cars.loc[cars.year > 1987.00, :].groupby('year')['odometer'].value_counts()


# In[ ]:


cars[['year', 'odometer']].corr()


# ## Maufacturer Model Type and Size features

# In[ ]:


cars.groupby('manufacturer')['price'].mean().sort_values().plot(kind='barh', figsize=(11, 11), legend='True', color='r')


# In[ ]:


cars.groupby('model')['price'].mean().sort_values()


# In[ ]:


cars.groupby('model')['price'].mean().sort_values().plot(figsize=(12, 4))


# In[ ]:


cars.groupby(['manufacturer', 'model'])['price'].mean().to_frame().T


# In[ ]:


print(cars.type.value_counts(dropna=False))


# In[ ]:


print(cars['size'].value_counts(dropna=False))


# ### It seems that those 4 features poses the highest challenge, as each of them incapsulate some information but not all of the main typing information, and while the 'size' feature holds to many NaN values, and the 'model' feature contains no less than 35,270 unique values, many of them identical but written in different ways, characters which renders those features unuseable for a predictive model, and while it is unclear whether manufacturer and type feature holds enough of the "car typing" information. Further, 'model' feature does not reveal much varience between the many different models of each manufacturer, as ilustrated by the visualization, and type feature also contains significant amount of NaN values.

# ## Condition and Title_status Features

# In[ ]:


print(cars.condition.value_counts(dropna=False))


# In[ ]:


sns.barplot(x=cars.loc[cars.year > 1987, :].condition, y=cars.loc[cars.year > 1987, :].price)


# In[ ]:


cars.loc[cars.year > 1987, :].groupby('condition')['year'].mean()


# ### While 40% of the condition feature values are NaN, concluding from feature values seem somewhat problematic. As the differences between effect of feature values on price is clear imputing based on mean price might be possible. 

# In[ ]:


print(cars.title_status.value_counts(dropna=False))


# ### The model we are after is for utility used cars, so we should eliminate salvaged and for parts only cars, as well as cars with missing titles. That will leave clean or rebuilt as the main categories (lien is just a financial issue that should be lifted prior to purchase). Therefore, we can transform this feature to a binary.

# ## Cylinders Fuel Transmission and Drive Features

# In[ ]:


print(cars.cylinders.value_counts(dropna=False))


# In[ ]:


cars.groupby('cylinders')['fuel'].value_counts()


# In[ ]:


cars.groupby('cylinders')['transmission'].value_counts()


# ### Cylinders feature does not correlate significantly with the other technical attributes, should be transformed to int dtype and handle the high amount of NaN values.

# In[ ]:


print(cars.fuel.value_counts(dropna=False))


# In[ ]:


cars.groupby('fuel')['type'].value_counts(dropna=False).head(30)


# ### The vast majority of the observations are valued 'gas', while most of the diesel cars are typed as big cars and can be distinguished this way, which leaves only electricas seperated from gas fuled cars. Probably this feature should be binary, gas or not.

# In[ ]:


print(cars.transmission.value_counts(dropna=False))


# ### The vast majority of the observations are valued 'automatic', probably this feature should be binary.

# In[ ]:


print(cars.drive.value_counts(dropna=False))


# In[ ]:


sns.barplot(x=cars.loc[cars.year > 1987, :].drive, y=cars.loc[cars.year > 1987, :].price)


# In[ ]:


cars.groupby('drive')['type'].value_counts(dropna=False).head(30)


# ### This feature looks very distinctive, but with many NaN values that should be handled.

# ## Paint_Color and Description

# In[ ]:


print(cars.paint_color.value_counts(dropna=False))


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=cars.loc[cars.year > 1987, :].paint_color, y=cars.loc[cars.year > 1987, :].price)


# ### Probably paint_color feature can be reduced to 3 values, Black, white or other, or even 2 values, black/white or other.

# In[ ]:


cars.loc[0, 'description']


# In[ ]:


cars.loc[2, 'description']


# In[ ]:


cars.loc[10000, 'description']


# ### The Description feature somtimes contains all of the car data (mostly when the seller is a professional dealership), and can be used for imputing, but somtimes can be empty of usfull data.

# # Preprocessing

# ## Feature Selection - Dropping Columns

# #### According to EDA findings, we can eliminate: 'model', 'size', 'state', 'lat' and 'long' columns.

# In[ ]:


print(cars.shape)
cars = del_by_label(cars, ['model', 'size', 'state', 'lat', 'long'], 1)
cars.shape


# #### Where 'condition' columns contains more than 55% NaN values, and very subjective in nature it should be dropped

# In[ ]:


print(cars.shape)
cars = del_by_label(cars, 'condition', 1)
cars.shape


# ## Feature Selection - Dropping Rows

# #### According to EDA findings, we can eliminate all rows where: cars.year < 1988.0, or cars.year == 2021.0, cars.year == 0.0.

# In[ ]:


print(cars.shape)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'year'] < 1988].index), 0)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'year'] == 2021].index), 0)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'year'] == 0].index), 0)
cars.shape


# #### According to EDA findings, we can eliminate all rows where cars.title_status == 'salvage'.

# In[ ]:


print(cars.shape)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'title_status'] == 'salvage'].index), 0)
cars.shape


# #### According to EDA findings, we can eliminate all rows where: cars.title_status == 'missing' or 'parts only'.

# In[ ]:


print(cars.shape)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'title_status'] == 'missing'].index), 0)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'title_status'] == 'parts only'].index), 0)
cars.shape


# #### According to EDA findings, we can eliminate all rows where: cars.cylinders == 'other'

# In[ ]:


print(cars.shape)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'cylinders'] == 'other'].index), 0)
cars.shape


# #### According to EDA findings, we can eliminate all rows  in 'odometer', 'year' where value == NaN

# In[ ]:


print(cars.shape)
cars = cars.dropna(axis=0, subset=['odometer', 'year'])
cars.shape


# ## Adding features

# ### Odometer provide information about the total use of the car, but it should also prove usefull to provide a feature of rhe relative use of the car, in relation to its age

# In[ ]:


cars.year = cars.year.replace(to_replace={0: 1})


# In[ ]:


cars.insert(loc=7, column='milage_per_year', value=(cars.odometer / (2020 - cars.year)), allow_duplicates=True)


# In[ ]:


cars.milage_per_year = cars.milage_per_year.fillna(value=0)
cars.milage_per_year = cars.milage_per_year.replace(to_replace={np.inf: 0, -np.inf: 0})


# ## Feature Engineering

# ### Transforming 'year' feature into an 'age' feature by adding 'age' based on 'year' and dropping 'year'.

# In[ ]:


cars.insert(loc=2, column='age', value=(2020 - cars.year), allow_duplicates=True)


# In[ ]:


cars = del_by_label(cars, 'year', 1)


# ### Reducing values in 'paint_color' feature

# In[ ]:


color_dict = {'white': 'white_black', 'black': 'white_black', 'silver': 'other', 'blue': 'other', 'red': 'other',
              'grey': 'other', 'green': 'other', 'custom': 'other', 'brown': 'other', 'orange': 'other', 'yellow': 'other',
              'purple': 'other', np.nan: 'other'}
cars.paint_color = cars.paint_color.replace(color_dict)


# ### Reducing values in 'cylinders' feature

# In[ ]:


cylinders_dict = {'6 cylinders': 6, '8 cylinders': 8, '4 cylinders': 4, '5 cylinders': 5, '10 cylinders': 10,
              '3 cylinders': 3, '12 cylinders': 12, np.nan: 'other'}
cars.cylinders = cars.cylinders.replace(cylinders_dict)


# ### Reducing values in 'fuel' feature

# In[ ]:


fuel_dict = {'gas': 'gas', 'diesel': 'other', 'other': 'other', 'hybrid': 'other', 'electric': 'other', np.nan: 'other'}
cars.fuel = cars.fuel.replace(fuel_dict)


# ### Reducing values in 'title_status' feature

# In[ ]:


title_dict = {'clean': 'clean', 'rebuilt': 'other', 'lien': 'other', np.nan: 'other'}
cars.title_status = cars.title_status.replace(title_dict)


# ### Reducing values in 'transmission' feature

# In[ ]:


transmission_dict = {'automatic': 'automatic', 'manual': 'other', 'other': 'other', np.nan: 'other'}
cars.transmission = cars.transmission.replace(transmission_dict)


# ### Converting 'drive' feature NaN values to 'other'

# In[ ]:


cars.drive.value_counts()


# In[ ]:


cars.drive = cars.drive.fillna(value='other')


# In[ ]:


cars.drive.value_counts()


# ## Transforming all categorical features with 4 or less values to a boolean array

# In[ ]:


cars = pd.get_dummies(data=cars, columns=['fuel', 'title_status', 'transmission', 'drive', 'paint_color'])


# ## Cutting out 'description' feature

# In[ ]:


description_col = cars.description
print(cars.shape)
cars = del_by_label(cars, ['description'], 1)
cars.shape


# In[ ]:


description_col.str.contains(pat='^V[0-9]|^V[0-9][0-9]', case=False, regex=True).sum()


# ## Scaling - Target

# #### Transforming 'price' to 'log_price' by adding 'log_price' based on 'price' and dropping 'price'.

# In[ ]:


cars['log_price'] = np.log1p(cars.price)


# In[ ]:


cars = del_by_label(cars, 'price', 1)


# ## Scaling - features

# ### To make it easier on all models we might use we will scale down 'odometer' and 'milage_per_year' features

# In[ ]:


cars.odometer = cars.odometer / 100


# In[ ]:


cars.milage_per_year = cars.milage_per_year / 100


# ## Reset Index

# In[ ]:


cars = cars.reset_index(drop=True)


# # Splitting the data-set to features and target, train and test, sub-sets

# In[ ]:


X_cars = cars.drop(labels='log_price', axis=1)
y_cars = cars.log_price


# In[ ]:


X_cars_train, X_cars_test, y_cars_train, y_cars_test = split(X_cars, y_cars, random_state=1)


# In[ ]:


print('X_cars_train shape:', X_cars_train.shape)
print('X_cars_test shape:', X_cars_test.shape)
print('y_cars_train shape:', y_cars_train.shape)
print('y_cars_testn shape:', y_cars_test.shape)


# #### Transforming multy valued categorical features by target median

# In[ ]:


class TargetMedianPerValTransformer():  
    
    def __init__(self, val_count, exclude=None):
        self.val_count = val_count
        self.exclude_list = exclude
        self.fit_dict = dict()
    
    def col_labels_for_fit(self, X):
        label_mask = ((X.nunique() > self.val_count) & (X.dtypes == object))
        for label in self.exclude_list:
            label_mask[label] = False
        return label_mask[label_mask].index
    
    def label_dict_generator(self, X, y):
        for label in self.fitted_feature_labels:
            if X[label].isna().any():
                X[label] = X[label].replace(to_replace=np.nan, value='other')
            grouper = X.groupby(label)
            self.fit_dict[label] = dict()
            for name, group in grouper:
                group_indices = grouper.get_group(name).index
                self.fit_dict[label][name] = y[group_indices].mean()
    
    def fit(self, X, y):
        self.fitted_feature_labels = self.col_labels_for_fit(X)
        self.label_dict_generator(X, y)
        return self
    
    def transform(self, X):
        for label in self.fitted_feature_labels:
            X[label] = X[label].map(self.fit_dict[label])
            if X[label].isna().any():
                X[label] = X[label].where(cond=(X[label] == np.nan), other=(X[label].median()))
        return X


# In[ ]:


transform_multy_val_car_features = TargetMedianPerValTransformer(val_count=4, exclude=['description'])


# In[ ]:


transform_multy_val_car_features.fit(X_cars_train, y_cars_train)


# In[ ]:


X_cars_train = transform_multy_val_car_features.transform(X_cars_train)


# In[ ]:


X_cars_test = transform_multy_val_car_features.transform(X_cars_test)


# ## Converting float64 dtype to float32 dtype for model fitting

# In[ ]:


# X_cars_train = X_cars_train.astype(dtype={'region': np.float32, 'age': np.float32,
#                    'manufacturer': np.float32, 'cylinders': np.float32,
#                    'odometer': np.float32, 'milage_per_year': np.float32,
#                    'type': np.float32})

# X_cars_test = X_cars_test.astype(dtype={'region': np.float32, 'age': np.float32,
#                    'manufacturer': np.float32, 'cylinders': np.float32,
#                    'odometer': np.float32, 'milage_per_year': np.float32,
#                    'type': np.float32})


# In[ ]:


X_cars_train.info()


# In[ ]:


X_cars_train.describe()


# In[ ]:


X_cars_test.describe()


# In[ ]:


X_cars_train.head(10)


# # predictive models 

# ## Decision Tree

# In[ ]:


cars_tree_model = DecisionTreeRegressor(criterion='mse',
                                        splitter='random',
                                        min_samples_split=7000,
                                        min_samples_leaf=2600)


# In[ ]:


cars_tree_model.fit(X_cars_train, y_cars_train)


# In[ ]:


print(cars_tree_model.max_features_)
print(cars_tree_model.n_features_)
print(cars_tree_model.n_outputs_)
print(cars_tree_model.feature_importances_)


# In[ ]:


for feature, importance in zip(X_cars_train.columns, cars_tree_model.feature_importances_):
    print(f'{feature:12}: {importance}')


# ### Visualizing the Tree

# In[ ]:


get_ipython().system('pip install pydot')
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz


# In[ ]:


def visualize_tree(model, md=10, width=1000):
    dot_data = StringIO()  
    export_graphviz(model, out_file=dot_data, feature_names=X_cars_train.columns, max_depth=md)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]  
    return Image(graph.create_png(), width=width)


# In[ ]:


# visualize_tree(cars_tree_model, md=12, width=1200)


# In[ ]:


# plot_tree(decision_tree=cars_tree_model, filled=True)


# ## Using Tree Model For Prediction

# In[ ]:


y_train_tree_pred = cars_tree_model.predict(X_cars_train)


# In[ ]:


tree_train_pred_dict = {'y_true': y_cars_train, 'y_pred': y_train_tree_pred}
pd.DataFrame(tree_train_pred_dict)


# In[ ]:


ax = sns.scatterplot(x=y_cars_train, y=y_train_tree_pred)
ax.plot(y_cars_train, y_cars_train, 'r')


# ## Model Score (RMSE)

# In[ ]:


RMSE_train = mse(y_cars_train, y_train_tree_pred)**0.5
RMSE_train


# ## Model Validation

# In[ ]:


y_test_tree_pred = cars_tree_model.predict(X_cars_test)


# In[ ]:


tree_test_pred_dict = {'y_true': y_cars_test, 'y_pred': y_test_tree_pred}
pd.DataFrame(tree_test_pred_dict)


# In[ ]:


RMSE_test = mse(y_cars_test, y_test_tree_pred)**0.5
RMSE_test

