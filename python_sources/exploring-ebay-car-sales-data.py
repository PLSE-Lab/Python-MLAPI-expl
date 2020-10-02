#!/usr/bin/env python
# coding: utf-8

# # Exploring Ebay Car Sales Data
# 
# In this guided project, I will use this [dataset](https://www.kaggle.com/piumiu/used-cars-database-50000-data-points) which is provided by [dataquest](https://www.dataquest.io). It is a *smaller* and *dirtier* version of [dataset](https://www.kaggle.com/orgesleka/used-cars-database/data) from [eBay Kleinanzeigen](https://www.ebay-kleinanzeigen.de/).
# 
# My goals are to clean and analyze the included used car listing.

# In[ ]:


import pandas as pd
import numpy as np

autos = pd.read_csv('../input/used-cars-database-50000-data-points/autos.csv', encoding = 'Latin-1')

autos.info()
autos.head()


# My observations about the dataset:
# 
# * There are 50,000 rows, 20 columns.
# * Some columns have NULL values, but less than 20% of total records.
# * Column names are [camelcase](https://en.wikipedia.org/wiki/Camel_case) instead of [snakecase](https://en.wikipedia.org/wiki/Snake_case#Examples_of_languages_that_use_snake_case_as_convention). They will be changed to camel case.
# * Some columns contain numeric values stored as text such as: `price` and `odometer`. These values also will be converted to numeric.
# 
# ## I. Cleaning Column Names:
# 
# To rename column names, we can use pandas [rename()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html) method.

# In[ ]:


autos.columns


# In[ ]:


autos.rename(columns = {'yearOfRegistration':'registration_year', 'monthOfRegistration':'registration_month', 
                       'notRepairedDamage':'unrepaired_damage', 'dateCreated':'ad_created'}, inplace = True)


# To change from camel case to snake case we can use function [underscore()](https://inflection.readthedocs.io/en/latest/_modules/inflection.html#underscore) from [Inflection](https://inflection.readthedocs.io/en/latest/index.html). It is a string transformation library. We will need import [re](https://docs.python.org/2/library/re.html) module before using the function.

# In[ ]:


import re

def underscore(word):
    """
    Make an underscored, lowercase form from the expression in the string.

    Example::

        >>> underscore("DeviceType")
        "device_type"

    As a rule of thumb you can think of :func:`underscore` as the inverse of
    :func:`camelize`, though there are cases where that does not hold::

        >>> camelize(underscore("IOError"))
        "IoError"

    """
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
    word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
    word = word.replace("-", "_")
    return word.lower()

for item in autos.columns:
    col_name = item
    col_name = underscore(col_name)
    autos.rename(columns = {item : col_name}, inplace = True)
    
autos.columns


# The reason why we should use snake case in Python, because [PEP8 Naming Conventions](https://www.python.org/dev/peps/pep-0008/#naming-conventions) suggests us to use lower case with underscore for function, method, variable and constant. Using naming convention is not only help to increase readability, but also let other developers understand your code easier.
# 
# ## II. Initial Exploration and Cleaning:
# 
# In this part, I will determine which columns should be ignored, columns have numeric values but stored as text.

# In[ ]:


autos.describe(include = 'all')


# * Column `seller`, `offer_type`, `abtest`, `unrepaired_damage`, `nr_of_pictures`, `postal_code` contain text values and almost all values are the same. These columns are candidates to be ignored because they do not have useful information for analysis.
# 
# * Column `registration_year` has very weird values (e.g. `9999` and `1000`). I will take a closer look at this later.
# 
# * `price` and `odometer` have numeric data stored as text that needs to be converted.

# In[ ]:


autos['price'] = autos['price'].str.replace('$','')
autos['price'] = autos['price'].str.replace(',','')
autos['price'] = autos['price'].astype(float)
autos['odometer'] = autos['odometer'].str.replace('km','')
autos['odometer'] = autos['odometer'].str.replace(',','')
autos['odometer'] = autos['odometer'].astype(int)
autos.rename(columns = {'odometer' : 'odometer_km'}, inplace = True)
autos.head()


# ## III. Exploring Columns:
# 
# After converting `odometer_km` and `price`, we have to detect outliers (values that unrealistically high or low). There are many ways to archive our goals. You can read this [article](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba) for more details. Now, I am going to use **Z-Score** method to discover outliers.
# 
# You can also watch these videos [Mode, Median, Mean, Range, and Standard Deviation](https://www.youtube.com/watch?v=mk8tOD0t8M0), [Z-Scores and Percentiles](https://www.youtube.com/watch?v=uAxyI_XfqXk&t=4s) for more visualized explanations.
# 
# Simply speaking, **Z-score** tell us how far a certain value from the mean.
# 
# ### Part 1. Exploring Price Column:

# In[ ]:


from scipy import stats

zp = np.abs(stats.zscore(autos['price']))
print(zp)


# Above numbers have not told us much. In most cases, if z-score of a value is less than -3 or greater than 3, that value will be identified as an outlier. We can use numpy [where()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html) to extract index of values that have z-score greater than 3.

# In[ ]:


for item in np.where(zp > 3):
    print(autos['price'].iloc[item])


# In[ ]:


autos['price'].min()


# Minium value in `price` column is `$0.0`. eBay is an auction site, the opening bid for an item could be `$1.0`. So we will keep all values between `$1.0` and `$3,889,999.0` by using [between()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.between.html) method:

# In[ ]:


autos = autos[autos['price'].between(0, 3890000, inclusive = False)]
autos.describe()


# In[ ]:


autos.shape


# ### Part 2. Exploring Odometer_km Column:
# 
# Just need to repeat above steps

# In[ ]:


zo = np.abs(stats.zscore(autos['odometer_km']))
print(zo)


# In[ ]:


for item in np.where(zo > 3):
    print(autos['odometer_km'].iloc[item])


# 836 records are not too many (around 1.72% of total amount). I will exclude them from my dataset.

# In[ ]:


autos = autos[autos['odometer_km'].between(5000, 150001, inclusive = False)]
autos.shape


# ### Part 3. Exploring the Date Columns:
# 
# We have 5 columns that represent date values:
# 
# - `date_crawled`: added by the crawler
# - `last_seen`: added by the crawler
# - `ad_created`: from the website
# - `registration_month`: from the website
# - `registration_year`: from the website
# 
# `registration_month` and `registration_year` are represented as numeric values. The other three columns are represented as timestamp.

# In[ ]:


autos[['date_crawled', 'ad_created', 'last_seen']][0:5]


# The first 10 characters represent the day (e.g. `2016-03-26`). We can extract the day by using `Series.str[:10]`, chains to [value_counts(normalize = True)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html) to generate a distribution, and then sort by the index with [sort_index()](https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.Series.sort_index.html). [Normalization](https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029) will help us to change values in the three columns to a common scale, therefore we can compare them. We also need to count Null values in these columns with [isnull()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.isnull.html) and [sum()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sum.html) to specify should we exclude missing values or not. [If the number of the cases is less than 5% of the sample](https://www.statisticssolutions.com/missing-values-in-data/), we can drop them by set `dropna = True` in `value_count()`.

# In[ ]:


autos[['date_crawled', 'ad_created', 'last_seen']].isnull().sum()


# In[ ]:


autos['date_crawled'].str[:10].value_counts(normalize = True).sort_index()


# Data was crawled daily from beginning of March 2016 to beginning of April 2016. The distribution between each days is almost identical.

# In[ ]:


autos['ad_created'].str[:10].value_counts(normalize = True).sort_index()


# Before March 2016, not many ads were created on eBay. From March 2016 to April 2016, ads were uploaded everyday. The frequency of days is also nearly equal. We can see that percentage of crawling and creating data are similar.

# In[ ]:


autos['last_seen'].str[:10].value_counts(normalize = True).sort_index()


# `last_seen` tells us the day ads were removed from eBay. It could be the car was sold. Most of the ads were end at April 2016 (around 50%). It's not likely there was a spike on sales, but more likely because of crawling period ending. Now, we continue with `registation_year`.

# In[ ]:


autos['registration_year'].describe()


# I once mentioned that `registration_year` has some weired values (e.g. `9999`). We can use z-score to remove those outliers.

# In[ ]:


zr = np.abs(stats.zscore(autos['registration_year']))
for item in np.where(zr > 3):
    print(autos['registration_year'].iloc[item])


# Any vehicles with registration year above 2016 are incorrect. We cannot have first registration greater than `last_seen`.

# In[ ]:


autos = autos[autos['registration_year'].between(1910, 2016)]
autos.shape


# In[ ]:


autos['registration_year'].value_counts(normalize = True).sort_index()


# From 1994 onwards, There was an increase of car registration. Especially in early years of the 21st century and then decreased gradually until 2016.
# 
# ## III. Exploring Price by Brand:
# 
# Next thing is exploring average of price by Brand. First of all, we will chose top 5 most popular Brands.

# In[ ]:


brand_count = autos['brand'].value_counts(normalize = True)
common_brand = brand_count[brand_count > 0.05].index
print(common_brand)


# In[ ]:


autos['brand'].value_counts(normalize = True)


# By far, Volkswagen is the most popluar brand which has number of cars several times higher than other manufacturers. German brand takes 4 of 5 places in the top 5 list. Next is to calculate average price of each brand.

# In[ ]:


mean_by_brand = {}

for brand in common_brand:
    brand_df = autos.loc[autos['brand'] == brand]
    mean_val = brand_df['price'].mean()
    mean_by_brand[brand] = int(mean_val)
    
print(mean_by_brand)


# Audi, Mercedes-Benz, and BMW are expensive. Ford and Opel are cheaper. Volkswagen is between, maybe because its popularity.

# ## IV. Exploring the Link between Price and Odometer:
# 
# To understand the link between average mileage and mean price we need compare two series objects. We can combine data from both series into a dataframe by using pandas [Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html) and [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).

# In[ ]:


mileage_by_brand = {}

for brand in common_brand:
    brand_df = autos.loc[autos['brand'] == brand]
    avg_val = brand_df['odometer_km'].mean()
    mileage_by_brand[brand] = int(avg_val)

print(mileage_by_brand)


# In[ ]:


mean_series = pd.Series(mean_by_brand).sort_values(ascending = False)
print(mean_series)


# In[ ]:


mileage_series = pd.Series(mileage_by_brand).sort_values(ascending = False)
print(mileage_series)


# In[ ]:


mmb = pd.DataFrame(mean_series, columns = ['mean_price'])
mmb['average_mileage'] = mileage_series
mmb


# Average mileage between brands are not too different. Instead, with high-class brand such as Audi, Mercedes-Benz and BMW, the average mileage are higher than the others. It seems that people who own luxury car, tend to use their vehicle longer. 

# *The purpose of this project is mainly to practice what I have learned from [dataquest.io](dataquest.io) - Python for Data Science: Pandas and Numpy Fundamentals. Many techniques, contents in this project were guided by dataquest.io and the following [solution](https://github.com/dataquestio/solutions/blob/master/Mission294Solutions.ipynb).*
