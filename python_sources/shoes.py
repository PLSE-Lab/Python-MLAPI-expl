#!/usr/bin/env python
# coding: utf-8

# **Hello and welcome. In this kernel we'll cover especially the following topic:**
# * Mean shoe prices by brand, color and category

# In[ ]:


import pandas as pd
import os
import numpy
import matplotlib.pyplot as plot
import seaborn as sb
import random
import re
from datetime import datetime
import matplotlib.colors as mc
import colorsys
from scipy import interpolate


# In[ ]:


filepath_women = "../input/Datafiniti_Womens_Shoes.csv"
df = pd.read_csv(filepath_women)


# Get first impression of data.

# In[ ]:


print(df.shape)
df.head(10)


# Check datatypes and null values.

# In[ ]:


df.info()


# For better overview check numerical and categorical columns separately.

# In[ ]:


df.describe()


# In[ ]:


categoricals = df.dtypes[df.dtypes == "object"].index.tolist()
df[categoricals].describe()


# With above information we are able to **delete unnecessary/incomplete columns**:
# * asins (amazon standard identification number):
#   too high null-ratio, content not relevant
# * ean (european article number):
#   too high null-ratio, content not relevant
# * prices.returnPolicy:
#   too high null-ratio
# * colors:
#   too high null-ratio
#   (following is code to get an idea of whether we can replace the lost information with column "prices.color")
# * dimension, manufacturer, manufacturerNumber, prices.availability, prices.condition, prices.merchant, prices.offer, prices.shipping, weight:
#    all have too high null-ratio

# In[ ]:


df[["colors", "prices.color"]].head(15)


# In[ ]:


df[["manufacturer", "manufacturerNumber"]].head(15)


# In[ ]:


columns_to_delete = ["asins", "colors", "dimension", "ean", "manufacturer", "manufacturerNumber", "weight", "prices.availability", "prices.condition", "prices.merchant", "prices.offer", "prices.returnPolicy", "prices.shipping"]
df.drop(columns_to_delete, axis = 1, inplace = True)


# We enjoy the better overview we already have by deleting some columns and take a look at some data columns again.

# In[ ]:


df.head(5)


# There are **three** things to address:
# 1. Some of the columns may be unnecessary due to no more than one unique value (prices.currency, primaryCategories)
# 2. There are four columns each containing dates (which are relevant?)
# 3. There seem to be more than one column carrying *key information* and no other relevant information (id, keys, upc, name)

# **1. Check the suspected columns and delete them possibly.**

# In[ ]:


print(df["prices.currency"].value_counts())
print(df["primaryCategories"].value_counts())


# In[ ]:


df.drop(["prices.currency", "primaryCategories"], axis = 1, inplace = True)


# **2. Date columns**

# In[ ]:


date_columns = ["dateAdded", "dateUpdated", "prices.dateAdded", "prices.dateSeen"]
df[date_columns].info()


# With further investigation we see that values in column "prices.dateSeen" contain more than one date, which doesn't seem to apply to the other columns. We'll drop "prices.dateSeen" which probably stores the current date every time the crawler checked up on the product.

# In[ ]:


df["prices.dateSeen"].value_counts()[:2]


# In[ ]:


df.drop(["prices.dateSeen"], axis = 1, inplace = True)
date_columns.remove("prices.dateSeen")


# We'd like to plot the other dates to get an idea of how they relate.
# Therefore we need to convert the datatypes to object type "datetime".

# In[ ]:


df_dates = df[date_columns].apply(func = lambda column : pd.to_datetime(column, errors = "ignore"), axis = 0)


# In[ ]:


#check number of missing values
print(df_dates["dateAdded"].isna().sum())
print(df_dates["dateUpdated"].isna().sum())
print(df_dates["prices.dateAdded"].isna().sum())


# In[ ]:


#replace missing values with median
timestamps = df_dates["prices.dateAdded"].map(na_action = "None", arg = lambda t : (t - numpy.datetime64('1970-01-01T00:00:00Z')) / numpy.timedelta64(1, 's'))
median = numpy.datetime64(datetime.utcfromtimestamp(timestamps.median()))
df_dates["prices.dateAdded"] = df_dates["prices.dateAdded"].fillna(median, axis = 0)


# In[ ]:


figure, ax = plot.subplots(figsize = (20,15))
figure.suptitle("Values of date columns", fontsize = 20)
plot.ylim(numpy.datetime64("2017-07-01"), numpy.datetime64("2018-05-01"))
plot.xlim(3000, 6000)
plot.scatter(x = range(len(df_dates.index)), y = df_dates["dateAdded"].values, c = "blue", s = 10, label = "dateAdded")
plot.scatter(x = range(len(df_dates.index)), y = df_dates["dateUpdated"].values, c = "green", s = 10, label = "dateUpdated")
plot.scatter(x = range(len(df_dates.index)), y = df_dates["prices.dateAdded"].values, c = "red", s = 10, label = "prices.dateAdded")
ax.legend()
x = plot.xticks([])


# In[ ]:


#check if any date column needs to be considered when grouping the data entries (by our interpretation of an product entity/ID)
non_date_columns = list(frozenset(df.columns) - frozenset(date_columns))
for column in df_dates.columns:
    print(column + ":")
    print(df.groupby(non_date_columns)[column].nunique().max())


# First of all we notice that the columns seem to be ordered. Secondly, we don't need to consider the date columns as potential danger when grouping the products. While it is hard to meet an interpretation of the different dates, the plot gave us a rough insight on what's going on in the columns. It seems that "dateAdded" stores the date on which the entry got added to the dataframe (respectively the product got encountered in the web), "dateUpdated" may encode meta information about when the entry got last updated in the dataframe. "prices.DateUpdated" may be updated whenever the crawler detects a change in the product's price value. Anyway, for our following analysis the dates are not relevant, therefore we decide to only keep one date column, with no missing values, let's decide for "dateUpdated".

# In[ ]:


df.drop(date_columns, axis = 1, inplace = True)
df = pd.concat([df, df_dates.dateUpdated], axis = 1)


# **3. Key columns**
# Now we investigate which columns are suitable keys/identifiers for the product, by the end of which we'll also be able to delete columns in case more than one column serve as pure identifier and don't store further relevant information.

# In[ ]:


def get_relationships(column):
    return df.groupby(column).apply(func = lambda frame : frame.apply(axis = 0, func = lambda col : col.nunique())).apply(
        axis = 0, func = lambda x : x.max()).values

relationship_matrix = pd.DataFrame([get_relationships(column) for column in df.columns], columns = df.columns, index = df.columns)


# Each row in the dataframe "relationship_matrix" indicates for a given column (the row index) how well this column serves as a key for any other column by printing the maximal number of distinct values in the other column that correspond to an identical value in the given column.

# In[ ]:


relationship_matrix


# We see, indeed, that several columns have somewhat of a key property. Since we assume this property on the "id" column, the easiest way to see that, is to look in the above dataframe in column "id". Every row, which stores "1" in the "id" column indicates, that a different value in the row index column implies a different "id" value, or put another way: the row index column is at least a such selective key as the "id" column. So these columns won't give us any benefit (over "id" column) by organizing the data into categories (~every unique row would have its own category), **unless** there is further information encoded, like for example in the "sourceURLs". In that case we wouldn't treat the entries as pure nominal variables, because we would be able to structure the variables (like an ordering for ordinal variables is nothing else than a structure).

# In[ ]:


#upc (universal product code)
df[["id", "keys", "name", "prices.sourceURLs", "sourceURLs", "upc", "imageURLs"]].head()


# In our case, as we aren't interested in any of the information contained in any of the columns, we decide to only keep the "id" column and drop the other columns. (Like we'll see later, information on product category in column "name" also stored in another column.)

# In[ ]:


df.drop(["keys", "name", "prices.sourceURLs", "sourceURLs", "upc", "imageURLs"], axis = 1, inplace = True)


# We've addressed all of our concerns and now we are ready to dive deeper into our data. After noticing that there are no null values left in our dataframe and doing some general stuff, we'll start working column by column and see if we'd like to modify it, i.e. structurize/extract its information into a common syntax.

# In[ ]:


df.rename(columns = {"prices.amountMax": "maxprice", "prices.amountMin": "minprice", "prices.color": "color", "prices.isSale": "sale",
                    "prices.size": "size", "dateUpdated": "date"}, inplace = True)


# In[ ]:


#convert string values to lower case
columns_nominal_categorical = ["brand", "categories", "color"]
for col in columns_nominal_categorical:
    df[col] = df[col].map(arg = lambda nominal : nominal.lower())


# We start working on the "categories".

# In[ ]:


df.categories.value_counts().head(20)


# An important thing to notice is that the third value (separated by commas) seems to store the information we want. We're trying to extract this value if possible and then search in every category where this is not possible if it contains an already elsewhere extraced value. During this process we can also add some reasonable values to the list, thereby enhancing the chance of mappings (on concise values) for until then unmapped category values.

# In[ ]:


def parse_category(category, words_categories, second_run = False, last_run = False):
    if(category.startswith("womens,shoes,")):
        words = category.split(",", 3)
        words_categories.append(words[2])
        return words[2]
    else:
        if second_run:
            hits = [cat for cat in words_categories if cat in category]
            if len(hits) > 0:
                return(hits[random.randint(0, len(hits) - 1)])
            else:
                return category
        else:
            if last_run and len(category.split(",")) >= 2:
                return "other"
            else:
                return category


# In[ ]:


random.seed(1)
words_categories = []
df_category = df["categories"].map(arg = lambda category : parse_category(category, words_categories))
df_category.value_counts()


# In[ ]:


words_categories = frozenset(words_categories)
words_categories = words_categories.union(frozenset(["work", "casual", "running", "dress"]))
df_category = df_category.map(arg = lambda category : parse_category(category, words_categories, second_run = True))
df_category.value_counts()


# In[ ]:


df_category = df_category.map(arg = lambda category : parse_category(category, words_categories, last_run = True))
df.categories = df_category


# In[ ]:


figure, ax = plot.subplots(figsize = (14,8))
figure.suptitle("Shoe categories", fontsize = 20)
plot.rcParams["ytick.labelsize"] = 15
plot.rcParams["xtick.labelsize"] = 12
p = df.categories.value_counts().head(10).plot.barh()
l = ax.set_xlabel("Number of products", fontsize = 15)


# Now focus on column "color". After checking certain values it seems that we can leave column as it is for the time being. Analogous we check other columns.

# In[ ]:


df.color.value_counts()[:15]


# In[ ]:


df.color.value_counts()[-15:]


# In[ ]:


figure, ax = plot.subplots(figsize = (14,8))
figure.suptitle("Shoe colors", fontsize = 20)
plot.rcParams["ytick.labelsize"] = 12
df.color.value_counts().head(10).plot.barh(color = "blue")
ax.set_ylabel("Color", fontsize = 15)
l = ax.set_xlabel("Number of products", fontsize = 15)


# **We're finally done with data cleaning and can start analysing**.
# 
# We'll start with looking at mean prices by brand, color and category. For this we define functions which aggregate prices by certain columns. **Notice** that we have to be careful when calculating the mean, because we want to weight each *product id* equally.
# 
# Example: A brand has two product ids, one pretty expensive pair of shoes with only one size and color, the other one much cheaper and with many variations (i.e. many rows in our dataframe). When aggregating naively by the brand, we'd get a much cheaper mean price then we want, because the cheaper product is weighted much higher (each *variation* is weighted equally) then the more expensive one.

# In[ ]:


#This function weights the prices equally by id
def aggregate_price_by_columns(df, par_columns):
    price_min = df.groupby(par_columns + ["id"]).minprice.mean().reset_index().groupby(par_columns).minprice.mean()
    price_max = df.groupby(par_columns + ["id"]).maxprice.mean().reset_index().groupby(par_columns).maxprice.mean()
    distinct_products = df.groupby(par_columns).id.nunique()
    result = pd.concat([price_min, price_max, distinct_products], axis = 1)
    price_interval = result.apply(func = lambda row : row.maxprice - row.minprice, axis = 1)
    result = pd.concat([result, price_interval], axis = 1).rename(columns = {"id": "#products", 0 : "span",
                                                                            "minprice": "min_mean", "maxprice": "max_mean"})
    return result.sort_values(by = "min_mean", axis = 0)

def aggregate_price_by_id(df):
    price_min = df.groupby("id").minprice.mean()
    price_max = df.groupby("id").maxprice.mean()
    result = pd.concat([price_min, price_max], axis = 1)
    price_interval = result.apply(func = lambda row : row.maxprice - row.minprice, axis = 1)
    result = pd.concat([result, price_interval], axis = 1).rename(columns = {0 : "span", "minprice": "min_mean", "maxprice": "max_mean"})
    return result.sort_values(by = "min_mean", axis = 0)


# In[ ]:


prices_by_id = aggregate_price_by_id(df)
figure, ax = plot.subplots(figsize = (15,8))
figure.suptitle("Shoe prices", fontsize = 20)
plot.xlim(0, 175)
plot.rcParams["ytick.labelsize"] = 12
plot.rcParams["xtick.labelsize"] = 12
sb.kdeplot(prices_by_id.min_mean.values, color = "blue", label = "mean minimum price")
ax.set_xlabel("Price", fontsize = 15)
ax.set_ylabel("Density", fontsize = 15)
sb.kdeplot(prices_by_id.max_mean.values, color = "green", label = "mean maximum price")
l = ax.legend()


# **Prices by brand**

# In[ ]:


prices_by_brand = aggregate_price_by_columns(df, ["brand"])
prices_by_brand.min_mean[-5:]


# The following code cell just helps us to encode information about the number of products in the color of the plot,
# the lighter the color (in the plot later) the fewer products by the brand are listed in the dataframe.

# In[ ]:


#returns color shade for scalar input
def lighten_color(color, amount=0.5):
    #get hexadecimal value
    c = mc.cnames[color]
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

#get max and mean number of products
mean_number_products = prices_by_brand["#products"].mean()
max_number_products = prices_by_brand["#products"].max()

#scale possible range for number of products on suitable range for input on func lighten_color
x = [1, mean_number_products, max_number_products]
y = [0.2, 1, 1.8]
f = interpolate.interp1d(x, y)

#calculate corresponding color for each brand's number of products
plot_color = [lighten_color("blue", f(number_products)) for number_products in prices_by_brand["#products"]]


# **First brand plot**: There are too many brands to plot the labels on the x-axis, we'll take a specific look at certain brands further below. For each brand the mean minimum price of their unique products is scattered and on top the interval to the mean maximum price (if it differs from the minimum) is plotted. The lighter the color the smaller the number of the brand's products (and therefore the less accurate the mean prices).

# In[ ]:


figure, ax = plot.subplots(figsize = (20,10))
figure.suptitle("Prices by brand(1)", fontsize = 20)
plot.rcParams["ytick.labelsize"] = 12
plot.ylim(0, 220)
plot.bar(prices_by_brand.index.values, prices_by_brand.span, bottom = prices_by_brand.min_mean, color = plot_color)
ax.set_ylabel("Price", fontsize = 15)
ax.set_xlabel("Brands", fontsize = 15)
plot.scatter(x = prices_by_brand.index.values, y = prices_by_brand.min_mean, color = plot_color)
ticks = plot.xticks([])


# **Second brand plot**: We focus on the brands with the most number of products.

# In[ ]:


prices_by_brand = prices_by_brand.sort_values(
    ascending = False, by = "#products", axis = 0).reset_index().loc[:9, :].set_index("brand").sort_values("min_mean", axis = 0)


# In[ ]:


figure, ax = plot.subplots(figsize = (20,10))
figure.suptitle("Prices by brand(2)", fontsize = 20)
plot.ylim(30, 90)
plot.bar(prices_by_brand.index.values, prices_by_brand.span, bottom = prices_by_brand.min_mean)
plot.rcParams["ytick.labelsize"] = 12
plot.rcParams["xtick.labelsize"] = 12
plot.scatter(x = prices_by_brand.index.values, y = prices_by_brand.min_mean, color = "blue")
l = ax.set_ylabel("Price", fontsize = 15)


# **Prices by color**

# In[ ]:


prices_by_color = aggregate_price_by_columns(df, ["color"]).sort_values(
    ascending = False, by = "#products", axis = 0).reset_index().loc[:9, :].set_index("color").sort_values("min_mean", axis = 0)
prices_by_color


# In[ ]:


plot_color = ['navajowhite', 'navy', 'floralwhite', 'burlywood', 'gray', 'black', 'brown', 'blue', 'red', 'tan']


# In[ ]:


figure, ax = plot.subplots(figsize = (20,10))
figure.suptitle("Prices by color", fontsize = 20)
plot.ylim(40, 80)
plot.bar(prices_by_color.index.values, prices_by_color.span, bottom = prices_by_color.min_mean, color = plot_color)
plot.rcParams["ytick.labelsize"] = 12
plot.rcParams["xtick.labelsize"] = 12
l = ax.set_ylabel("Price", fontsize = 15)


# **Prices by category**

# In[ ]:


prices_by_category = aggregate_price_by_columns(df, ["categories"]).sort_values(
    ascending = False, by = "#products", axis = 0).reset_index().loc[:9, :].set_index("categories").sort_values("min_mean", axis = 0)


# In[ ]:


figure, ax = plot.subplots(figsize = (20,10))
figure.suptitle("Prices by category", fontsize = 20)
plot.rcParams["ytick.labelsize"] = 12
plot.rcParams["xtick.labelsize"] = 15
plot.ylim(30, 140)
plot.bar(prices_by_category.index.values, prices_by_category.span, bottom = prices_by_category.min_mean)
l = ax.set_ylabel("Price", fontsize = 15)


# **Correlation between sale and number of sizes**
# 
# First we need to parse the number of available sizes from the "sizes" column. We assume that the entries in the column are separated by commas and represent the sizes still available and not the sizes that have been available in the beginning.

# In[ ]:


df.sizes.value_counts()[:10]


# In[ ]:


print([size for size in df.sizes.values if not "," in size])


# In[ ]:


number_sizes = pd.DataFrame(df.sizes.map(arg = lambda size_list : len(size_list.split(",")))).rename(columns = {"sizes": "#sizes"})
price_span = pd.DataFrame(df.apply(func = lambda row : row.maxprice - row.minprice, axis = 1)).rename(columns = {0: "span"})
df_sizes = pd.concat([df, number_sizes, price_span], axis = 1)


# In[ ]:


df_sizes.corr()


# Though the correlation coefficient of "#sizes" and "sale" is negative like expected, the value is absolutely **insignificant**.
# 
# However, the "sale" column correlates stronger with the "maxprice" column, more expensive products seem to be more often in sale than cheaper ones. Also it seems that more expensive shoes come with a less number of available sizes.
