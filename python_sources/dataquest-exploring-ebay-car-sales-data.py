#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In this guided project, we work with a dataset of used cars from *eBay Kleinanzeigen*, a classifieds section of the German eBay website. While the full dataset consisting of over 370,000 listings was cleaned an uploaded to [Kaggle](https://www.kaggle.com/orgesleka/used-cars-database/data), this guided project will use a subset of 50,000 observations that has also been dirtied by the DataQuest team in order to more closely mimic what could be expected from the originally scraped data set.
# 
# The set contains the following columns.
# - `dateCrawled`: When the ad was first crawled. All other field values for the corresponding row were scraped on this date.
# - `name`: Name of the car listing.
# - `seller`: Whether the seller is a private owner or a dealer.
# - `offerType`: The type of listing.
# - `price`: The price on the ad to sell the car.
# - `abtest`: Whether the listing is included in an A/B test.
# - `vehicleType`: The vehicle type.
# - `yearOfRegistration`: The year in which the car was first registered.
# - `gearbox`: The transmission type.
# - `powerPS`: The power of the car in [PS](https://www.carwow.co.uk/guides/glossary/what-is-horsepower).
# - `model`: The car model name.
# - `odometer`: The odometer reading on the car, in kilometers.
# - `monthOfRegistration`: The month in which the car was first registered.
# - `fuelType`: What type of fuel the car uses.
# - `brand`: The brand of the car.
# - `notRepairedDamage`: Whether or not the car has damage which is not yet repaired.
# - `dateCreated`: The date on which the eBay listing was created.
# - `nrOfPictures`: The number of pictures in the listing.
# - `postalCode`: The postal code for the location of the vehicle.
# - `lastSeenOnline`: When the crawler last saw this listing online.
# 
# The goal of this project is to clean the data and then use pandas to perform some basic initial analysis of the listings. To start, we first import the NumPy and pandas libraries, and then will attempt read the CSV file which contains the data and load it into a pandas DataFrame.

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


auto_filepath = "../input/used-cars-database-50000-data-points/autos.csv"


# In[ ]:


autos = pd.read_csv(auto_filepath)


# When trying to import the file using the default encoding of UTF-8, we get a `UnicodeDecodeError`. This means that we should attempt to use the next two most popular encodings, Latin-1 and Windows-1252, to see if that resolves the error.

# In[ ]:


# Attempt to use Latin-1 encoding to resolve UnicodeDecodeError
autos = pd.read_csv(auto_filepath, encoding = "Latin-1")


# It seems as if using Latin-1 encoding allowed us to successfully read and load the dataset into a pandas DataFrame. Next we explore the first few rows of the data as well as the info provided by the `DataFrame.info()` method.

# In[ ]:


autos.head()


# Looking at the first few rows of the data, one thing that stands out immediately is the fact that many of the qualitative variables, such as `vehicleType`, `gearbox`, `fuelType`, and `notRepairedDamage`, use German vocabulary. This makes sense given the fact that the dataset was obtained by scraping German eBay listings. To better interpret the meanings of the values in the columns which use German, we should translate them into English. In addition, there are columns, such as `price` and `odometer`, which represent numeric quantities but are currently stored as strings. These columns will need to be converted and then renamed to indicate the units which are being used. Also, the `dateCrawled` and `lastSeen` columns appear to currently be stored as strings as well. We may wish to convert them to pandas `Timestamp` objects to make use of the robust [set of tools available in pandas to work with time series and dates](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html).

# In[ ]:


autos.info()


# Looking at the information provided by `DataFrame.info()`, we see that a number of columns contain null entries. Those columns are `vehicleType`, `gearbox`, `model`, `fuelType`, and `notRepairedDamage`. For each column which contains null entries, we will need to decide between dropping the corresponding row, leaving the null entry as-is, or replacing it with a descriptive value.

# # Cleaning Column Names

# While the column names are perfectly workable in their current state, there are a few things that we can clean up in order to make them easier to work with.
# 
# 1. Change the following column names to be less clunky.
#     - `yearOfRegistration` to `registration_year`
#     - `monthOfRegistration` to `registration_month`
#     - `notRepairedDamage` to `unrepaired_damage`
#     - `dateCreated` to `ad_created`
# 2. Change the rest of the column names from camelcase to snakecase, which is the prefered style for Python.

# In[ ]:


auto_columns = autos.columns.to_series()
columns_dict = {"yearOfRegistration": "registration_year", "monthOfRegistration": "registration_month",
               "notRepairedDamage": "unrepaired_damage", "dateCreated": "ad_created"}
# First replace clunky column names
auto_columns.replace(columns_dict, inplace = True)
# Use a regular expression to insert underscores in between camelcase letters
auto_columns = auto_columns.str.replace(r"([a-z])([A-Z])", lambda m: "_".join(m.group(1, 2)))
# Make all column names lowercase
auto_columns = auto_columns.str.lower()
auto_columns


# Now that we have changed the column names to make them easier to work with and also have them follow Python's preferred style conventions we replace the original column names with the modified ones and re-inspect the first few rows of the `autos` dataframe.

# In[ ]:


autos.columns = auto_columns
autos.head()


# # Initial Exploration and Cleaning

# Now that we have cleaned up the column names, it is time to continue our initial exploration of the data to determine what the next cleaning steps will be. First we will look at some basic descriptive statistics for our columns.

# In[ ]:


autos.describe(include = "all")


# In[ ]:


autos["seller"].value_counts()


# In[ ]:


autos["offer_type"].value_counts()


# First, we notice that the `seller` and `offer_type` columns are almost exclusively a single value. That means they are good candidates to drop as they won't have useful information for analysis. It also appears that the column `nr_of_pictures` is all zeros, since the minimum and maximum values in the column are both zero, so I will investigate that to confirm.

# In[ ]:


autos["nr_of_pictures"].value_counts()


# The column `nr_of_pictures` is indeed all zeros, so that is another good candidate to drop.
# 
# Continuing along, the minimum and maximum values in the `registration_year` column should also be addressed, as they are 1000 and 9999, respectively. Cars didn't exist in the year 1000, let alone eBay, and the year 9999 is far off into the future.

# In[ ]:


# Calculate the most recent date an observation was collected.
pd.to_datetime(autos["date_crawled"]).max()


# In[ ]:


# Count the number of cars with registration years prior to 1906 or after 2016
autos.loc[~autos["registration_year"].between(1906, 2016), "registration_year"].count()


# In[ ]:


autos.loc[~autos["registration_year"].between(1906, 2016), "registration_year"].value_counts()


# Looking a little more closely we first note that that the most recent year the data was scraped was 2016, so there any listings which have a `registration_year` value of 2017 or later should be addressed. There are 1,972 entries which have a registration year earlier than 1906 ([the first year in which German license plates were issued with a lettering plan](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Germany#History)) or after 2016. Since this is a only about 4% of the entries, dropping the corresponding rows wouldn't have a large negative impact on further analysis down the line. The majority of the entries with questionable registration year have a registration year of either 2017 or 2018, so might make sense to simply impute those values (along with the three values of 2019), if we had a way of confirming that those wrong years were simply the results of typos, and then drop the rows with the more egregious errors.

# In[ ]:


# Count the number of cars with power rating less than 1ps or greater than 2000ps
autos.loc[~autos["power_ps"].between(1, 2000), "power_ps"].count()


# In[ ]:


autos.loc[~autos["power_ps"].between(1, 2000), "power_ps"].value_counts()


# Next, the `power_ps` column has suspicious minimum (0ps) and maximum (17,700ps) values as well. It's impossible for a car to have a power rating of 0ps (though it some of the earliest cars had [power ratings of less than 1 ps](https://jalopnik.com/ten-of-the-least-powerful-road-cars-ever-made-1674061551)), and even the most [powerful car](https://en.wikipedia.org/wiki/List_of_automotive_superlatives#Power) available as of January 2020 (the [Lotus Evija](https://en.wikipedia.org/wiki/Lotus_Evija)) has a power rating of 2,000ps. One possibility is that the cars with a power rating of 0ps were listed without a power rating, so the value of zero actually represents a missing value. About 11% of the entries have power ratings outside the range of 1ps to 2,000ps, with the vast majority (5,500 entries) having a power listed as 0ps. That's a decent chunk of the data, so while it is pretty safe to drop the rows with the other impossible power values, depending on what we want to do with the data we may wish to impute the rows with a power of 0ps using something like the mean, median, or mode power value of the data set.

# In[ ]:


autos["registration_month"].value_counts()


# Moving on to the `registration_month` column, we see that has a suspicious minimum value of 0. There is no zero month, so that could either be due to typos for September (9) or October (10), or accidentally using zero indexing when counting months. It could also represent cars for which a registration month was not given or found by the crawler. About 10% of the rows have a `registration_month` value of 0, so once again while we could drop those rows, we should consider what kind of analysis we wish to do going forward in order to decide between dropping the rows in question, imputing their `registration_month` values, or leaving them as-is.
# 
# Lastly, while the minimum value of 1067 for the `postal_code` column initially looked potentially suspicious, it is reasonable given the [postal code system in Germany](https://en.wikipedia.org/wiki/List_of_postal_codes_in_Germany).
# 
# Before moving on to the `price` and `odometer` columns, which are numeric values stored as text, we once again recall that the `vehicle_type`, `gearbox`, `model`, `fuel_type`, and `unrepaired_damage` columns contain some null values. None of the columns contains more than 20% null values, but will will still need to clean those up if at some point if we want to do analysis involving those columns.

# In[ ]:


# First strip leading $
cleaned_price = autos["price"].str.strip("$")
# Remove commas
cleaned_price = cleaned_price.str.replace(",", "")
# Convert to float64 dtype
cleaned_price = cleaned_price.astype(float)
autos["price"] = cleaned_price


# To clean up the `price` column, all we needed to do was strip the `$` character and then remove all commas before casting it to the `float64` dtype.

# In[ ]:


# First strip trailing "km"
cleaned_odometer = autos["odometer"].str.strip("km")
# Remove commas
cleaned_odometer = cleaned_odometer.str.replace(",", "")
# Convert to float64
cleaned_odometer = cleaned_odometer.astype(float)
autos["odometer"] = cleaned_odometer


# Cleaning up the `odometer` column was quite similar to cleaning the `price` column, except instead of stripping the `$` character we needed to strip the trailing `km`. Now that we have cleaned up the `odometer` column, we rename it to `odometer_km` so we can keep the information that the odometer readings for each row were recorded in kilometers.

# In[ ]:


autos.rename(columns = {"odometer": "odometer_km"}, inplace = True)


# # Exploring the Odometer and Price Columns

# Now that we have computed the `odometer_km` and `price` columns from strings into floats, we can analyze them to look for potential outliers that we may want to either remove or impute, depending on the further analysis we wish to do. It is important, however, when analyzing these columns for potential outliers, to ask ourselves if any outliers we encounter might provide meaningful information before excluding those values or imputing them. Especially if our domain knowledge tells us that the outliers, while rare, are still possible, then they may be worth keeping as-is. First, we will explore the `odometer_km` column.

# In[ ]:


autos["odometer_km"].nunique()


# In[ ]:


autos["odometer_km"].describe()


# In[ ]:


autos["odometer_km"].value_counts()


# In[ ]:


(autos["odometer_km"] >= 100000).sum()


# We see that there are 13 distinct values in the `odometer_km` column, ranging from 5,000km to 150,000km. This is a reasonable range of odometer readings for used cars, since it corresponds to a range of approximately 3,100-93,000 miles. The majority of cars (about 60%) in the set have an odometer reading of 150,000km, and almost 80% of the cars have an odometer reading of at least 100,000km. While I would expect more unique values in this column, I assume that the small number is due to rounding choices made by the person who originally collected this data or by limitations in the odometer reading options users have when posting a listing. Since there don't appear to be any outliers in the `odometer_km` column, we move on to the `price` column.

# In[ ]:


autos["price"].nunique()


# There are only 2,357 distinct values in the `price` column. Again this is a fairly small number relative to the size of the dataset, but I assume it is likely due to some of the more common preferences people have when setting prices (e.g. rounding to even multiples of 5, 10, 100, 500, 1000, etc.).

# In[ ]:


np.round(autos["price"].describe(), 2)


# The `price` column has both unrealistic minimum and maximum values that immediately stand out. At the low end, it is impossible to post any item on eBay with a price of zero. At the high end, a price of \$99,999,999, while certainly a possible list price -- in 2005 a \$168 million gigayacht was [listed and reportedly sold on eBay](https://www.ebayinc.com/company/our-history/) -- it is still quite unrealistic. Looking through posts from eBay about the most expensive and interesting purchases made through the site ([2019](https://www.ebayinc.com/stories/news/ebay-unveils-the-most-interesting-and-expensive-purchases-of-2019/), [2018](https://www.ebayinc.com/stories/news/revealing-ebays-20-most-expensive-purchases-of-2018/), [2017](https://www.ebay.com/motors/blog/the-most-expensive-cars-sold-on-ebay-motors-in-2017/), [2016](https://www.ebay.com/motors/blog/most-expensive-cars-ebay-2016/), [2010](https://www.ebay.com/motors/blog/most-expensive-cars-sold-ebay/)), while an imperfect reference since it only discusses listings that were actually sold, suggests that the highest realistic price for a car would be \$4,000,000, and that typically the most expensive cars are listed for \$2,000,000 or less.

# In[ ]:


# Get value counts for the prices that are less than 0.01 and greater than 4000000
autos.loc[~autos["price"].between(0.01, 4000000), "price"].value_counts().sort_index()


# In[ ]:


# Count the number of cars with prices less than 0.01 and greater than 4000000
autos.loc[~autos["price"].between(0.01, 4000000), "price"].count()


# We see that the vast majority of cars with a unrealistic prices have a price of zero, with 1,421 such vehicles, and in total only about 3% (1,429) of the listings have extreme outlier prices (prices of either zero or larger than 4,000,000). That is a very small fraction of the listings, so even if those prices are accurate, it still makes sense to exclude those listings if we want to do any analyses that involve the price. Let's exclude those outlier prices to see how that affects the descriptive statistics for the remaining values.

# In[ ]:


np.round(autos.loc[autos["price"].between(0.01, 4000000), "price"].describe(), 2)


# After excluding the most egregious prices, there still appears to be some rather extreme values at both the high and the low ends.

# In[ ]:


autos.loc[autos["price"].between(0.01, 4000000), "price"].value_counts().sort_index().head(25)


# At the low end, there are prices that are below \$100. While those are quite low prices for a car, they are still within reason if there are sellers who are either trying to get rid of a truly junky car or are trying to use a very low price to catch the attention of potential buyers browsing the eBay listings.

# In[ ]:


autos.loc[autos["price"].between(0.01, 4000000), "price"].value_counts().sort_index().tail(25)


# There are a small handful of cars at the high end. The prices seem to increase fairly incrementally up to \$350,000, and then there is a large jump from there. Since the cars with prices higher than \$350,000 could still be legitimate luxury vehicles or otherwise valuable cars, and since there are so few of them, we could inspect those rows by hand to see if the vehicle name, model, and brand provide any information to help us decide whether or not to exclude them, or if we might want to impute those price values with something more reasonable.

# In[ ]:


autos.loc[autos["price"] > 350000, ["name", "model", "brand", "registration_year", "price"]]


# While there are some cars that clearly shouldn't have such a high price (such as the Ford Focus in the first row, unless it was posted with a price of \$999,999 as a joke) and others have prices that are conceivable (such as the Ferrari FXX in the last row), other cars would require more research to determing whether or not they have reasonable prices in this data set. Doing this might not be worth the time and effort unless we need accurate price information to answer questions about cars at the top end of the pricing spectrum. If we are more interested in analyzing the vast majority of cars with prices of \$350,000 or less, then it would be a lot easier to simply drop these rows.

# # Exploring the Date Columns

# Now that we have explored the `odometer_km` and `price` columns and identified some possible further tasks for cleaning up the data, we continue our exploration by looking at the five columns which represent date values: `date_crawled`, `last_seen`, `ad_created`, `registration_month`, and `registration_year`. As we saw in our initial exploration, the `date_crawled`, `last_seen`, and `ad_created` columns are currently reprented as strings, while the `registration_month` and `registration_year` columns are stored as integers. We don't need to do any additional preprocessing to compute helpful summary statistics for the `registration_month` and `registration_year` columns, so we will first focus our attention to the date columns which are stored as strings.

# In[ ]:


autos[["date_crawled", "ad_created", "last_seen"]].head()


# Looking at the head of those columns, it appears the strings are formatted as timestamps, so we can make use of the built-in tools that pandas has for working with dates and times. While we will only be using very basic functionality, there are helpful detailed explanations for the the date/time functionality in pandas in the [official documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html) and in the [*Python Data Science Handbook* by Jake VanderPlas](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html). We will start off by converting those columns to pandas `datetime` objects using the `pandas.to_datetime()` function. More info about all of the optional arguments for that function can be found [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html#pandas.to_datetime).

# In[ ]:


autos[["date_crawled", "ad_created", "last_seen"]] = autos[["date_crawled", "ad_created", "last_seen"]].apply(pd.to_datetime)


# In[ ]:


autos[["date_crawled", "ad_created", "last_seen"]].info()


# Now that we have converted those columns to `datetime` objects, we can quite conveniently calculate the distributions of the values in a large variety of ways. For each column, we will start off by investigating the distributions based on full date (day, month, year), but we can also explore the distributions based on [year, day of month, day of week, hour of day, and more](https://pandas.pydata.org/pandas-docs/stable/reference/series.html#datetime-properties). In order to convert each full timestamp into the format `YYYY-MM-DD`, we take the [floor of each timestamp based on using days as my frequency](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.floor.html#pandas.Series.dt.floor).

# In[ ]:


autos["date_crawled"].dt.floor("D").value_counts(normalize = True).sort_index()


# We can see that the data was scraped over the period of approximately one month, from March 5, 2016 to April 7, 2016, with the distribution being fairly uniform. Since the data was fairly uniformly scraped over the course of the month, analyzing the distribution based on day of week wouldn't be particular interesting, but we can still look at the distribution based on hour.

# In[ ]:


autos["date_crawled"].dt.hour.value_counts(normalize = True).sort_index()


# It appears that most of the data was scraped pretty uniformly outside the hours 2:00am to 6:59am. Now we move on to the `ad_created` column.

# In[ ]:


autos["ad_created"].dt.floor("D").value_counts(normalize = True).sort_index().head()


# In[ ]:


autos["ad_created"].dt.floor("D").value_counts(normalize = True).sort_index().tail()


# In[ ]:


autos["ad_created"].dt.year.value_counts(normalize = True)


# In[ ]:


autos.loc[autos["ad_created"].dt.year == 2016, "ad_created"].dt.month.value_counts(normalize = True).sort_index()


# The ads were all created between June 11, 2015 and April 7, 2016 (the last date on which data was scraped). The vast majority of listings are from March and April 2016. Since I did so with the `date_crawled` column, we will also look at the distribution of when the ads were created based on hour.

# In[ ]:


autos["ad_created"].dt.hour.value_counts(normalize = True).sort_index()


# In[ ]:


autos["ad_created"].dt.time.value_counts(normalize = True).sort_index()


# All of listings have a creation time of 00:00:00 (exactly midnight), which doesn't make sense. I assume it may be due to either only the date of creation being accessible to the crawler that scraped the data or a choice made by the person who collected the data to exclude the finer-grained time information for this column. For the purposes of overall analysis it doesn't make much a difference, though not having fine-grained time information does prevent us from exploring the possibility of potentially interesting time-based patterns related to when ads are listed. Lastly, we look over the `last_seen` column.

# In[ ]:


autos["last_seen"].dt.floor("D").value_counts(normalize = True).sort_index()


# In[ ]:


autos["last_seen"].dt.floor("D").value_counts(normalize = True).sort_index().tail().sum()


# Just like the `date_crawled` column, the `last_seen` column covers the period from March 5, 2016 through April 7, 2016 during which the data was scraped. Over half of the entries were last seen in the last five days of data scraping. While the date on which an entry was last seen can be used to determine if an entry was removed (due to the car being purchased, the listing expiring, or the seller taking the listing down for other reasons) by comparing it to the corresponding date in the `date_crawled` column for that row, we shouldn't make those assumptions for cars which were seen in the last few days of scraping (April 5, 6, 7). It is likely that if a car was last seen on those days, it is simply due to the fact that that is the end of the scraping window. This is especially likely if the crawler didn't check every single listing it had previously seen on every day of crawling. 
# 
# Finally, as noted above in the Initial Exploration and Cleaning section, there are some odd values in both the `registration_month` and `registration_year` columns.

# In[ ]:


autos["registration_month"].value_counts(normalize = True).sort_index()


# First, about 10% of the listings have a registration month of `0`, which is an impossible month. This could either be due to typos in the original listings, or in this context a value of zero could indicate that the listing didn't include the registration month.

# In[ ]:


autos["registration_year"].describe()


# In[ ]:


# Calculate the most recent date an observation was collected.
autos["date_crawled"].max()


# In[ ]:


# Count the number of cars with registration years prior to 1906 or after 2016
autos.loc[~autos["registration_year"].between(1906, 2016), "registration_year"].count()


# In[ ]:


autos.loc[~autos["registration_year"].between(1906, 2016), "registration_year"].value_counts()


# Second, 1,972 entries (about 4% of the entries) have impossible registration years (prior to 1906 or after 2016, the year in which the data was collected).

# # Dealing with Incorrect Registration Year Data

# In the previous section we noted that about 4% of the entries have impossible registration years. Since that is a small proportion of the data it is fairly safe to simply drop those rows, though imputing those values might also be an option if we wish to take the time to analyze them a little more closely. For now, we'll explore the distribution of the `registration_year` column after removing the impossible years.

# In[ ]:


autos.loc[autos["registration_year"].between(1906, 2016), "registration_year"].describe()


# The remaining registration years range from 1910 to 2016, with at least least 75% of the registrations coming from 1999 or later.

# In[ ]:


autos.loc[autos["registration_year"].between(1906, 2016), "registration_year"].quantile(0.1)


# In[ ]:


autos.loc[autos["registration_year"].between(1995, 2016), "registration_year"].value_counts(normalize = True).sort_index()


# Looking a little closer, 90% of the registrations come from 1995 or later, and those registrations are roughly evenly distributed.

# In[ ]:


autos.loc[autos["registration_year"].between(1906, 1994), "registration_year"].value_counts(normalize = True).sort_index()


# In[ ]:


autos.loc[autos["registration_year"].between(1906, 1994), "registration_year"].value_counts(normalize = True).sort_index().loc[1970:].sum()


# The pre-1995 registrations are much less evenly distributed, with about 60% of those registrations still coming from the 1990s, and almost 94% of them coming from the 1970s or later.

# In[ ]:


autos.loc[autos["registration_year"].between(1906, 2016), "registration_year"].value_counts(normalize = True).sort_index().loc[1970:].sum()


# In fact, 99.5% of the cars have registration years from 1970 or later.

# # Initial Exploration and Cleaning Summary

# After reading the data from the `autos.csv` file and looking at the first few rows, we cleaned up the column names to make them less clunky and also convert them from camelcase to snakecase in order to follow Python's style recommendations. We then explored each column for null values, numeric data that needs conversion into numeric types, and outliers among the numeric columns. We also looked for columns that contained almost exclusively a single value, since those aren't likely to be helpful for any analysis we wish to do. The columns that needed particular attention were as follows.
# 
# - `seller`, `offer_type`, `nr_of_pictures`: Contain almost exclusively a single value. We should drop these columns.
# - `vehicle_type`, `gearbox`, `model`, `fuel_type`, `unrepaired_damage`: Contain missing values, though none has more than 20% missing values. Depending on what we want to do with this data, we can drop the rows with missing values, impute the missing values, or leave them as-is. In addition, the `vehicle_type`, `gearbox`, `fuel_type`, and `unrepaired_damage` columns contain categorical data that uses German words, so those can be translated into English for easier comprehension.
# - `date_crawled`, `ad_created`, `last_seen`: Convert from string to pandas `datetime` type.
# - `price`: Convert from string to float. About 3% of values are impossible (a price of \$0) and 14 values are potentially unrealistic (prices exceeding \$350,000). It is safe to drop the rows with impossible prices, but we should think more carefully before dropping or imputing the unrealistic ones depending on the analysis we wish to do.
# - `registration_year`: About 4% of the values are impossible (prior to 1906 or after 2016). It is fairly safe to drop the rows with these impossible registration years.
# - `registration_month`: About 10% of the values are impossible (0). We should consider either dropping the corresponding rows, imputing the values, or leaving them as-is, depending on the analysis we wish to do.
# - `power_ps`: About 11% of the entries are impossible (either 0 or exceeding 2000). We should consider dropping the rows with impossible power ratings or imputing the values depending on how we want to use this column.
# - `odometer_km`: Convert from string to float and rename column to preserve units information.
# 
# For the time being, we will drop the `seller`, `offer_type`, and `nr_of_pictures` columns. Then we drop the rows with impossible `price` and `registration_year` values. We also drop the rows with the most unrealistic `price` values (exceeding \$4,000,000) at this point in time. Finally, we will translate the `vehicle_type`, `gearbox`, `fuel_type`, and `unrepaired_damage` columns from German to English.

# In[ ]:


# Drop seller, offer_type, nr_of_pictures columns
autos.drop(columns = ["seller", "offer_type", "nr_of_pictures"], inplace = True)


# In[ ]:


# Filter out rows with impossible registration year 
# and impossible/unrealistic price
price_year_filter = autos["registration_year"].between(1906, 2016) & autos["price"].between(0.01, 4000000)
autos = autos[price_year_filter]


# After dropping the rows and columns we chose to drop, we turn our attention to translating the categorical columns from German into English. First up is the `vehicle_type` column.

# In[ ]:


autos["vehicle_type"].value_counts()


# There eight different vehicle types. While some, such as coupe or SUV don't need to be translated into English, the others require translation. Using a mix of Google Translate, the [English](https://en.wikipedia.org/wiki/Car_classification)/[German](https://de.wikipedia.org/wiki/Fahrzeugklasse) Wikipedia pages for car classification, the [English](https://en.wikipedia.org/wiki/Car_body_style)/[German](https://de.wikipedia.org/wiki/Karosseriebauform) Wikipedia pages for car body styles and [eBay Kleinanzeigen](https://www.ebay-kleinanzeigen.de/), we have the following translations:
# 
# - andere $\Leftrightarrow$ other
# - bus $\Leftrightarrow$ van
# - cabrio $\Leftrightarrow$ convertible
# - kleinwagen $\Leftrightarrow$ supermini (European classification) or subcompact (American classification)
# - kombi $\Leftrightarrow$ station wagon
# - limousine $\Leftrightarrow$ sedan

# In[ ]:


vehicle_types = {"andere": "other", "bus": "van", "cabrio": "convertible",
                "coupe": "coupe", "kleinwagen": "subcompact", "kombi": "station wagon",
                "limousine": "sedan", "suv": "suv"}
autos["vehicle_type"] = autos["vehicle_type"].map(vehicle_types)


# In[ ]:


autos["gearbox"].value_counts()


# There are only two different gearbox types, which are very straightforward to translate.

# In[ ]:


autos["gearbox"] = autos["gearbox"].map({"manuell": "manual", "automatik": "automatic"})


# In[ ]:


autos["fuel_type"].value_counts()


# There are seven different fuel types. Again, using Google translate gives us the following translations. We also refer to the [Wikipedia page about alternative fuel vehicles](https://en.wikipedia.org/wiki/Alternative_fuel_vehicle) for explanations of CNG (compressed natural gas) and LPG (liquefied petroleum gas). There are only three types that we need to translate.
# 
# - andere $\Leftrightarrow$ other
# - benzin $\Leftrightarrow$ gasoline
# - elektro $\Leftrightarrow$ electric

# In[ ]:


fuel_types = {"andere": "other", "benzin": "gasoline", "cng": "cng", "diesel": "diesel",
             "elektro": "electric", "hybrid": "hybrid", "lpg": "lpg"}
autos["fuel_type"] = autos["fuel_type"].map(fuel_types)


# In[ ]:


autos["unrepaired_damage"].value_counts()


# Lastly, the `unrepaired_damage` column is very straightforward to translate.

# In[ ]:


autos["unrepaired_damage"] = autos["unrepaired_damage"].map({"ja": "yes", "nein": "no"})


# To wrap up our cleaning, we look over the head of the cleaned data.

# In[ ]:


autos.head()


# One potentially interesting further preprocessing step would be to analyze the `name` column for any keywords that could be extracted as new columns.

# # Exploring Price and Mileage by Brand

# Now that we have finished preprocessing the `autos` dataset, we can finally do some analysis. To start, we will do some basic aggregating by the `brand` column to explore differences in price and mileage between brands. Before performing any aggregations, we explore the `brand` column itself.

# In[ ]:


autos["brand"].nunique()


# In[ ]:


autos["brand"].value_counts(normalize = True)


# In[ ]:


autos["brand"].value_counts(normalize = True).head(10).sum()


# We see that while there are 40 different brands represented in this dataset, the top ten most common ones (Volkswagen, BMW, Opel, Mercedes Benz, Audi, Ford, Renault, Peugeot, Fiat, and Seat) make up almost 80% of the listings. The five brands are all German ones, and aside from Ford all of the top ten brands are European ones. This makes sense, since cars from German and other European brands would be the most readily available for German drivers to purchase and then potentially list on eBay when they are ready to sell their cars. For completeness, I will still aggregate over all 40 brands, but I will focus the majority of my analysis toward those in the top ten. To perform my aggregations, I will use the ["group by" features in Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html). Additional info can once again be found in [VanderPlas's book](https://jakevdp.github.io/PythonDataScienceHandbook/03.08-aggregation-and-grouping.html).

# In[ ]:


top_10_brands = autos["brand"].value_counts().head(10).index


# In[ ]:


# Compute mean of odometer_km and price columns
# Round to the nearest whole number for easier interpretation
# Restrict focus to the top ten most common brands
# Sort from highest mean price to lowest
autos.groupby("brand")[["odometer_km", "price"]].mean().loc[top_10_brands].sort_values("price", ascending = False).round(0)


# Among the top ten most common brands, there aren't any major differences between mean mileage, with all brands having a mean mileage between 115,000km and 135,000km. There are, however, some pretty striking differences in average price between brands. The three most expensive brands, Audi, Mercedes Benz, and BMW, are each about three times more expensive than the three least expensive brands, Renault, Fiat, and Opel. This difference in average price makes sense given the fact that the expensive brands are luxury ones which carry an associated price premium while the cheap brands make economy cars. The most popular brand, Volkswagen, sits in the middle ground between the two extremes, with an average listing price of \$5,604.

# In[ ]:


# Compute median of price column
# Round to the nearest whole number for easier interpretation
# Restrict focus to the top ten most common brands
# Sort from highest median price to lowest
autos.groupby("brand")["price"].agg(["min", "median", "max"]).loc[top_10_brands].sort_values("median", ascending = False).round(0)


# Since we noted that there are some unrealistically high prices in the data, even after filtering out the most egregiously unrealistic ones, we also can compare the median price between brands, since the median is less affected by outliers. The same pattern appears, though it is even more pronounced, with the median Audi price being almost quintuple the median Renault price.

# # Exploring Price by Other Factors

# In addition to comparing car prices across brands, we can also compare them across other factors. First, we will see if there are any patterns between price and mileage. Since we saw earlier that there is a fairly small number of distinct values in the `odometer_km` column, I will simply group by it directly.

# In[ ]:


autos.groupby("odometer_km")["price"].describe().sort_index().round(2)


# In[ ]:


autos.loc[autos["odometer_km"] < 50000, "price"].mean()


# There is a very clear pattern when looking at both the mean and median price: as the odometer reading goes up, the price goes down. The dropoff is fairly minimal for cars with 5,000-50,000km; with the average prices for each odometer category hovering within a few thousand dollars of the overall average for that group (about \$16,000). After 50,000km, however, the average prices drop precipitously.

# In[ ]:


autos.fillna({"unrepaired_damage": "unknown"}).groupby("unrepaired_damage")["price"].describe().round(2)


# Finally we compare prices between cars which have unrepaired damage and those which do not. I also replaced all of the null entries in the `unrepaired_damage` column with the value `"unknown"` so we can also consider prices for cars in that group. There is a very clear difference in average price between the three groups. Cars with unrepaired damage have an average price that is about a third of the average price for undamaged cars and cars with damage that has been repaired. In addition, even though cars with an unknown damage status have an average price about 50% higher than cars with unrepaired damage, that average price is still less than half the average price for cars without unrepaired damage.

# # Exploring Popular Brand/Model Combinations

# The last basic analysis we perform is identifying the most common brand/model combinations. This is very easy to do using the "group by" features in pandas.

# In[ ]:


# Group by brand and then get the value counts for the models in each brand
# Include null values in the value counts
# Then sort by values in descending order
autos.groupby("brand")["model"].value_counts(dropna = False).sort_values(ascending = False).head(10)


# Here we have the top ten most common brand/model combinations, which are all cars from German brands. If we wish to easily see what fraction of the overall dataset is represented by these common brand/model combinations, we [flatten the multi-indexed series](https://jakevdp.github.io/PythonDataScienceHandbook/03.05-hierarchical-indexing.html#Index-setting-and-resetting).

# In[ ]:


# Flatten multi-indexed series
flat_brand_model = autos.groupby("brand")["model"].value_counts(dropna = False).reset_index(name = "count")
flat_brand_model["proportion"] = flat_brand_model["count"]/autos.shape[0]
flat_brand_model.sort_values(["proportion"], ascending = False).head(10)


# In[ ]:


flat_brand_model["proportion"].sort_values(ascending = False).head(10).sum()


# Volkswagen Golfs and BMW 3 Series are by far the most common cars in the data set, and in total the top ten most common brand/model combinations make up about 36% of the listings in the set.

# # Conclusion and Potential Next Steps

# Over the course of this notebook, we went through the basic steps of preprocessing a dataset of used cars from eBay Kleinanzeigen. This involved loading the data into pandas; cleaning up the column names; and exploring the columns to identify ones which are likely candidates to drop immediately, ones which need to be converted into an appropriate type, and ones which contain potentially problematic values (null values and outliers). After processing the columns we identified, we then performed some basic analysis and aggregation to compare different categories of cars. In particular, we compared prices across different brands, odometer values, and states of repair. We also compared odometer readings across brands and explored the most common brand/model combinations.
# 
# Beyond the work we have done in this notebook, there are a number of potential next steps for interesting preprocessing and analysis. Some of them include the following.
# 
# - Analyze the `name` column for keywords that could be extracted and added as columns to the dataset.
# - Categorize cars based on kilometers driven per year at time of listing and then analyze patterns in prices among those categories.
# - Translate postal codes into geographic regions and compare the other variables (listing price, car brand/model/type) to see what differences or similarities there are between listings from various regions of Germany.
# - Add data visualizations to further summarize the analysis.
# - Explore various machine learning models to try and predict listing price based on the other variables.
# - Try to identify interesting clusters in the data using unsupervised machine learning techniques.
