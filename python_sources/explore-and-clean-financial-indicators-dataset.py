#!/usr/bin/env python
# coding: utf-8

# # EXPLORATORY DATA ANALYSIS
# 
# In this notebook, I will show some quick and easy exploratory steps that I usually take when working on these types of dataset (namely finance-related datasets). Hopefully you'll find this notebook somewhat useful.
# 
# I will be focusing my attention on the file `2014_Financial_Data.csv`, which is available as past of the dataset **200+ Financial Indicators of US stocks (2014-2018)**. Needless to say, the reported steps can be applied to every `.csv` file.
# 
# The Python packages that we will be using are:
# 
# 1. for data manipulation
#  - pandas
#  - numpy
# 2. for plotting purposes
#  - matplotlib
#  - seaborn
# 3. for finance related operations
#  - [pandas_datareader](https://pandas-datareader.readthedocs.io/en/latest/)
# 
# With that being said, let's begin by importing the required packages.

# In[ ]:


# Data manipulation
import pandas as pd
import numpy as np

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Finance related operations
from pandas_datareader import data

# Import this to silence a warning when converting data column of a dataframe on the fly
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

get_ipython().run_line_magic('matplotlib', 'inline')


# # STEP 1: LOAD DATA
# 
# First things first, we need to load the data from the `.csv` file. This is easily done with `pandas`. I like to keep the ticker of the stocks as index of the dataframe, so I specify `index_col=0` when loading the data.
# 
# Furthermore, it is convenient to immediately drop those rows where *all* the values are `NaN`.

# In[ ]:


# Load data
df = pd.read_csv('../input/200-financial-indicators-of-us-stocks-20142018/2014_Financial_Data.csv', index_col=0)

# Drop rows with no information
df.dropna(how='all', inplace=True)


# # STEP 2: FIRST LOOK AT THE DATASET
# 
# ## STEP 2.1: general info, categorical variables
# 
# It is useful to take a quick look at the initial state of the dataset:
# 
# 1. use `.info()`, `.describe()` method to get a first sense of the dimensions of the dataset and value of numeric variables;
# 2. focus on the categorical variables and also the class of the samples, to see if they are balanced.

# In[ ]:


# Get info about dataset
df.info()

# Describe dataset variables
df.describe()


# We now know that we have:
# - 3808 samples
# - 224 columns
#  - 222 numeric --> they are the financial indicators
#  - 1 int       --> this is the class column
#  - 1 object    --> this is categorical (`Sector`)
# 
# Next, we will take a look at the distribution of the class values, and the distribution of the categorical variable `Sector`.

# In[ ]:


# Plot class distribution
df_class = df['Class'].value_counts()
sns.barplot(np.arange(len(df_class)), df_class)
plt.title('CLASS COUNT', fontsize=20)
plt.show()

# Plot sector distribution
df_sector = df['Sector'].value_counts()
sns.barplot(np.arange(len(df_sector)), df_sector)
plt.xticks(np.arange(len(df_sector)), df_sector.index.values.tolist(), rotation=90)
plt.title('SECTORS COUNT', fontsize=20)
plt.show()


# The plots above show that:
# 1. **the samples are not balanced in terms of class**. Indeed, 2174 samples belong to class `0`, which as explained in the documentation of the dataset correspond to stocks that are *not buy-worthy*. At the same time, 1634 samples belong to class `1`, meaning they are *buy-worthy* stocks. This should be accounted for when splitting the data between training and testing data (it is useful to use the `stratify` option available within `sklearn.model_selection.train_test_split`).
# 2. there is a total of 11 sectors, 5 of them with about 500+ stocks each, while the remaining 6 sectors have less than 300 stocks. In particular, the sectors *Utilities* and *Communication Services* have around 100 samples. This has to be kept in mind if we want to use this data with ML algorithms: there are very few samples, which could lead to overfitting, etc.

# ## STEP 2.2: price variation, look for outliers/errors
# 
# It is always important to make sure that the target data *makes sense*. I am particularly curious to see if the column `2015 PRICE VAR [%]`, which lists the percent price variation of each stock during the year 2015, contains any *mistake* (for instance mistypings or unreasonable values). A quick plot of this column (for each sector), will allow us to assess the situation.
# 
# In layman's terms, here we are looking for major peaks/valleys, which indicate stocks that increased/decreased in value by an incredible amount with respect to the overall sector's trend.

# In[ ]:


# Extract the columns we need in this step from the dataframe
df_ = df.loc[:, ['Sector', '2015 PRICE VAR [%]']]

# Get list of sectors
sector_list = df_['Sector'].unique()

# Plot the percent price variation for each sector
for sector in sector_list:
    
    temp = df_[df_['Sector'] == sector]

    plt.figure(figsize=(30,5))
    plt.plot(temp['2015 PRICE VAR [%]'])
    plt.title(sector.upper(), fontsize=20)
    plt.show()


# Thanks to this check, we can clearly see that there are indeed some major peaks in the following sectors:
# 
# * Consumer Defensive
# * Basic Materials
# * Healthcare
# * Consumer Cyclical
# * Real Estate
# * Energy
# * Financial Services
# * Technology
# 
# This means that, for one reason or another, some stocks experienced incredible gains. However, how can be sure that each of these gains is organic (i.e. due to trading activity)?
# 
# We can take a closer look at this situation by plotting the price trend for those **stocks that increased their value by more than 500% during 2015**. While it is possible for a stock to experience such gains, I'd still like to verify it with my eyes.
# 
# Here, we will use `pandas_datareader` to pull the *Adjusted Close* daily price, during 2015, of the required stocks. To further investigate these stocks, I think it is worth to plot the *Volume* too.

# In[ ]:


# Get stocks that increased more than 500%
gain = 500
top_gainers = df_[df_['2015 PRICE VAR [%]'] >= gain]
top_gainers = top_gainers['2015 PRICE VAR [%]'].sort_values(ascending=False)
print(f'{len(top_gainers)} STOCKS with more than {gain}% gain.')
print()

# Set
date_start = '01-01-2015'
date_end = '12-31-2015'
tickers = top_gainers.index.values.tolist()

for ticker in tickers:
    
    # Pull daily prices for each ticker from Yahoo Finance
    daily_price = data.DataReader(ticker, 'yahoo', date_start, date_end)
    
    # Plot prices with volume
    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    
    ax0.plot(daily_price['Adj Close'])
    ax0.set_title(ticker, fontsize=18)
    ax0.set_ylabel('Daily Adj Close $', fontsize=14)
    ax1.plot(daily_price['Volume'])
    ax1.set_ylabel('Volume', fontsize=14)
    ax1.yaxis.set_major_formatter(
            matplotlib.ticker.StrMethodFormatter('{x:.0E}'))

    fig.align_ylabels(ax1)
    fig.tight_layout()
    plt.show()


# As we can see, most of the `top_gainers` stocks did not experienced an organic growth during 2015. This is highlighted by a portion of the price trend being completely flat, due to the absence of trading activity.
# 
# So, I reckon that only the last 2 stocks from the `top_gainers`, namely **NYMX** and **AVXL**, should be kept in the dataframe and we should drop the others.

# In[ ]:


# Drop those stocks with inorganic gains
inorganic_stocks = tickers[:-2] # all except last 2
df.drop(inorganic_stocks, axis=0, inplace=True)


# So, it is worth to check again the target data in the column `2015 PRICE VAR [%]`, in order to assess the impact of dropping the *fake* top gainers. 

# In[ ]:


# Check again for gain-outliers
df_ = df.loc[:, ['Sector', '2015 PRICE VAR [%]']]
sector_list = df_['Sector'].unique()

for sector in sector_list:
    
    temp = df_[df_['Sector'] == sector] # get all data for one sector

    plt.figure(figsize=(30,5))
    plt.plot(temp['2015 PRICE VAR [%]'])
    plt.title(sector.upper(), fontsize=20)
    plt.show()


# Now that's much better! We don't have any major peak, and the remaining ones are somewhat reasonable values.
# 
# Still, even if we removed all those fake top gainers, *we cannot be fully certain that the remaining stocks have undergone an organic trading process during 2015*. 

# # STEP 3: HANDLE MISSING VALUES, 0-VALUES
# 
# The next check we need to perform concerns the presence of missing values (`NaN`). At the same time, I think it is also useful to check the quantity of `0`-valued entries. What I like to do is simply plot a bar chart of the count of both missing values and 0-valued entries, in order to take a first look at the situation. (Due to the large quantity of financial indicators available, I will make quite a big plot)
# 
# Before doing that, we can drop the categorical columns from the dataframe `df`, since we won't be needing them now.

# In[ ]:


# Drop columns relative to classification, we will use them later
class_data = df.loc[:, ['Class', '2015 PRICE VAR [%]']]
df.drop(['Class', '2015 PRICE VAR [%]'], inplace=True, axis=1)

# Plot initial status of data quality in terms of nan-values and zero-values
nan_vals = df.isna().sum()
zero_vals = df.isin([0]).sum()
ind = np.arange(df.shape[1])

plt.figure(figsize=(50,10))

plt.subplot(2,1,1)
plt.title('INITIAL INFORMATION ABOUT DATASET', fontsize=22)
plt.bar(ind, nan_vals.values.tolist())
plt.ylabel('NAN-VALUES COUNT', fontsize=18)

plt.subplot(2,1,2)
plt.bar(ind, zero_vals.values.tolist())
plt.ylabel('ZERO-VALUES COUNT', fontsize=18)
plt.xticks(ind, nan_vals.index.values, rotation='90')

plt.show()


# We can see that:
# 1. There are quite a lot of missing values
# 2. There are also a lot of 0-valued entries. For some financial indicators, almost every entry is set to 0.
# 
# To understand the situation from a more quantitative perspective, it is useful to count the occurrences of both missing-values and 0-valued entries, and sort them in descending order. This allows us to establish the *dominance* level for both missing values and 0-valued entries.

# In[ ]:


# Find count and percent of nan-values, zero-values
total_nans = df.isnull().sum().sort_values(ascending=False)
percent_nans = (df.isnull().sum()/df.isnull().count() * 100).sort_values(ascending=False)
total_zeros = df.isin([0]).sum().sort_values(ascending=False)
percent_zeros = (df.isin([0]).sum()/df.isin([0]).count() * 100).sort_values(ascending=False)
df_nans = pd.concat([total_nans, percent_nans], axis=1, keys=['Total NaN', 'Percent NaN'])
df_zeros = pd.concat([total_zeros, percent_zeros], axis=1, keys=['Total Zeros', 'Percent Zeros'])

# Graphical representation
plt.figure(figsize=(15,5))
plt.bar(np.arange(30), df_nans['Percent NaN'].iloc[:30].values.tolist())
plt.xticks(np.arange(30), df_nans['Percent NaN'].iloc[:30].index.values.tolist(), rotation='90')
plt.ylabel('NAN-Dominance [%]', fontsize=18)
plt.grid(alpha=0.3, axis='y')
plt.show()

plt.figure(figsize=(15,5))
plt.bar(np.arange(30), df_zeros['Percent Zeros'].iloc[:30].values.tolist())
plt.xticks(np.arange(30), df_zeros['Percent Zeros'].iloc[:30].index.values.tolist(), rotation='90')
plt.ylabel('ZEROS-Dominance [%]', fontsize=18)
plt.grid(alpha=0.3, axis='y')
plt.show()


# The two plots above clearly show that to improve the quality of the dataframe `df` we need to:
# 1. fill the missing data
# 2. fill or drop those indicators that are heavy zeros-dominant.
# 
# **What levels of nan-dominance and zeros-dominance are we going to tolerate?**
# 
# I usually determine a threshold level for both nan-dominance and zeros-dominance, which corresponds to a given percentage of the total available samples (rows): **if a column has a percentage of nan-values and/or zero-valued entries higher than the threshold, I drop it**.
# 
# For this specific case we know that we have about 3800 samples, so I reckon we can set:
# * nan-dominance threshold = 5-7%
# * zeros-dominance threshold = 5-10%
# 
# Once the threshold levels have been set, I iteratively compute the `.quantile()` of both `df_nans` and `df_zeros` in order to find the number of financial indicators that I will be dropping. In this case, we can see that:
# * We need to drop the top 50% (`test_nan_level=1-0.5=0.5`) nan-dominant financial indicators in order to not have columns with more than 226 `nan` values, which corresponds to a nan-dominance threshold of 5.9% (aligned with our initial guess).
# * We need to drop the top 40% (`test_zeros_level=1-0.4=0.6`) zero-dominant financial indicators in order to not have columns with more than 283 `0` values, which corresponds to a zero-dominance threshold of 7.5% (aligned with our initial guess).

# In[ ]:


# Find reasonable threshold for nan-values situation
test_nan_level = 0.5
print(df_nans.quantile(test_nan_level))
_, thresh_nan = df_nans.quantile(test_nan_level)

# Find reasonable threshold for zero-values situation
test_zeros_level = 0.6
print(df_zeros.quantile(test_zeros_level))
_, thresh_zeros = df_zeros.quantile(test_zeros_level)


# Once the threshold levels have been set, I can proceed and drop from `df` those columns (financial indicators) that show dominance levels higher than the threshold levels, in terms of both missing values and 0-valued entries.
# 
# So, we reduced the number of financial indicators available in the dataframe `df` to 62. By doing so, we removed all those columns characterized by heavy nan-dominance and zeros-dominance. 
# 
# *We should always keep in mind that this is quite a brute force approach, and there is the possibilty of having dropped useful information.*

# In[ ]:


# Clean dataset applying thresholds for both zero values, nan-values
print(f'INITIAL NUMBER OF VARIABLES: {df.shape[1]}')
print()

df_test1 = df.drop((df_nans[df_nans['Percent NaN'] > thresh_nan]).index, 1)
print(f'NUMBER OF VARIABLES AFTER NaN THRESHOLD {thresh_nan:.2f}%: {df_test1.shape[1]}')
print()

df_zeros_postnan = df_zeros.drop((df_nans[df_nans['Percent NaN'] > thresh_nan]).index, axis=0)
df_test2 = df_test1.drop((df_zeros_postnan[df_zeros_postnan['Percent Zeros'] > thresh_zeros]).index, 1)
print(f'NUMBER OF VARIABLES AFTER Zeros THRESHOLD {thresh_zeros:.2f}%: {df_test2.shape[1]}')


# # STEP 4: CORRELATION MATRIX, CHECK MISSING VALUES AGAIN
# 
# The correlation matrix is an important tool that can be used to quickly evaluate the linear correlation between variables, in this case financial indicators. As clearly explained [here](https://www.investopedia.com/ask/answers/032515/what-does-it-mean-if-correlation-coefficient-positive-negative-or-zero.asp), a positive linear correlation value between two variables means that they move in a similar way; a negative linear correlation value between two variables means that they move in opposite ways. Finally, if the correlation value is close to 0, then their trends are not related.
# 
# Looking at the figure below, we can see that there is a chunk of financial indicators that show no linear correlation whatsoever. Those financial indicators are the heavy nan-dominant ones (as highlighted in the barplot below). This means that this chart will change once we will fill the `nan` values.

# In[ ]:


# Plot correlation matrix
fig, ax = plt.subplots(figsize=(20,15)) 
sns.heatmap(df_test2.corr(), annot=False, cmap='YlGnBu', vmin=-1, vmax=1, center=0, ax=ax)
plt.show()


# We can evaluate the impact of our choices in terms of threshold levels by plotting again the count of missing values and 0-valued entries occurring in the remaining financial indicators. The situation has clearly improved, even if a few financial indicators mantain high levels of nan-dominance, which is evident when looking at the correlation matrix above.

# In[ ]:


# New check on nan values
plt.figure(figsize=(50,10))

plt.subplot(2,1,1)
plt.title('INFORMATION ABOUT DATASET - CLEANED NAN + ZEROS', fontsize=22)
plt.bar(np.arange(df_test2.shape[1]), df_test2.isnull().sum())
plt.ylabel('NAN-VALUES COUNT', fontsize=18)

plt.subplot(2,1,2)
plt.bar(np.arange(df_test2.shape[1]), df_test2.isin([0]).sum())
plt.ylabel('ZERO-VALUES COUNT', fontsize=18)
plt.xticks(np.arange(df_test2.shape[1]), df_test2.columns.values, rotation='90')

plt.show()


# # STEP 5: HANDLE EXTREME VALUES
# 
# Analyzing the `df` with the method `.describe()` we can see that some financial indicators show a large discrepancy between max value and 75% quantile. Furthermore, we also have standard deviation values that are very large! This could be a sign of the presence of outliers: to be conservative, I will drop the top 3% and bottom 3% of the data for each financial indicator.

# In[ ]:


# Analyze dataframe
df_test2.describe()


# In[ ]:


# Cut outliers
top_quantiles = df_test2.quantile(0.97)
outliers_top = (df_test2 > top_quantiles)

low_quantiles = df_test2.quantile(0.03)
outliers_low = (df_test2 < low_quantiles)

df_test2 = df_test2.mask(outliers_top, top_quantiles, axis=1)
df_test2 = df_test2.mask(outliers_low, low_quantiles, axis=1)

# Take a look at the dataframe post-outliers cut
df_test2.describe()


# Looking at the statistical description of the dataframe `df` post outliers removal, we can see that **we managed to decrease the standard deviation values considerably**, and also the discrepancy between max value and 75% quantile is smaller.

# # STEP 6: FILL MISSING VALUES
# 
# We can now fill the missing values, but how? There are several methods we could use to fill the missing values:
# * fill `nan` with 0
# * fill `nan` with mean value of column
# * fill `nan` with mode value of column
# * fill `nan` with previous value
# * ...
# 
# In this case, I think it is appropriate to fill the missing values with the mean value of the column. However, we must not forget the intrinsic characteristics of the data we are working with: **we have a many stocks from many different sectors**. It is fair to expect that each sector is characterized by macro-trends and macro-factors that may influence some financial indicators in different ways. So, I reckon that we should keep this separation somehow.
# 
# From a practical perspective, this translates into **filling the missing value with the mean value of the column, grouped by each sector**.

# In[ ]:


# Replace nan-values with mean value of column, considering each sector individually.
df_test2 = df_test2.groupby(['Sector']).transform(lambda x: x.fillna(x.mean()))


# Once that's done, we can plot again the correlation matrix in order to evaluate the impact of our choices.

# In[ ]:


# Plot correlation matrix of output dataset
fig, ax = plt.subplots(figsize=(20,15)) 
sns.heatmap(df_test2.corr(), annot=False, cmap='YlGnBu', vmin=-1, vmax=1, center=0, ax=ax)
plt.show()


# As we can see, the chunk of financial indicators that was characterized by linear correlation equal to 0 is now more organic (i.e. correlation is either positive or negative), thanks to the fact that we replaced the missing values with the respective mean value of the column (per-sector).

# # STEP 7: ADD TARGET DATA
# 
# As you recall, we dropped the target data from the dataframe. However, we need it back in order to use this dataset with ML algorithms. This can be easily achieved thanks to a couple `.join()` lines.
# 
# Finally, we can wrap this notebook up by printing both `.info()` and `.describe()` of the final dataframe `df_out`.

# In[ ]:


# Add the sector column
df_out = df_test2.join(df['Sector'])

# Add back the classification columns
df_out = df_out.join(class_data)

# Print information about dataset
df_out.info()
df_out.describe()


# In this notebook, we explored the financial indicators for a list of stocks relative to 2014. After an initial investigation regarding general aspects of the dataset, we performed some data cleaning steps in order to improve the usability of the dataset.
# 
# Feel free to fork this notebook and add your own touch.
