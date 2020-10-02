#!/usr/bin/env python
# coding: utf-8

# Hello everyone, first kernel here :)
# I will explore the data provided for the black friday sales and attempt to draw some basic conclusions using descriptive statistics.
# 
# I begin by loading the necessary libs:

# In[ ]:


# Loading the essentials, numpy and pandas

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import scipy.stats as ss

# Next up, os for listing and walking through directories
import os

# Plt and seaborn for graphing
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import seaborn as sns

print(os.listdir("../input"))


# Time to load the data into a dataframe and look at the variables at my disposal.

# In[ ]:


df_blackfriday = pd.read_csv("../input/BlackFriday.csv")

df_blackfriday.head()


# Already we see some missing values in the 2nd and 3rd product categories - let's take a better look at NaN values.

# In[ ]:


nan_to_total_ratio = df_blackfriday.isna().sum() / df_blackfriday.shape[0]
print(nan_to_total_ratio)


# Seems there are missing values mainly in the aforementioned product categories. I am not sure we can safely replace these NA values with zeros assuming that this means that a particular product does not belong to a category, especially considering the fact that these are the only missing values in the dataset and the curator warned that there are missing values. Perhaps I should drop these columns out of the analysis for now.

# In[ ]:


# Save these for later use
product_category_2_series = df_blackfriday['Product_Category_2']
product_category_3_series = df_blackfriday['Product_Category_3']

df_blackfriday = df_blackfriday.drop(['Product_Category_2', 'Product_Category_3'], axis=1)


# Let's take a look at the distributions of ** purchasers** per various categories rather than the distribution of purchases. In order to get meaningful information on the gender, age etc. distributions we should look at the purchases grouped by user ID since the dataset is actually comprised of individual transactions. Thanks to [this user](https://www.kaggle.com/dabate) with [his comment](https://www.kaggle.com/shamalip/black-friday-data-exploration#433093) (first comment in the linked kernel) for pointing it out - I have not initially realized this to be the case and assumed each data row represented a single purchaser.
# 
# Before displaying the distribution graphs, I want to check if the gender, age, occupation, city category, length of stay in a city and marital status for each user have not changed during the collection of this dataset. While the noise in information that is the consequence of these changes for single purchasers is most likely minimal in our case, I wanted to be precise.
# 
# Also, I wanted to check if purchases of the same item (product_id) occured more than once for an individual user.
# 
# Below is a snippet of code used to check all of the above. It's not a very elegant solution so please comment below if you know of a better way to do this:
# 

# In[ ]:


# Store the relevant categories
categories = df_blackfriday.columns[2:-1]

# Define function for our check
# Function used so I can easily exit two nested loops with the 'return' keyword
def check_duplicates_and_differences(category, groupby_object):
    # Iterate over a GroupBy object, where keys are User IDs,
    # and groups are groups of categorical values sharing a User ID
    for key,group in groupby_object:
        value_list = []
        # Iterate over individual values in group
        for i, value in enumerate(group):
            # Branch code here: in the case of Product ID, we check for existence of duplicate values,
            # else we check for differences between current and previous value
            if category == 'Product_ID':
                if value in value_list:
                    print("Found duplicate value: {0} in user_id: {1} of category: {2}".format(value,group,category))
                    # Break the loops if any duplicates are found within a category
                    return
                value_list.append(value)
            else:
                if i>=1 and value != value_list[i-1]:
                    print("Variable {0} of user {1} changes from {2} to {3}".format(category, key, value_list[i-1], value))
                    # Break the loops if any changes are found within a category
                    return
                value_list.append(value)

for category in categories:
    # First, group our category by User ID
    grouped = df_blackfriday[category].groupby(df_blackfriday['User_ID'])

    check_duplicates_and_differences(category, grouped)


# Oops... seems I've included the Product_Category_1 column in the search, which is non-sensical - it is expected that users will purchase products from different subcategories. Nonetheless, the output confirmed that the above code detects changes.
# 
# As far as the other categories go, we can see that the values don't change which means we can safely draw conclusions from our distribution graphs. On the other hand, it seems no identical purchases have been made by any of the users. 
# 
# I realized also that in the case of Product ID it might've been good enough to just look at the distribution of **purchases** rather than purchasers, but the number of individual buyers of a certain product might be a good feature for a later regression problem of predicting prices.
# 
# Time for the distribution graphs. Note that I distinguish between the Product Category variable and the rest - this column doesn't need to be grouped by user since our aim is to find out how much a certain product is purchased.
# 
# Adding to this, I thought it would be a good idea to plot another graph for each categorical variable: the amount of purchases in dollars per category.
# 

# In[ ]:


ncols = 2

# Assuming ncols, calculate number of rows based on number of categories times two
# since we have 2 graphs per cat, then use these values for our grid size
nrows = math.ceil(len(categories)*2/ncols)
grid_size = (nrows,ncols)
print(grid_size)
# Multiplier to convert grid size to appropriate size in inches
inch_multiplier = 4

grid_inches = tuple(map(lambda x: x * inch_multiplier, grid_size))

plt.figure(figsize=grid_inches)

# 2-step iteration over number of categories times two
for index,category in zip(range(1, len(category)*2, 2), categories):
    plt.subplot(grid_size[0],grid_size[1],index)
    # Create a common color mapping for both cases so we can more easily compare the differences
    subcategories = df_blackfriday[category].unique()
    rgb_values = sns.color_palette("Set1", len(subcategories))
    color_map = dict(zip(subcategories, rgb_values))
    
    # First, plot distribution of users for category
    if not category == 'Product_Category_1':
        # Group by user and aggregate values by replacing them with the first value - can do this since
        # I've shown that the values in these categories don't change per user
        df_grouped_by_user = df_blackfriday[category].groupby(df_blackfriday['User_ID']).agg('first')
        # Use normalize parameter to obtain fractions
        df_grouped_by_user = df_grouped_by_user.sort_values().value_counts(normalize=True)
        df_grouped_by_user.plot(kind='bar', title=category, color=df_grouped_by_user.index.map(color_map))
        plt.ylabel('No. of users')
    else:
        df_grouped = df_blackfriday[category].sort_values().value_counts(normalize=True)
        df_grouped.plot(kind='bar', title=category, color=df_grouped.index.map(color_map))

    # Second, plot amount of purchases in dollars per category
    plt.subplot(grid_size[0],grid_size[1],index+1)
    
    df_grouped_by_cat = df_blackfriday['Purchase'].groupby(df_blackfriday[category]).agg('sum')
    # Divide Series elements by sum of all purchases to get fractions that we can 
    # compare with normalized count distributions
    df_grouped_by_cat = df_grouped_by_cat.divide(df_blackfriday['Purchase'].sum())
    df_grouped_by_cat = df_grouped_by_cat.sort_values(ascending=False)
    df_grouped_by_cat.plot(kind='bar', title=category, color=df_grouped_by_cat.index.map(color_map))
    plt.ylabel('Purchases')
        
plt.subplots_adjust(wspace = 0.2, hspace = 0.5, top=3)


# We can see that:
# * Almost 70% of all purchasers are men
# * 35% of the purchasers are aged between 26 and 35 years
# * Occupation no. 4 is most represented with about 12% of all purchasers, followed closely by occupations no. 0 and no. 7 with about 11.5% share
# * Nearly 60% of purchasers are single
# * More than 50% of purchases come from cities in the C category.
# * Most of the purchasers only stayed a year in the current city (about 25%).
# * Most of the products purchased come from subcategory no. 5 with about 28% of the share
# 
# Comparing the purchasing amount fractions with the count distributions, it seems that the fractions and ordering of subcategories don't match up in the cases of City Category, Gender and Product Category 1. Although most purchasers come from cities in the City Category C, the ones hailing from B cities seem to spend more. In the case of Gender, the distribution tilts even more in favor of men in terms of purchasing power (80% purchases vs. 70% purchasers). Finally, Product Category 1 purchases seem to differ mainly when it comes to product subcategory no. 1 which dominates the purchases with more than a 35% share, while being second to subcategory no. 5 in count distributions with only about 25% share.
# 

# Let's look at the correlations between our variables. In this dataset we have a mix of categorical and continuous variables so we can't simply use pandas .corr() method since it is primed for calculating correlation coefficients for continuous variables with three different approaches: Pearson, Kendall and Spearman. 
# 
# With that in mind, I searched for existing solutions in python for calculating the various measurements of association and correlation for categorical -categorical  and continuous - categorical variables. We will be using [Cramer's V](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V) measure of association for the categorical - categorical case.
# First, props to [this answer at stack exchange](https://stackoverflow.com/a/39266194) for the following function:

# In[ ]:


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    # Use sum twice because of the way it works for dataframes
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


# I also define a function for the [correlation ratio](https://en.wikipedia.org/wiki/Correlation_ratio) for the case of categorical vs continuous variables;

# In[ ]:


def correlation_ratio(dataframe, nominal_series_name, numerical_series_name):
    categories_means = []
    categories_weights = []
    total_mean = np.average(dataframe[numerical_series_name])
    total_variance = np.var(dataframe[numerical_series_name])
    for category in dataframe[nominal_series_name].unique():
        category_series = dataframe.loc[dataframe[nominal_series_name] == category][numerical_series_name]
        category_mean = np.average(category_series)
        categories_means.append(category_mean)
        categories_weights.append(len(category_series))

    categories_weighted_variance = np.average((categories_means - total_mean)**2, weights=categories_weights)
    eta = categories_weighted_variance / total_variance
    return eta


# We use these functions to define a 3rd function to create our correlation matrix (thanks to [dython library](https://github.com/shakedzy/dython) for inspiration):

# In[ ]:


def create_corr_matrix(dataframe, nominal_columns, numerical_columns):
    columns = dataframe.columns
    # Forcing dtype np.float64 seems to be important for sns.heatmap() method to work with the output correlation matrix
    corr_matrix = pd.DataFrame(index=columns, columns=columns, dtype=np.float64)
    for i in range(0, len(columns)):
        for j in range(i, len(columns)):
            if i == j:
                corr_matrix.at[columns[j], columns[i]] = 1.00
            else:
                if columns[i] in nominal_columns:
                    if columns[j] in nominal_columns:
                        # Categorical to categorical correlation
                        confusion_matrix = pd.crosstab(dataframe[columns[i]], dataframe[columns[j]])
                        corr_coef = cramers_corrected_stat(confusion_matrix)
                        corr_matrix.at[columns[j], columns[i]] = corr_coef
                        corr_matrix.at[columns[i], columns[j]] = corr_coef
                    else:
                        # Categorical to continuous correlation
                        corr_coef = correlation_ratio(dataframe,columns[i], columns[j])
                        corr_matrix.at[columns[j], columns[i]] = corr_coef
                        corr_matrix.at[columns[i], columns[j]] = corr_coef
                else:
                    if columns[j] in nominal_columns:
                        # Continuous to categorical correlation
                        corr_coef = correlation_ratio(dataframe, columns[j], columns[i])
                        corr_matrix.at[columns[j], columns[i]] = corr_coef
                        corr_matrix.at[columns[i], columns[j]] = corr_coef
                    else:
                        # Continuous to continuous correlation - using Spearman coefficient here
                        corr_coef, pval = ss.spearmanr(dataframe[columns[j]], dataframe[columns[i]])
                        corr_matrix.at[columns[j], columns[i]] = corr_coef
                        corr_matrix.at[columns[i], columns[j]] = corr_coef
    return corr_matrix   


# Finally, use these functions to construct our correlation heatmap:

# In[ ]:


# Drop User_ID, Product_ID and our only continuous variable, Purchase
nominal_columns = df_blackfriday.columns.drop(['User_ID','Product_ID','Purchase'])
numerical_columns = ['Purchase']
df_blackfriday_nouser = df_blackfriday.drop(['User_ID', 'Product_ID'], axis=1)
corr_matrix = create_corr_matrix(df_blackfriday_nouser, nominal_columns, numerical_columns)

plt.subplots(figsize=(18,8))

# Use vmin = 0 since we only have one numerical column, and the non-continuous association measures used
# here range from 0 to 1
sns.heatmap(corr_matrix, vmin=0, square=True)


# The correlation ratio was only calculated for our Purchase variable vs all the other ones since this is the only continuous variable in our analysis. The highest correlation ratio seems to be with the Product Category 1 categorical variable with a correlation ratio value north of 0.6. This is probably because the subcategories of product category 1 represent different types of consumer products, with each product type also having some price variance due to products coming from different manufacturers.
# 
# Looking at the Cramer's V measures, the most associated pairs of categorical variables, given [Cohen's (1988) guidelines for behavioral sciences](http://rcompanion.org/handbook/H_10.html) for which there's a medium strength of association (or higher, depending on the number of degrees of freedom) seem to be Occupation - Age, Occupation - Gender and Age - Marital Status. Although we cannot interpret the direction of the association (positive or negative) with Cramer's V measure, the Age and Marital Status association arises probably because of the increased likelihood of being in a marriage the older someone is. The association between Occupation and Gender isn't too surprising either since many occupations tend to be dominated by men or women.  Finally, the association between Occupation and Age is an interesting one - I wasn't sure why this might've occured. One idea is this occurs because of the various 'newer' occupations in the IT field that are dominated by younger people (assuming this is the case).
# 
# 
