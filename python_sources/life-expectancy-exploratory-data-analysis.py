#!/usr/bin/env python
# coding: utf-8

# # Life Expectancy: Exploratory Data Analysis
# Goal: Find a set of features that affect Life Expectancy.
# 1. Data Cleaning
# 2. Data Exploration
# 3. Feature Engineering
# 4. Summary

# ## Imports and Dataset Load

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/life-expectancy-who/Life Expectancy Data.csv')


# In[ ]:


df.head()


# Now that the dataset is loaded into the DataFrame, `df`, it is now time to begin the EDA. The first step of any EDA is data cleaning (AKA data wrangling, data munging, data cleansing).

# # Section 1: Data Cleaning

# In order to properly clean the data, it is important to understand the variables presented in the data. There are a number of things important to know about each variable:
# 1. What does the variable mean and what type of variable is it (Nominal/Ordinal/Interval/Ratio)?
# 2. Does the variable have missing values? If so, what should be done about them?
# 3. Does the variable have outliers? If so, what should be done about them?
# 
# Each of these questions will be answered in turn for all the variables. And those answers can be found in this section.

# ### 1.1: Dataset Description/Variable Descriptions

# #### Dataset Description
# This dataset is comprised of data from all over the world from various countries aggregated by the World Health Organization (WHO for short). The data is an aggregate of many indicators for a particular country in a particular year. In essence, the data is multiple indicators in a time series separated by country. A more in depth look into the context, content, acknowledgments, and inspiration for this dataset can be found [here](https://www.kaggle.com/kumarajarshi/life-expectancy-who).

# Before getting into the variable descriptions, the string values for the columns/variables themselves are not very 'clean' so the following is a quick cleaning of the column/variable titles.

# In[ ]:


orig_cols = list(df.columns)
new_cols = []
for col in orig_cols:
    new_cols.append(col.strip().replace('  ', ' ').replace(' ', '_').lower())
df.columns = new_cols


# #### Variable Descriptions
# Format: variable (type) - description
# - country (Nominal) - the country in which the indicators are from (i.e. United States of America or Congo)
# - year (Ordinal) - the calendar year the indicators are from (ranging from 2000 to 2015)
# - status (Nominal) - whether a country is considered to be 'Developing' or 'Developed' by WHO standards
# - life_expectancy (Ratio) - the life expectancy of people in years for a particular country and year
# - adult_mortality (Ratio) - the adult mortality rate per 1000 population (i.e. number of people dying between 15 and 60 years per 1000 population); if the rate is 263 then that means 263 people will die out of 1000 between the ages of 15 and 60; another way to think of this is that the chance an individual will die between 15 and 60 is 26.3%
# - infant_deaths (Ratio) - number of infant deaths per 1000 population; similar to above, but for infants
# - alcohol (Ratio) - a country's alcohol consumption rate measured as liters of pure alcohol consumption per capita
# - percentage_expenditure (Ratio) - expenditure on health as a percentage of Gross Domestic Product (gdp)
# - hepatitis_b (Ratio) - number of 1 year olds with Hepatitis B immunization over all 1 year olds in population
# - measles (Ratio) - number of reported Measles cases per 1000 population
# - bmi (Interval/Ordinal) - average Body Mass Index (BMI) of a country's total population
# - under-five_deaths (Ratio) - number of people under the age of five deaths per 1000 population
# - polio (Ratio) - number of 1 year olds with Polio immunization over the number of all 1 year olds in population
# - total_expenditure (Ratio) - government expenditure on health as a percentage of total government expenditure
# - diphtheria (Ratio) - Diphtheria tetanus toxoid and pertussis (DTP3) immunization rate of 1 year olds
# - hiv/aids (Ratio) - deaths per 1000 live births caused by HIV/AIDS for people under 5; number of people under 5 who die due to HIV/AIDS per 1000 births
# - gdp (Ratio) - Gross Domestic Product per capita
# - population (Ratio) - population of a country
# - thinness_1-19_years (Ratio) - rate of thinness among people aged *10-19* (Note: variable should be renamed to *thinness_10-19_years* to more accurately represent the variable)
# - thinness_5-9_years (Ratio) - rate of thinness among people aged 5-9
# - income_composition_of_resources (Ratio) - Human Development Index in terms of income composition of resources (index ranging from 0 to 1)
# - schooling (Ratio) - average number of years of schooling of a population

# As stated above it would be useful to change the name of the variable `thinness_1-19_years` to `thinness_10-19_years` as it is a more accurate depiction of what the variable means.

# In[ ]:


df.rename(columns={'thinness_1-19_years':'thinness_10-19_years'}, inplace=True)


# Now that the descriptions of the dataset and variables have been made, a look at the missing values of each variable should be done.

# ### 1.2: Missing Values

# There are few things that must be done concerning missing values:
# 1. Detection of missing values
#     - Find nulls
#     - Could a null be signified by anything other than null? Zero values perhaps?
# 2. Dealing with missing values
#     - Fill nulls? Impute or Interpolate
#     - Eliminate nulls?

# #### 1.2.1: Missing Values Detection

# ##### Finding possible inexplicit nulls
# These nulls would be missing values that aren't necessarily easy to find using the `df.info()` method.
# - What values could be null?
# - What values could be erroneous?

# **Inexplicit Nulls**\
# The easiest and quickest method here would be to do a quick `df.describe()` and look at each variable on its own to see if the values make sense given the description of the variable.

# In[ ]:


df.describe().iloc[:, 1:]


# Things that may not make sense from above:
# - Adult mortality of 1? This is likely an error in measurement, but what values make sense here? May need to change to null if under a certain threshold.
# - Infant deaths as low as 0 per 1000? That just isn't plausible - I'm deeming those values to actually be null. Also on the other end 1800 is likely an outlier, but it is possible in a country with very high birthrates and perhaps a not very high population total - this can be dealt with later.
# - BMI of 1 and 87.3? Pretty sure the whole population would not exist if that were the case. A BMI of 15 or lower is seriously underweight and a BMI of 40 or higher is morbidly obese, therefore a large number of these measurements just seem unrealistic...this variable might not be worth digging into at all.
# - Under Five Deaths, similar to infant deaths just isn't likely (perhaps even impossible) to have values at zero.
# - GDP per capita as low as 1.68 (USD) possible? Doubtful - but perhaps values this low are outliers.
# - Population of 34 for an entire country? Hmm...

# In[ ]:


plt.figure(figsize=(15,10))
for i, col in enumerate(['adult_mortality', 'infant_deaths', 'bmi', 'under-five_deaths', 'gdp', 'population'], start=1):
    plt.subplot(2, 3, i)
    df.boxplot(col)


# There are a few of the above that could simply be outliers, but there are some that almost certainly have to be errors of some sort. Of the above variables, changes to null will be made for the following since these numbers don't make any sense:
# 1. Adult mortality rates lower than the 5th percentile
# 2. Infant deaths of 0
# 3. BMI less than 10 and greater than 50
# 4. Under Five deaths of 0

# In[ ]:


mort_5_percentile = np.percentile(df.adult_mortality.dropna(), 5)
df.adult_mortality = df.apply(lambda x: np.nan if x.adult_mortality < mort_5_percentile else x.adult_mortality, axis=1)
df.infant_deaths = df.infant_deaths.replace(0, np.nan)
df.bmi = df.apply(lambda x: np.nan if (x.bmi < 10 or x.bmi > 50) else x.bmi, axis=1)
df['under-five_deaths'] = df['under-five_deaths'].replace(0, np.nan)


# ##### All missing values (all explicit now)
# Easy way to do this is with `df.info()`:

# In[ ]:


df.info()


# It appears that there are a decent amount of null values, may be of more use to break down the data into those that contain nulls in order to take a closer look. The function below attempts to do just that - it only returns the columns that contain (explicit) nulls, keeps a running total of those columns with nulls as well as their location in the dataframe, returns the count of nulls in a specified column and the percent of nulls out of all the values in the column.

# In[ ]:


def nulls_breakdown(df=df):
    df_cols = list(df.columns)
    cols_total_count = len(list(df.columns))
    cols_count = 0
    for loc, col in enumerate(df_cols):
        null_count = df[col].isnull().sum()
        total_count = df[col].isnull().count()
        percent_null = round(null_count/total_count*100, 2)
        if null_count > 0:
            cols_count += 1
            print('[iloc = {}] {} has {} null values: {}% null'.format(loc, col, null_count, percent_null))
    cols_percent_null = round(cols_count/cols_total_count*100, 2)
    print('Out of {} total columns, {} contain null values; {}% columns contain null values.'.format(cols_total_count, cols_count, cols_percent_null))


# In[ ]:


nulls_breakdown()


# #### 1.2.2: Dealing with Missing Values

# Nearly half of the BMI variable's values are null, it is likely best to remove this variable altogether.

# In[ ]:


df.drop(columns='bmi', inplace=True)


# Alright, so it looks like there are a lot of columns containing null values, since this is time series data assorted by country, the best course of action would be to interpolate the data by country. However, when attempting to interpolate by country it doesn't fill in any values as the countries' data for all the null values are null for each year, therefore imputation by year may be the best possible method here. Imputation of each year's mean is done below.

# In[ ]:


imputed_data = []
for year in list(df.year.unique()):
    year_data = df[df.year == year].copy()
    for col in list(year_data.columns)[3:]:
        year_data[col] = year_data[col].fillna(year_data[col].dropna().mean()).copy()
    imputed_data.append(year_data)
df = pd.concat(imputed_data).copy()


# One more look at the null values...

# In[ ]:


nulls_breakdown(df)


# Appears that this method took care of the null values. Hopefully meaningful results can still be garnered using this method. Next up, outliers...

# ### 1.3: Outliers

# Similar to missing values, there are a few things that need done in order to deal with outliers:
# 1. Detect the outliers
#     - Boxplots/histograms
#     - Tukey's Method
# 2. Deal with outliers
#     - Drop outliers?
#     - Limit/Winsorize outliers?
#     - Transform the data using log/inverse/square root/etc?

# #### 1.3.1: Outliers Detection

# First a boxplot and histogram will be created for each continuous variable in order to visually see if outliers exist.

# In[ ]:


cont_vars = list(df.columns)[3:]
def outliers_visual(data):
    plt.figure(figsize=(15, 40))
    i = 0
    for col in cont_vars:
        i += 1
        plt.subplot(9, 4, i)
        plt.boxplot(data[col])
        plt.title('{} boxplot'.format(col))
        i += 1
        plt.subplot(9, 4, i)
        plt.hist(data[col])
        plt.title('{} histogram'.format(col))
    plt.show()
outliers_visual(df)


# Visually, it is plain to see that there are a number of outliers for all of these variables - including the target variable, life expectancy. The same will be done statistically using Tukey's method below - outliers being considered anything outside of 1.5 times the IQR.

# In[ ]:


def outlier_count(col, data=df):
    print(15*'-' + col + 15*'-')
    q75, q25 = np.percentile(data[col], [75, 25])
    iqr = q75 - q25
    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
    outlier_count = len(np.where((data[col] > max_val) | (data[col] < min_val))[0])
    outlier_percent = round(outlier_count/len(data[col])*100, 2)
    print('Number of outliers: {}'.format(outlier_count))
    print('Percent of data that is outlier: {}%'.format(outlier_percent))


# In[ ]:


for col in cont_vars:
    outlier_count(col)


# It appears there are a decent amount of outliers in this dataset. Now that they have been detected, what should be done with them?

# #### 1.3.2: Dealing with Outliers

# There are a number of ways to deal with outliers in a dataset, the usual options are as follows:
# 1. Drop Outliers (best avoided in order to keep as much information as possible)
# 2. Limit values to upper and/or lower bounds (Winsorize the data)
# 3. Transform the data (log/inverse/square root/etc.)
#     - advantage: can 'normalize' the data and eliminate outliers
#     - disadvantage: cannot be done to variables containing values of 0 or below

# Since each variable has a unique amount of outliers and also has outliers on different sides of the data, the best route to take is probably winsorizing (limiting) the values for each variable on its own until no outliers remain. The function below allows me to do exactly that by going variable by variable with the ability to use a lower limit and/or upper limit for winsorization. By default the function will show two boxplots side by side for the variable (one boxplot of the original data, and one with the winsorized change). Once a satisfactory limit is found (by visual analysis), the winsorized data will be saved in the `wins_dict` dictionary so the data can easily be accessed later.

# In[ ]:


def test_wins(col, lower_limit=0, upper_limit=0, show_plot=True):
    wins_data = winsorize(df[col], limits=(lower_limit, upper_limit))
    wins_dict[col] = wins_data
    if show_plot == True:
        plt.figure(figsize=(15,5))
        plt.subplot(121)
        plt.boxplot(df[col])
        plt.title('original {}'.format(col))
        plt.subplot(122)
        plt.boxplot(wins_data)
        plt.title('wins=({},{}) {}'.format(lower_limit, upper_limit, col))
        plt.show()


# In[ ]:


wins_dict = {}
test_wins(cont_vars[0], lower_limit=.01, show_plot=True)
test_wins(cont_vars[1], upper_limit=.04, show_plot=False)
test_wins(cont_vars[2], upper_limit=.05, show_plot=False)
test_wins(cont_vars[3], upper_limit=.0025, show_plot=False)
test_wins(cont_vars[4], upper_limit=.135, show_plot=False)
test_wins(cont_vars[5], lower_limit=.1, show_plot=False)
test_wins(cont_vars[6], upper_limit=.19, show_plot=False)
test_wins(cont_vars[7], upper_limit=.05, show_plot=False)
test_wins(cont_vars[8], lower_limit=.1, show_plot=False)
test_wins(cont_vars[9], upper_limit=.02, show_plot=False)
test_wins(cont_vars[10], lower_limit=.105, show_plot=False)
test_wins(cont_vars[11], upper_limit=.185, show_plot=False)
test_wins(cont_vars[12], upper_limit=.105, show_plot=False)
test_wins(cont_vars[13], upper_limit=.07, show_plot=False)
test_wins(cont_vars[14], upper_limit=.035, show_plot=False)
test_wins(cont_vars[15], upper_limit=.035, show_plot=False)
test_wins(cont_vars[16], lower_limit=.05, show_plot=False)
test_wins(cont_vars[17], lower_limit=.025, upper_limit=.005, show_plot=False)


# The plot above is an example of how the winsorization is visually inspected (the rest are not shown for brevity).

# All the variables have now been winsorized as little as possible in order to keep as much data in tact as possible while still being able to eliminate the outliers. Finally, small boxplots will be shown for each variable's winsorized data to show that the outliers have indeed been dealt with.

# In[ ]:


plt.figure(figsize=(15,5))
for i, col in enumerate(cont_vars, 1):
    plt.subplot(2, 9, i)
    plt.boxplot(wins_dict[col])
plt.tight_layout()
plt.show()


# Now that the outliers have been dealt with, the data cleaning section is complete.

# # Section 2: Data Exploration

# Before diving into exploration, a new dataframe with the winsorized data should be created.

# In[ ]:


wins_df = df.iloc[:, 0:3]
for col in cont_vars:
    wins_df[col] = wins_dict[col]


# With that out of the way, the main areas of interest in this section are as follows:
# 1. Univariate Analysis
#     - Continuous variables
#     - Categorical Variables
# 2. Bivariate Analysis
#     - Continuous to Continuous variables
#     - Continuous to Categorical variables
#     - Categorical to Categorical variables

# ### 2.1: Univariate Analysis

# Univariate analysis is looking at the data for each variable on its own. This is generally done best by using histograms for continuous data, count/barplots for categorical data and of course by getting the descriptive stats by using `.describe()`.

# **Descriptive Statistics**

# In[ ]:


wins_df.describe()


# In[ ]:


wins_df.describe(include='O')


# **Visual Distributions**

# In[ ]:


plt.figure(figsize=(15, 20))
for i, col in enumerate(cont_vars, 1):
    plt.subplot(5, 4, i)
    plt.hist(wins_df[col])
    plt.title(col)


# The winsorization had a large effect on some variables while not having too much of an effect on others. Even though all of these variables were winsorized in some fashion, some variables are much more obviously winsorized than others. What about the categorical variables, how many of each of these are there in the data (in essence, what is their distribution?)

# In[ ]:


plt.figure(figsize=(15, 25))
wins_df.country.value_counts(ascending=True).plot(kind='barh')
plt.title('Count of Rows by Country')
plt.xlabel('Count of Rows')
plt.ylabel('Country')
plt.tight_layout()
plt.show()


# This isn't the most appealing graph, but it displays that the mass majority of countries have 16 rows (16 years) worth of data. This is important to know mostly to make sure that certain countries are not being overrepresented.

# In[ ]:


wins_df.year.value_counts().sort_index().plot(kind='barh')
plt.title('Count of Rows by Year')
plt.xlabel('Count of Rows')
plt.ylabel('Year')
plt.show()


# Again, not the most useful plot, but does display that each year has the same amount of rows, except for 2013, which contains 10 more rows than the rest (the countries with only one row from the prior graph's data must be from 2013 alone). This shouldn't have a detrimental effect on analysis.

# In[ ]:


plt.figure(figsize=(10, 5))
plt.subplot(121)
wins_df.status.value_counts().plot(kind='bar')
plt.title('Count of Rows by Country Status')
plt.xlabel('Country Status')
plt.ylabel('Count of Rows')
plt.xticks(rotation=0)

plt.subplot(122)
wins_df.status.value_counts().plot(kind='pie', autopct='%.2f')
plt.ylabel('')
plt.title('Country Status Pie Chart')

plt.show()


# This graph, though simple, is important. The above displays that the majority of our data comes from countries listed as 'Developing' - 82.57% to be exact. It is likely that any model used will more accurately depict results for 'Developing' countries over 'Developed' countries as the majority of the data lies within countries that are 'Developing' rather than 'Developed'.

# ### 2.2: Bivariate Analysis

# There are a number of things that should be examined here:
# 1. Continuous variables compared to the life expectancy (target variable) and to one another
# 2. Categorical variables compared to the life expectancy (target variable)
# 3. Comparison of Country Status and Year to Continuous variables (country has an extremely large number of values with small sample sizes, so country comparisons aren't especially helpful for this dataset)

# #### 2.2.1: Continuous to Continuous Analysis

# In[ ]:


wins_df[cont_vars].corr()


# In[ ]:


mask = np.triu(wins_df[cont_vars].corr())
plt.figure(figsize=(15,6))
sns.heatmap(wins_df[cont_vars].corr(), annot=True, fmt='.2g', vmin=-1, vmax=1, center=0, cmap='coolwarm', mask=mask)
plt.ylim(18, 0)
plt.title('Correlation Matrix Heatmap')
plt.show()


# Note: the values above show rounding at the final two digits, for more exact values, reference the correlation matrix.

# The above heatmap is very useful! It very easily displays a number of important correlations between variables. Some general takeaways from the graphic above:
# - Life Expectancy (target variable) appears to be relatively highly correlated (negatively or positively) with:
#     - Adult Mortality (negative)
#     - HIV/AIDS (negative)
#     - Income Composition of Resources (positive)
#     - Schooling (positive)
# - Life expectancy (target variable) is extremely lowly correlated to population (nearly no correlation at all)
# - Infant deaths and Under Five deaths are extremely highly correlated
# - Percentage Expenditure and GDP are relatively highly correlated
# - Hepatitis B vaccine rate is relatively positively correlated with Polio and Diphtheria vaccine rates
# - Polio vaccine rate and Diphtheria vaccine rate are very positively correlated
# - HIV/AIDS is relatively negatively correlated with Income Composition of Resources
# - Thinness of 5-9 Year olds rate and Thinness of 10-15 Year olds rate is extremely highly correlated
# - Income Composition of Resources and Schooling are very highly correlated

# *Looking ahead: after combining/removing variables that are very highly or extremely highly correlated with one another as well as variables that are very lowly correlated with one another, the best course of action may be to perform dimensionality reduction using PCA in the feature engineering stage.*

# #### 2.2.2: Categorical to Life Expectancy Comparison

# First, looking at how life expectancy has changed over the years may be helpful.

# In[ ]:


sns.lineplot('year', 'life_expectancy', data=wins_df, marker='o')
plt.title('Life Expectancy by Year')
plt.show()


# There appears to definitely be a positive trend over time, but is 15 years of data enough to make the year relevant to a model?

# In[ ]:


wins_df.year.corr(wins_df.life_expectancy)


# There definitely appears to be a correlation, but are the differences between the years significant enough to be considered different? A t-test comparison will be used to find out.

# In[ ]:


years = list(wins_df.year.unique())
years.sort()


# In[ ]:


yearly_le = {}
for year in years:
    year_data = wins_df[wins_df.year == year].life_expectancy
    yearly_le[year] = year_data


# In[ ]:


for year in years[:-1]:
    print(10*'-' + str(year) + ' to ' + str(year+1) + 10*'-')
    print(stats.ttest_ind(yearly_le[year], yearly_le[year+1], equal_var=False))


# Based on the above t-tests, year to year the differences between Life Expectancy do not appear to be significant.

# What about status? There is definitely a difference in the amount of count of values between these two variables (found in the prior univariate analysis), but how about the difference between them with respect to Life Expectancy?

# In[ ]:


wins_df.groupby('status').life_expectancy.agg(['mean'])


# It appears that 'Developed' countries have a much higher average Life Expectancy. But similar to the year comparisons above, is this difference significant? Again, a t-test comparison will be used to find out.

# In[ ]:


developed_le = wins_df[wins_df.status == 'Developed'].life_expectancy
developing_le = wins_df[wins_df.status == 'Developing'].life_expectancy
stats.ttest_ind(developed_le, developing_le, equal_var=False)


# Based on the result of the above t-test, there appears to be a very significant difference between 'Developing' and 'Developed' countries with respect to their Life Expectancy. Since this is the case, a comparison between the status variable and all other continuous variables should be made before moving to the feature engineering phase.

# #### 2.2.3: Status Variable Compared to other Continuous Variables

# Since the status variable only contains two different values, it is likely best to compare a number of descriptive statistics for those two values with respect to all the other continuous variables.

# In[ ]:


wins_df_cols = list(wins_df.columns)
interested_vars = [wins_df_cols[2]]
for col in wins_df_cols[4:]:
    interested_vars.append(col)


# In[ ]:


wins_df[interested_vars].groupby('status').agg('mean')


# From the above, it appears that many of these values are likely correlated to whether a country is 'Developed' or 'Developing'. Again, t-tests are the best way to find out if differences are significant here.

# In[ ]:


developed_df = wins_df[wins_df.status == 'Developed']
developing_df = wins_df[wins_df.status == 'Developing']
for col in interested_vars[1:]:
    print(5*'-' + str(col) + ' Developed/Developing t-test comparison' + 5*'-')
    print('p-value=' +str(stats.ttest_ind(developed_df[col], developing_df[col], equal_var=False)[1]))


# From the above, it is plain to see that there is a significant difference between the following variables with respect to a country's status:
# - Adult Mortality
# - Alcohol
# - Percentage Expenditure
# - Hepatitis B
# - Measles
# - Under Five Deaths
# - Polio
# - Total Expenditure
# - Diphtheria
# - HIV/AIDS
# - GDP
# - Population
# - Thinness of 10 to 19 Year Olds
# - Thinness of 5 to 9 Year Olds
# - Income Composition of Resources
# - Schooling

# This implies that the status of a country is likely highly correlated to the above variables - also from earlier, it is significant in the difference between Life Expectancy as well. This variable should likely be included in our features in the next section. 

# And now that the main comparisons have been made between all the relevant variables, it is now time to move on to the feature engineering phase of the EDA.

# # Section 3: Feature Engineering

# First off, since it is apparent that the status of a country should be included in some way in the final features of the data, one hot encoding will be conducted in order to include it in the future model.

# In[ ]:


feat_df = wins_df.join(pd.get_dummies(wins_df.status)).drop(columns='status').copy()


# In[ ]:


feat_df.iloc[:, 2:].corr().iloc[:, -2:].T


# From the above it can be observed that whether a country is 'Developed' or not is certainly correlated with a number of variables, but not extremely highly. However, it does have a very low correlation with Infant Deaths, Under Five Deaths and Population.

# Next, the categorical columns, 'year' and 'country' will be dropped as they don't have significant differences among life expectancy.

# In[ ]:


feat_df.drop(columns=['country', 'year'], inplace=True)


# From the prior analysis, there are a number of variables that are very or extremely highly correlated with one another. In those cases, the variable which is most highly correlated to Life Expectancy (target variable) will be kept while the others will be dismissed.

# In[ ]:


def feat_heatmap():
    mask = np.triu(feat_df.corr())
    plt.figure(figsize=(15,6))
    sns.heatmap(feat_df.corr(), annot=True, fmt='.2g', vmin=-1, vmax=1, center=0, cmap='coolwarm', mask=mask)
    plt.ylim(len(feat_df.columns), 0)
    plt.title('Features Correlation Matrix Heatmap')
    plt.show()
feat_heatmap()


# The following are very/extremely highly correlated (correlation > .7 or correlation < -.7):
# - Infant Deaths/Under Five Deaths (drop Infant Deaths - Under Five Deaths is more highly correlated to Life Expectancy)
# - GDP/Percentage Expenditure (drop Percentage Expenditure - GDP is more higher correlated to Life Expectancy)
# - Polio/Diphtheria (drop Polio - Diphtheria is more highly correlated to Life Expectancy)
# - Thinness 5-9/Thinness 10-19 (drop Thinness 10-19 as correlations to other variables are slightly higher)
# - Income Composition of Resources/Schooling (drop Schooling - Income Composition of Resources is more highly correlated with Life Expectancy)
# - Developing/Developed (drop Developing - these two are the same just opposite of one another)

# In[ ]:


feat_df.drop(columns=['infant_deaths', 'percentage_expenditure','polio','thinness_10-19_years','schooling','Developing'], inplace=True)


# In addition to the above variables, it may also be useful to drop variables which are not very correlated with any of the other variables, the only variable where that is the case is 'Population'.

# In[ ]:


feat_df.drop(columns=['population'], inplace=True)


# Another look at the correlation heatmap...

# In[ ]:


feat_heatmap()


# #### Did someone say PCA?

# It may be useful to run a Principal Components Analysis (PCA) on this data to reduce the amount of dimensions (features). But there are a number of assumptions/requirements when it comes to PCA:
# - Continuous data: the data used should be of a continuous type
# - Sample size: the sample size should have between 5-10 samples per feature
# - Normalized data: the data is generally normally distributed
# - Correlation: there should be correlation between the features
# - Linearity: it is assumed that relationships between features are linear
# - Outliers: PCA is sensitive to outliers, therefore outliers should not be present

# The features set currently satisfies 3 of the above assumptions: sample size, correlation, outliers. The linearity assumption may not be true, the data is not currently normalized and not all the data is continuous - the developed indicator is categorical. First the 'Developed' variable should be removed.

# In[ ]:


pca_df = feat_df.drop(columns='Developed').copy()


# PCA is an unsupervised technique so the target variable is not needed and can be dropped.

# In[ ]:


pca_df.drop(columns='life_expectancy', inplace=True)


# In[ ]:


len(pca_df.columns)


# In[ ]:


X = scale(pca_df)
sklearn_pca = PCA()
Y = sklearn_pca.fit_transform(X)
print('Explained variance by Principal Components:', sklearn_pca.explained_variance_ratio_)
print('Eigenvalues:', sklearn_pca.explained_variance_)


# In order to capture at least 90% of the variance, 8 components would still be needed, only reducing the amount of features by 3. This is with the assumption that the variables are linearly related as well. If the components with explained variance of one or greater are used (generally not the greatest idea - especially if explanation for variance isn't especially high for those components), then it would be down to three principal components. What does the scree plot have to say?

# In[ ]:


plt.plot(sklearn_pca.explained_variance_)
plt.show()
print('PC1 Explained Variance:', str(round(sklearn_pca.explained_variance_ratio_[0]*100, 2))+'%')


# Based on the scree plot above, it would suggest that only PC1 be kept, this is likely not a great idea as PC1 only accounts for 36.69% of the total variance of the variables.

# #### In this case, perhaps more features > less features

# Ultimately, I would probably start modeling using the features prior to the PCA method. Those features are as follows:
# 1. Adult Mortality
# 2. Alcohol
# 3. Hepatitis B
# 4. Measles
# 5. Under-Five Deaths
# 6. Total Expenditure
# 7. Diphtheria
# 8. HIV/AIDS
# 9. GDP
# 10. Thinness 5-9 Years
# 11. Income Composition Of Resources
# 12. Developed

# *Note: it may be of use to normalize many of these variables' values, that all depends on the model - the above is simply a good start and easily explainable.*

# All of the above variables contain a seemingly meaningful correlation to the target variable (Life Expectancy) while also not being overly correlated with one another.

# The reason more variables have not been removed is because there doesn't seem to be a good reason for further removal. Ultimately, keeping more indicators is likely better as long as they are unique enough from one another (and there isn't an overabundance of them).

# # Section 4: In Summary

# In summation, the dataset started with 21 unclean variables (including the target) and has been pared down to 12 features to describe the target variable (Life Expectancy). This is very likely only the beginning of the possible things that could be done with this dataset, but nonetheless it serves as a solid foundation for modeling. What follows is a general overview of what has been done in this project.

# The first step was to clean the data, this included detecting and dealing with both missing values and outliers. The variables and dataset were given a general description so that a better understanding of what the variables mean could be gathered. Then both explicit and inexplicit missing values were detected. Inexplicit missing values were values that didn't make sense for a variable given the nature of the data. There were a number of seemingly nonsensical values found given many variables' descriptions. Those inexplicit missing values were then converted to explicit missing values or nulls. Interpolation would have likely been the best method to deal with the now explicit null values (since it is time series data), but interpolation in this case would not have garnered any results. Therefore, the next best thing was done instead, imputation based on the means of all countries by year. Once missing values were sorted, the next step was detecting and dealing with outliers. Extreme value detection was done primarily by using standard box and whisker plots with a standard IQR threshold of 1.5. Using this technique, each variable's data was winsorized on a one by one basis to eliminate outliers while limiting the loss of data. Once this step was complete, exploration of the data could be conducted. 

# The now clean dataset was then analyzed using univariate and bivariate techniques. One of the univariate techniques used was to inspect continuous variables using histograms in order to get an idea of their distributions. The general descriptive statistics were also found for the continuous variables. After that, categorical count plots were created to get an idea of the 'distribution' of categorical data. From that analysis it was discovered that the majority of the data fell under the 'Developing' country status. With the univariate analysis complete, it was time to move on to bivariate analysis. Bivariate analysis definitely laid most of the groundwork for understanding the relationships not only between the target variable (Life Expectancy) and the other variables, but also every variable compare to one another. The primary method used in the bivariate analysis was by the use of the correlation matrix in conjunction with the heatmap visual from the Seaborn library. This took care of the main comparisons between continuous to continuous data and was the main foundation for feature selection. But before moving on to feature engineering, some categorical variables were compared to the target variable. It was found that 'Life Expectancy' with respect to year did not garner significant enough difference to use in analysis. However, it was found that the 'Status' of a country did have a significant effect on 'Life Expectancy'. In addition to 'Life Expectancy' it also appeared to be significantly different for a number of other continuous variables. It is for this reason that new indicator variables, 'Developed' and 'Developing', were created in the next section, feature engineering.

# Finally, feature engineering. First, the categorical variables 'Year' and 'Country' were removed as they didn't provide significant differences between data. Then with a general understanding of the variables and the relationship of those variables to one another, it was relatively simple to remove a number of 'highly correlated to one another' variables. The primary method was to use the correlation matrix heatmap in order to detect variables that were highly correlated to another, then from those pairs keep the variable which was more highly correlated with the target. Using this method, the number of variables has been dropped from 20 down to 12 features. The dimensionality reduction method of PCA was used, but didn't seem to garner very useful results. Ultimately, the features kept were those prior to the operation of PCA. It is likely that further transformation of the features should be done, but without knowing which model will be used this set of basic features appears to be the best and most representative set for the target variable of 'Life Expectancy'. With that being said, below is the final heatmap for all the features ultimately selected above (as well as the target variable).

# In[ ]:


feat_heatmap()

