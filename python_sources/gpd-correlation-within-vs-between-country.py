#!/usr/bin/env python
# coding: utf-8

# This Kernel does some basic analysis on the suicide dataset. The goal is to set up a pipeline to analyse such datasets. The ultimate goal here is to get a better view on possible correlations between gpd/capita, age, gdp/capita rate of change, and suicide per 100'000, and its rate of change. To investigate the possible connection between gdp per capita and suicide rates, two approaches can be taken, which we will see yield very different results. One approach is to take the data from all listed countries (101) together over the years and make a scatter plot for each pair of (suicide/100k -- gdp/capita). This data allows us to calculate a correlation that is based on data from *different* countries across around 30 years. The problem with this procedure is that countries are very divers and other possible parameters correlated to suicide rates probably differ from country to country, and consequently are not controled for. The second approach looks at each country individually and considers the *time series* of gdp/capita and suicide rates over 30 years. This within-country approach thus singles out gdp/capita as a time-evolving parameter. It comes to no surprise that both approach yield different results, even to such an extend that the correlation has opposite sign.
# 
# This kernel is an attempt to create a generic pipeline to analyse data in order to answer crucial questions. As this is my first commited Kernel, I'm sure I can improve. I'm open to any feedback for future work.
# 
# ( This kernel can be seen as an introduction to basic dataset analysis, using EDA, visualisation, correlation, multi-indexing, pivoting, row-by-row manipulation, ...)

# The first thing that needs to happen is importing the most widely used packages, which should be standard procedure for any data analysis. We import *numpy* for mathematics, *pandas* for its data-friendly dataframes, and *seaborn* and *matplotlib* for visualisation. In this kernel, we import *os* to load the dataset *master.csv*. A first glimpse into the data is obtained by using the *.head()* method.
# 

# In[ ]:


# import the needed libraries.
import numpy as np # mathematics
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # used for plotting dataframes
import matplotlib.pyplot as plt # basic plotting library
plt.style.use('ggplot') # plot styling

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))


# import data from .csv file:
df = pd.read_csv("../input/master.csv")
df.head(10) # show the first 10 rows of the dataset
#df.tail(10)


# **A few observations can be made from simply looking at this piece of data:**
# 
# * There is a low number of columns, so we can visually see which parameters are available to us: age, sex, year, country,...
# * Looks like the data consists of both categorical data (sex, age, generation,...) and numerical (population, suicides_no,...)
# * The country and year column allow for multi-indexing to get a better overview of the data
# * The data is sorted alphabetically on country, then on year
# * There are six age categories present
# * The generation column gives the possibility to keep track of the same people over time while their age changes
# * Final year is 2014 (this turns out not to be the case by using code to check this)
# * There is no overall suicide number per country per year, we should make this one ourselves
# * From the table above it is not easy to see which countries and years are included.
# 
# 
# The next step is to write some code to check and quantify our visual observations, and get some more information from the dataset. When datasets become enormous, such systematic, code based analysis will be the only viable option to get a grasp of the data.

# In[ ]:


# Get some quantitative insight in the data. Only viable option if big data.

df.info() # this gives insight into the type of data which is in our dataframe

df_columns = df.columns # get column names which signify the parameters
print(df_columns)

print(df['age'].nunique()) # this gives us the unique age categories in the "age" column
print(df['country'].nunique()) # there is data of 101 countries
print(df['generation'].unique()) # this gives us the unique generation categories in the "generation" column

print(df['year'].max()) # the last year of data is 2016, this was not seen by looking at df.head() or df.tail().

df_country = df[df["country"]=="Belgium"] # print out first few lines for specific country. Leave out .head() to get full section of df

print(df_country.groupby(["year"]).mean().head(30)) # if you groupby.mean(), then only the columns that can be meaned will remain
                                                    # gives overall number of suicides per year, no subcategories


list_of_countries = df['country'].unique()
print(list_of_countries) # get a list of all unique countries
list_of_generations = df['generation'].unique()
print(list_of_generations) # get a list of all unique generations
  


# After having looked over the data, we can take some time to think of interesting questions to answer from the data given. I think this step is crucial to be able to get as much as possible out of a dataset. Most of the time, it is asking great questions that will lead to the best insights.
# * Does GDP *between* countries correlate with suicide rates?
# * Does GDP change *within one country* correlate with suicide rates?--And does this agree with previous one?
# * Does GDP rise/fall in gdp affect men more than women? Is it different for distinct age groups (suspect yes)?
# * Does the dominant age group shift throughout time? This can be checked by tracking generations.  Men/women distinction?
# * Does absolute population correlate with suicides/100k?
# 
# In the next blocks of code, these questions are answered.
# 
# The first approach is to investigate whether there is a correlation between the gdp/capita of a country, and its suicide rates. The first approach is to single out a specific country and look at the time series of both gdp/capita and suicides/100k population
# 
# This can then be compared with a scatter plot of gpd/capita and suicides per 100k for all countries, for each year separately. In this fashion, we keep all the other country specific parameters that could influence suicide rates constant, as cultures vary extremely across countries. Consequently, the picture we get is more accurate than the one obtained where we look *between* country correlation (more on this later). Also note that we start out by splitting up our dataset in male and female, because we know that suicide numbers substantially differ between the two sexes. 

# In[ ]:


## Get to a plot that shows suicide/100k pop compared to gdp_per_capitca and pct_change of gdp for a specific country. ##

# 1: Split the dataframe into male and female
df_male = df[df['sex']=='male']
df_female = df[df['sex']=='female']

# 2: Remove unnecessary columns.
df_male = df_male.drop(['HDI for year','sex', 'country-year'], axis=1) 
df_female = df_female.drop(['HDI for year','sex', 'country-year'], axis=1)

# 3: restructure dataframe for a picture without any subcategories such as age, generation, ...
# --NOTE-- here that we cannot simply take the mean() of the suicide rates, since these numbers are based on different population numbers, i.e. the mean of the suicide rates
# of the different age categories is not equal to the mean of all age categories considered together. As mentioned, we need to take a weighted average here.

df_male_overall = df_male.groupby(['country','year']).mean() # this is the wrong mean for suicide rates, not for gdp per capita
df_male_overall_test_col = df_male.groupby(['country','year']).apply(lambda x: np.average(x['suicides/100k pop'], weights=x['population'])) #gives the right weighted mean column
df_male_overall['suicides/100k pop (weighed)'] = df_male_overall_test_col
df_male_overall = df_male_overall.drop(['suicides_no','suicides/100k pop'], axis=1)

df_female_overall = df_female.groupby(['country','year']).mean() # this is the wrong mean for suicide rates, not for gdp per capita
df_female_overall_test_col = df_female.groupby(['country','year']).apply(lambda x: np.average(x['suicides/100k pop'], weights=x['population'])) #gives the right weighed mean column
df_female_overall['suicides/100k pop (weighed)'] = df_female_overall_test_col
df_female_overall = df_female_overall.drop(['suicides_no','suicides/100k pop'], axis=1)

df_male_overall['pct_change_gdp'] = df_male_overall['gdp_per_capita ($)'].pct_change() # add a column with the percentage change of gdp
df_female_overall['pct_change_gdp'] = df_female_overall['gdp_per_capita ($)'].pct_change() # add a column with the percentage change of gdp

df_male_overall['pct_change_suicides'] = df_male_overall['suicides/100k pop (weighed)'].pct_change() # add a column with the percentage change of gdp
df_female_overall['pct_change_suicides'] = df_female_overall['suicides/100k pop (weighed)'].pct_change() # add a column with the percentage change of gdp


#print(df_female_overall) # Check whether the dataframe now has the structure we need for the plots.

# 4: plot two different scale plots with two y-axes for a specific country
country = 'Belgium'

# gdp_per_capita

ax = df_male_overall.loc[country].plot(y='suicides/100k pop (weighed)', legend=False, color='blue', figsize=(10,5),
                                       title = 'gdp per captica & suicide/100k: 1985-2015, male/female', colormap='jet') # .loc[] makes sure we only consider one country
ax.set_ylabel('suicides/100k')
ax2 = ax.twinx()
ax2.set_ylabel('gdp per capita')
df_male_overall.loc[country].plot(y='gdp_per_capita ($)', ax=ax2, legend=False, color='green')
df_female_overall.loc[country].plot(y='suicides/100k pop (weighed)', ax=ax, legend=False, color='red')
ax.figure.legend()

# pct_change
ax = df_male_overall.loc[country].plot(y='pct_change_suicides', legend=False, color='blue', figsize=(10,5),
                                       title='pct_change gdp/capita & pct_change suicides: 1985-2015, male/female')
ax2 = ax.twinx()
ax.set_ylabel('pct_change suicides/100k')
ax2.set_ylabel('pct_change gdp per capita')
df_male_overall.loc[country].plot(y='pct_change_gdp', ax=ax2, legend=False, color='green')
df_female_overall.loc[country].plot(y='pct_change_suicides', ax=ax, legend=False, color='red')
ax.figure.legend()

plt.show()

pct_change_corr_male = df_male_overall.loc[country]['pct_change_suicides'].corr(df_male_overall.loc[country]['pct_change_gdp'])
gdp_corr_male = df_male_overall.loc[country]['suicides/100k pop (weighed)'].corr(df_male_overall.loc[country]['gdp_per_capita ($)'])

pct_change_corr_female = df_female_overall.loc[country]['pct_change_suicides'].corr(df_female_overall.loc[country]['pct_change_gdp'])
gdp_corr_female = df_female_overall.loc[country]['suicides/100k pop (weighed)'].corr(df_female_overall.loc[country]['gdp_per_capita ($)'])


print('Correlation between pct_change_suicide and pct_change_gdp in {}: {:.3f} for males'.format(country, pct_change_corr_male))
print('Correlation between pct_change_suicide and pct_change_gdp in {}: {:.3f} for females'.format(country, pct_change_corr_female))
print('Correlation between suicide rate and gdp/capita in {}: {:.3f} for males'.format(country,gdp_corr_male))
print('Correlation between suicide rate and gdp/capita in {}: {:.3f} for females'.format(country,gdp_corr_female))


# Our first findings, both visually and quantitatively, show that there is a negative correlation between suicide rates and gdp/capita for the specific country of Belgium. It's also immediately clear that suicide rates are more than double, or triple for males than for females.
# The *change* in gdp/capita compared to the change in suicide rates is also analyzed to see whether it is the change in standard of living that influences suicides more than the actual standard of living itself. For Belgium it is clear that the correlation between the *change* in gdp/capita and the *change* in suicide rates is far lower than that between the absolute numbers. However, this relation is completely reversed for the United States, where the changes in gdp/capita and suicide rates are highly correlated.

# Now that we have a working code to generate this plot for one specific country, we can look at what the plot looks like for a weighted average over all the countries, in part to be able to judge where specific countries fall compared to others. The procedure is largely the same.

# In[ ]:


# 5 average over all countries -- think if taking mean destroys a parameter (only country) THIS CANNOT BE DONE, cannot take mean of means --> same procedure as before

df_male_world = df_male.groupby('year').mean() # take the mean for each year over all the countries
df_female_world = df_female.groupby('year').mean()

# create a column that contains the weighted averages and add it to the datafeame

df_male_world_col_suicides = df_male_overall.groupby('year').apply(lambda x: np.average(x['suicides/100k pop (weighed)'], weights = x['population'])) 
df_male_world_col_gdp = df_male_overall.groupby('year').apply(lambda x: np.average(x['gdp_per_capita ($)'], weights = x['population']))
df_male_world['suicides/100k pop (weighed)'] = df_male_world_col_suicides
df_male_world['gdp_per_capita (weighed)'] = df_male_world_col_gdp

df_female_world_col_suicides = df_female_overall.groupby('year').apply(lambda x: np.average(x['suicides/100k pop (weighed)'], weights = x['population']))
df_female_world_col_gdp = df_female_overall.groupby('year').apply(lambda x: np.average(x['gdp_per_capita ($)'], weights = x['population']))
df_female_world['suicides/100k pop (weighed)'] = df_female_world_col_suicides
df_female_world['gdp_per_capita (weighed)'] = df_female_world_col_gdp

df_male_world.drop(['population','gdp_per_capita ($)','suicides/100k pop','suicides_no'],axis=1, inplace=True) # drop unnecessary columns
df_female_world.drop(['population','gdp_per_capita ($)','suicides/100k pop','suicides_no'],axis=1, inplace=True)

df_male_world['pct_change_gdp'] = df_male_world['gdp_per_capita (weighed)'].pct_change() # add pct_change_gdp column
df_male_world['pct_change_suicide'] = df_male_world['suicides/100k pop (weighed)'].pct_change()

df_female_world['pct_change_gdp'] = df_female_world['gdp_per_capita (weighed)'].pct_change() # add pct_change_gdp column
df_female_world['pct_change_suicide'] = df_female_world['suicides/100k pop (weighed)'].pct_change()

# 6 plot this data

ax = df_male_world.plot(y='suicides/100k pop (weighed)', legend=False, color='blue', figsize=(10,5), title='World average gdp/capita & suicides/100k: 1985-2015, male/female')
ax2 = ax.twinx()
ax.set_ylabel('suicides/100k pop')
ax2.set_ylabel('gdp/capita')
df_male_world.plot(y='gdp_per_capita (weighed)', ax=ax2, legend=False, color='green')
df_female_world.plot(y='suicides/100k pop (weighed)', ax=ax, legend=False, color='red')
ax.figure.legend()

# pct_change
ax = df_male_world.plot(y='pct_change_suicide', legend=False, color='blue', figsize=(10,5), title='World average pct change in gdp/capitca & suicides/100k: 1985-2015, male/female')
ax2 = ax.twinx()
ax.set_ylabel('pct_change suicides/100k pop')
ax2.set_ylabel('pct_change gdp/capita')
df_male_world.plot(y='pct_change_gdp', ax=ax2, legend=False, color='green')
df_female_world.plot(y='pct_change_suicide', ax=ax, legend=False, color='red')
ax.figure.legend()

pct_change_corr_world_male = df_male_world['pct_change_suicide'].corr(df_male_world['pct_change_gdp'])
gdp_corr_world_male = df_male_world['suicides/100k pop (weighed)'].corr(df_male_world['gdp_per_capita (weighed)'])

pct_change_corr_world_female = df_female_world['pct_change_suicide'].corr(df_female_world['pct_change_gdp'])
gdp_corr_world_female = df_female_world['suicides/100k pop (weighed)'].corr(df_female_world['gdp_per_capita (weighed)'])

print('World Correlation between pct_change_suicide rate and pct_change_gdp: {:.3f} for males'.format( pct_change_corr_world_male))
print('World Correlation between pct_change_suicide rate and pct_change_gdp: {:.3f} for females'.format( pct_change_corr_world_female))
print('World Correlation between suicide rate and gdp/capita: {:.3f} for males'.format(gdp_corr_world_male))
print('World Correlation between suicide rate and gdp/capita: {:.3f} for females'.format(gdp_corr_world_female))


# Now that we have an overall idea of how the gdp per capita and gdp change influences suicide rates within the same country, thus controling for numerous cultural and political parameters, let's see how it compares with the same correlations between countries, for a given year and for all years takes together. To do this, we simply make a scatter plot of all countries, for each year, between the gdp/capita and the suicide rates.

# In[ ]:


# First block of code is for one specific year, which can be seen as just practice in manipulating dataframes, but adds no value to the current analysis.
'''
year = 2005
df_male_overall.loc[(slice(None), year),:].plot.scatter('gdp_per_capita ($)','suicides/100k pop (weighed)', figsize=(10,5))
df_female_overall.loc[(slice(None), year),:].plot.scatter('gdp_per_capita ($)','suicides/100k pop (weighed)', figsize=(10,5))


corr_male_1 = df_male_overall['suicides/100k pop (weighed)'].corr(df_male_overall['gdp_per_capita ($)'])
print(corr_male_1)
'''

corr_male = np.array([])
corr_female = np.array([])
list_of_years = df['year'].unique() # get range object of years to iterate over

# create for-loop where for each year the correlation is calculated and put in an array
for year in list_of_years:
    corr_between_country_male = df_male_overall.loc[(slice(None), year),:]['suicides/100k pop (weighed)'].corr(df_male_overall.loc[(slice(None), year),:]['gdp_per_capita ($)'])
    corr_between_country_female = df_female_overall.loc[(slice(None), year),:]['suicides/100k pop (weighed)'].corr(df_female_overall.loc[(slice(None), year),:]['gdp_per_capita ($)'])

    corr_male = np.append(corr_male, corr_between_country_male)
    corr_female = np.append(corr_female, corr_between_country_female)

print('Correlation between gdp/capita and suicide rates for all countries over 30 year period is {:.3f} for males'.format(corr_male.mean())) # take mean() of array with correlations over the years
print('Correlation between gdp/capita and suicide rates for all countries over 30 year period is {:.3f} for females'.format(corr_female.mean()))

df_male_overall.plot.scatter('gdp_per_capita ($)', 'suicides/100k pop (weighed)', figsize=(10,5), title='All countries gdp/capita - suicide rates: 1987-2016, male')
df_female_overall.plot.scatter('gdp_per_capita ($)', 'suicides/100k pop (weighed)', figsize=(10,5),title='All countries gdp/capita - suicide rates: 1987-2016, female')


# Both the visual scatter plots and the calculated correlation show that there is a positive correlation between gdp/capita and suicide rates, both for males and females.

# From the previous analysis we can conclude that when considering the correlation between gdp/capita and suicides/100k there is a noticable difference when using the evolution in time within countries, and looking between countries. It is arguably far more accurate to track the change through time within one country, since this excludes the huge differences between countries that could also affect suicide rates. The within country approach shows a worldwide *negative* correlation between the gdp/capita and the suicides/100k, which makes sense. However, when looking at the between country data, it shows a *positive* correlation between the two variables.
# 
# In the next part of this analysis, we take a closer look at how the suicide rates are distributed across different ages and generations, and if there is a difference to be seen.
# 
# To look at the more diverse categories, it willbe helpful to convert values of a specific column, say the *age* column, to seperate columns, using what is called pivoting a dataframe.
# 

# In[ ]:


#1 start from the original dataset again
df.head()
df_sub = df.drop(['suicides_no','country-year','HDI for year',' gdp_for_year ($) '],axis=1) # drop columns not needed for this analysis

# 2 single out a specific country
country = 'Belgium'

df_sub_country = df_sub[df_sub["country"]==country]
df_sub_country.head()

#https://pstblog.com/2016/10/04/stacked-charts for little guide on stacked barplots

# 3 prepare the data for a pivot operation
df_sub_country_male = df_sub_country[df_sub_country['sex']=='male'] # split is now necessary to avoid duplicate values in pivot
df_sub_country_female = df_sub_country[df_sub_country['sex']=='female']

df_sub_country_reduced_male = df_sub_country_male.drop(['sex','population','gdp_per_capita ($)','generation','country'],axis=1) # drop unnecessary columns
df_sub_country_reduced_female = df_sub_country_female.drop(['sex','population','gdp_per_capita ($)','generation','country'],axis=1)

# 4 pivot the dataframe
#print(df_sub_country_reduced_male.head()) # dataframe before pivoting

pivot_df_male = df_sub_country_reduced_male.pivot_table(index='year', columns='age', values='suicides/100k pop') # pivot so age categories become columns
pivot_df_female = df_sub_country_reduced_female.pivot_table(index='year', columns='age', values='suicides/100k pop')

pivot_df_male = pivot_df_male[['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']] # change order of columns
pivot_df_female = pivot_df_female[['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']]

#print(pivot_df_male.head()) # dataframe after pivoting

# 5 plot
ax = pivot_df_male.plot.bar(stacked=True, figsize=(10,5), width=0.9, 
                       title='Evolution of suicide rates for each age category. 1985-2016, males') # we make a stacked bar plot to show the distribution per age category over time
plt.ylabel('suicide/100k pop')
ax2 = pivot_df_female.plot.bar(stacked=True, figsize=(10,5), width=0.9,
                       title='Evolution of suicide rates for each age category. 1985-2016, females')
plt.ylabel('suicide/100k pop')

# We can now do the same for generation subcategory (a lot of copy-pasta)
# 1 start from df_sub_country_male/female

df_sub_country_male

# 2 drop colunms, keep generation

df_sub_country_reduced_male_gen = df_sub_country_male.drop(['sex','population','gdp_per_capita ($)','age','country'],axis=1) # drop unnecessary columns
df_sub_country_reduced_female_gen = df_sub_country_female.drop(['sex','population','gdp_per_capita ($)','age','country'],axis=1)

pivot_df_male_gen = df_sub_country_reduced_male_gen.pivot_table(index='year', columns='generation', values='suicides/100k pop') # pivot so age categories become columns
pivot_df_female_gen = df_sub_country_reduced_female_gen.pivot_table(index='year', columns='generation', values='suicides/100k pop')


# change order of columns
pivot_df_male_gen = pivot_df_male_gen[['G.I. Generation', 'Silent', 'Boomers', 'Generation X', 'Millenials','Generation Z']] 
pivot_df_female_gen = pivot_df_female_gen[['G.I. Generation', 'Silent', 'Boomers', 'Generation X', 'Millenials','Generation Z']]

# 3 plot where each bar has the same height to clearly show the evolution of the share of each generation to the suicide rates.

# rescale every row so sum is 100
pivot_df_male_gen.fillna(0, inplace=True) # replace NaN with zero
pivot_df_female_gen.fillna(0, inplace=True)

pivot_df_norm_male = pivot_df_male_gen.div(pivot_df_male_gen.sum(axis=1), axis=0)
pivot_df_norm_female = pivot_df_female_gen.div(pivot_df_female_gen.sum(axis=1), axis=0)

pivot_df_norm_male.plot.bar(stacked=True, figsize=(10,5), width=0.9
                            , title='Tracking suicide rates of each generation over 30 years, males')
plt.ylabel('suicide/100k pop, scaled to 1')
pivot_df_norm_female.plot.bar(stacked=True, figsize=(10,5), width=0.9
                             , title='Tracking suicide rates of each generation over 30 years, females')
plt.ylabel('suicide/100k pop, scaled to 1')



# From the first figure we can see that for the specific case of Belgium, the largest drop in suicide rates occured for males of over 75 years of age. In general it looks like the older the age, the higher the suicide rates. This is also supported by the second two figures, where, for one generation, the suicide rates tend to increase as time evolves.

# To conclude:
