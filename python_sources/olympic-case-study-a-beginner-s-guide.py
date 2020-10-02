#!/usr/bin/env python
# coding: utf-8

# # Olympic Case Study - A Beginner's Guide

# Hello Viewers, Thanks for taking a moment to view my kernel on Olympic case study. 
# 
# This is an interesting kernel aimed at analysing the data on Olympic games held until the year 2016. This is a starter's guide for exploratory data analysis and basic data visualizations. 
# 
# This kernel also helps you to understand how to frame questions initially before analysing the dataset. Once you define your own questions on what to explore in the particular dataset, it would immensely help you.
# 
# Our dataset has the following features.
# 
# * ID - Unique number for each athlete;
# * Name - Athlete's name;
# * Sex - M or F;
# * Age;
# * Height - In centimeters;
# * Weight - In kilograms;
# * Team - Team name;
# * NOC;
# * Games - Year and season;
# * Year;
# * Season - Summer or Winter;
# * City - Host city;
# * Sport - Sport;
# * Event - Event;
# * Medal - Gold, Silver, Bronze, or NA.

# ## Importing Libraries and Dataset

# The libraries used in this kernel are,
# * Numpy
# * Pandas
# * Matplotlib

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
noc = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')
original = df.copy()


# It is a best practise to keep a copy of the original dataset. Let's merge both the datasets with NOC field as the primary key.

# In[ ]:


df = pd.merge(left = df, right = noc, on = 'NOC')


# In[ ]:


print(df.shape)


# In[ ]:


df.info()


# In[ ]:


df.describe()


# Let's quickly drop the duplicate rows in this dataset as they impact our analysis. Also the columns like ID and notes would be of no use for our task. Hence they can also be removed.

# In[ ]:


df.drop_duplicates(inplace = True) 


# In[ ]:


df.drop(columns = ['ID', 'notes'], inplace = True)


# In[ ]:


num_cols = df.select_dtypes(['int64', 'float64']).columns.tolist()
obj_cols = df.select_dtypes('O').columns.tolist()


# In[ ]:


print('No. of Unique Values')
for i in obj_cols:
    print(i, ':', df[i].nunique())


# In[ ]:


df.sample(10)


# Let's check for the missing values. I'm defining a simple function which takes a dataframe as an argument and gives out another dataframe with no. of missing values and their percentage.

# In[ ]:


def missing(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False) / len(df) * 100, 2)
    return pd.concat([total, percent], axis = 1, keys = ['Total', '% Missing'])


# In[ ]:


missing(df)


# Medals can be left as it is because not all the players who participate in the game would win. But the height, weight and age columns also have missing values. Even though the number of missing values in height and weight features are high, let us drop this as we will be still having a dataset with reasonable number of observations which suffice for the analysis.

# In[ ]:


df.dropna(subset = ['Weight', 'Height', 'Age'], inplace = True)


# Though we have age, it is of a numeric type. Let's bin this feature as the categorization of numeric variables would simplify the analysis and more intuitive.

# In[ ]:


bins = [0, 29, 45, 60, 100]
df['Age Group'] = pd.cut(df['Age'], bins = bins)


# An interesting feature we could create from height and weight features is BMI. Let's create a feature for BMI using this formula.
# 
# * BMI = Weight (kg) / Height (m) ** 2

# In[ ]:


df['bmi'] = df['Weight'] / ((df['Height'] / 100) ** 2)


# In[ ]:


df['bmi group'] = np.where(df['bmi'] <= 18.5, 'Underweight',
                          np.where(df['bmi'] < 25, 'Normal',
                                  np.where(df['bmi'] < 30, 'Overweight',
                                          np.where(df['bmi'] >= 30, 'Obese', 'NA'))))


# Okay, Let's define our own questions before moving forward. Framing questions before deep diving into the analysis would help us to stick to the flow. We would not be wasting time in less important points, rather we would be focused in finding the answers one by one. This will make our analysis more structured as well.

# ## Insights to be derived

# The following questions apply for both summer and winter Olympics.
# 
# ### Country:
# * What are the top 5 countries which have won most medals?
# * What are top 5 countries which has sent more players to the Olympic games?
# * Did the top 5 countries won medals consistently for the last 5 Olympic seasons?
# * Does an increase in no. of players from a particular country increases the no. of medals for the country?
# * Which country has the best male to female ratio of players?
# * Are there any countries which have participated in every game in the most recent Olympic held?
# 
# ### City:
# * Which are the top 5 cities which hosted the most number of Olympic games?
# 
# ### Sport:
# * Are there any new sports introduced in the most recent summer Olympic held?
# * Which are the sports being played since the first summer Olympic season?
# 
# ### Player:
# * Who are the top 5 players that won most gold, silver and bronze medals? 
# * Are young players tend to win more medals than the elder ones?
# * Is BMI a factor influencing the player's ability to win a medal?

# In[ ]:


print("Number of Olympic Seasons Held :", df['Year'].nunique())
print("Number of Countries Participated :", df['NOC'].nunique())
print("Number of Players Participated :", df['Name'].nunique())
print("Number of Sports Conducted :", df['Sport'].nunique())
print("Number of Medals Won :\n", df['Medal'].dropna().value_counts())


# ### What are the top 5 countries which have won most gold, silver and bronze medals?

# Let's first have a subset of the original dataframe. This subset contains the observations where atleast a medal is won.

# In[ ]:


medals_won = df.dropna(subset = ['Medal'])


# In[ ]:


plt.rcParams['figure.figsize'] = (8,6)

medals_won['region'].value_counts().nlargest(5).plot(kind = 'bar', linewidth = 1, facecolor = 'seagreen', edgecolor = 'k')
plt.title('Top 5 Countries with most medals')
plt.xlabel('Country')
plt.ylabel('# Medals')
plt.show()


# ### What are the top 5 countries which sent more players?

# In[ ]:


df.groupby('region')['Name'].count().nlargest(5).plot(kind = 'bar', linewidth = 1, facecolor = 'seagreen', edgecolor = 'k')
plt.title('Top 5 Countries which sent most players')
plt.xlabel('Country')
plt.ylabel('# Players')
plt.show()


# ### Did the top 5 countries won medals consistently for the last 5 Olympic seasons?

# To check whether the top 5 countries which won the most medals were consistent across the last 5 seasons, let's create a subset with only the top 5 countries.

# In[ ]:


top5countries_medals = medals_won.loc[medals_won['region'].isin(['USA', 'UK', 'Russia', 'France', 'Germany']), :]


# In[ ]:


plt.rcParams['figure.figsize'] = (12,6)

pd.pivot_table(index = 'Year', columns = 'region', values = 'Medal', aggfunc = 'count',data = top5countries_medals).iloc[-5:, :].plot()
plt.title('Performance of Top 5 Countries Over Last 5 Seasons')
plt.xlabel('Country')
plt.ylabel('#Medals')
plt.show()


# ### Which country has the best male - female ratio?

# Let's create a pivot table to see the count of players by gender in every region.

# In[ ]:


mf_ratio = pd.pivot_table(index = 'region', columns = 'Sex', values = 'Name', data = df, aggfunc = 'nunique')
mf_ratio.head()


# In[ ]:


mf_ratio['MRatio'] = ((mf_ratio['M'] / mf_ratio['M'])  * 100).astype(int)
mf_ratio['FRatio'] = ((mf_ratio['F'] / mf_ratio['M'])  * 100).astype(int)
mf_ratio['Overall'] = round(mf_ratio['MRatio'] / mf_ratio['FRatio'], 2)


# In[ ]:


mf_ratio[(mf_ratio['Overall'] > 0.90) & (mf_ratio['Overall'] < 1.1)]


# These are the countries which have the M:F ratio almost equal to 1. Why are we looking for a value almost equal to 1?
# The value of 1 represents that the no. of males participated is equal to the no. of the females participated. As the value of 1 can't be obtained ideally, a buffer of -0.1 to 0.1 is chosen.
# Let's filter this with the countries which has sent more than 100 players as it makes more sense.

# In[ ]:


mf_ratio.loc[((mf_ratio['Overall'] > 0.90) & (mf_ratio['Overall'] < 1.1) & (mf_ratio['M'] + mf_ratio['F'] > 100))].sort_values(by = 'Overall', ascending = False)


# ### Are there any countries which have participated in every game in the most recent Olympic held?

# Let's create a subset with observations only for the recent season of the Olympics held .i.e. 2016.

# In[ ]:


recent_olympic = df.loc[df['Year'] == 2016, :]


# In[ ]:


recent_olympic['Sport'].nunique()


# There were 34 games held during this Olympic season. Now let's check which are the countries who have played 34 games during 2016. This would help us understand the countries which have played in every sport.

# In[ ]:


country_by_sport = pd.pivot_table(index = 'region', values = 'Sport', aggfunc = 'nunique', data = recent_olympic)


# In[ ]:


print(country_by_sport[country_by_sport['Sport'] == 34].index)


# Brazil is the only country which played in every sport in the 2016 Olympic season.

# ### Which are the top 5 cities which hosted the most number of Olympic games?

# In[ ]:


plt.rcParams['figure.figsize'] = (8,6)

df['City'].value_counts().nlargest(5).plot(kind = 'bar', linewidth = 1, facecolor = 'seagreen', edgecolor = 'k')
plt.title('Top 5 Cities Hosted Most Matches')
plt.xlabel('City')
plt.ylabel('# Matches')
plt.show()


# ### Are there any new sports introduced in the most recent summer Olympic held?

# In[ ]:


last_olympic = df.loc[df['Year'] == 2012, :]


# In[ ]:


last_olympic['Sport'].nunique()


# In[ ]:


recent_olympic['Sport'].nunique()


# This is an interesting question. The optimum way I could think of finding this is by using set analysis. Sets carry only the unique values by default. Let's create two sets, the one which has games held during the recent Olympics and the other with that held during the previous season.

# In[ ]:


last = set(last_olympic['Sport'])
recent = set(recent_olympic['Sport'])


# The difference operation of two sets would give out the values that are present in one set, but not in the other set. This is what we are looking for, right? Let's check it out.

# In[ ]:


print('Games Introduced in the recent Summer Olympics : ', recent.difference(last))


# ### Which are the sports being played since the first summer Olympic season?

# Let's create two sets as we did earlier - one for the last summer olympics and the other for the first season and then compute the intersection to check the games which are being played since the first Olympic season.

# In[ ]:


first_olympic = df.loc[df['Year'] == 1896, :]


# In[ ]:


first = set(first_olympic['Sport'])
recent = set(recent_olympic['Sport'])


# In[ ]:


print('Sports Played Since the First Summer Olympic Season', '\n')
print(recent.intersection(first))


# ### Who are the top 5 players that won most gold, silver and bronze medals?

# In[ ]:


plt.rcParams['figure.figsize'] = (8,6)

medals_won['Name'].value_counts().nlargest(5).plot(kind = 'bar', linewidth = 1, facecolor = 'seagreen', edgecolor = 'k')
plt.title('Top 5 Players with most medals')
plt.xlabel('Players')
plt.ylabel('# Medals')
plt.show()


# ### Are young players tend to win more medals than the elder ones?

# Let's create a new dataframe which has the total number of players and who have won the medals. This will help us to compute the percentage of players in every age group who have won medals.

# In[ ]:


medals_by_age = pd.DataFrame()


# In[ ]:


medals_by_age['Total Players'] = df['Age Group'].value_counts()
medals_by_age['Players Won Medals'] = medals_won['Age Group'].value_counts()
medals_by_age['Percent'] = round(medals_by_age['Players Won Medals'] / medals_by_age['Total Players'] * 100, 2)


# In[ ]:


medals_by_age.sort_index()


# The elder participants seem to won in less numbers, as expected. The increase in age probably would have resulted in less agility.

# ### Is BMI a factor influencing the player's ability to win a medal?

# Let's create a new dataframe like we did previosly. This time we will consider BMI Group as a primary feature and compute the total and percentage of players who have won medals.

# In[ ]:


medals_by_bmi = pd.DataFrame()


# In[ ]:


medals_by_bmi['Total Players'] = df['bmi group'].value_counts()
medals_by_bmi['Players Won Medals'] = medals_won['bmi group'].value_counts()
medals_by_bmi['Percent'] = round(medals_by_bmi['Players Won Medals'] / medals_by_bmi['Total Players'] * 100, 2)


# In[ ]:


medals_by_bmi


# There is no inference in particular. In Olympics, there are games which require the participants to either have more or less weight. 
# 
# For example, if we take sports like wresting this would be played by participants who weigh more. For sports like athletics, it is the reverse case. Given this fact, there is no underlying trend between the BMI of the player and the winning ability.

# This notebook is an initial version of the analysis. I'll update this kernel with more inferences and incorporate some advanced visualization and data wrangling techniques. Do watch out this space for interesting updates.
# 
# Please upvote and comment if you like this kernel. Thanks in advance!

# In[ ]:




