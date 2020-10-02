#!/usr/bin/env python
# coding: utf-8

# # Suicides Dataset
# For this dataset I will focus on three different points of view.
# - Differences over sex and age of suicides/100k pop.
# - Relationship between GDP and suicides over the years.
# - Comparison between poorest and richest countries.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Data Exploring
# First let's examine and clean the data.

# In[2]:


df = pd.read_csv('../input/master.csv')
df.sample(5)


# In[3]:


df.info()


# In[4]:


# Strip whitespace since some of the columns would become harder to name.
df.columns = df.columns.str.strip()

# Drop unnecessary columns for the analysis objectives.
df.drop(columns=['suicides_no', 'country-year', 'HDI for year', 'gdp_for_year ($)', 'generation'], inplace=True)

df.sample(5)


# In[5]:


# Rename columns for easier readability.
df.rename(columns={'suicides/100k pop':'suicides', 'gdp_per_capita ($)':'gpd'}, inplace=True)

df.head()


# # Data Analysis
# Now that our data is ready, let's perform the analysis previously mentioned.
# 
# ## Suicides by sex and age

# In[6]:


suicides_sex_age = df[['sex', 'age', 'suicides']].groupby(['sex', 'age']).mean()
suicides_sex_age


# In[7]:


# Reorder age index for visualization.
suicides_sex_age.reset_index(inplace=True)
suicides_sex_age['age'] = suicides_sex_age['age'].str.replace(' years', '')

age_sort = {'5-14': 0, '15-24': 1, '25-34': 2, '35-54': 3, '55-74': 4, '75+': 5}
suicides_sex_age['sort'] = suicides_sex_age['age'].map(age_sort)
suicides_sex_age.sort_values(by='sort', inplace=True)
suicides_sex_age.drop('sort', axis=1, inplace=True)

suicides_sex_age


# In[48]:


age_groups = suicides_sex_age['age'].unique()
male_suicides = suicides_sex_age[suicides_sex_age['sex'] == 'male']['suicides']
female_suicides = suicides_sex_age[suicides_sex_age['sex'] == 'female']['suicides']

plt.bar(age_groups, male_suicides, label='Male')
plt.bar(age_groups, female_suicides, label='Female')

plt.title('Average suicides across the world by sex')
plt.xlabel('Age group')
plt.ylabel('Suicides per 100k population')
plt.legend()
plt.show()


# Males commit suicide significantly higher than females, with and average difference of 15 suicides per 100k of population. Interestingly this difference becomes greater at older ages.

# ## Suicides vs GPD

# In[9]:


suicides_vs_gpd = df[['suicides', 'year', 'gpd']].groupby('year').mean()
suicides_vs_gpd.reset_index(inplace=True)

suicides_vs_gpd.head()


# In[10]:


fig, ax1 = plt.subplots()

# Plot the suicides over the years.
lns1 = ax1.plot(suicides_vs_gpd['year'], suicides_vs_gpd['suicides'], 'C0', label='Suicides')

# Create a shared axis for plotting on a different scale the GPD.
ax2 = ax1.twinx()
lns2 = ax2.plot(suicides_vs_gpd['year'], suicides_vs_gpd['gpd'], 'C1', label='GPD')

# Join both legends into the same box.
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=2)

# Set the labels.
ax1.set_ylabel('Suicides per 100k population')
ax2.set_ylabel('GDP per Capita')
ax1.set_xlabel('Years')

plt.tight_layout()
plt.show()


# As we would expect as the world got richer, the number of suicides went down. Something unexpected happened the year 2016, where the suicides per 100k of population were as higher as in 2002.

# ## Comparison: Rich vs Poor

# In[16]:


suicides_poor_rich = df[['year', 'country', 'gpd', 'suicides']]

# Sort the the countries by their average gpd over the years.
# Then get the list of the countries ordered.
countries_by_gpd = suicides_poor_rich.groupby('country').mean().sort_values('gpd', ascending=False).index


# In[17]:


# Get the top and bottom 5 countries of the list.
top_countries = countries_by_gpd[:5]
bot_countries = countries_by_gpd[-5:]

# Append them for the future filter.
countries_to_compare = top_countries.append(bot_countries)
countries_to_compare


# In[18]:


# Filter the rows that only are one of those countries.
suicides_poor_rich = suicides_poor_rich.loc[suicides_poor_rich['country'].isin(countries_to_compare)]
suicides_poor_rich.sample(5)


# In[19]:


# Create a filter for splitting those countries into two groups.
country_filter = {country:'TOP' for country in top_countries}
country_filter.update({country:'BOT' for country in bot_countries})

country_filter


# In[20]:


# Apply the filter.
suicides_poor_rich['country'] = suicides_poor_rich['country'].map(country_filter)
suicides_poor_rich.sample(5)


# In[52]:


# Simply, plot the results.
sns.lineplot(x='year', y='suicides', data=suicides_poor_rich, hue='country', ci=None)
plt.legend(labels=['BOT', 'TOP'])

plt.title('Comparison between top and bottom economies')
plt.xlabel('Year')
plt.ylabel('Suicides per 100k pop')
plt.show()


# As we can see, the poorest countries got even below the richest countries for almost two decades. That can be due to the fact that the life-expectancy is lower on those countries, and, as we saw ealier the huge part of the total suicides are commited at the lastests stages of life.

# # Conclusions
# We found some interesting insights about how suicides are correlated to age, sex and gpd. Also, we found some anomalies such as the suicides for older age groups, or the dramatic increase in suicides from 2015 to 2016. These last two will be interesnting to analyse in further projects.
# 
# I hope you liked the analysis.
