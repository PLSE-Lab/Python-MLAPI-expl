#!/usr/bin/env python
# coding: utf-8

# # Suicide Rates Overview from 1985 to 2016 

# ## Introduction 

# This project is about the suicide rate among different countries. I'll do some simple exploratory data analysis work using these data.Look forward to your good advice and suggestion!

# > **The columns meaning is below:**  
# $country:$ The country where the suicide happened.  
# $year:$ The year where the suicide happened.  
# $sex:$ The sex of the dead.  
# $age:$ The interval of age.  
# $suicide no:$ The count of the dead.  
# $population:$ The population of the country.  
# $suicides/100k pop:$ The percent of suicide in every 100000 population.  
# $country year:$ The combination of country and year.  
# $HDI for year:$ Human development index.  
# $gdp for year:$ The GDP of the year.  
# $gdp per capita:$ The GDP of the capita.  
# $generation:$ The generation which the dead belongs to. 

# ## Questions 

# > **Q1. The change of suicide rate by gender form 1985 to 2016.**  
# > **Q2. The change of suicide rate by age form 1985 to 2016.**  
# > **Q3. Analyze the four highest suicide number countrieas' data.**  
# > **Q4. Analyze the suicide rate by generation.**  
# > **Q5. The correlation beween numerical data.**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Data Understanding 

# In[ ]:


# load the data file
df = pd.read_csv("../input/master.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


# Check out the brief information of each column
df.info()


# In[ ]:


# Check out if there is any duplicated data
df.duplicated().sum()


# In[ ]:


# Check out the unique data of some columns
df.nunique()


# In[ ]:


df['age'].unique()


# In[ ]:


df['generation'].unique()


# In[ ]:


# Returns valid descriptive statistics for each column of data
df.describe()


# - Rename the columns `suicides/100k pop`, `country-year` and `HDI for year` to `suicides_100k_pop`, `country_year` and `HDI_for_year`.  
# - Delete "($)" from `gdp_for_year ($)` and `gdp_per_capita ($)`.  
# - Change the `sex` values: "male" to "M", "female" to "F", remove the "years" in `age`.  
# - Replace the "NaN" in `HDI for year` to 0.  
# - `gdp_for_year ($)` must be changed to type "int".

# ## Data Cleaning 

# In[ ]:


# Rename columns
df.rename(columns = {'suicides/100k pop': 'suicides_100k_pop', 'country-year': 'country_year', 'HDI for year': 'hdi_for_year'}, inplace=True)


# In[ ]:


df.rename(columns = {' gdp_for_year ($) ': 'gdp_for_year', 'gdp_per_capita ($)': 'gdp_per_capita'}, inplace=True)


# In[ ]:


# Change the sex values to simpler forms
df.loc[df['sex'] == 'male', 'sex'] = 'M'
df.loc[df['sex'] == 'female', 'sex'] = 'F'


# In[ ]:


# Change the age values to simpler forms
df['age'] = df.loc[df['age'].str.contains('years'), 'age'].apply(lambda x: x[:-6])


# In[ ]:


# Replace the "NaN" to 0
df.fillna(0, inplace=True)


# ## Exploratory data analysis

# ### Q1. The change of suicide rate by gender form 1985 to 2016.

# In[ ]:


data = df.copy()


# In[ ]:


subtable = data.pivot_table('suicides_no', index=['year'], columns=['sex'], aggfunc='sum', margins=True)
subtable['F_prop'] = subtable['F'] / subtable['All']*100
subtable['M_prop'] = subtable['M'] / subtable['All']*100


# In[ ]:


suicide_sex = pd.DataFrame(subtable)
suicide_sex.drop('All', inplace=True)


# In[ ]:


labels = suicide_sex.index
M = suicide_sex['M_prop']
F = suicide_sex['F_prop']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(16,8))
sns.set(style="white")
rects1 = ax.bar(x - width/2, M, width, label='Male', color = 'royalblue', alpha=.8)
rects2 = ax.bar(x + width/2, F, width, label='Female', color = 'goldenrod', alpha=.8)
ax1 = ax.twinx()
rects3 = ax1.plot(x, M, 'b')
rects4 = ax1.plot(x, F, 'orange')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Years', fontsize=14)
ax.set_ylabel('Suicide Percent', fontsize=14)
ax.set_title('Suicide Percent by Gender', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30)
ax.legend()
ax1.legend(loc=1)

plt.show()


# - **_In every year's total suicide number, the male percent is quite larger than female and increases slowly from 1985 to 2001._**

# In[ ]:


population = data.groupby('year')['suicides_no', 'population'].sum()
population['suicides_prop'] = population['suicides_no'] / population['population'] * 100


# In[ ]:


sns.set_style("white")
labels = population.index
y1 = population['suicides_no']
y2 = population['suicides_prop']
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(16,8))
ax.bar(x, y1,alpha=.8,color='royalblue')
ax.set_xlabel('Years', fontsize=14)
ax.set_ylabel('Total Suicide',fontsize=14)
ax.set_title('The Trend of Suicides Number Changes', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30)
ax1 = ax.twinx()
ax1.plot(x, y2, 'firebrick')
ax1.set_ylabel('Suicide Percent',fontsize=14)

plt.show()


# - **_The total number of suicide is increasing in the 32 years, but the percentage trend is not like this. The percentage is increasing drastically before 1995 and decreasing after 1995._**

# ### Q2. The change of suicide rate by age form 1985 to 2016. 

# In[ ]:


subtable_1 = data.pivot_table('suicides_no', index=['year'], columns=['age'], aggfunc='sum')


# In[ ]:


suicide_age = subtable_1.div(subtable_1.sum(axis=1), axis=0)
suicide_age.drop(2016, inplace=True)


# In[ ]:


suicide_age.head()


# In[ ]:


sns.set_style("whitegrid")
labels = suicide_age.index
y1 = suicide_age['15-24']
y2 = suicide_age['25-34']
y3 = suicide_age['35-54']
y4 = suicide_age['5-14']
y5 = suicide_age['55-74']
y6 = suicide_age['75+']
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(16,8))
ax.plot(x, y4, color='c', marker='*', label='5-14')
ax.plot(x, y1, color='r', marker='*', label='15-24')
ax.plot(x, y2, color='b', marker='*', label='25-34')
ax.plot(x, y3, color='g', marker='*', label='35-54')
ax.plot(x, y5, color='k', marker='*', label='55-74')
ax.plot(x, y6, color='m', marker='*', label='75+')
ax.legend()
ax.set_title("The Percentage of Suicide by Age", fontsize=16)
ax.set_xlabel("Years", fontsize=14)
ax.set_ylabel("Percent", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30)
plt.show()


# - **_The higheast rate of suicide is the adult whose age is between 35 and 54, and the number is increasing slowly from 1985 to 2001. The othors have no obvious changes._**

# ### Q3. Analyze the four highest suicide number countrieas' data.

# In[ ]:


country_names = list(pd.DataFrame(data.groupby('country')['suicides_no'].sum().sort_values()[-4:]).index)


# In[ ]:


suicide_country = data[data['country'].isin(country_names)]
suicide_country = suicide_country.pivot_table('suicides_no', index='year', columns='country', aggfunc='sum')


# In[ ]:


suicide_country.dropna(axis=0, how='any', inplace=True)


# In[ ]:


country_population = data[data['country'].isin(country_names)].pivot_table('population', index='year', columns='country', aggfunc='sum').dropna(axis=0, how='any')


# In[ ]:


suicide_country = suicide_country.div(country_population) * 100000


# In[ ]:


labels = suicide_country.index
F = suicide_country['France']
J = suicide_country['Japan']
R = suicide_country['Russian Federation']
U = suicide_country['United States']

x = np.arange(len(labels))  # the label locations
width = 0.2 # the width of the bars

fig, ax = plt.subplots(figsize=(16,8))
sns.set(style="white")
rects1 = ax.bar(x, F, width, label='France', color = 'tomato', alpha=.8)
rects2 = ax.bar(x + width, J, width, label='Japan', color = 'chocolate', alpha=.8)
rects3 = ax.bar(x + 2*width, R, width, label='Russian Federation', color = 'gold', alpha=.8)
rects4 = ax.bar(x + 3*width, U, width, label='United States', color = 'olive', alpha=.8)
ax1 = ax.twinx()
rects5 = ax1.plot(x, F, 'tomato')
rects6 = ax1.plot(x, J, 'chocolate')
rects7 = ax1.plot(x, R, 'gold')
rects8 = ax1.plot(x, U, 'olive')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Years', fontsize=14)
ax.set_ylabel('Suicide/10k Percent', fontsize=14)
ax.set_title('Suicide Percent by Country', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30)
ax.legend()
ax1.legend(loc=1)

plt.show()


# > - **_Russian Federation has the highest suicide rate especially from 1992 to 2007, Japan's suicide rate increases suddenly in 1997 and the rate keeps in a high level after 1997 even higher than France._**  
# > - **_After 2000 the suicide rate in United States increases slowly and the rate in France decreased slowly._**

# In[ ]:


suicide_country_age = data[data['country'].isin(country_names)].pivot_table('suicides_no', index='age', columns='country', aggfunc='sum')
suicide_country_age


# In[ ]:


labels = suicide_country_age.index
explode = (0, 0, 0.1, 0, 0, 0)
colors = ('gold', 'orange', 'olive', 'yellowgreen', 'palegreen', 'beige')
fig1, ax= plt.subplots(2, 2, figsize = (12,12))
# France
ax[0,0].pie(suicide_country_age['France'], explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax[0,0].set_title('France', fontsize=20)
ax[0,0].axis('equal')
# Japan
ax[0,1].pie(suicide_country_age['Japan'], explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax[0,1].set_title('Japan', fontsize=20)
ax[0,1].axis('equal')
# Russian Federation
ax[1,0].pie(suicide_country_age['Russian Federation'], explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax[1,0].set_title('Russian Federation', fontsize=20)
ax[1,0].axis('equal')
# United States
ax[1,1].pie(suicide_country_age['United States'], explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax[1,1].set_title('United States', fontsize=20)
ax[1,1].axis('equal')

plt.show()


# > **_The age between 36 and 54 has the highest suicide rate in the four countries, than is the age between 55 and 74._**

# ### Q4. Analyze the suicide rate by generation.

# In[ ]:


suicide_generation = data.pivot_table('suicides_no', index='generation', columns='sex', aggfunc='sum')
suicide_generation_population = data.pivot_table('population', index='generation', columns='sex', aggfunc='sum')


# In[ ]:


suicide_generation = suicide_generation.div(suicide_generation_population) * 100000
suicide_generation


# In[ ]:


labels = suicide_generation.index
M = suicide_generation['M']
F = suicide_generation['F']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(16,8))
sns.set(style="white")
rects1 = ax.bar(x - width/2, M, width, label='Male', color = 'skyblue', alpha=.8)
rects2 = ax.bar(x + width/2, F, width, label='Female', color = 'darkorange', alpha=.8)
ax.set_xlabel('Generation', fontsize=14)
ax.set_ylabel('Suicide Percent', fontsize=14)
ax.set_title('Suicide Percent by Generation', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show();


# > **_G.I. Generation has the highest suicide rate._**

# ### Q5. The correlation beween numerical data.

# In[ ]:


data.head()


# In[ ]:


# select numerical data
data_1 = pd.DataFrame(data.groupby('country_year')['suicides_no'].sum())
data_2 = pd.DataFrame(data.groupby('country_year')['population'].sum())
data_3 = pd.DataFrame(data.groupby('country_year')['gdp_per_capita'].mean())
data_4 = pd.DataFrame(data.groupby('country_year')['hdi_for_year'].mean())


# In[ ]:


# conbine data_1, _2, _3
suicide = pd.concat([data_1, data_2, data_3, data_4], axis=1, join='inner')
suicide['suicide_10k_pop'] = suicide['suicides_no'] / suicide['population'] * 100000
suicide.head()


# In[ ]:


sns.set(style="darkgrid")
sns.jointplot(y = "suicide_10k_pop", x = "gdp_per_capita", data=suicide, kind="reg", color="slateblue", space=0.5)
plt.title("The Scatter of Suicide/10k Pop and GDP/capita", fontsize=18)
plt.xlabel("GDP/capita", fontsize=15)
plt.ylabel("Suicide/10k Pop", fontsize=15)
plt.show();


# In[ ]:


sns.set(style="darkgrid")
sns.jointplot(y = "suicide_10k_pop", x = "hdi_for_year", data=suicide[suicide['hdi_for_year'] != 0], kind="reg", color="slateblue", space=0.5)
plt.title("The Scatter of Suicide/10k Pop and HDI/year", fontsize=18)
plt.xlabel("HDI/year", fontsize=15)
plt.ylabel("Suicide/10k Pop", fontsize=15)
plt.show();


# > - **_There is no correlation between Suicide/10k Pop and GDP/capita._**  
# > - **_There is weak correlation between Suicide/10k Pop and HDI/year._**

# In[ ]:




