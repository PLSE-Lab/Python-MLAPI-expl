#!/usr/bin/env python
# coding: utf-8

# **Beginner: Italy Migration Data Visualization (Seaborn and Matplotlib)**

# **Table of Contents**
# 
# 1. Info on this dataset
# 2. Exploring the dataset and cleaning the data
# 3. Visualizing the top 10 **most** represented nationalities of immigrants to Italy using Seaborn
# 4. Visualizing the top 5 **least** represented nationalities of immigrants to Italy using Seaborn
# 5. Visualizing the trend of Argentinian immigration to Italy between 1990 and 2013 using Matplotlib 
# 6. Visualizing a comparison of the trends of Romanian and Moroccan immigration to Italy between 1990 and 2013 using Matplotlib 
# 7. Visualizing the frequency distribution of immigrants from 179 nationalities to Italy between 1990 and 2013 using Matplotlib 
# 8. Visualizing the frequency distribution of immigrants from Germany, Tunisia, and Nigeria to Italy between 1990 and 2013 using Matplotlib 
# 9. Visualizing the number of Egyptian immigrants to Italy for each year between 1990 and 2013 using Matplotlib
# 10. Visualizing the percentages of immigrants to Italy from each continent between 1990 and 2013
# 11. Visualizing the trends of immigration to Italy between 1990 and 2013 using Matplotlib 
# 12. Visualizing Argentinian and Brazilian immigration to Italy between 1990 and 2013 using Matplotlib

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **1. Introduction to the dataset**
# 
# The dataset contains annual data on the flows of international migrants as recorded by Italy. The data presents both inflows and outflows according to the place of birth, citizenship or place of previous / next residence both for foreigners and nationals.
# 
# This dataset is part of a compilation of data on international migration flows which is useful for analytical purposes, but also serves to raise awareness about the problems of comparability among available statistics. Countries collecting and publishing data on the flows of international migrants use different definitions to identify migrants and use different concepts to determine their origin and destination.

# In[ ]:


import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
print('setup complete') 


# **2. Exploring the Dataset and cleaning the data**
# 
# Once we have imported the necessary libraries, let's read the dataset. 
# 
# Note that for the purposes of this kernel we will only use the sheet named "Italy by Citizenship".

# In[ ]:


italy_imm_filepath = '/kaggle/input/italy-immigration-data-by-the-un/Italy.xlsx'
italy_imm_data = pd.read_excel(italy_imm_filepath, 
                              sheet_name='Italy by Citizenship', 
                              skiprows = range(20), 
                              skipfooter = 2)
italy_imm_data.sample(40) 


# One thing we can immediately notice by looking at the dataset is that quite a large number of cells are marked with '..' 
# It might either indicate that no data is available for that specific year or that the count is 0. 
# 
# Let's assume that the count is '0', and therefore transform the dataset accordingly.

# In[ ]:


italy_imm_data.replace(['..'],0, inplace = True)


# As we can see the dataset contains data on both immigrants and emigrants. Let's only have a look at the immigrants category.

# In[ ]:


italy_imm_data = italy_imm_data[italy_imm_data.Type == 'Immigrants'] 
italy_imm_data 


# Let's subsitute the name of the column 'OdName' with what it really represents: 'Country'.

# In[ ]:


italy_imm_data.rename(columns={'OdName':'Country'}, inplace=True) 


# If we look at the dataset more closely, we can see that it also contain data on Italian citizens immigrating back to Italy. 
# Let's drop it and exclusively look at other nationalities. 

# In[ ]:


italy_imm_data = italy_imm_data[italy_imm_data.Country != 'Italy'] 
italy_imm_data 


# Let's look at the dataset again. 
# 
# Three columns seem to be not useful for our analysis: 'AREA', 'REG', 'DEV'. Furthermore if we try to sum all the numeric values contained in the columns representing the years, we might end up in trouble. 
# 
# Let's drop them. 

# In[ ]:


italy_imm_data = italy_imm_data.drop(['AREA', 'REG', 'DEV'], axis=1) 


# As we can see, for each country we have the number of immigrants per year from 1980 to 2013. 
# 
# Let's add a column called 'Total'.

# In[ ]:


italy_imm_data['Total'] = italy_imm_data.sum(axis=1) 


# Let's check that the new column has been created.

# In[ ]:


italy_imm_data.columns


# **3. Visualizing the top 10 most represented nationalities of immigrants to Italy using Seaborn**
# 
# Let's have a look at which are the top 10 most represented nationalities of immigrants to Italy using a Seaborn barplot. 
# 
# First thing first, we need to group the countries by their total.

# In[ ]:


df1 = italy_imm_data.groupby(["Total"]) 


# Now, let's sort the values.

# In[ ]:


df2= df1.apply(lambda x: x.sort_values(['Country']))


# Let's have a look at the grouped and sorted values.

# In[ ]:


df2 


# Now we are ready to visualize the data. 

# In[ ]:


plt.figure(figsize=(12,8)) 
sns.set(style="white") 
sns.barplot(x=df2.Total.tail(10), y=df2.Country.tail(10), 
            palette="BuGn_r", edgecolor=".2");


# As we can see, Romania is by far the most represented nationality among immigrants to Italy between 1980 and 2013.

# **4. Visualizing the top 5 least represented nationalities of immigrants to Italy using Seaborn**
# 
# We have seen the top 10 most represented nationalities, but which are the top 5 least represented nationalities?
# 
# Let's have a look below using Seaborn.

# In[ ]:


plt.figure(figsize=(12,8))
sns.set(style="white") 
splot = sns.barplot(y=df2.Total.head(5), x=df2.Country.head(5), 
            palette="cubehelix", edgecolor=".2");

for p in splot.patches:
  splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'baseline', xytext = (0, 10), textcoords = 'offset points')


# Citizens of Nauru seems to be not very represented among immigrants to Italy between 1980 to 2013. 

# **5. Visualizing the trend of Argentinian immigration to Italy between 1990 and 2013 using Matplotlib**
# 
# It's now time to look at the data from a specific country: let's pick Argentina. 
# 
# To do this, it will be convenient to set the column 'Country' to index of the dataset. 

# In[ ]:


italy_imm_data.set_index('Country', inplace=True) 


# If we have a quick look at the dataset again, the first thing we would notice is that there is basically no data available between 1980 and 1990. 

# In[ ]:


italy_imm_data.head(10) 


# Here we go. 
# 
# Let's act accordingly and let's first convert the column names into strings to avoid confusion.
# 
# Then let's create the range 'years' (1990 to 2014) which will prove useful for plotting the data using Matplotlib.

# In[ ]:


italy_imm_data.columns = list(map(str, italy_imm_data.columns))
years = list(map(str, range(1990, 2014)))

Great! We can plot the data now using Matplotlib.
# In[ ]:


argentina = italy_imm_data.loc['Argentina', years]
plt.style.use(['fivethirtyeight'])

argentina.index = argentina.index.map(int)
argentina = argentina.astype(int)

fig = plt.figure(figsize=(13, 8))
ax = fig.add_axes([1,1,1,1])
argentina.plot(kind='line', ax=ax)
plt.text(1997.8, 5590, 'What happened at this time?')

ax.set_title('Immigration from Argentina')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')

plt.show()


# What we can immediately notice by visualizing the data is a major spike in Argentinian immigration to Italy around 2003/2004. This must be consequential to the crisis which has affected Argentina between 1998 and 2002.

# **6. Visualizing a comparison of the trends of Romanian and Moroccan immigration to Italy between 1990 and 2013 using Matplotlib**
# 
# How does immigration to Italy of two differnent nationalities compare? Let's have a look at the Romanian and Moroccan immigration between 1990 and 2013. 

# In[ ]:


romania_morocco = italy_imm_data.loc[['Romania', 'Morocco'], years]
romania_morocco.head()


# We need to transpose the data or we will not be able to plot properly.

# In[ ]:


romania_morocco = romania_morocco.T
romania_morocco.head()


# Let's plot it using Matplotlib.

# In[ ]:


romania_morocco.index = romania_morocco.index.map(int)
romania_morocco = romania_morocco.astype(int)

fig = plt.figure(figsize=(13, 8))
ax = fig.add_axes([1,1,1,1])
romania_morocco.plot(kind='line', ax=ax)

ax.set_title('Immigrants from Romania and Morocco')
ax.set_ylabel('Number of immigrants')
ax.set_xlabel('Years')

plt.show()


# Massive difference between the two nationalities. Six-times more Romanians than Moroccans have moved to Italy in 2007/2008.

# **7. Visualizing the frequency distribution of immigrants from 179 nationalities to Italy between 1990 and 2013 using Matplotlib**
# 
# If we look at all the countries in our dataset, what is the frequency distribution for the year 2013? 
# 
# Running the code below will tell us that, for example, in 2013:
# 
# * 179 nationalities have between 0 and 5822.7 immigrants
# * 9 nationalities have between 11645.4 and 17468.1 immigrants, and so forth.. 

# In[ ]:


count, bin_edges = np.histogram(italy_imm_data['2013'])

print(count)
print(bin_edges)


# Let's plot this data with a histogram using Matplotlib.

# In[ ]:


fig = plt.figure(figsize=(13,8))
ax = fig.add_axes([1,1,1,1])
italy_imm_data['2013'].plot(kind='hist', ax=ax)

ax.set_title('Histogram of Immigration from 179 Countries to Italy in 2013') 
ax.set_ylabel('Number of Countries') 
ax.set_xlabel('Number of Immigrants') 

plt.show()


# **8. Visualizing the frequency distribution of immigrants from Germany, Tunisia, and Nigeria to Italy between 1990 and 2013 using Matplotlib**
# 
# Let's repeat the exercise - this time looking at the frequency distribution of German, Tunisian, and Nigerian immigrants to Italy over the period between 1990 and 2013.

# In[ ]:


italy_imm_data.loc[['Germany', 'Tunisia', 'Nigeria'], years].T.columns.tolist()


# In[ ]:


italy_imm_data_t = italy_imm_data.loc[['Germany', 'Tunisia', 'Nigeria'], years].T

fig = plt.figure(figsize=(13,8))
ax = fig.add_axes([1,1,1,1])
italy_imm_data_t.plot(kind='hist', ax=ax)

ax.set_title('Immigration from Germany, Tunisia, and Nigeria from 1990 - 2013')
ax.set_ylabel('Number of Years')
ax.set_xlabel('Number of Immigrants') 

plt.show()


# The numbers for Germany are quite interesting. 

# **9. Visualizing the number of Egyptian immigrants to Italy for each year between 1990 and 2013 using Matplotlib**
# 
# Let's have a look at how many Egyptians have moved to Italy each year from 1990 to 2013. 

# In[ ]:


egypt = italy_imm_data.loc['Egypt', years]
egypt = egypt.astype(int)

fig = plt.figure(figsize=(13,8))
ax = fig.add_axes([1,1,1,1])
egypt.plot(kind='bar', ax=ax)

ax.set_xlabel('Year') 
ax.set_ylabel('Number of immigrants') 
ax.set_title('Egyptian immigrants to Italy from 1990 to 2013')

plt.show()


# What happened in 2001? Did someone forget to report the data for that year? 

# **10. Visualizing the percentages of immigrants to Italy from each continent between 1990 and 2013**
# 
# Which continents are the most represented among immigrants to Italy?
# 
# Let's have a look.

# In[ ]:


continents = italy_imm_data.groupby('AreaName', axis=0).sum()
print(type(italy_imm_data.groupby('AreaName', axis=0)))
continents.head()


# In[ ]:


colors_list = ['green', 'red', 'yellow', 'blue', 'orange', 'black']
explode_list = [0.1, 0.1, 0, 0.1, 0.1, 0] 

continents['Total'].plot(kind='pie',
                           figsize=(15, 8),
                           autopct='%1.1f%%', 
                           startangle=90,    
                           shadow=True,       
                           labels=None,         
                           pctdistance=1.14,     
                           colors=colors_list,  
                           explode=explode_list 
                           )

plt.title('Immigration to Italy by Continent [1990 - 2013]', y=1.14) 
plt.axis('equal') 

plt.legend(labels=continents.index, loc='upper left') 

plt.show()


# Unsurprisingly, Europe leads the way, followed by Asia and Africa. Oceania is last with only 0.1%.

# **11. Visualizing the trends of immigration to Italy between 1990 and 2013 using Matplotlib (scatterplots and bubble plots)**
# 
# Let's have a look at how immigration to Italy trended between 1990 and 2013.

# In[ ]:


tot = pd.DataFrame(italy_imm_data[years].sum(axis=0))
tot.index = map(int, tot.index)
tot.reset_index(inplace = True)
tot.columns = ['year', 'total']
tot


# A scatterplot using Matplotlib will help us with this. 

# In[ ]:


fig = plt.figure(figsize=(13,8))
ax = fig.add_axes([1,1,1,1])
tot.plot(kind='scatter', x='year', y='total', ax=ax)

ax.set_title('Total Immigration to Italy from 1990 - 2013')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Immigrants')

plt.show()


# There indeed seems to be an upward trend. 
# 
# Let's perform a linear regression. 

# In[ ]:


x = tot['year']      
y = tot['total']   
fit = np.polyfit(x, y, deg=1)
fit


# Let's plot the line. 

# In[ ]:


fig = plt.figure(figsize=(13,8))
ax = fig.add_axes([1,1,1,1])
tot.plot(kind='scatter', x='year', y='total', ax=ax)

ax.set_title('Total Immigration to Italy from 1990 - 2013')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Immigrants')

ax.plot(x, fit[0] * x + fit[1], color='red')


# Following this trend perhaps we can predict how many people moved to Italy in 2014, 2015, etc.

# **12. Visualizing Argentinian and Brazilian immigration to Italy between 1990 and 2013 using Matplotlib**
# 
# By plotting a bubble plot we should be able to see the immigration trends for both coutries during the in scope period. 
# 
# To do so, we first need to create a new dataframe with the data for Argentina and Brazil.

# In[ ]:


italy_imm_data_t = italy_imm_data[years].T
italy_imm_data_t.index = map(int, italy_imm_data_t.index)
italy_imm_data_t.index.name = 'Year'
italy_imm_data_t.reset_index(inplace=True)
italy_imm_data_t


# Let's normalize the weights now by using feature scaling so that all values are between 0 and 1. 

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scale_bra = MinMaxScaler()
scale_arg = MinMaxScaler()
norm_brazil = scale_bra.fit_transform(italy_imm_data_t['Brazil'].values.reshape(-1, 1))
norm_arg = scale_arg.fit_transform(italy_imm_data_t['Argentina'].values.reshape(-1, 1))


# Let's plot it. 

# In[ ]:


italy_imm_data_t['weight_arg'] = norm_arg
italy_imm_data_t['weight_brazil'] = norm_brazil

fig = plt.figure(figsize=(13,9))
ax = fig.add_axes([1,1,1,1])


italy_imm_data_t.plot(kind='scatter', x='Year', y='Brazil',
            alpha=0.5,                  # transparency
            s=norm_brazil * 2000 + 10,  # pass in weights 
            ax=ax)


italy_imm_data_t.plot(kind='scatter', x='Year', y='Argentina',
            alpha=0.5,
            color="blue",
            s=norm_arg * 2000 + 10,
            ax=ax)

ax.set_ylabel('Number of Immigrants')
ax.set_title('Immigration from Brazil and Argentina from 1990 - 2013')
ax.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='x-large')

plt.show()


# How would you interpret this graph? We have already analyzed Argentinian immigration to Italy and tried to explain the spike in the trend by looking at the 1998-2002 crisis. However, what would justify the spike in Brazilian immigration between 2005 and 2010?
