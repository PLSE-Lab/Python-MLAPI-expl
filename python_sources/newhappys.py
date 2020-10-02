#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


data_2015 = pd.read_csv('/kaggle/input/world-happiness/2017.csv')
data_2016 = pd.read_csv('/kaggle/input/world-happiness/2019.csv')
data_2017 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')
data_2018 = pd.read_csv('/kaggle/input/world-happiness/2016.csv')
data_2019 = pd.read_csv('/kaggle/input/world-happiness/2018.csv')


# In[ ]:


data_2015['Year'] = 2015
data_2016['Year'] = 2016
data_2017['Year'] = 2017
data_2018['Year'] = 2018
data_2019['Year'] = 2019


# In[ ]:


data_2015.info()


# In[ ]:


data_2016.info()


# In[ ]:


data_2017.info()


# In[ ]:


data_2018.info()


# In[ ]:


data_2019.info()


# In[ ]:


# We created a dictionary based on 2017-2018 for non-Country and Region data.

Country_Region_dich = {}

for i in data_2017.index:
    Country_Region_dich[data_2017.loc[i,'Country']] = data_2017.loc[i,'Region']


for i in data_2018.index:
    Country_Region_dich[data_2018.loc[i,'Country']] = data_2018.loc[i,'Region']


# # 2015

# In[ ]:


data_2015.columns


# In[ ]:


# The names of the columns have been changed

data_2015.rename(columns={"Country": "Country", 
                          'Happiness.Rank':'Happiness.Rank',
                          'Happiness.Score': "Happiness.score",
                          'Health..Life.Expectancy.': "Health" ,
                          "Freedom" : "Freedom",
                          'Trust..Government.Corruption.':"Corruption",
                          "Generosity" : "Generosity",
                          }, inplace=True)


# In[ ]:


# unwanted columns names deleted
data_2015 = data_2015.drop(columns = ['Whisker.high','Whisker.low','Economy..GDP.per.Capita.','Family','Dystopia.Residual'])


# In[ ]:


# Country check from Country Region_dict

for i in data_2015['Country']:    
    if i not in Country_Region_dich.keys():
        print(i)


# In[ ]:


# incorrect data changed

data_2015.Country.replace('Taiwan Province of China','Taiwan',inplace=True)
data_2015.Country.replace('Hong Kong S.A.R., China','Hong Kong',inplace=True)
data_2015['Region'] = np.nan    #  Region columns added


# In[ ]:


# Region information for 2015 found appropriate Region data from Country Region_dict

for i in data_2015.Country:
    data_2015.loc[    data_2015.Country == i  ,   'Region' ]     = Country_Region_dich[i]


# # 2016

# In[ ]:


# data_2016


# In[ ]:


data_2016.columns


# In[ ]:


# The names of the columns have been changed
data_2016.rename(columns={"Country or region": "Country",
                          "Region": "Region",
                          'Overall rank':'Happiness.Rank',
                          "Score": "Happiness.score",
                          "Healthy life expectancy": "Health",
                          "Freedom to make life choices" : "Freedom",
                          "Perceptions of corruption":"Corruption",
                          "Generosity" : "Generosity",
                          }, inplace=True)


# In[ ]:


#unwanted columns names deleted
data_2016 = data_2016.drop(columns = ['GDP per capita','Social support',])


# In[ ]:


# Country check from Country Region_dict

for i in data_2016['Country']:    # Country kontrolu yaptik
    if i not in Country_Region_dich.keys():
        print(i)


# In[ ]:


# incorrect data changed
Country_Region_dich['Trinidad & Tobago'] = 'Latin America and Caribbean'
Country_Region_dich['Northern Cyprus'] = 'Middle East and Northern Africa'
Country_Region_dich['North Macedonia'] = 'Central and Eastern Europe'
Country_Region_dich['Gambia'] = 'Sub-Saharan Africa'


# In[ ]:


data_2016['Region'] = np.nan


# In[ ]:


# data_2016.Country.replace('Taiwan Province of China','Taiwan',inplace=True)
# data_2016.Country.replace('Hong Kong S.A.R., China','Hong Kong',inplace=True)


# In[ ]:


# Region information for 2015 found appropriate Region data from Country Region_dict

for i in data_2016.Country:
    data_2016.loc[    data_2016.Country == i  ,   'Region' ]     = Country_Region_dich[i]


# # 2017

# In[ ]:


# data_2017


# In[ ]:


data_2017.columns


# In[ ]:


# The names of the columns have been changed
data_2017.rename(columns={"Country": "Country", 
                          "Region": "Region",
                          'Happiness Rank':'Happiness.Rank',
                          "Happiness Score": "Happiness.score",
                          "Health (Life Expectancy)": "Health",
                          "Freedom" : "Freedom",
                          "Trust (Government Corruption)":"Corruption",
                          "Generosity" : "Generosity",
                          }, inplace=True)


# In[ ]:


#unwanted columns names deleted
data_2017 = data_2017.drop(columns = ['Standard Error','Economy (GDP per Capita)','Family','Dystopia Residual'])


# In[ ]:


# Country check from Country Region_dict

for i in data_2017['Country']:    
    if i not in Country_Region_dich.keys():
        print(i)


# # 2018

# In[ ]:


# data_2018


# In[ ]:


data_2018.columns


# In[ ]:


# The names of the columns have been changed
data_2018.rename(columns={"Country": "Country",
                          "Region": "Region",
                          'Happiness Rank':'Happiness.Rank',
                          "Happiness Score": "Happiness.score",
                          "Health (Life Expectancy)": "Health",
                          "Freedom" : "Freedom",
                          "Trust (Government Corruption)":"Corruption",
                          "Generosity" : "Generosity",
                          }, inplace=True)


# In[ ]:


#unwanted columns names deleted
data_2018 = data_2018.drop(columns = ['Lower Confidence Interval','Upper Confidence Interval','Economy (GDP per Capita)','Family','Dystopia Residual'])


# In[ ]:


# Country check from Country Region_dict

for i in data_2018['Country']:    
    if i not in Country_Region_dich.keys():
        print(i)


# # 2019

# In[ ]:


# data_2019


# In[ ]:


data_2019.columns


# In[ ]:


# The names of the columns have been changed
data_2019.rename(columns={"Country or region": "Country",
                          "Region": "Region",
                          'Overall rank':'Happiness.Rank',
                          "Score": "Happiness.score",
                          "Healthy life expectancy": "Health",
                          "Freedom to make life choices" : "Freedom",
                          "Perceptions of corruption":"Corruption",
                          "Generosity" : "Generosity",
                          }, inplace=True)


# In[ ]:


#unwanted columns names deleted
data_2019 = data_2019.drop(columns = ['GDP per capita','Social support',])


# In[ ]:


data_2019['Region'] = np.nan


# In[ ]:


# Country check from Country Region_dict

for i in data_2019['Country']:    # Country kontrolu yaptik
    if i not in Country_Region_dich.keys():
        print(i)


# In[ ]:


# Region information for 2015 found appropriate Region data from Country Region_dict

for i in data_2019.Country:
    data_2019.loc[    data_2019.Country == i  ,   'Region' ]     = Country_Region_dich[i]


# In[ ]:


# data_2019


# # The order of variables was made

# In[ ]:


data_2015.columns


# In[ ]:


data_2015 = data_2015[['Country', 'Region', 'Happiness.Rank', 'Happiness.score', 'Health',
       'Freedom', 'Corruption', 'Generosity', 'Year']]


# In[ ]:


data_2016 = data_2016[['Country', 'Region', 'Happiness.Rank', 'Happiness.score', 'Health',
       'Freedom', 'Corruption', 'Generosity', 'Year']]


# In[ ]:


data_2017 = data_2017[['Country', 'Region', 'Happiness.Rank', 'Happiness.score', 'Health',
       'Freedom', 'Corruption', 'Generosity', 'Year']]


# In[ ]:


data_2018 = data_2018[['Country', 'Region', 'Happiness.Rank', 'Happiness.score', 'Health',
       'Freedom', 'Corruption', 'Generosity', 'Year']]


# In[ ]:


data_2019 = data_2019[['Country', 'Region', 'Happiness.Rank', 'Happiness.score', 'Health',
       'Freedom', 'Corruption', 'Generosity', 'Year']]


# In[ ]:


# data_2015.columns == data_2019.columns


# # concat done

# In[ ]:


data = pd.concat([data_2015,data_2016,data_2017,data_2018,data_2019], ignore_index=True)


# *** operations to be performed on data **
# 1. Combine these 5 datasets in one df to be "Country, Region, (happiness.score = score), Health, Freedom, Corruption, Generosity" columns based on countries. You can discard other variables. (Note: You can create additional columns if necessary while doing merge operations.)
# 2. Some datasets do not have the region information of the countries, search these countries and add them to the missing places in the region column.
# 3. Count the NaN values.
# 4. Fill NaN values with the average of that variable.
# 5. Get the information and describe information on the data.
# 6. Calculate how many different regions there are.
# 7. Which is the variable that affects happiness most. Check if there is a difference by years.
# 8. In which region (Region) there are few countries.
# 9. Find the happiest and the most unhappy 3 countries by taking the average of 5 years of HAPPINESS.
# 10. Find the best and worst countries by taking the average of 5 years CORRUPTION.
# 11.  Looking for 5 years, find the highest and the lowest average REGION.
# 12. Find out which region is the most unhealthy.
# 13. Average the variables of Happiness, Freedom and Corruption by grouping the countries according to their regions.
# 
# 

# # Get the information and describe information on the data.

# In[ ]:


data.info()


# In[ ]:


data.describe().T


# # Count the NaN values.

# In[ ]:


data.isnull().sum()


# # Fill NaN values with the average of that variable.

# In[ ]:


data = data.fillna(data.mean())   # eksik verilere bulundugu sutunun ort atadik


# In[ ]:


data.isnull().sum()


# # Calculate how many different regions there are.

# In[ ]:


set(data.Region)


# In[ ]:


len(set(data.Region))


# # Which is the variable that affects happiness most. Check if there is a difference by years.

# In[ ]:


data.corr()


# # How many countries are available in which region.

# In[ ]:


data.groupby('Region')['Country'].count()


# # Find the happiest and the most unhappy 3 countries by taking the average of 5 years of HAPPINESS.

# In[ ]:


data.groupby("Country")["Happiness.score"].mean().sort_values(ascending = False).head(3)


# In[ ]:


data.groupby("Country")["Happiness.score"].mean().sort_values(ascending = False).tail(3)


# # Find the best and worst countries by taking the average of 5 years CORRUPTION.

# In[ ]:


data.groupby("Country")["Corruption"].mean().sort_values()


# In[ ]:


data.groupby("Country")["Corruption"].mean().sort_values().tail(1)


# In[ ]:


data.groupby("Country")["Corruption"].mean().sort_values().head(1)


# # Based on 5 years, find the highest and lowest region of freedom average.

# In[ ]:


data.groupby("Region")["Freedom"].mean().sort_values(ascending = False) 


# In[ ]:


data.groupby("Region")["Freedom"].mean().sort_values(ascending = False).head(1) 


# In[ ]:


data.groupby("Region")["Freedom"].mean().sort_values(ascending = False).tail(1) 


# # Find out which region is the most unhealthy.

# In[ ]:


data.groupby("Region")["Health"].mean().sort_values().head(1)


# # Group the countries according to their regions and take the average of Happiness, Freedom and Corruption variables.

# In[ ]:


data.groupby('Region')['Happiness.score','Freedom','Corruption'].mean()

