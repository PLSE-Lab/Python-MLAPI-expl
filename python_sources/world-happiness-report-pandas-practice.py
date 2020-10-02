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


# First, read the csv files.
data_2015 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")
data_2016 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")
data_2017 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")
data_2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")
data_2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")


# # 1-2. Merging the tables and completing region info
# - Take a copy of the file, drop the unused variables, add year variable, rearranging the variable names and their order. 

# In[ ]:


#data_2015


# In[ ]:


#data_2015.columns


# In[ ]:


data1_2015 = data_2015.copy()
data1_2015.drop(['Happiness Rank','Region','Standard Error','Economy (GDP per Capita)','Family','Dystopia Residual'], axis=1, inplace=True)
data1_2015.columns = ['Country','Happiness','Health','Freedom','Corruption','Generosity']
data1_2015['Year'] = '2015'
data1_2015 = data1_2015[['Year','Country','Happiness','Health','Freedom','Corruption','Generosity']]
data1_2015


# In[ ]:


#data_2016


# In[ ]:


#data_2016.columns


# In[ ]:


data1_2016 = data_2016.copy()
data1_2016.drop(['Region','Happiness Rank','Lower Confidence Interval', 'Upper Confidence Interval',
                 'Economy (GDP per Capita)', 'Family','Dystopia Residual'], axis=1, inplace=True)
data1_2016.columns = ['Country','Happiness','Health','Freedom','Corruption','Generosity']
data1_2016['Year'] = '2016'
data1_2016 = data1_2016[['Year','Country','Happiness','Health','Freedom','Corruption','Generosity']]
data1_2016


# In[ ]:


#data_2017


# In[ ]:


#data_2017.columns


# In[ ]:


data1_2017 = data_2017.copy()
data1_2017.drop(['Happiness.Rank','Whisker.high','Whisker.low', 'Economy..GDP.per.Capita.', 
                 'Family','Dystopia.Residual'], axis=1, inplace=True)
data1_2017.columns = ['Country','Happiness','Health','Freedom','Generosity','Corruption']
data1_2017['Year'] = '2017'
data1_2017 = data1_2017[['Year','Country','Happiness','Health','Freedom','Corruption','Generosity']]
data1_2017


# In[ ]:


# This code checks how many rows in country column match in two DFs. 
#data1_2017['Country'].isin(data_reg['Country']).value_counts()


# In[ ]:


#data_2018


# In[ ]:


#data_2018.columns


# In[ ]:


data1_2018 = data_2018.copy()
data1_2018.drop(['Overall rank','GDP per capita','Social support'], axis=1, inplace=True)
data1_2018.columns = ['Country','Happiness','Health','Freedom','Generosity','Corruption']
data1_2018['Year'] = '2018'
data1_2018 = data1_2018[['Year','Country','Happiness','Health','Freedom','Corruption','Generosity']]
data1_2018


# In[ ]:


#data_2019


# In[ ]:


#data_2019.columns


# In[ ]:


data1_2019 = data_2019.copy()
data1_2019.drop(['Overall rank','GDP per capita','Social support'], axis=1, inplace=True)
data1_2019.columns = ['Country','Happiness','Health','Freedom','Generosity','Corruption']
data1_2019['Year'] = '2019'
data1_2019 = data1_2019[['Year','Country','Happiness','Health','Freedom','Corruption','Generosity']]
data1_2019


# In[ ]:


# Merge 5 DFs vertically with concat method after making sure that all variable names and their order match. 
data = pd.concat([data1_2015,data1_2016,data1_2017,data1_2018,data1_2019], ignore_index=True)


# In[ ]:


# Check the country list for duplicate country names with different expressions.
sorted(list(data.Country.unique()))


# In[ ]:


# Replacing different names for the same countries for consistency.
print(data[data.Country=='Hong Kong S.A.R., China'])
print(data[data.Country=='Northern Cyprus'])
print(data[data.Country=='North Macedonia'])
print(data[data.Country=='Trinidad & Tobago'])
print(data[data.Country=='Taiwan Province of China'])
data.loc[385,'Country'] = 'Hong Kong'
data.loc[527,'Country'] = 'North Cyprus'
data.loc[689,'Country'] = 'North Cyprus'
data.loc[709,'Country'] = 'Macedonia'
data.loc[507,'Country'] = 'Trinidad and Tobago'
data.loc[664,'Country'] = 'Trinidad and Tobago'
data.loc[347,'Country'] = 'Taiwan'


# In[ ]:


# We merge the region info and main data frame with how = 'left' command for not losing any row. 
data_reg = data_2015.iloc[:,0:2] # Take the country and region info from 2015 to complete region info.
data = pd.merge(data, data_reg, on = 'Country', how = 'left')
data = data[['Year','Country','Region','Happiness','Health','Freedom','Corruption','Generosity']]
data


# # 3. Displaying number of NaN values
# - We have 18 NaN values in region since the country/region info of 2015 data frame that we used as reference for other tables do not completely match with the other data frames. 
# - We have 1 NaN value in corruption variable. 

# In[ ]:


data.isna().sum()


# In[ ]:


# List of countries with NaN values in the Region column.
data[data.Region.isna()].Country


# # 4. Filling NaN values
# - We have only 1 NaN value in Corruption, we will replace this with the mean of this column.
# - We cannot do a similar thing in Region, because it is not numeric. We can use region info of 2016 and internet research.

# In[ ]:


nan_region = data[data.Region.isna()].Country.unique() # Total list of countries that do not have region info.
list1 = []
for i in list(nan_region):
    if i in data1_2016.Country.values:
        list1.append(i)
print(list1)  # The list of countries whose region info can be taken from 2016 DF. 
print(data[data.Country=='Belize'])
print(data[data.Country=='Somalia'])
print(data[data.Country=='Namibia'])
print(data[data.Country=='South Sudan'])
print(data[data.Country=='Gambia'])
print(data[data.Country=='Puerto Rico'])
data.loc[209,'Region'] = 'Latin America and Caribbean' #Replace NaN values with region info for these countries. 
data.loc[364,'Region'] = 'Latin America and Caribbean'  
data.loc[518,'Region'] = 'Latin America and Caribbean'
data.loc[233,'Region'] = 'Sub-Saharan Africa'
data.loc[407,'Region'] = 'Sub-Saharan Africa'
data.loc[567,'Region'] = 'Sub-Saharan Africa'
data.loc[737,'Region'] = 'Sub-Saharan Africa'
data.loc[270,'Region'] = 'Sub-Saharan Africa'
data.loc[425,'Region'] = 'Sub-Saharan Africa'
data.loc[588,'Region'] = 'Sub-Saharan Africa'
data.loc[738,'Region'] = 'Sub-Saharan Africa'
data.loc[300,'Region'] = 'Sub-Saharan Africa'
data.loc[461,'Region'] = 'Sub-Saharan Africa'
data.loc[623,'Region'] = 'Sub-Saharan Africa'
data.loc[781,'Region'] = 'Sub-Saharan Africa'
data.loc[254,'Region'] = 'Sub-Saharan Africa'
data.loc[745,'Region'] = 'Sub-Saharan Africa'
data.loc[172,'Region'] = 'Latin America and Caribbean' 


# In[ ]:


# NaN value in Corruption column.
data[data.Corruption.isna()]


# In[ ]:


# Replace the NaN value in Corruption column with the mean.
data['Corruption'].fillna(data.Corruption.mean(), inplace=True)


# In[ ]:


# After dealing with NaN values we have no NaN values. 
data.isna().sum()


# # 5. General info and descriptive stats of the data

# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.describe().T


# # 6. Regions in the data frame

# In[ ]:


# Regions in the data frame.
data['Region'].unique()


# In[ ]:


# Number of regions in the data frame. 
len(data['Region'].unique())


# # 7. Relation between Happiness and other variables

# In[ ]:


# General correlation values between all the variables
correlation = data.corr()
correlation


# In[ ]:


# Correlation values bigger than 0.5
correlation.abs()[correlation.abs()>0.5]
# We can make an initial inference that there is a positive relation between happiness and health and freedom. 
# The correlation between happiness and healt is highest.


# In[ ]:


# Correlation values by years
correlation_yr = data.groupby('Year').corr()
correlation_yr


# In[ ]:


# Correlation values bigger than 0.5
correlation_yr.abs()[correlation_yr.abs()>0.5]
# We can make an initial inference that there is not any significant change in the positive relation between happiness and health 
# and freedom over the years. 


# # 8. Number of countries by region

# In[ ]:


# Use nunique() to count unique countries since there are repeating countries for each year. 
data.groupby('Region')['Country'].nunique()


# # 9. Top 3 happiest and unhappiest countries

# In[ ]:


# Calculate the mean of happiness score grouped by countries
score_mean = data.groupby('Country')['Happiness'].mean()


# In[ ]:


# The top 3 happiest countries
score_mean.sort_values(ascending=False).head(3)


# In[ ]:


# The top 3 unhappiest countries
score_mean.sort_values(ascending=True).head(3)


# # 10. Top 3 countries with the best and worst corruption scores

# In[ ]:


# Calculate the mean of corruption grouped by countries
corruption_mean = data.groupby('Country')['Corruption'].mean()


# In[ ]:


# Top 3 countries with the best scores for corruption
corruption_mean.sort_values(ascending=False).head(3)


# In[ ]:


# Top 3 countries with the worst scores for corruption
corruption_mean.sort_values(ascending=True).head(3)


# # 11. The highest and lowest freedom scores by region

# In[ ]:


freedom_mean = data.groupby('Region')['Freedom'].mean()


# In[ ]:


# The region with highest freedom score
freedom_mean.sort_values(ascending=False).head(1)


# In[ ]:


# The region with lowest freedom score
freedom_mean.sort_values(ascending=False).tail(1)


# # 12. The region with lowest healt score

# In[ ]:


# The region with lowest health score
health_mean = data.groupby('Region')['Health'].mean()
health_mean.sort_values(ascending=False).tail(1)


# # 13. The means of happiness, freedom and corruption by regions

# In[ ]:


data.groupby('Region').aggregate({'Happiness':'mean','Freedom':'mean', 'Corruption':'mean'}) 

