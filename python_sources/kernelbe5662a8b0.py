# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#thenames = ['country', 'year', 'sex', 'age group', 'count of suicides', 'population', 'suicide rate', 'country-year' 'composite key', 'HDI for year', 'gdp_for_year', 'gdp_per_capita', 'generation' ]
data=pd.read_csv('../input/master.csv')
#print(data)
data=data.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo','population':'Population','suicides/100k pop':'Suicides100kPop','country-year':'CountryYear','HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYearMoney','gdp_per_capita ($)':'GdpPerCapitalMoney','generation':'Generation'})
min_year=min(data.Year)
max_year=max(data.Year)
print('Min Year :',min_year)
print('Max Year :',max_year)

#1985 min year,2016 max year.

data_country=data[(data['Year']==min_year)]

country_1985=data[(data['Year']==min_year)].Country.unique()
country_1985_male=[]
country_1985_female=[]

for country in country_1985:
    country_1985_male.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='male')]))
    country_1985_female.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='female')])) 
    
#We found the ratio of men and women who committed suicide in some countries in 1985 and we are now charting.

plt.figure(figsize=(10,10))
sns.barplot(y=country_1985,x=country_1985_male,color='red')
sns.barplot(y=country_1985,x=country_1985_female,color='yellow')
plt.ylabel('Countries')
plt.xlabel('Count Male vs Female')
plt.title('1985 Year Suicide Rate Gender')
plt.show()

#Very odd all the rates came on an equal level. So let's do max year.

data_country=data[(data['Year']==max_year)]

country_2016=data[(data['Year']==max_year)].Country.unique()
country_2016_male=[]
country_2016_female=[]

for country in country_2016:
    country_2016_male.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='male')]))
    country_2016_female.append(len(data_country[(data_country['Country']==country)&(data_country['Gender']=='female')])) 
    
#We found the ratio of men and women who committed suicide in some countries in 1985 and we are now charting.

plt.figure(figsize=(10,10))
sns.barplot(y=country_2016,x=country_2016_male,color='red')
sns.barplot(y=country_2016,x=country_2016_female,color='yellow')
plt.ylabel('Countries')
plt.xlabel('Count Male vs Female')
plt.title('2016 Year Suicide Rate Gender')
plt.show()
# Any results you write to the current directory are saved as output.