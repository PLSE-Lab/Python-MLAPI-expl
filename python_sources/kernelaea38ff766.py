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


raw_data=pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
raw_data


# he Human Development Index (HDI) is a composite statistic of life expectancy, education, and per capita income indicators, which are used to rank countries into four tiers of human development.WE MISS LOTS OF VALUES SO THATS WHY WE SHOULD DROP HDI COLUMN AND ADD NEW FEATURES SUCH AS REGION

# In[ ]:


raw_data.info()


# In[ ]:


null_values=raw_data[raw_data['HDI for year'].isna()==False]
null=null_values.year
import collections
collections.Counter(null)


# In[ ]:





# In[ ]:





# In[ ]:





# ****MY GOAL IS TO BUILD ML MODEL,******WHICH WILL PREDICT SUICIDE RATE,SPECIFIC TO CERTAIN CHARACTERISTICS OF CERTAIN COUNTRY.
# LETS TAKE A LOOK TO THE COUNTRIES
# 

# In[ ]:


countries=raw_data['country-year'].unique()
pd.options.display.max_columns=None
pd.options.display.max_rows=50
countries=pd.DataFrame(countries)
display(countries)


# AS YOU SEE HERE,THERE ARE ALMOST ALL COUNTRIES.Our goal now is to remove countries and let our model to predict suicide rate just by providing other features,such as GDP per capita.I havent dropped **year** column yet,because there is tendency of suicide in last decade,due to the influence of social media in Internet in Europe and USA.

# In[ ]:


raw_data['country'].unique()


# In[ ]:


region=pd.Series
europe=[]
asia=[]
samerica=[]
namerica=[]
for country in raw_data['country']:
    if country==europe.any():
        region.append('Europe')
    elif country==asia.any():
        region.append('Asia')
    elif country==samerica.any():
        region.append('South America')
    else:
        region.append('North America')
raw_data=pd.concat([raw_data,region],axis=1)    


# **lets just analyse years and suicide rates and compare the results**
# 

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(raw_data['year'],raw_data['suicides/100k pop'])


# In[ ]:


plt.scatter(raw_data['year'],raw_data['suicides/100k pop'])


# oh,it seems that i am wrong.Hmmm,maybe one more attempt!
# 

# In[ ]:


import seaborn as sns
sns.boxplot(x='year', y='suicides/100k pop', data=raw_data)


# **oh,there are lots of outliers.Nonetheless,as you see here(or maybe not),the most suicidal time period is considered between 1991-1997.But,what if,we separate developed countries from developing ones?There are lots of countries,which still do not have full access to Internet.The HDI of above 0.700 considered as high.**

# In[ ]:


youngsters_data=raw_data[raw_data['age']==('5-14 years'or'15-24 years')]
sns.boxplot(x='year', y='suicides/100k pop', data=youngsters_data)


# In[ ]:





# In[ ]:





# In[ ]:


developed_countries=raw_data[raw_data['HDI for year']>0.700]
sns.boxplot(x='year', y='suicides/100k pop', data=developed_countries)


# In[ ]:


plt.scatter(developed_countries['year'],developed_countries['suicides/100k pop'])


# NOPE,QUALITY OF LIFE OVERWEIGHTS SOCIAL MEDIA.LETS TAKE A LOOK TO COUNTRIES WITH MEDIUM AND LOW HDI
# 

# In[ ]:


underdeveloping_countries=raw_data[raw_data['HDI for year']<0.700]
sns.boxplot(x='year', y='suicides/100k pop', data=underdeveloping_countries)


# In[ ]:


developed_countries_young=youngsters_data[youngsters_data['HDI for year']>0.700]
sns.boxplot(x='year', y='suicides/100k pop', data=developed_countries_young)


# In[ ]:





# there is no tendency,but 1995 year seems to me as a **outlier** year.WHAT IF WE CAN JUST REMOVE STATISTICS FROM THIS YEAR AT ALL?GOOD IDEA,RIGHT
# 
# 

# In[ ]:


needed_data=raw_data.drop(['country-year','country'],axis=1)
needed_data=needed_data[needed_data['year']!=(1993|1994|1995|1996|1997)]
needed_data                                              


# In[ ]:


needed_data['year']=='1995'
pd.options.display.max_rows=None
display(needed_data['year']=='1995')


# WE ARE NOW FREE FROM OUTLIERS.OR NOT?WE ALSO HAVE OUTLIERS ON SUICIDE RATES.

# In[ ]:


pd.options.display.max_rows=100


# In[ ]:


plt.hist(needed_data['suicides/100k pop'])


# MORE THAN 50 IS CONSIDERED AS OUTLIERS.MAYBE NOT,BUT IT WILL MAKE NOISE TO ML MODEL,SO LETS RESPECT OUR FUTURE MODEL.

# In[ ]:


needed_data.drop(['year'],axis=1,inplace=True)
needed_data=needed_data[needed_data['suicides/100k pop']<50]


# **LETS BE MORE OPTIMISTIC,RIGHT**

# In[ ]:


satisfactory_data=needed_data.copy()


# 

# In[ ]:


satisfactory_data


# In[ ]:


satisfactory_data['HDI for year'].isna().sum()


# OH MAN,THIS IS A BAD SIGN.THERE ARE LOTS OF INSTANCES HAVE **NaN** VALUES,SO WE CAN NOT JUST DROP rows,that have NaN values,so we can either drop whole column or substitute NaN to median or mean values

# 

# In[ ]:


satisfactory_data['sex'].value_counts()


# In[ ]:


satisfactory_data['age'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# if remove the column HDI,then use CONTINENTS
# if hdi normal,no continents
# if possible use both of them
# split year column to non internet and after internet period
# 
