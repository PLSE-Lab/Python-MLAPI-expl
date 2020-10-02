#!/usr/bin/env python
# coding: utf-8

# # Suicide Rates Overview 1985 to 2016

# ### Compares socio-economic info with suicide rates by year and country

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[ ]:


data = pd.read_csv('../input/master.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# There are in total 27820 records and 12 attributes 

# In[ ]:


data.columns


# In[ ]:


data.info()


# In[ ]:


data.describe()


# 
#     1. HDI for year has many missing data. As a result, better remove from dataset. 
#     2. gdp_for_year is in text format, better change it back to numeric(floating 64) 
#     3. country-year provides no extra information. Cn be deleted  

# In[ ]:


data[' gdp_for_year ($) '] = data[' gdp_for_year ($) '].apply(lambda x: x.replace(',','')).astype(float)
data=data.drop(['HDI for year','country-year'],axis=1)


# In[ ]:


data.head()


# Next step is to rename attribure names.

# In[ ]:


data=data.rename(columns={'suicides/100k pop':'suicide_100k',' gdp_for_year ($) ':'gdp_yr','gdp_per_capita ($)':'gdp_cap'})


# In[ ]:


data.head()


# In[ ]:


data.shape


# After cleaning, there are in total 10 attributes 

# Now is time to explore data 

# In[ ]:


country = data.groupby('country')['suicides_no','population'].agg('sum')
country['suicides_100k']=country['suicides_no']/country['population']*100000
country.sort_values(by='suicides_100k',ascending=False,inplace=True)
print("Highest suicide rate by country")
print(country.head())
print()
print("Lowest suicide rate by country")
print(country.tail())


# Although summing up population across all years is not meaningful, however this can simply show the severeness of suicide in different countries without considering the year effect by comparing the total number of suicide and total population in the whole year.
# <br>
# <br> The top 5 countries with suicide rate are Lithuania, Russia, Sri Lanka, Belarus and Hungary. 
# <br> The lowest 5 countries with suicide rate are Saint Kitts and Nevis, Dominica, Oman, Jamaica and Antigua and Barbuda. 
# 
# 3 out of 5 top countries with suicide rate are in eastern Europe. This can be an interesting discovery. 

# In[ ]:


year = data.groupby('year')['suicides_no','population'].agg('sum')
year['suicides_100k']=year['suicides_no']/year['population']*100000
year.sort_values(by='suicides_100k',ascending=False,inplace=True)
year


# With the improvement of living standard, the suicide rate should be decreasing across year. From the data across year, the highest suicide rates are in 90s.  

# In[ ]:


gender = data.groupby('sex')['suicides_no','population'].agg('sum')
gender['suicides_100k']=gender['suicides_no']/gender['population']*100000
gender.sort_values(by='suicides_100k',ascending=False,inplace=True)
gender


# There are numerous studies on gender differences in suicide. From  [Wikipedia](https://en.wikipedia.org/wiki/Gender_differences_in_suicide), the the suicide rate for male is around 1.8 times more often than female. And from data, across all years and countries, the suicide rate for male is around 2.5 times higher than female.
# <br>Again, there is difference on suicide rate across gender in different countries. It's good to see if eastern Europe also has particular findings.

# In[ ]:


country_gender = data.groupby(['country','sex'])['suicides_no','population'].agg('sum').reset_index()
country_gender['suicides_100k']=country_gender['suicides_no']/country_gender['population']*100000

country_gender.sort_values(by='suicides_100k',ascending=False,inplace=True)
country_gender.head()


# In[ ]:


country_gender2 = pd.crosstab(index=country_gender.country,columns=country_gender.sex,values=country_gender.suicides_100k,aggfunc='sum')
country_gender2['m_vs_f']=country_gender2['male']-country_gender2['female']
country_gender2.sort_values(by='m_vs_f',ascending=False,inplace=True)
country_gender2.head()


# The above table shows the difference between suicide rate across gender and country.<br>
# The effect of gender differnce on suicide is significant in some eastern Europe as the difference is over 40 per 100k population. 

# In[ ]:


age = data.groupby(['age'])['suicides_no','population'].agg('sum')
age['suicides_100k']=age['suicides_no']/age['population']*100000
age.sort_values(by='suicides_100k',ascending=False,inplace=True)
age


# The suicide rates increase with the increase of victum ages. 

# Another study is the relation between gdp per capita and suicide rate. <br>There is a possibility that the better economy can reduce the suicide rate.
# <br>To study this, scatter plot between gdp per capita and suicide no are plotted. Because gdp per capita is changing across year, therefore there will be one data for each country-year combination.

# First is check if the gdp per capita is unique for country-year combination

# In[ ]:


country_year_gdp = data[['country','year','gdp_cap']]
country_year_gdp.drop_duplicates(inplace=True)
print(country_year_gdp.shape)

country_year = country_year_gdp.drop_duplicates(subset=['country','year'])
print(country_year.shape)


# So gdp per capita is unique for country-year combination
# <br>Next step is to create the scatter plot.

# In[ ]:


country_year = data.groupby(['country','year']).agg({'suicides_no':'sum','population':'sum','gdp_cap':'min'}).reset_index()
country_year['suicides_100k']=country_year['suicides_no']/country_year['population']*100000
country_year.head()


# In[ ]:


plt.figure(figsize=(15,15))
plt.scatter(x=country_year.gdp_cap, y=country_year.suicides_100k)
plt.title('Suicide rate per 100k population across gdp per capita')
plt.show()


# To study more in detailed, separate two groups of data, one with gdp per cap lower than 20k, and one with gdp per cap higher than 20k 

# In[ ]:


plt.figure(figsize=(15,15))
plt.scatter(x=country_year[country_year['gdp_cap']<20000].gdp_cap, y=country_year[country_year['gdp_cap']<20000].suicides_100k)
plt.plot(np.unique(country_year[country_year['gdp_cap']<20000].gdp_cap),
         np.poly1d(np.polyfit(country_year[country_year['gdp_cap']<20000].gdp_cap,
                              country_year[country_year['gdp_cap']<20000].suicides_100k,1))
         (np.unique(country_year[country_year['gdp_cap']<20000].gdp_cap)))

plt.title('Suicide rate per 100k population across gdp per capita (with gdp less than 20k USD)')
plt.show()


# In[ ]:


plt.figure(figsize=(15,15))
plt.scatter(x=country_year[country_year['gdp_cap']>=20000].gdp_cap, y=country_year[country_year['gdp_cap']>=20000].suicides_100k)
plt.plot(np.unique(country_year[country_year['gdp_cap']>=20000].gdp_cap),
         np.poly1d(np.polyfit(country_year[country_year['gdp_cap']>=20000].gdp_cap,
                              country_year[country_year['gdp_cap']>=20000].suicides_100k,1))
         (np.unique(country_year[country_year['gdp_cap']>=20000].gdp_cap)))

plt.title('Suicide rate per 100k population across gdp per capita (with gdp higher than or equal to 20k USD)')
plt.show()


# Although the two best fit lines show different relationship between suicide rate and gdp per capita, the slopes are not very deep. Also from the two scatter plots, one can see the distrubition of scatters are quite broad. As a result, it is hard to connect between suicide rate and gdp per capita.

# In[ ]:




