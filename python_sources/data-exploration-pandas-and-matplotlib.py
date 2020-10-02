#!/usr/bin/env python
# coding: utf-8

# # Suicide Rates - Data Exploration
# 
# 
# I am new to using Python and am going to practise the skills I have learnt from Coursera by completing a data exploration of the Suicide dataset. 
# 
# Firstly, I load in the libraries and preview the dataset.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = pd.read_csv('../input/master.csv')

s[s['year'] == 2010].head()


# We have data on country suicide rates, split by gender, year, and age, with additional information on measures such as GDP and HDI (human development index). 
# 
# We can wrangle the data to initially look at the global suicide rate over time
# 
# ## Global suicide rates over time

# In[ ]:


glo = s.groupby(['year']).agg({'suicides_no' : 'sum','population':'sum',}).reset_index()
glo['suicides_percapita'] = glo.suicides_no / glo.population * 100000
plt.figure(figsize = (15,8))
_ = plt.plot(glo.year,glo.suicides_percapita,'-o')
_ = plt.title('Suicides per 100k of population')


# Global suicide rates seem to have been rising through the late eighties and early nineties, with a peak in the mid nineties, before falling off throughout the rest of the century and early twenty first century.
# 
# ## Is there a relationship between suicide and gender?

# In[ ]:


# split by men and woman
gender = s.groupby(['year','sex']).agg({'suicides_no' : 'sum','population' : 'sum'}).reset_index()
gender['suicides_percapita'] = gender.suicides_no / gender.population * 100000

def gender_plot(sex): 
    data = gender.loc[gender['sex'] == sex,['year','suicides_percapita']]
    plt.plot(data['year'],data['suicides_percapita'])

plt.figure(figsize = (15,8))
for i in ('male','female'):
    gender_plot(i)

_ = plt.legend(('male','female'))
_ = plt.title('Suicides per 100K of population \n Split by Gender')


# Globally, male suicide is much more prominent that female suicide, with the incidence per capita being three times higher on average over the time period than women's per capita rate (6 per 100K women, vs 21 per 100K men). 
# 
# It also seems it is the male suicide rate which drives the increase through the mid nineties, followed by a fall; the women's rate seems to be either flat or falling over the time period. 
# 
# ## Is there a relationship between suicide and age?

# In[ ]:


age = s.groupby(['year','age']).agg({'population' : 'sum','suicides_no': 'sum'}).reset_index()
age['suicides_percapita'] = age.suicides_no / age.population * 100000


def age_plot(agegroup):
    data = age.loc[age['age'] == agegroup,['year','suicides_percapita']]
    _ = plt.plot(data['year'],data['suicides_percapita'])

agegroups = [ '5-14 years',
                     '15-24 years',
                     '25-34 years',
                     '35-54 years',
                     '55-74 years',
                     '75+ years']

plt.figure(figsize = (15,8))

for i in agegroups:
    age_plot(i)
_ = plt.legend(agegroups)
_ = plt.title('Suicides per 100K of population \n Split by Age Group')


# Suicide rates are higher in older age groups. 
# 
# It is also worth noting that the oldest age group (75+) is the only age group to see an almost persistent decline in suicide per capita over the period studied, and the only age group to not experience some increase into the mid-nineties (other than the under 14s age group, which appears to be flat at very low rates per capita).  
# 
# ## What about considering both age and gender?

# In[ ]:


yas = s.groupby(['year','age','sex']).agg({'population' : 'sum','suicides_no':'sum'}).reset_index()
yas['suicides_percapita'] = yas.suicides_no / yas.population * 100000

def yas_plot(age,sex,position):
    data = yas.loc[(yas['sex'] == sex) & (yas['age'] == age),['year','age','sex','suicides_percapita']]
    axs[position].plot(data['year'],data['suicides_percapita'])
    
fig, axs = plt.subplots(1,2,sharey = True,figsize = (15,8))

for i in agegroups:
    yas_plot(i,'male',0)

for i in agegroups:
    yas_plot(i,'female',1)

axs[0].legend(agegroups)
axs[0].title.set_text('Suicides per 100K of Male population \n Split by Age Group')
axs[1].legend(agegroups)
axs[1].title.set_text('Suicides per 100K of Female population \n Split by Age Group')


# Firstly, we can see once again that men's suicide rates are higher than women for all age groups. We can also see that it is true for both men and women that older age groups have a higher incidence of suicide. 
# 
# ## Is there a relationship between suicide rates and income?
# 
# For a given year, we can create a scatter plot with income on the X axis, suicide rate per capita on the Y axis and plot each country as  a single point, to see if there is a pattern. We could also change the size of each point to represent in nominal terms now many suicides happen in that country.

# In[ ]:


y2010 = s.loc[s['year'] == 2010].groupby(['country']).agg({'population' : 'sum','suicides_no' : 'sum'})

gdp = s.loc[s['year'] == 2010].drop_duplicates(subset = ['country','gdp_per_capita ($)']).loc[:,['country','gdp_per_capita ($)']]
gdp.set_index('country',inplace = True)

y2010 = pd.merge(y2010,gdp,left_index = True,right_index = True,how = 'left')
y2010['suicides_percapita'] = y2010.suicides_no / y2010.population  * 100000

plt.figure(figsize = (15,8))
_ = plt.scatter(y2010['gdp_per_capita ($)'],y2010['suicides_percapita'],s = y2010['suicides_no']/5,alpha = 0.7)
_ = plt.xlabel('Income')
_ = plt.ylabel('Suicides per 100K of population')
_ = plt.title('Income and suicide rates by country in 2010 \n (bubble size = nominal number of suicides)')


# There doesn't seem to be a clear cut relationship; poor countries (furthest left on X axis) have both high and low suicide rates per capita. There is also variation in the wealthier countries' rates too. 
