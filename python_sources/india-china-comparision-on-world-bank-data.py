#!/usr/bin/env python
# coding: utf-8

# # Analysing World Bank Data (1960-2016) of population between India and China
# 
# India and China are the countries with the largest and second largest population in the world respectively. While both countries are approaching the same population, one country's approach to handle population is much more different than the other. 
# 
# NOTE: Some of my deductions from the following data may be incorrect, so feel free to correct me.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


population = pd.read_csv('../input/country_population.csv')
fertility_rate = pd.read_csv('../input/fertility_rate.csv')
expectancy = pd.read_csv('../input/life_expectancy.csv')


# ## Comparing population growth
# 
# Population growth of a country is a crucial metric since countries have a fixed area and more people means needing to accomodate more amount of people in the same amount of area. This raises problems like increasing farm and dairy produce and at the same time finding new area to build houses.

# In[ ]:


population.head()


# In[ ]:


india = population.loc[population['Country Name'] == 'India']
china = population.loc[population['Country Name'] == 'China']


# ## Comparing population of India and China per year (1960-2016)

# In[ ]:


x = range(1959, 2016)
size = india.size
y_india = india.iloc[0, 4:size]
y_china = china.iloc[0, 4:size]

plt.plot(x, y_india)
plt.plot(x, y_china)
plt.xlabel('Year (1960 - 2016)')
plt.ylabel('population (billions)')
plt.legend(['India', 'China'])
plt.show()


# It seems like India's population has been steadily growing while in China the growth rate has been decreasing which confirms that the one-child limit set from 1979 to 2015 has worked to control the growth of population, whereas no such policy has been implemented in India and it seems like the population of India might even exceed that of China in the near future, since china now has a 2 child limit. This poses a large threat on India since India's geographical area (2,973,193.0 sqaure km) is much smaller than that of China (9,326,410.0 sqaure km)

# ## Comparing population growth rate per year

# In[ ]:


india_growth = []
for i in range(56):
    india_growth.append(y_india[i + 1] - y_india[i])

china_growth = []
for i in range(56):
    china_growth.append(y_china[i + 1] - y_china[i])
    
plt.plot(x[1:], india_growth)
plt.plot(x[1:], china_growth)
plt.legend(["India", "China"])
plt.show()


# ## Comparing fertility rate
# Fertility rate is roughly defined as number of children born per woman and is a key statistic to observe the affect of the 1-child limit

# In[ ]:


fertility_rate.head()


# In[ ]:


# fertility rate of china
china_fer = fertility_rate.loc[fertility_rate['Country Name'] == "China"].iloc[0, 4:size]
# fertility rate of India
india_fer = fertility_rate.loc[fertility_rate['Country Name'] == "India"].iloc[0, 4:size]


# In[ ]:


# comparing fertility rate of both countries
plt.plot(x, india_fer)
plt.plot(x, china_fer)
plt.xlabel('Year (1960 - 2016)')
plt.ylabel('child per woman')
plt.legend(['India', 'China'])
plt.show()


# India's approach of gradually decreasing fertilty rate seems more favourable for stability of the society. China's approach of immediate action and reducing fertilty rate to 1 seems harsh and would tend to bring about resistance, but the 1-child limit was introduced around 1980, so what caused the immediate decrease in fertilty rate about 10 years earlier is unclear right now.

# ## Population density (People per square kilomteters)
# The biggest problem with increase in population is the limited amount of geographical area available in the country. The population density needs to be in a safe range so as to keep the country's expenses in check. For example, if this ratio is very large and there is no land to produce food, most of it will need to be imported and will decrease self-sufficiency of the country.

# In[ ]:


china_area = 9326410.0
india_area = 2973193.0
plt.plot(x, y_india/india_area)
plt.plot(x, y_china/china_area)
plt.xlabel('Year (1960 - 2016)')
plt.ylabel('population density (people per sq. km.)')
plt.legend(['India', 'China'])
plt.show()


# As you can see, India's population density is much, much higher than that of China. At some point, it might start to cause troubles when above a certain limit combined with the per capita GDP. For example, Monaco has a density of 18,589 but has a relatively high per capita GDP of 168,000 USD. Wheras in the case of Bangladesh, the per capita GDP is 1,754 USD and the density is 1,145 which may have contributed to the allegedly lower quality of life. I know it's not for me to decide anything but the numbers speak for themselves, I hope conditions change for the better everywhere.
