#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


indicators = pd.read_csv('../input/Indicators.csv')


# In[ ]:


# Get the number of patents both for residents and non residents
patents = indicators[(indicators.IndicatorCode == 'IP.PAT.RESD') | (indicators.IndicatorCode == 'IP.PAT.NRES')]
patents[:5]


# In[ ]:


population = indicators[indicators.IndicatorCode == 'SP.POP.TOTL']
population[:5]


# ## Total number of patent applications since 1960
# Looking at the graph below, it seems that Japan leads the race with the US in the second place.

# In[ ]:


china_tot = patents[patents.CountryName == 'China'].sum()
usa_tot = patents[patents.CountryName == 'United States'].sum()
japan_tot = patents[patents.CountryName == 'Japan'].sum()
korea_tot = patents[patents.CountryName == 'Korea, Rep.'].sum()
uk_tot = patents[patents.CountryName == 'United Kingdom'].sum()
russia_tot = patents[patents.CountryName == 'Russian Federation'].sum()
india_tot = patents[patents.CountryName == 'India'].sum()

# Prepare the input for the graph
vals = np.array([int(china_tot.Value), int(usa_tot.Value), int(japan_tot.Value), int(korea_tot.Value), int(uk_tot.Value), int(russia_tot.Value), int(india_tot.Value)])
countries = np.array(['China', 'USA', 'Japan', 'Korea', 'UK', 'Russia', 'India'])


# In[ ]:


plt.figure(figsize=(14,7))
sns.barplot(countries, vals, palette="Set3")
plt.title("Total number of patent applications since 1960", fontsize=14)
plt.ylabel("Number of applications", fontsize=14)


# ## What if we zoom in the last 10 years?
# USA is now taking the first place with China in second.

# In[ ]:


china_tot10 = patents[(patents.CountryName == 'China') & (patents.Year > 2005)].sum()
usa_tot10 = patents[(patents.CountryName == 'United States') & (patents.Year > 2005)].sum()
japan_tot10 = patents[(patents.CountryName == 'Japan') & (patents.Year > 2005)].sum()
korea_tot10 = patents[(patents.CountryName == 'Korea, Rep.') & (patents.Year > 2005)].sum()
uk_tot10 = patents[(patents.CountryName == 'United Kingdom') & (patents.Year > 2005)].sum()
russia_tot10 = patents[(patents.CountryName == 'Russian Federation') & (patents.Year > 2005)].sum()
india_tot10 = patents[(patents.CountryName == 'India') & (patents.Year > 2005)].sum()

# Preparing the input for the graph
vals = np.array([int(china_tot10.Value), int(usa_tot10.Value), int(japan_tot10.Value), int(korea_tot10.Value), int(uk_tot10.Value), int(russia_tot10.Value), int(india_tot10.Value)])


# In[ ]:


plt.figure(figsize=(14,7))
plt.title("Total number of patent applications since 2006", fontsize=14)
plt.ylabel("Number of applications", fontsize=14)
sns.barplot(countries, vals, palette="Set3")


# ## The time series reveals the trend
# Grouping by the year and taking the sum will add both the resident and non-resident 
# applications per country.

# In[ ]:


china_t = patents[patents.CountryName == 'China'].groupby('Year').sum()
usa_t = patents[patents.CountryName == 'United States'].groupby('Year').sum()
japan_t = patents[patents.CountryName == 'Japan'].groupby('Year').sum()
korea_t = patents[patents.CountryName == 'Korea, Rep.'].groupby('Year').sum()
uk_t = patents[patents.CountryName == 'United Kingdom'].groupby('Year').sum()
russia_t = patents[patents.CountryName == 'Russian Federation'].groupby('Year').sum()
india_t = patents[patents.CountryName == 'India'].groupby('Year').sum()


# In[ ]:


plt.figure(figsize=(14,7))
plt.plot(china_t, label='China')
plt.plot(usa_t, label='USA')
plt.plot(korea_t, label='Korea Rep.')
plt.plot(japan_t, label='Japan')
plt.plot(uk_t, label='UK')
plt.plot(russia_t, label='Russia')
plt.plot(india_t, label='India')
plt.xlabel('Years',  fontsize=12)
plt.ylabel('# of Patent Applications',  fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Patent Applications")


# ## What about the applications per capita?
# Some countries are larger than other others and it is excpected to have larger number of patent applications than the smaller ones. Let's take a look at how many patents did each country produce per a hundrend thousand residents.

# In[ ]:


# The reason to group by the year and take the sum is to create a similar structure
# as with the time series above so that we can merge using the same index
china_pop = population[population.CountryName == 'China'].groupby('Year').sum()
usa_pop = population[population.CountryName == 'United States'].groupby('Year').sum()
japan_pop = population[population.CountryName == 'Japan'].groupby('Year').sum()
korea_pop = population[population.CountryName == 'Korea, Rep.'].groupby('Year').sum()
uk_pop = population[population.CountryName == 'United Kingdom'].groupby('Year').sum()
russia_pop = population[population.CountryName == 'Russia'].groupby('Year').sum()
india_pop = population[population.CountryName == 'India'].groupby('Year').sum()


# In[ ]:


# There might be an easier way to do this merge
china100k = pd.merge(china_t, china_pop, left_index=True, right_index=True)
usa100k = pd.merge(usa_t, usa_pop, left_index=True, right_index=True)
japan100k = pd.merge(japan_t, japan_pop, left_index=True, right_index=True)
korea100k = pd.merge(korea_t, korea_pop, left_index=True, right_index=True)
uk100k = pd.merge(uk_t, uk_pop, left_index=True, right_index=True)
russia100k = pd.merge(russia_t, russia_pop, left_index=True, right_index=True)
india100k = pd.merge(india_t, india_pop, left_index=True, right_index=True)


# In[ ]:


plt.figure(figsize=(14,7))
plt.plot(1e6*china100k.Value_x/china100k.Value_y, label='China')
plt.plot(1e6*usa100k.Value_x/usa100k.Value_y, label='USA')
plt.plot(1e6*korea100k.Value_x/korea100k.Value_y, label='Korea Rep.')
plt.plot(1e6*japan100k.Value_x/japan100k.Value_y, label='Japan')
plt.plot(1e6*uk100k.Value_x/uk100k.Value_y, label='UK')
plt.plot(1e6*russia100k.Value_x/russia100k.Value_y, label='Russia')
plt.plot(1e6*india100k.Value_x/india100k.Value_y, label='India')
plt.xlabel('Years',  fontsize=14)
plt.ylabel('# of Patent Applications per 100k population',  fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Patent Applications per 100k population", fontsize=14)


# ## Conclusion
# It seems that China is the emerging power when it comes to the patent applications that are filed in a yearly basis.
# Korea has the highest number of applications per resident but being a smaller country, it produces less applications than China, US and Japan.