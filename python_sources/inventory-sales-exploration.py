#!/usr/bin/env python
# coding: utf-8

# ## Initial Data Exploration
# 
# On this kernel we will take a look at the ratio between inventory and sales and how it changed throughout the years
# 
# ### Objective
# Explore the data

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sales_ratio = pd.read_csv('../input/total-business-inventories-to-sales-ratio_1.csv')


# In[ ]:


sales_ratio.info()


# The description of the data says that the columns realtime_start and realtime_end are automated columns from the day of the data extraction. Therefore we can remove both columns.
# 
# Also, we will index the data on date and assign all numeric values to numeric, since our info tells us that is an object.

# In[ ]:


sales_ratio.drop(['realtime_start', 'realtime_end'], axis=1, inplace=True)
#Indexing on date
sales_ratio['date'] = pd.to_datetime(sales_ratio['date'])
sales_ratio['value'] = pd.to_numeric(sales_ratio['value'], errors='coerce')
sales_ratio.set_index('date', inplace=True)

sales_ratio.head()


# In[ ]:


sales_ratio.info()


# From all the columns converted, only a fraction of them have valid values, so we will remove all invalid values.
# It also appear that we don't have a frequency yet. So we'll take care of that here too.

# In[ ]:


sales_ratio.dropna(inplace=True)
sales_ratio = sales_ratio.asfreq('M', method='ffill')
sales_ratio.info()


# In[ ]:


# I could have used describe() here, but I find it not intuitive for whomever may be reading your analisys
print(f'Mean {np.round(np.mean(sales_ratio.value),2)}')
print(f'Standard Deviation {np.round(np.std(sales_ratio.value),2)}')
print(f'Median {np.median(sales_ratio.value)}')
print(f'Min {np.min(sales_ratio.value)}')
print(f'Max {np.max(sales_ratio.value)}')
print(f'25% of all values are below: {np.percentile(sales_ratio.value, 25)}')
print(f'50% of all values are below: {np.percentile(sales_ratio.value, 50)}')
print(f'75% of all values are below: {np.percentile(sales_ratio.value, 75)}')


# As our mean is smaller than the median, it may show some skewedness on our data. A visualization may help us identify what is going on.

# In[ ]:


sns.set_style('whitegrid')
sns.set_palette('tab20')


# In[ ]:


sns.distplot(sales_ratio.value,bins=12, color='b', kde_kws={'color': 'r'})
sns.despine(left=True)


# Seems like our data is slightly right skewed (the median is bigger than the mean) and bimodal. This can happen in two scenarios, the first is that we have two different types of stores, or that the value has changed with time.
# 
# As we are looking at a time series, our hipothesis is that the value has changed with time.

# In[ ]:


plt.figure(figsize=(10,4))
g = sales_ratio.plot()
sns.despine(left=True)
g.set_title('Inventory/sales ratio through the years')
g.set_xlabel('Year')
g.set_ylabel('Inventory/Sales Ratio')
plt.show()


# Our hipothesis seems right. We can easily notice a decreasing pattern until the years of 2007 to 2009, where it peaked and then slowly started to go up again.
# 
# We can also verify if this ratio has some kind of seasonality. For this, we will evaluate the last 2 years month by month.

# In[ ]:


sales_ratio['month'] = sales_ratio.index.month
sales_ratio.tail(5)


# In[ ]:


y = sales_ratio['value'].resample('M').mean()
decomposition = sm.tsa.seasonal_decompose(y,model='additive')
decomposition.plot()


# In[ ]:



g = sns.boxplot(x=sales_ratio['2016-01-01':'2018-01-01'].index.month, y=sales_ratio['2016-01-01':'2018-01-01'].value)
sns.despine(left=True)
g.set_title('Monthly mean of the last 2 years (2016/2017)')
g.set_xlabel('Month')
g.set_ylabel('Mean')
g.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.show()


# Here, we can notice that there is a certain seasonality. It appears that people don't tend to keep inventory for winter, which seems counter intuitive.
# 
# Last but not least, let's look at the last 10 years

# In[ ]:


plt.figure(figsize=(10,6))
g = sns.barplot(x=sales_ratio['2008-01-01':'2018-01-01'].index.year, y=sales_ratio['2008-01-01':'2018-01-01'].value,ci=None, saturation=0.65, palette='tab20')
g.set_title('Mean of the last 10 years (2008/2017)')
g.set_ylabel('Mean')
g.set_xlabel('Year')
sns.despine(left=True)


# ### Conclusion
# 
# Economic growth and crisis can cause a lot of impact in inventory management. We can notice that major economic shifts (2007/2008 Crisis and the 2016 Elections) have affected the market in mostly negative ways. Although it tends to estabilize after a while.
# 
# That was a fun little research that I plan to extend in the near future.
# 
