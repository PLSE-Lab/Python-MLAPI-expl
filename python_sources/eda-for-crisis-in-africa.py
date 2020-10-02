#!/usr/bin/env python
# coding: utf-8

# ![](https://www.worldatlas.com/r/w1200-h701-c1200x701/upload/e5/76/06/currency-zimbabwe-dollar.jpg)

# In this kernel let's explore some of the economic crises that the countries in Africa faces. As I go on in this journey and learn new topics, I will incorporate them with each new updates. Please upvote if you like this kernel and provide feedback!

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import the dataset
data = pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')


# In[ ]:


#Let's have a look at currency rate
sns.set_style('whitegrid')
plt.figure(figsize = (8,5))
sns.lineplot(x = 'year', y = 'exch_usd', hue = 'country', data = data, palette = 'colorblind')
plt.xlabel('Year')
plt.ylabel('Exchange Rate')
display()


# **Observations:**
# 1. Some countries have relatively lower exchange rate than other countries. Countries like South Africa, Zambia, Egypt and Morocco has relatively lower exchange rate (It is hard to interpret with the above graph, Let's break it down the exchange rate for each country in the next graph)
# 2. The exchange rate is almost zero for all the countries before 1940. This might be because the value is not recorded or a new currency had been adopted by the countries. (Further analysis required)
# 3. There are tremendous spikes in the exchange rate Angola and Zimbabwe. This might indicate an economic breakdown.
# ### Let's break it down further

# In[ ]:


#Exchange rates before and after independece
sns.set_style('whitegrid')
fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(18,12), dpi= 60)
axes = axes.flatten()
for i, ax in zip(data['country'].unique(), axes):
  sns.lineplot(x = 'year', y = 'exch_usd', hue = 'independence', data = data[data['country'] == i], ax = ax)
  ax.set_xlabel('Year')
  ax.set_ylabel('Exchange Rate')
  ax.set_title('{}'.format(i))
  ax.get_legend().remove()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=1)
fig.subplots_adjust(top=0.95)
for i in range(13,16):
  fig.delaxes(axes[i])
plt.tight_layout()


# **Observations:**
# 1. All the countries had a good exchange rate before independence. This is because, most of the countries would have opted for new currency system after independece. For example, Tunisian dinar was introduced in 1960 and the Algerian dinar was introduced in 1964 (Ref:Wikipedia).
# 2. Egypt has been an independent country since 1850s. However it's exchange rate has started increasing from 1970s. Let's consider Egypt as a special case in respective to independence.
# 3. The exchange rate had gone up after the independence for almost all the countries expect Tunisia. Except Tunisia and Ivory coast, the exchange rate for all the countries have been increasing from the independence with some fluctuations.
# 4. There are some sudden spikes in the exchange rate. Angolan Kwanza - In 1999, a second currency was introduced in Angola called the kwanza and it suffered early on from high inflation (Wikipedia). Tunisian dinar was introduced in 1960, hence a spike.

# In[ ]:


#Hyperinflation
#Relationship between Inflation annual and Inflation crisis
sns.set_style('whitegrid')
fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(18,12), dpi= 60)
axes = axes.flatten()
for i, ax in zip(data['country'].unique(), axes):
  sns.lineplot(x = 'year', y = 'inflation_annual_cpi', data = data[data['country'] == i], ax = ax, color = 'cornflowerblue')
  ax.set_xlabel('Year')
  ax.set_ylabel('Inflation Rate')
  ax.set_title('{}'.format(i))
  inflation = data[(data['country'] == i) & (data['inflation_crises'] == 1)]['year'].unique()
  for i in inflation:
    ax.axvline(x=i, color='indianred', linestyle='--', linewidth=.9)
fig.subplots_adjust(top=0.95)
for i in range(13,16):
  fig.delaxes(axes[i])
plt.tight_layout()


# **Observations:**
# 1. The dotted lines represents inflation crisis.
# 2. It is obvious that whenever there is a higher inflation rate it causes an inflation crisis. Eventhough the plots for countries like Angola and Zimbabwe shows an inflation crisis even when the inflation rate is relatively lower becuase the y axis (inflation rate) for Angolo is already very higher and Zimbabwe's highest inflation rate is 20 million so the graph couldn't able to capture it.

# In[ ]:


#Number of inflation crisis by Country
data.groupby('country').agg({'inflation_crises':'sum'}).sort_values('inflation_crises', ascending = False)


# > **Observations:**
# 1. Angolo, Zambia and Zimbabwe suffered more number of inflations compared to all other african countries.
# 2. South Africa has the lowest number of inflation crisis.

# In[ ]:


#Let's have a look at exchange rate and currency crisis
sns.set_style('whitegrid')
fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(18,12), dpi= 60)
axes = axes.flatten()
for i, ax in zip(data['country'].unique(), axes):
  sns.lineplot(x = 'year', y = 'exch_usd', data = data[data['country'] == i], ax = ax, color = 'mediumslateblue')
  ax.set_xlabel('Year')
  ax.set_ylabel('Exchange Rate')
  ax.set_title('{}'.format(i))
  currency = data[(data['country'] == i) & (data['currency_crises'] == 1)]['year'].unique()
  for i in currency:
    ax.axvline(x=i, color='indianred', linestyle='--', linewidth=.9)
fig.subplots_adjust(top=0.95)
for i in range(13,16):
  fig.delaxes(axes[i])
plt.tight_layout()
display()


# **Observations**
# 1. The dotted lines represents currency crisis.
# 2. There is no visual evidence to support any relationship between exchange rate and currency crisis.

# In[ ]:


#Number of inflation crisis by Country
data.groupby('country').agg({'currency_crises':'sum'}).sort_values('currency_crises', ascending = False)


# **Observations:**
# 1. Angolo, Zimbabwe and Zambia suffered more number of currency crises compared to all other african countries (Same as the inflation crisis).
# 2. However, South Africa has 16 currency crises even though it had only one inflation crisis.**

# In[ ]:


#Relationship between Exchange Rate and Inflation rate
fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(18,12), dpi= 60)
axes = axes.flatten()
for i, ax in zip(data['country'].unique(), axes):
  sns.lineplot(x = 'year', y = 'exch_usd', data = data[data['country'] == i], ax = ax, color = 'indianred', label = 'Exchange Rate')
  ax2 = ax.twinx()
  sns.lineplot(x = 'year', y = 'inflation_annual_cpi', data = data[data['country'] == i], ax = ax2, color = 'slateblue', label = 'Inflation Rate')
  ax.set_xlabel('Year')
  ax.set_ylabel('Exchange Rate')
  ax.get_legend().remove()
  ax2.set_ylabel('Inflation Rate')
  ax2.get_legend().remove()
  ax.set_title('{}'.format(i))
handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(handles + handles2, labels + labels2, loc=1)
fig.subplots_adjust(top=0.95)
for i in range(13,16):
  fig.delaxes(axes[i])
plt.tight_layout()
display()


# **Observations:**
# 1. There is no visual evidence to support that there is a relationship between exchange rate and inflation rate.
# 2. Occasionally there is a sudden increase in the exchange rate after a steep rise & fall in the inflation rate. However, this can only be seen in countries like Angolo, Nigeria, Tunisia, Zambia and Zimbabwe

# In[ ]:


#Mapping the values in banking_crisis to 0 and 1
dict = {'no_crisis': 0, 'crisis': 1}
data['banking_crisis'] = data['banking_crisis'].map(dict)


# In[ ]:


#Visualizing different types of crisis
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(18,12), dpi= 60)
axes = axes.flatten()
cols = ['currency_crises','inflation_crises','banking_crisis','systemic_crisis']
for i, ax in zip(cols, axes):
  sns.countplot(y = 'country', ax = ax, data = data, hue = i, palette = 'Paired')
plt.tight_layout()
display()


# **Observations:**
# 1. Zimbabwe has higher number of crisis (Systemic, Currency, Inflation, Banking)
# 2. Zambia has comparatively more number of Currency and Inflation crises but lesser number of Banking and Systemic crises.
# 3. Angola on the other hand has comparatively lesser number of Currency and Inflation crises but higher number of Banking and Systemic crises.

# In[ ]:


#Visualizing different types of debts
fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(18,7), dpi= 60)
axes = axes.flatten()
cols = ['domestic_debt_in_default','sovereign_external_debt_default']
for i, ax in zip(cols, axes):
  sns.countplot(x = 'country', ax = ax, data = data, hue = i)
plt.tight_layout()
display()


# **Observations:**
# 1. The only two countries that suffered from domestic debt in default are Angolo and Zimbabwe.
# 2. However countries like Central African Republic, Ivory Coast and Zimbabwe suffered more number of soverign external debt default than any other african countries.
# 3. Mauritius is the only country that did not have any sovereign external debt default.

# In[ ]:




