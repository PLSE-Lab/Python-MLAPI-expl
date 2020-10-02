#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("darkgrid")

sns.set(rc={'figure.figsize':(20, 10)})


# In[ ]:


data = pd.read_csv('../input/gapminder/gapminder.tsv', sep="\t")


# In[ ]:


data.head()


# In[ ]:


bd = data.loc[data['country'] == 'Bangladesh']
bd.head()


# In[ ]:


afg = data.loc[data['country'] == 'Afghanistan']
afg.tail()


# In[ ]:


asia = data[data['continent'] == 'Asia']
asia.head()


# In[ ]:


asiaWithSparsePop = asia[asia['pop'] < 1.453083e+07]
sns.boxplot(x="lifeExp", y="country", data=asiaWithSparsePop)


# In[ ]:


asia['pop'].describe()


# In[ ]:


bd.head()


# In[ ]:


bdgdp = sns.boxplot(x='country', y='lifeExp', data=bd)


# In[ ]:


bd.describe()


# In[ ]:


bd.shape


# In[ ]:


bd = bd.assign(gdpQuantiles = lambda x: pd.qcut(x['gdpPercap'] , q=4, labels=['min', 'low', 'medium', 'max']))
sns.boxplot(x='gdpQuantiles', y='year', data=bd)


# In[ ]:


sns.pairplot(data)


# In[ ]:


sns.distplot(data['lifeExp'])


# In[ ]:


data['lifeExp'].describe()


# In[ ]:


bd['lifeExp']


# In[ ]:


sns.jointplot(x="gdpPercap", y="pop", data=data)


# In[ ]:


sns.distplot(bd['lifeExp'])


# In[ ]:


america = data[data['continent'] == 'Americas']
sns.distplot(america['lifeExp'])


# In[ ]:


sns.scatterplot(x='lifeExp', y='gdpPercap', size='pop', hue='year', data=data)


# In[ ]:


data.describe()


# In[ ]:


gdp = data[data['gdpPercap'] < 9325.462346]
gdp = gdp.assign(popQuartiles = lambda x: pd.qcut(x=x['pop'], q=4))
sns.set(rc={'figure.figsize':(20, 15)})
sns.scatterplot(x='lifeExp', y='gdpPercap', size='popQuartiles', hue='year', data=gdp)


# In[ ]:


data.groupby('continent').describe()


# In[ ]:


sns.barplot(y='country', x='pop', data=data[data['continent'] == 'Asia'])


# In[ ]:


countriesWithLowPopulation = data[data['pop'] < 2.793664e+06 ]
countriesWithLowPopulation.count()


# In[ ]:


lowPopAsia = countriesWithLowPopulation[countriesWithLowPopulation['continent'] == 'Asia']
sns.barplot(y='country', x='pop', data=lowPopAsia)

