#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()

# Disabling warnings
import warnings
warnings.simplefilter("ignore")


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Import our dataset
df = pd.read_csv('../input/master.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


df.rename(columns={'HDI for year': 'HDI_for_year', 'country-year':'country_year', 
                   'suicides/100k pop': 'suicides/100k_pop', ' gdp_for_year ($) ':'gdp_for_year', 
                   'gdp_per_capita ($)':'gdp_per_capita'}, inplace=True);


# In[ ]:


df.columns


# In[ ]:


# Handling Missing Data
df.dropna(1, inplace=True);


# In[ ]:


df.describe()


# In[ ]:


df.index = df.year


# In[ ]:


# Data Visualization And Analysis

# Plot Total No. Of Suicides Per Year From 1985 To 2016.
data_per_year['suicides_no'].plot()
plt.title('Total No. Of Suicides Per Year From 1985 To 2016')
plt.ylabel('No. Suicides')
plt.xlabel('Year')
plt.xlim((df.year.min() - 1), (df.year.max() + 1))
plt.show()


# In[ ]:


# Polt Total No. Of Suicides Per Year From 1985 To 2016 Hue To Gendar.
df.pivot_table('suicides_no', index='year', columns='sex', aggfunc='sum').plot()
plt.title('Total No. Of Suicides Per Year From 1985 To 2016 Hue To Gendar')
plt.ylabel('No. Suicides')
plt.xlabel('Year')
plt.xlim((df.year.min() - 1), (df.year.max() + 1))
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='year', y='suicides_no', data=df, hue='sex') 


# In[ ]:


# Bar Plot No. of Suicides per country last 30 years.

# Main Var. for Ploting
sui_no = df.groupby(['country']).suicides_no.sum()
countries = []
for (i, m) in df.groupby('country'):
    countries.append(i)
countries = np.array(countries);

# ploting
plt.figure(figsize=(10,20))
sns.barplot(y=countries, x=sui_no)
plt.xlabel('No. of suicides')
plt.ylabel('Countries')
plt.title('Total No. of Suicides per country from 1987 to 2016')
plt.xlim(0, 1e6)
plt.show()


# In[ ]:


# Bar Plot No. of Suicides per Age last 30 years.

# Set Variables.
age_sui = df.pivot_table('suicides_no', index='age', aggfunc='sum')
x = age_sui.index.values
y = age_sui.values
y = y.reshape(6,)

# Ploting
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=x, y=y)
ax.set(title='No. Of Suicides Per Age', ylabel='No. of suicides', xlabel='Age');
plt.xticks(rotation=45);
plt.show()


# In[ ]:


gen_sui = df.pivot_table('suicides_no', index='generation', aggfunc='sum')
x = gen_sui.index.values
y = gen_sui.values
y = y.reshape(6,)

fig, ax = plt.subplots(figsize=(10, 6))
explode = (0.05,0.05,0.05,0.1,0.05,0.05)
ax.pie(y, explode=explode, labels=x, autopct='%1.1f%%', shadow=True, startangle=0)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[ ]:


plt.figure(figsize=(10,7))
sns.stripplot(x="year", y='suicides/100k_pop', data=df)
plt.title('No. Of Suicides/100k Population')
plt.xlabel('Year')
plt.ylabel('Suicides/100k Population')
plt.xticks(rotation=60)
plt.show()


# In[ ]:


sns.distplot(df['suicides/100k_pop'])
plt.show()


# In[ ]:


sns.set_color_codes()
sns.distplot(df['country'].value_counts().values)
plt.show()


# In[ ]:


sns.pairplot(df, hue="sex")
plt.show()


# In[ ]:





# <h2>There is a supplement version you can check it and give your recommendation : <a href='https://www.kaggle.com/elcaiseri/classification-of-high-low-suicide-rates'>Classification of high, low Suicide Rates
# </a> </h2>
# 
# <h2>Thanks For being here</h2>

# In[ ]:




