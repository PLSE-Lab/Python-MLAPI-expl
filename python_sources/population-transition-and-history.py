#!/usr/bin/env python
# coding: utf-8

# # Population transition and history

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/data.csv')
print(df.columns)
df.head()


# Since this dataset seems to include a lot of NaN records, let's extract records which holds some non null values.

# In[ ]:


nonnull_counts = df.isnull().sum(axis=1) / len(df.columns) < 0.05

nonnull_index = [i for (i, v) in enumerate(nonnull_counts) if v]

effective_records = df.iloc[nonnull_index]
print(effective_records.shape)
effective_records.head()


# # Overview
# 
# Plot all records by countries.

# In[ ]:


years = [str(y) for y in range(1974, 2016)]

cols = 2
rows = int(170 / cols) # 171 is the number of effective_records

from matplotlib.ticker import ScalarFormatter
xmajor_formatter = ScalarFormatter()
xmajor_formatter.set_powerlimits((-3, 4))
fig, ax = plt.subplots(rows, cols, figsize=(9, 300))
for r in range(rows):
    for c in range(cols):
        ax[r, c].plot(effective_records.iloc[r * cols + c][years])
        indicator_name = effective_records.iloc[r * cols + c]['Indicator Name']
        country_name = effective_records.iloc[r * cols + c, 0]
        ax[r, c].set_title(indicator_name + ' in ' + country_name)
        ax[r, c].yaxis.set_major_formatter( xmajor_formatter )


# We can see all data shown here is about population growth. Other data seems to lack of records. All countries are 192 now. So around 90% of countries are covered in this data.

# In[ ]:


for i in range(171):
    plt.scatter(years, effective_records.iloc[i][years].values)


# Now we look into the countries whose population decreased in recent 30 years.

# In[ ]:


population_decreased_countries = effective_records[effective_records['2015'] - effective_records['1975'] < 0.0]

population_decreased_countries['Decreased Ratio'] = (population_decreased_countries['2015'] - population_decreased_countries['1975']) / population_decreased_countries['1975']

population_decreased_countries.sort('Decreased Ratio')


# These countries are mainly in East Europe. So we can assume this is related to USSR.
# Checking population growth again, all countries in this list lost their population just from 1989, when Soviet Union Dissolution was occurred.

# In[ ]:


rows = 5
cols = 2
fig, ax = plt.subplots(rows, cols, figsize=(9, 15))

for r in range(rows):
    for c in range(cols):
        country_name = population_decreased_countries.iloc[r * cols + c, 0]
        ax[r, c].plot(population_decreased_countries.iloc[r * cols + c][years])
        ax[r, c].set_title("{}".format(country_name))
        ax[r, c].yaxis.set_major_formatter( xmajor_formatter )


# Let me check other countries.

# In[ ]:


rwanda = effective_records[effective_records.iloc[:, 0] == 'Rwanda']

plt.plot(years, rwanda.iloc[0][years], label='Population')
plt.axvline(x=1994, linewidth=3, label='Genocide')
plt.axvline(x=1991, linewidth=10, label='Civil War', color='red')
plt.title('Rwanda')
plt.legend()


# Rwanda experienced civil war from 1990 and genocide at 1994. So we can see the population decreased at this span.
# Same trend can be seen in Kosovo population.
# 
# 

# In[ ]:


kosovo = effective_records[effective_records.iloc[:, 0] == 'Kosovo']

plt.plot(years, kosovo.iloc[0][years], label='Population')
plt.axvline(x=1996, linewidth=3, label='Begin of Civil War')
plt.axvline(x=1999, linewidth=3, label='End of Civil War')
plt.title('Kosovo')
plt.legend()


# In[ ]:


kazakhstan = effective_records[effective_records.iloc[:, 0] == 'Kazakhstan']

plt.plot(years, kazakhstan.iloc[0][years], label='Population')
plt.axvline(x=1989, linewidth=3, label='Dissolution of USSR')
plt.title('Kazakhstan')
plt.legend()


# Kazakhstan was also a member of USSR but their population grew again after 2000. 

# In[ ]:


czech = effective_records[effective_records.iloc[:, 0] == 'Czech Republic']

plt.plot(years, czech.iloc[0][years], label='Population')
plt.axvline(x=1989, linewidth=3, label='Velvet Revolution')
plt.axvline(x=1993, linewidth=3, label='Separate from Slovakia', color='red')
plt.axvline(x=2004, linewidth=3, label='Join EU', color='green')
plt.title('Czech Republic')
plt.legend(loc=2)


# Czech Republic has variegated population transition. As well as other USSR countries it decreased population from 1989. In addition, Czech decided to be apart from Slovakia at 1993. It might accelerate the population decrease, but after joining EU the people can come to Czech freely. It might contribute population growth again.

# # Conclution
# 
# I want to add more countries and attache their history further.
