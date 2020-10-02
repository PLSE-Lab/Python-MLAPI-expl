#!/usr/bin/env python
# coding: utf-8

# ## USA cars
# This is an entry-level analysis for practice.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Import the necessary libraries and the data

# In[ ]:


import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq
import plotly.graph_objects as go
import wordcloud


# In[ ]:


# The dataset before cleaning
cars = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')
print(cars.shape)
cars.head()


# In[ ]:


cars['country'] = cars['country'].str.strip()

# For title_status there is only 2 option, and salvage insurance takes only 6,5% of the database
# so I won't work with it, because it could distort the results.
cars = cars[cars['title_status'] == 'clean vehicle']

# There is a total of 7 data from the 2499 where the country is canada. I will work only with usa.
cars = cars[cars['country'] == 'usa']

# The Unnamed: 0 column is only another index column, and I won't use the lot, because
# the vin code will be my identifier, so I drop them too.
cars.drop(columns = ['Unnamed: 0', 'lot','condition','title_status', 'country'], inplace = True)
print(cars.shape)
cars.head()


# ### Correlation in our numeric data

# In[ ]:


sns.set(rc={'figure.figsize':(12,8)})

car_corr = cars.corr()
sns.heatmap(car_corr, annot = True, annot_kws={'size':50}, center = 0, cmap = 'magma')
plt.show()


# After a brief insight with heatmap, we should do some more computation for our data. On the next plot, every subplot gets a touple in the form of (correlation coefficient, p-value). The correlation coefficient shows the strength and the direction of the correlation, and the p-value shows whether it is significant. Let the null hypothesis be there is no linear correlation and the significance level 0.05 .

# In[ ]:


fig, axes = plt.subplots(1, 3, figsize = (24, 6))

plt.suptitle('Correlation in our numeric data',fontsize = 18, y = 1.05)

sns.regplot(x = 'price', y = 'year', data = cars, marker = '.', ci= False,
            line_kws={'color': '#0A333A'}, scatter_kws={'color':'#7FBCC6'}, ax = axes[0])
cr1 = scipy.stats.pearsonr(cars['price'], cars['year'])
axes[0].set_title(cr1, pad = 20)

sns.regplot(x = 'price', y = 'mileage', data = cars, marker = '.', ci= False,
            line_kws={'color': '#0A333A'}, scatter_kws={'color':'#7FBCC6'}, ax = axes[1])
cr2 = scipy.stats.pearsonr(cars['price'], cars['mileage'])
axes[1].set_title(cr2, pad = 20)

sns.regplot(x = 'year', y = 'mileage', data = cars, marker = '.', ci= False,
            line_kws={'color': '#0A333A'}, scatter_kws={'color':'#7FBCC6'}, ax = axes[2])
cr3 = scipy.stats.pearsonr(cars['year'], cars['mileage'])
axes[2].set_title(cr3, pad = 20)


plt.show()


# Year vs Price: The greater the price, the newer a car. The correlation is relatively weak.
# 
# Mileage vs Price: The greater the price, the lower the mileage. The correlation is relatively weak.
# 
# Mileage vs Year: The newer the car, the lower the mileage. The correlation is moderately strong.

# ### Subdivision by brand

# #### Most expensive brand
# Firstly, let's see how expensive a brand, if we use the average prices. Before the task, we should clean our dataset froum outliers. I will keep a brand only, if it has more than 10 items. In this case I can work with more than 98% of the data, and I have to work only with 48% of the brands. 

# In[ ]:


temp = pd.DataFrame(cars.groupby(['brand']).count()['vin'])
temp.sort_values('vin', ascending = False, inplace = True)
# temp[temp['vin'] > 10].sum().values / temp.sum().values == 0.98239588
# temp[temp['vin'] > 10].count().values / temp.count().values == 0.48148148
brand_list = temp[temp['vin'] > 10].index.values


# In[ ]:


av_prices = []
for i in brand_list:
    x = cars[cars['brand']==i]
    av_price = sum(x.price)/len(x)
    av_prices.append(av_price)
data = pd.DataFrame({'brand_list': brand_list,'av_prices':av_prices})
new_index = (data['av_prices'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

sns.barplot(y=sorted_data['brand_list'], x=sorted_data['av_prices'], palette = 'GnBu_d')
plt.xlabel('Average Price ($)', fontsize = 14)
plt.ylabel('Brand', fontsize = 14)
plt.title('Average price per brand', fontsize = 16)
plt.show()


# #### Most frequent brand
# Let's work some more with our filtered data.

# In[ ]:


counts = []
for i in brand_list:
    x = cars[cars['brand']==i]
    count = len(x.vin)
    counts.append(count)
data2 = pd.DataFrame({'brand_list': brand_list,'counts':counts})
new_index2 = (data2['counts'].sort_values(ascending=False)).index.values
sorted_data2 = data2.reindex(new_index2)

sns.barplot(y=sorted_data2['brand_list'], x=sorted_data2['counts'], palette = 'GnBu_d')
plt.xlabel('# of brands', fontsize = 14)
plt.ylabel('Brand', fontsize = 14)
plt.title('Number of brands', fontsize = 16)
plt.show()


# As we van see, 90% of the cars belongs to the top 4 brand. We could examine that, is there any connection between the price, and the frequency?

# #### Price and frequency
# Let the null hypothesis be there is no linear correlation and the significance level 0.05

# In[ ]:


pf = sorted_data.merge(sorted_data2, on = 'brand_list')
sns.lmplot(x = 'av_prices', y = 'counts', data = pf,
          line_kws={'color': '#0A333A'}, scatter_kws={'color':'#7FBCC6'})
plt.show()
scipy.stats.pearsonr(pf['av_prices'], pf['counts'])


# The correlation coefficient is 0.34, which could mean a weak positive correlation, but the p-value, which is 0.255 is greater than our significance level, so we can't refuse the null hypothesis.

# #### Model in brand
# The next figure shows the brand and price further broken down by the model attribute. After an eyeball test, it seems the model can be a significant variable. It could be interesting to repeat the tests so far with the model instead of brand. Also it could be interesting to test with both attributes applied, but there would be a lot of rare data, so I won't do it now.

# In[ ]:


cars['count'] = 1
brand_list = temp[temp['vin'] > 20].index.values
cars_b = cars[np.in1d(cars['brand'],brand_list)]
c_sun = px.sunburst(cars_b, path = ['brand','model'], values = 'count', color = 'price', 
            width = 750, height = 750, color_continuous_scale = 'Teal')
cars.drop(columns = 'count', inplace = True)
c_sun.show()


# ### Cars by state
# Firstly, we should add a new column to our cars table, which contains the state codes.

# In[ ]:


page_url = 'https://www.infoplease.com/us/postal-information/state-abbreviations-and-state-postal-codes'
uClient = uReq(page_url)
page_soup = soup(uClient.read(), "html.parser")
uClient.close()

containers = page_soup.tbody.findAll('tr')


# In[ ]:


out_filename = 'state_code.csv'
headers = 'state,code\n'

f = open(out_filename, "w")
f.write(headers)

for container in containers:
    cont = container.findAll('td')
    
    state = cont[0].text
    code = cont[2].text

    f.write(state.strip().lower() + ',' + code + '\n')
    
f.close()


# In[ ]:


sc = pd.read_csv('state_code.csv')
cars = cars.merge(sc, on = 'state')
cars.head()


# In[ ]:


carsc = pd.DataFrame(cars.groupby(['code']).count()['vin'])

fig = go.Figure(data=go.Choropleth(locations=carsc.index, z = carsc['vin'],
                                   locationmode = 'USA-states', colorscale = 'Teal',
                                   colorbar_title = '# of cars'))

fig.update_layout(title_text = 'Cars advertised by state', geo_scope='usa')

fig.show()


# The most cars are in Pennsylvania, Florida and Texas. We worked a lot with prices so far, so we could make a map about the average prices by states. On that map I will keep only those states, where there are at least 10 cars to filter out the outlier values.

# In[ ]:


states = carsc[carsc['vin'] >= 10].index.values

carsp = cars[np.in1d(cars['code'],states)]
carsp = pd.DataFrame(carsp.groupby(['code']).mean()['price'])


fig = go.Figure(data=go.Choropleth(locations=carsp.index, z = carsp['price'],
                                   locationmode = 'USA-states', colorscale = 'Teal',
                                   colorbar_title = 'USD'))

fig.update_layout(title_text = 'Average orice of the cars by states', geo_scope='usa')

fig.show()

