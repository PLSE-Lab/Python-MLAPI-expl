#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### 1. Load the dataset.
# The original data was saved with an index, so specify this when loading.

# In[ ]:


data = pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv', index_col=[0])
data.tail()


# ### 2. Rename the columns.
# An Exchange Rate is just a price - the price (in local currency) for 1 unit of foreign currency.  For example, on 2000-01-03 the price of USD1 was AUD1.5172, ie. it cost an Australian AUD1.5172 to buy one unit of US currency.  All of the prices in the dataset have been expressed this way, ie. the cost (in local) to buy 1 US dollar.  Consequently, we can simplify the column headings to be the 3-character ISO curreny code.

# In[ ]:


data.rename({'Time Serie':'COBDate',
           'AUSTRALIA - AUSTRALIAN DOLLAR/US$':'AUD',
           'EURO AREA - EURO/US$':'EUR',
           'NEW ZEALAND - NEW ZELAND DOLLAR/US$':'NZD',
           'UNITED KINGDOM - UNITED KINGDOM POUND/US$':'GBP',
           'BRAZIL - REAL/US$':'BRL',
           'CANADA - CANADIAN DOLLAR/US$':'CAD',
           'CHINA - YUAN/US$':'CNY',
           'HONG KONG - HONG KONG DOLLAR/US$':'HKD',
           'INDIA - INDIAN RUPEE/US$':'INR',
           'KOREA - WON/US$':'KRW',
           'MEXICO - MEXICAN PESO/US$':'MXN',
           'SOUTH AFRICA - RAND/US$':'ZAR',
           'SINGAPORE - SINGAPORE DOLLAR/US$':'SGD',
           'DENMARK - DANISH KRONE/US$':'DKK',
           'JAPAN - YEN/US$':'JPY',
           'MALAYSIA - RINGGIT/US$': 'MYR',
           'NORWAY - NORWEGIAN KRONE/US$':'NOK',
           'SWEDEN - KRONA/US$':'SEK',
           'SRI LANKA - SRI LANKAN RUPEE/US$':'LKR',
           'SWITZERLAND - FRANC/US$':'CHF',
           'TAIWAN - NEW TAIWAN DOLLAR/US$':'TWD',
           'THAILAND - BAHT/US$':'THB',
          }, axis='columns', inplace=True)
data.head()


# ### 3. Remove rows with missing data and adjust datatypes.
# This dataset contains an 'ND' string to indicate missing data, so ensure those rows are removed as well.  Given 'ND' was present, all numeric values will have been cast as strings ('object' in pandas) so this needs to be corrected.

# In[ ]:


print(data.shape)

data.dropna(inplace=True)
data = data[~data.eq('ND').any(1)]

print(data.shape)

print(data.dtypes)

for column in data.columns:
    if column == 'COBDate':
        data[column] = data[column].astype('datetime64')
    else:
        data[column] = data[column].astype('float64')
        
print(data.dtypes)


# ### 4. Reduce the dataset to the areas of interest.
# For the task at hand I'm onky going to look at the currencies of the USA's top 4 trading partners, as revealed by the US Census Bureau Statistics (https://www.census.gov/foreign-trade/statistics/highlights/top/index.html)' ie. CAD, MXN, CNY and JPY.  I will be looking to see if three of these exchange rates (MXN' CNY and JPY) can predicted the exchange rate of its major trading partner, CAD.
# 
# Additionally, I will restrict the dataset to the most recent calendar year (2019) so that the results cover the most recent time period.

# In[ ]:


predictors = ['MXN','CNY','JPY']
predicted  = 'CAD'

## Uncomment the below statement if analysing all (non-predicted) currencies.
#predictors = [ 'AUD', 'EUR', 'NZD', 'GBP', 'BRL', 'CNY', 'HKD',
#       'INR', 'KRW', 'MXN', 'ZAR', 'SGD', 'DKK', 'JPY', 'MYR', 'NOK', 'SEK',
#       'LKR', 'CHF', 'TWD', 'THB']

data = data[data.COBDate >= '2019-01-01'].reset_index(drop=True)
print(data.shape)
data = pd.concat([data[predictors], data[predicted]], axis='columns').reset_index(drop=True)
data.head()


# ### 5. Plot the data.
# 
# We can visualize the data by plotting individual predictors against the predicted currency and see if there are any discernible patterns.

# In[ ]:


import matplotlib.pyplot as plt

def chart_it(x,y):
    plt.scatter(x, y)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.title('Relationship between ' + y.name + ' and ' + x.name)
    plt.show()

[chart_it(data[column], data[predicted]) for column in predictors];


# ### 6. Perform Principal Component Analysis (PCA) to investigate the possibility of reducing the number of predictor dimensions.
# We can use a Scree Plot and observe the 'elbow' to see where we may cut-off our dimensions. 

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

scaled = MinMaxScaler().fit_transform(data[predictors])
n = scaled.shape[1]
pca = PCA(n_components = n)
pca.fit(scaled)

x = list(range(1,n+1))
y = list(pca.explained_variance_ratio_)
plt.plot(x, y, 'o-', linewidth=1)
plt.xticks(x, x)
plt.xlabel('Principal Component')
plt.yticks(np.arange(0,1.1,step=0.1), np.arange(0,110,step=10))
plt.ylabel('Percentage of Variance Explained')
plt.title('Scree Plot')
plt.show()


# ### 7. Examine Projected Predictors' relationships with predicted variable.
# SciKit-Learn's X_new = pca.fit_transform(X) method projects (the higher dimension data) X onto the lower-dimensional space (X_new).  By plotting each projection in X_new against the predicted variable, we can examine the suitability of each projection (a.k.a. Principal Component) against the regressor variable.

# In[ ]:


def chart_it(x,y, chart_type='scatter'):
    plt.scatter(x, y)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.title('Relationship between ' + y.name + ' and ' + x.name)
    plt.xticks(np.arange(-1, 1.2, step=0.2))
    plt.show()

projections = pca.fit_transform(scaled)
for i in range(projections.shape[1]):
    data['PC' + str(i+1)] = projections[:,i]
    chart_it(data['PC' + str(i+1)], data[predicted])

