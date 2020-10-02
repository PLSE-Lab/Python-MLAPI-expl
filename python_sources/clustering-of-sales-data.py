# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
from collections import Counter
from sklearn.cluster import KMeans, DBSCAN
from pylab import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

files_data = []
for file in os.listdir("../input"):
    file_data = pd.read_csv('../input/' + file)
    files_data.append(file_data)

stores_data = pd.concat(files_data)

stores_data['Discount'] = stores_data['MRP'] - stores_data['Sales Price']
stores_data['Discount Rate'] = np.where( stores_data['MRP'] != 0, 
                    (stores_data['Discount'] * 100) / stores_data['MRP'], 0)
    
stores_data.loc[(stores_data['MRP'] == 0) & (stores_data['Sales Price'] != 0), 'MRP Anamoly'] = True
stores_data.loc[~((stores_data['MRP'] == 0) & (stores_data['Sales Price'] != 0)), 'MRP Anamoly'] = False

stores_data['Sales Anamoly'] =  stores_data['Sales Price'] > stores_data['MRP']

stores_data['Sale Date'] = stores_data['Sale Date'].astype('datetime64[ns]')
stores_data['Month'] = pd.DatetimeIndex(stores_data['Sale Date']).month
stores_data['Year'] = pd.DatetimeIndex(stores_data['Sale Date']).year


def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return 0


# print(stores_data.head())

stores_data_sep_feb = stores_data[( (stores_data['Month'].isin([9, 10, 11, 12])) & (stores_data['Year'] == 2017) ) | ( (stores_data['Month'].isin([1, 2])) & (stores_data['Year'] == 2018) )]
# print(stores_data_sep_feb)

grouped_by_brand = stores_data_sep_feb.groupby('Brand Code')
# aggregator = {'Store Code': pd.Series.nunique, 'Month': pd.Series.nunique}
# print(grouped_by_brand.agg(aggregator))

age_of_brands = (grouped_by_brand['Sale Date'].max() - grouped_by_brand['Sale Date'].min()).dt.days
X = pd.DataFrame(age_of_brands)
X.columns = ['Age of Brand']
# print(X.head())
# print(type(X))
X.loc[X['Age of Brand'] == 0, 'Age of Brand'] = 1
# age_of_brands = np.where( (grouped_by_brand['Sale Date'].max() > grouped_by_brand['Sale Date'].min()), 
#                     grouped_by_brand['Sale Date'].max() - grouped_by_brand['Sale Date'].min(), pd.Timedelta(days=1) )
# X = pd.DataFrame(age_of_brands)
# print(X)
# print(X.dtypes)
# print(stores_data_sep_feb[stores_data_sep_feb['Brand Code']=='BRAND164'])

X['Average Sales Qty per month per store'] = grouped_by_brand['Sales Qty'].sum() / \
                (grouped_by_brand['Month'].apply(pd.Series.nunique) * \
                grouped_by_brand['Store Code'].apply(pd.Series.nunique))
                
# print(X['Average Sales Qty per month per store'])

months_year = [ (9, 2017), (10, 2017), (11, 2017), (12, 2017), (1, 2018), (2, 2018) ]
averaging_cols = [ ('Sales Price', 'Avg'), ('Discount Rate', 'Wavg'), ('SKU Code', 'Uniq') ]

for month, year in months_year:
    grouped_by_brand_mon = stores_data[(stores_data['Month'] == month) & (stores_data['Year'] == year)].groupby('Brand Code')
    for col, op in averaging_cols:
        column_name = col + ' ' + str(month) + ' ' + str(year)
        if op == 'Avg':
            X[column_name] = grouped_by_brand_mon[col].mean()
        elif op == 'Wavg':
            X[column_name] = grouped_by_brand_mon.apply(wavg, col, 'Sales Qty')
        elif op == 'Uniq':
            X[column_name] = grouped_by_brand_mon[col].apply(pd.Series.nunique, col)

# print(X.iloc[21, 2])
X.fillna(0, inplace=True)
# print(X.iloc[21, 2])
print(X.head())
X_norm = (X - X.mean()) / (X.max() - X.min())
X.to_csv('X.csv')
X_norm.to_csv('X Normalized.csv')

optimal_k_value = 4
kMeans = KMeans(optimal_k_value, max_iter=1000, n_init=20).fit(X_norm)
X['Cluster'] = kMeans.labels_
X.sort_values('Cluster').to_csv('Clustered data.csv')

#inertia_vals = []
#for k in range(2, 10, 1):
#    kMeans = KMeans(k, max_iter=1000).fit(X_norm)
#    # print(kMeans.labels_)
#    X['Cluster'] = kMeans.labels_
#    inertia_vals.append(kMeans.inertia_)
#    # print(X.sort_values('Cluster'))
#    file_name = 'K Clustered Data ' + str(k) + '.csv'
#    print(file_name)
#    X.sort_values('Cluster').to_csv(file_name)


# k_vals = arange(2, 70, 2)
# plot(k_vals, inertia_vals)

# xlabel('k Value')
# ylabel('Inertia')
# grid(True)
# show()

# eps_vals = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1, 1.5, 2, 3]
# num_neighbors = [3, 5, 8, 10]
# for eps_val in eps_vals:
#    for n in num_neighbors:
#        dbscan = DBSCAN(eps=eps_val, min_samples=n).fit(X_norm)
#        # print(dbscan.labels_)
#        count = Counter(np.array(dbscan.labels_)).get(-1)
#        X['Cluster'] = dbscan.labels_
#        
#        # print(X.sort_values('Cluster'))
#        file_name = str(count) + ' Clustered Data ' + str(eps_val) + ' ' + str(n) + '.csv'
#        print(file_name)
#        X.sort_values('Cluster').to_csv(file_name)
# print(X.head())