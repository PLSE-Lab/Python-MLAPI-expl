#!/usr/bin/env python
# coding: utf-8

# 
# ### Acknowledgements
# 
#  * **D C Aichara**:  https://www.kaggle.com/dcaichara/feature-engineering-and-lightgbm
#  * **Dan Ofer**: https://www.kaggle.com/danofer/baseline-feature-engineering-geotab-69-5-lb
#  * **Fatih Bilgin**: https://www.kaggle.com/fatihbilgin/data-visualization-and-eda-for-geotab-bigquery
#  * **Leonardo Ferreira**: https://www.kaggle.com/kabure/insightful-eda-modeling-lgbm-hyperopt
#  * **John Miller**: https://www.kaggle.com/jpmiller/eda-to-break-through-rmse-68
#  * **Bojan Tunguz**: https://www.kaggle.com/tunguz/adversarial-geotab
#  * **Bruno Gorresen Mello**: https://www.kaggle.com/bgmello/how-one-percentile-affect-the-others
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import datetime as dt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[ ]:


plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
pd.set_option('display.max_columns', 99)
start = dt.datetime.now()


# In[ ]:


validation_splits = pd.DataFrame([
    ['Atlanta', 33.791, 33.835],
    ['Boston', 42.361, 42.383],
    ['Chicago', 41.921, 41.974],
    ['Philadelphia', 39.999, 40.046],
], columns=['City', 'l1', 'l2'])

direction_encoding = {
    'N': 0,
    'NE': 1 / 4,
    'E': 1 / 2,
    'SE': 3 / 4,
    'S': 1,
    'SW': 5 / 4,
    'W': 3 / 2,
    'NW': 7 / 4
}

road_encoding = {
    'Road': 1,
    'Street': 2,
    'Avenue': 2,
    'Drive': 3,
    'Broad': 3,
    'Boulevard': 4
}
monthly_rainfall = {
    'Atlanta1': 5.02, 'Atlanta5': 3.95, 'Atlanta6': 3.63, 'Atlanta7': 5.12,
    'Atlanta8': 3.67, 'Atlanta9': 4.09, 'Atlanta10': 3.11, 'Atlanta11': 4.10,
    'Atlanta12': 3.82, 'Boston1': 3.92, 'Boston5': 3.24, 'Boston6': 3.22,
    'Boston7': 3.06, 'Boston8': 3.37, 'Boston9': 3.47, 'Boston10': 3.79,
    'Boston11': 3.98, 'Boston12': 3.73, 'Chicago1': 1.75, 'Chicago5': 3.38,
    'Chicago6': 3.63, 'Chicago7': 3.51, 'Chicago8': 4.62, 'Chicago9': 3.27,
    'Chicago10': 2.71, 'Chicago11': 3.01, 'Chicago12': 2.43,
    'Philadelphia1': 3.52, 'Philadelphia5': 3.88, 'Philadelphia6': 3.29,
    'Philadelphia7': 4.39, 'Philadelphia8': 3.82, 'Philadelphia9': 3.88,
    'Philadelphia10': 2.75, 'Philadelphia11': 3.16, 'Philadelphia12': 3.31
}
monthly_temperature = {
    'Atlanta1': 43, 'Atlanta5': 69, 'Atlanta6': 76, 'Atlanta7': 79,
    'Atlanta8': 78, 'Atlanta9': 73, 'Atlanta10': 62, 'Atlanta11': 53,
    'Atlanta12': 45, 'Boston1': 30, 'Boston5': 59, 'Boston6': 68, 'Boston7': 74,
    'Boston8': 73, 'Boston9': 66, 'Boston10': 55, 'Boston11': 45,
    'Boston12': 35, 'Chicago1': 27, 'Chicago5': 60, 'Chicago6': 70,
    'Chicago7': 76, 'Chicago8': 76, 'Chicago9': 68, 'Chicago10': 56,
    'Chicago11': 45, 'Chicago12': 32, 'Philadelphia1': 35, 'Philadelphia5': 66,
    'Philadelphia6': 76, 'Philadelphia7': 81, 'Philadelphia8': 79,
    'Philadelphia9': 72, 'Philadelphia10': 60, 'Philadelphia11': 49,
    'Philadelphia12': 40}


# ### Read and combine train and test
# 
# This way we can make sure to apply the same transformations to train and test.

# In[ ]:


train = pd.read_csv(
    '../input/bigquery-geotab-intersection-congestion/train.csv')
test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv')
train.shape, test.shape

train['IsTrain'] = 1
test['IsTrain'] = 0
full = pd.concat([train, test], sort=True)


# In[ ]:


# Validation Groups
full = full.merge(validation_splits, on='City')
full['ValidationGroup'] = 1
full.loc[full.Latitude <= full.l1, 'ValidationGroup'] = 0
full.loc[full.Latitude > full.l2, 'ValidationGroup'] = 2
full.drop(['l1', 'l2'], axis=1, inplace=True)


# In[ ]:


cols = [c for c in test.columns if c not in ['Path']]
train.loc[train.DistanceToFirstStop_p80 > 0, cols + ['DistanceToFirstStop_p80']].head()
test[cols].head()


# ### Add a few features
# * Add areas
# * Flag missing values
# * Encode direction
# * Add City-Month-Weekend-Hour group

# In[ ]:


full['Latitude3'] = full.Latitude.round(3)
full['Longitude3'] = full.Longitude.round(3)
full['EntryStreetMissing'] = 1 * full.EntryStreetName.isna()
full['ExitStreetMissing'] = 1 * full.ExitStreetName.isna()

full['CMWH'] = full.City + '_'                + full.Month.astype(str) + '_'                + full.Weekend.astype(str) + '_'                + full.Hour.astype(str)

full.EntryHeading = full.EntryHeading.replace(direction_encoding)
full.ExitHeading = full.ExitHeading.replace(direction_encoding)
full['DiffHeading'] = full['EntryHeading'] - full['ExitHeading']


# ### Add weather features

# In[ ]:


full['city_month'] = full["City"] + full["Month"].astype(str)
full["Rainfall"] = full['city_month'].replace(monthly_rainfall)
full["Temperature"] = full['city_month'].replace(monthly_temperature)
full.drop('city_month', axis=1, inplace=True)


# ### Road encoding

# In[ ]:


def road_encode(x):
    for road in road_encoding.keys():
        if road in x:
            return road_encoding[road]
    return 0

full = full.fillna(dict(EntryStreetName='Unknown Something',
                        ExitStreetName='Unknown Something'))

full['EntryType'] = full['EntryStreetName'].apply(road_encode)
full['ExitType'] = full['ExitStreetName'].apply(road_encode)


# ### Combine city with street and intersection

# In[ ]:


full.EntryStreetName = full.City + ' ' + full.EntryStreetName
full.ExitStreetName = full.City + ' ' + full.ExitStreetName
full['Intersection'] = full.City + ' ' + full.IntersectionId.astype(str)

full['SameStreet'] = 1 * (full.EntryStreetName == full.ExitStreetName)


# ### Standardize Lat/Lon asnc calculate distance from city center

# In[ ]:


# Geolocation
for col in ['Latitude', 'Longitude']:
    scaler = StandardScaler()
    full[col] = scaler.fit_transform(full[col].values.reshape(-1, 1))

# Distance from CityCenter
full = full.merge(
    full.groupby('City')[['Latitude', 'Longitude']].mean(),
    left_on='City', right_index=True, suffixes=['', 'Dist']
)
full.LatitudeDist = (5 * np.abs(full.Latitude - full.LatitudeDist)).round(3)
full.LongitudeDist = (5 * np.abs(full.Longitude - full.LongitudeDist)).round(3)
full['CenterDistL1'] = (5 * (full.LatitudeDist + full.LongitudeDist)).round(3)
full['CenterDistL2'] = (3 * np.sqrt(
    (full.LatitudeDist ** 2 + full.LongitudeDist ** 2))).round(3)


# ### Use frequency encoding

# In[ ]:


def add_frequency(df, column):
    cnt = df.groupby(column)[['RowId']].count()
    cnt.loc[cnt.RowId > 10, 'RowId'] = 10 * (
            cnt.loc[cnt.RowId > 10, 'RowId'] // 10)
    cnt.columns = [f'{column}Count']
    return df.merge(cnt, left_on=column, right_index=True)

full = add_frequency(full, 'Longitude3')
full = add_frequency(full, 'Latitude3')
full = add_frequency(full, 'ExitStreetName')
full = add_frequency(full, 'EntryStreetName')
full = add_frequency(full, 'Intersection')
full = add_frequency(full, 'Path')

# Frequency Encoding with unique intersections
def add_unique_intersections(df, column):
    cnt = df.groupby(column)[['Intersection']].nunique()
    cnt.loc[cnt.Intersection > 10, 'Intersection'] = 5 * (
            cnt.loc[cnt.Intersection > 10, 'Intersection'] // 5)
    cnt.columns = [f'{column}UniqueIntersections']
    return df.merge(cnt, left_on=column, right_index=True)

full = add_unique_intersections(full, 'Longitude3')
full = add_unique_intersections(full, 'Latitude3')
full = add_unique_intersections(full, 'ExitStreetName')
full = add_unique_intersections(full, 'EntryStreetName')


# ### Apply LabelEncoder on categorical features

# In[ ]:


columns_to_encode = [
    'City',
    'EntryStreetName',
    'ExitStreetName',
    'Intersection',
    'CMWH'
]
for c in columns_to_encode:
    encoder = LabelEncoder()
    full[c] = encoder.fit_transform(full[c])


# ### Save results

# In[ ]:


full.to_csv('features_v3.csv.gz', compression='gzip', index=False)


# ### Check feature stats

# In[ ]:


train = full[full.IsTrain == 1].copy()
test = full[full.IsTrain == 0].copy()

column_stats = pd.concat([
    pd.DataFrame(full.count()).rename(columns={0: 'cnt'}),
    pd.DataFrame(train.count()).rename(columns={0: 'train_cnt'}),
    pd.DataFrame(test.count()).rename(columns={0: 'test_cnt'}),
    pd.DataFrame(full.nunique()).rename(columns={0: 'unique'}),
    pd.DataFrame(train.nunique()).rename(columns={0: 'train_unique'}),
    pd.DataFrame(test.nunique()).rename(columns={0: 'test_unique'}),
], sort=True, axis=1)
column_stats['seen_in_train%'] = (
            100 * column_stats.train_unique / column_stats.unique).round(1)
column_stats = column_stats.sort_values(by='unique')
column_stats.to_csv('col_stats.csv')
column_stats


# In[ ]:


end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))

