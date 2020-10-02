# %% [code]
# %% [code]
from datetime import datetime
import networkx as nx
import osmnx as ox
import requests
import matplotlib.cm as cm
import matplotlib.colors as colors
%matplotlib inline
ox.config(use_cache=True, log_console=True)
ox.__version__
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
import dask.dataframe as dd
import os
from tqdm import tqdm

# %% [code]


# %% [code]

TRAIN_PATH = '/kaggle/input/new-york-city-taxi-fare-prediction/train.csv'
#%%time
# Assume we only know that the csv file is somehow large, but not the exact size
# we want to know the exact number of rows

# Method 1, using file.readlines. Takes about 20 seconds.
with open(TRAIN_PATH) as file:
    n_rows = len(file.readlines())

print (f'Exact number of rows: {n_rows}')

# %% [code]
from tqdm import tqdm
chunksize = 5_000_000
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())
df_list = [] # list to hold the batch dataframe

for df_chunk in tqdm(pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize)):
     
    # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
    # Using parse_dates would be much slower!
    df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
    df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    df_list.append(df_chunk) 
train_df = pd.concat(df_list)
    

# %% [code]
nyc_taxi_fare=train_df.reset_index()
nyc_taxi_fare.to_feather('nyc_taxi_data_raw.feather')
# Delete the dataframe list to release memory
del df_list

# See what we have loaded

# %% [code]
nyc_taxi_fare = pd.read_feather('/kaggle/output/kaggle/working/nyc_taxi_data_raw.feather')#,nrows = 6000, parse_dates=["pickup_datetime"])

# %% [code]
nyc_taxi_fare.shape

# %% [code]
display(nyc_taxi_fare.head())
display(nyc_taxi_fare.tail())

# %% [code]
display(nyc_taxi_fare.passenger_count.value_counts())
#display(nyc_taxi_fare.passenger_count.unique())

# %% [code]
nyc_taxi_fare.describe()

# %% [code]
nyc_taxi_fare['year'] = pd.DatetimeIndex(nyc_taxi_fare['pickup_datetime']).year

# %% [code]
nyc_points=nyc_taxi_fare[(nyc_taxi_fare.pickup_latitude!=0)&(nyc_taxi_fare.pickup_latitude<41)&(nyc_taxi_fare.pickup_latitude>40.4)&(nyc_taxi_fare.dropoff_latitude>40.4)&
                         (nyc_taxi_fare.dropoff_longitude<-73)&(nyc_taxi_fare.pickup_longitude<-73)&(nyc_taxi_fare.year==2014)&(nyc_taxi_fare.fare_amount>0)]

# %% [code]
nyc_points.shape

# %% [code]
points= nyc_points.apply(lambda row: Point(row.pickup_longitude,row.pickup_latitude),axis=1)
# turn the file into a geodataframe
nyc_points1=gpd.GeoDataFrame(nyc_points,geometry=points)

# %% [code]
nyc_points1.shape

# %% [code]
import os
# os.chdir(r'/kaggle/working')

# nyc_points1.to_csv(r'nyc_points1_csv.csv')

# from IPython.display import FileLink
# FileLink(r'nyc_points1_csv.csv')

# %% [code]
# Imprt file to df
# n = pd.read_csv("/kaggle/working/nyc_points1_csv.csv")#nrows = 100_000 parse_dates=["pickup_datetime"]
# n.head()

# %% [code]
nyc_taxi_fare.shape

# %% [code]
nyc_points1.head()

# %% [code]
# put a coordinates map to the file 
nyc_points1.crs={'init':'epsg:4326'}
nyc_points1.head()

# %% [code]
display(nyc_points1.crs)
display(nyc_points1.head())
display(nyc_points1.shape)

# %% [code]
nyc_points1['year'] = pd.DatetimeIndex(nyc_points1['pickup_datetime']).year
nyc_points1['month'] = pd.DatetimeIndex(nyc_points1['pickup_datetime']).month
nyc_points1['hour'] = pd.DatetimeIndex(nyc_points1['pickup_datetime']).hour
nyc_points1['date'] = pd.DatetimeIndex(nyc_points1['pickup_datetime']).date
nyc_points1['week'] = pd.DatetimeIndex(nyc_points1['pickup_datetime']).week
nyc_points1['dayofweek'] = pd.DatetimeIndex(nyc_points1['date']).dayofweek

# %% [code]
nyc_points1.shape

# %% [code]
nyc_points1.BoroName1.value_counts()

# %% [code]
n_a.BoroName.value_counts()

# %% [code]
nyc_points1.crs

# %% [code]

#train_df.to_feather('nyc_taxi_data_raw.feather')

# %% [code]
nyc_points1.groupby('month')['fare_amount'].count().plot.bar(color='YELLOW')
plt.style.use("fivethirtyeight")
plt.title("Histogram of fare_amount by month")
plt.xlabel("month")
plt.ylabel("fare_amount")

# %% [code]
from matplotlib import pyplot as plt
rate1=nyc_points1[(nyc_points1.dayofweek==0)]
rate2=nyc_points1[(nyc_points1.dayofweek==1)]
rate3=nyc_points1[(nyc_points1.dayofweek==2)]
rate4=nyc_points1[(nyc_points1.dayofweek==3)]
rate5=nyc_points1[(nyc_points1.dayofweek==4)]
rate6=nyc_points1[(nyc_points1.dayofweek==5)]
rate7=nyc_points1[(nyc_points1.dayofweek==6)]
#credit_rank['e']=round(credit_rank['dev*limit_bal_']/100000,0)
a=rate1.groupby('hour')['fare_amount'].count().plot()
b=rate2.groupby('hour')['fare_amount'].count().plot()
c=rate3.groupby('hour')['fare_amount'].count().plot()
d=rate4.groupby('hour')['fare_amount'].count().plot()
e=rate5.groupby('hour')['fare_amount'].count().plot()
f=rate6.groupby('hour')['fare_amount'].count().plot()
g=rate7.groupby('hour')['fare_amount'].count().plot()
#(credit_rank.groupby('dev*limit_bal_')['default'].sum() / credit_rank.groupby('dev*limit_bal_')['default'].count()).plot()
plt.style.use('bmh')
plt.title("Count of taxi per hour by day")
plt.xlabel("hour")
plt.ylabel("Count")
plt.legend(["0","1","2","3","4","5","6"],loc=4)

# %% [code]
from matplotlib import pyplot as plt
rate1=nyc_points1[(nyc_points1.dayofweek==0)]
rate2=nyc_points1[(nyc_points1.dayofweek==1)]
rate3=nyc_points1[(nyc_points1.dayofweek==2)]
rate4=nyc_points1[(nyc_points1.dayofweek==3)]
rate5=nyc_points1[(nyc_points1.dayofweek==4)]
rate6=nyc_points1[(nyc_points1.dayofweek==5)]
rate7=nyc_points1[(nyc_points1.dayofweek==6)]
#credit_rank['e']=round(credit_rank['dev*limit_bal_']/100000,0)
a=rate1.groupby('hour')['fare_amount'].mean().plot()
b=rate2.groupby('hour')['fare_amount'].mean().plot()
c=rate3.groupby('hour')['fare_amount'].mean().plot()
d=rate4.groupby('hour')['fare_amount'].mean().plot()
e=rate5.groupby('hour')['fare_amount'].mean().plot()
f=rate6.groupby('hour')['fare_amount'].mean().plot()
g=rate7.groupby('hour')['fare_amount'].mean().plot()
#(credit_rank.groupby('dev*limit_bal_')['default'].sum() / credit_rank.groupby('dev*limit_bal_')['default'].count()).plot()
plt.style.use('bmh')
plt.title("Mean of taxi per hour by day")
plt.xlabel("hour")
plt.ylabel("Mean price")
plt.legend(["0","1","2","3","4","5","6"],loc=1)

# %% [code]
nyc_points1['isWeekend'] = nyc_points1['dayofweek']

#Weekday name using dictionary
isWeekend_dict = {0:0,1:0,2:0,3:0,4:0,5:1,6:1}

nyc_points1.isWeekend.replace(isWeekend_dict,inplace=True)

# %% [code]
from matplotlib import pyplot as plt
rate1=nyc_points1[(nyc_points1.dayofweek==0)]
rate2=nyc_points1[(nyc_points1.dayofweek==1)]
rate3=nyc_points1[(nyc_points1.dayofweek==2)]
rate4=nyc_points1[(nyc_points1.dayofweek==3)]
rate5=nyc_points1[(nyc_points1.dayofweek==4)]
rate6=nyc_points1[(nyc_points1.dayofweek==5)]
rate7=nyc_points1[(nyc_points1.dayofweek==6)]
#credit_rank['e']=round(credit_rank['dev*limit_bal_']/100000,0)
a=rate1.groupby('month')['fare_amount'].mean().plot()
b=rate2.groupby('month')['fare_amount'].mean().plot()
c=rate3.groupby('month')['fare_amount'].mean().plot()
d=rate4.groupby('month')['fare_amount'].mean().plot()
e=rate5.groupby('month')['fare_amount'].mean().plot()
f=rate6.groupby('month')['fare_amount'].mean().plot()
g=rate7.groupby('month')['fare_amount'].mean().plot()
#(credit_rank.groupby('dev*limit_bal_')['default'].sum() / credit_rank.groupby('dev*limit_bal_')['default'].count()).plot()
plt.style.use('bmh')
plt.title("Mean of taxi per month by day")
plt.xlabel("month")
plt.ylabel("Mean price")
plt.legend(["0","1","2","3","4","5","6"],loc=4)

# %% [code]
#nyc_points1.groupby(['isWeekend','hour'])['fare_amount'].agg(['mean'])
pd.crosstab(nyc_points1.hour,nyc_points1.dayofweek,values=nyc_points1.fare_amount,aggfunc='mean').round(0)
pd.crosstab(nyc_points1.hour,nyc_points1.month,values=nyc_points1.fare_amount,aggfunc='mean').round(0)
#ddd=pd.crosstab(df4.bill_pay_percentile,df4.default)

# %% [code]
display(nyc_points1.isWeekend.value_counts())
display(nyc_points1.dayofweek.value_counts())

# %% [code]
# pip install arcgis
# pip install geopandas
# pip install Rtree
# pip install osmnx
# pip install geopy

# %% [code]
#points= t.apply(lambda row: Point(row.pickup_longitude,row.pickup_latitude),axis=1)
# nyc_points=gpd.GeoDataFrame(t(nrows = 6000),geometry=points)
# nyc_points.crs={'init':'epsg:4326'}
# nyc_points.head()

# %% [code]
nyc_points2=nyc_points1.head(2_000_000)

# %% [code]
nyc_points2.head()

# %% [code]
nyc_points2.crs

# %% [code]
# 1-  Listing points
#listings = pd.read_csv(“data/stockholm_listings.csv”)
# 2 - convert to Geopandas Geodataframe
#gdf_nyc_taxi_fare = gpd.GeoDataFrame(nyc_taxi_fare,   geometry=gpd.points_from_xy(nyc_taxi_fare.pickup_longitude, nyc_taxi_fare.pickup_latitude))

# %% [code]
import geoplot
ny = gpd.read_file(gpd.datasets.get_path('nybb'))
ny.crs

# %% [code]
ny.head()

# %% [code]
ny=ny.to_crs(crs=nyc_points1.crs)
ny.plot(column="BoroName")

# %% [code]
nyc_points2['new_col1'] = range(0,len(nyc_points2))
nyc_points2.index=nyc_points2['new_col1']

# %% [code]
nyc_points2.head()

# %% [code]
# you have to run n1-n4 every data alone because its very heavy
n1=nyc_points2[nyc_points2.index<500001]
taxi1 = gpd.sjoin(n1,ny,how='inner',op='within')

n2=nyc_points2[(nyc_points2.index<1000001)&(nyc_points2.index>500000)]
taxi2 = gpd.sjoin(n2,ny,how='inner',op='within')

n3=nyc_points2[(nyc_points2.index<1500001)&(nyc_points2.index>1000000)]
taxi3 = gpd.sjoin(n3,ny,how='inner',op='within')

n4=nyc_points2[(nyc_points2.index<2000001)&(nyc_points2.index>1500000)]
taxi4 = gpd.sjoin(n4,ny,how='inner',op='within')

# %% [code]
taxi4.head()

# %% [code]
n_a=pd.concat([taxi1,taxi2,taxi3,taxi4],axis=0)

import os
os.chdir(r'/kaggle/working')

n_a.to_csv(r'taxi_pol.csv')

# from IPython.display import FileLink
# FileLink(r'taxi_pol.csv')

# %% [code]
print(n_a.shape)

# %% [code]
table1=n_a.groupby(['month','dayofweek','hour','BoroName'])['fare_amount'].count().to_frame(name='count').reset_index()
table1.head(100000)
table1.to_csv('/kaggle/input/agregates/table1_csv.csv') # relative position 
#dt.to_csv('C:/Users/abc/Desktop/file_name.csv')

# %% [code]

points= nyc_points.apply(lambda row: Point(row.dropoff_longitude,row.dropoff_latitude),axis=1)
# turn the file into a geodataframe
nyc_points1=gpd.GeoDataFrame(nyc_points,geometry=points)

# %% [code]
nyc_points1.shape

# %% [code]
import os
# os.chdir(r'/kaggle/working')

# nyc_points1.to_csv(r'nyc_points1_csv.csv')

# from IPython.display import FileLink
# FileLink(r'nyc_points1_csv.csv')

# %% [code]
# Imprt file to df
# n = pd.read_csv("/kaggle/working/nyc_points1_csv.csv")#nrows = 100_000 parse_dates=["pickup_datetime"]
# n.head()

# %% [code]
nyc_taxi_fare.shape

# %% [code]
nyc_points1.head()

# %% [code]
# put a coordinates map to the file 
nyc_points1.crs={'init':'epsg:4326'}
nyc_points1.head()

# %% [code]
display(nyc_points1.crs)
display(nyc_points1.head())
display(nyc_points1.shape)



# %% [code]
nyc_points1.shape

# %% [code]
nyc_points1.BoroName1.value_counts()

# %% [code]
n_a.BoroName.value_counts()

# %% [code]
nyc_points1.crs

# %% [code]

#train_df.to_feather('nyc_taxi_data_raw.feather')







# %% [code]
#nyc_points1.groupby(['isWeekend','hour'])['fare_amount'].agg(['mean'])
pd.crosstab(nyc_points1.hour,nyc_points1.dayofweek,values=nyc_points1.fare_amount,aggfunc='mean').round(0)
pd.crosstab(nyc_points1.hour,nyc_points1.month,values=nyc_points1.fare_amount,aggfunc='mean').round(0)
#ddd=pd.crosstab(df4.bill_pay_percentile,df4.default)

# %% [code]
display(nyc_points1.isWeekend.value_counts())
display(nyc_points1.dayofweek.value_counts())

# %% [code]
# pip install arcgis
# pip install geopandas
# pip install Rtree
# pip install osmnx
# pip install geopy

# %% [code]
#points= t.apply(lambda row: Point(row.pickup_longitude,row.pickup_latitude),axis=1)
# nyc_points=gpd.GeoDataFrame(t(nrows = 6000),geometry=points)
# nyc_points.crs={'init':'epsg:4326'}
# nyc_points.head()

# %% [code]
nyc_points2=nyc_points1.head(2_000_000)

# %% [code]
nyc_points2.head()

# %% [code]
nyc_points2.crs

# %% [code]
# 1-  Listing points
#listings = pd.read_csv(“data/stockholm_listings.csv”)
# 2 - convert to Geopandas Geodataframe
#gdf_nyc_taxi_fare = gpd.GeoDataFrame(nyc_taxi_fare,   geometry=gpd.points_from_xy(nyc_taxi_fare.pickup_longitude, nyc_taxi_fare.pickup_latitude))

# %% [code]
import geoplot
ny = gpd.read_file(gpd.datasets.get_path('nybb'))
ny.crs

# %% [code]
ny.head()

# %% [code]
ny=ny.to_crs(crs=nyc_points1.crs)
ny.plot(column="BoroName")

# %% [code]
nyc_points2['new_col1'] = range(0,len(nyc_points2))
nyc_points2.index=nyc_points2['new_col1']

# %% [code]
nyc_points2.head()

# %% [code]
# you have to run n1-n4 every data alone because its very heavy
n1=nyc_points2[nyc_points2.index<500001]
taxi1 = gpd.sjoin(n1,ny,how='inner',op='within')

n2=nyc_points2[(nyc_points2.index<1000001)&(nyc_points2.index>500000)]
taxi2 = gpd.sjoin(n2,ny,how='inner',op='within')

n3=nyc_points2[(nyc_points2.index<1500001)&(nyc_points2.index>1000000)]
taxi3 = gpd.sjoin(n3,ny,how='inner',op='within')

n4=nyc_points2[(nyc_points2.index<2000001)&(nyc_points2.index>1500000)]
taxi4 = gpd.sjoin(n4,ny,how='inner',op='within')

# %% [code]
taxi4.head()

# %% [code]
n_a=pd.concat([taxi1,taxi2,taxi3,taxi4],axis=0)

import os
os.chdir(r'/kaggle/working')

n_a.to_csv(r'taxi_pol2.csv')

# from IPython.display import FileLink
# FileLink(r'taxi_pol.csv')

# %% [code]
print(n_a.shape)

# %% [code]
table2=n_a.groupby(['month','dayofweek','hour','BoroName'])['fare_amount'].count().to_frame(name='count').reset_index()
table2.head(100000)
table2.to_csv('/kaggle/input/agregates/table2_csv.csv') # relative position
# %% [code]

