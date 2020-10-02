#!/usr/bin/env python
# coding: utf-8

# ## Missing hours in weather data
# Exploring the data a bit you can see that there are several missing timestamps on the given weather data files (air temperatures, etc)
# 
# I explore here few ways to fill those gaps using interpolation per site (weather station)
# 
# This kernels should provide an easy way to change your pipeline to use these files, so it outputs the same file structure (gzipped) and you only need to change the data source of your kernel to use the cleaned up data.

# In[ ]:


# !pip install --upgrade numpy==1.17.3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, gc
import random
import datetime

from tqdm import tqdm_notebook as tqdm

# matplotlib and seaborn for plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_squared_log_error


# In[ ]:


path = '../input/ashrae-energy-prediction'
# Input data files are available in the "../input/" directory.
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Check the meter averages per weekday

# In[ ]:


building = pd.read_csv(f'{path}/building_metadata.csv')

train = pd.read_csv(f'{path}/train.csv', parse_dates=['timestamp'])
train = train[train.meter == 0]  # electricity only
del train['meter']
train = train.merge(building, on='building_id', how='left')


# In[ ]:


ematrix = train.groupby(['site_id', 'timestamp']).meter_reading.mean().to_frame('reading').reset_index()
ematrix = ematrix.pivot(index='timestamp', columns='site_id', values='reading')
ematrix.describe()


# In[ ]:


# norm
ematrix = ematrix - ematrix.mean()
ematrix = ematrix / ematrix.std()
ematrix.describe()


# In[ ]:


ematrix.reset_index(inplace=True)
ematrix['weekday'] = ematrix.timestamp.dt.weekday
ematrix['hour'] = ematrix.timestamp.dt.hour
ematrix.set_index('timestamp', inplace=True)


# In[ ]:


hourly_stats = ematrix.groupby(['weekday', 'hour']).mean()
ax = hourly_stats.plot()
ax.figure.set_size_inches(18, 4)
_ = ax.set_title('Average readings per hour per weekday')


# In[ ]:


building.groupby('site_id').primary_use.agg(lambda x:x.value_counts()[:3].to_dict()).to_dict()


# # Check missing temperature data points

# In[ ]:


#load training data 2016
weather = pd.read_csv(f'{path}/weather_train.csv', parse_dates=['timestamp'])
# pivot to plot
wmatrix = weather.pivot(index='timestamp', columns='site_id', values='air_temperature')
# site with largest amount of missing data points
site_id = wmatrix.count().idxmin()
# plot perid
start_date, end_date = datetime.date(2016, 1, 1), datetime.date(2016, 1, 9)
f,ax = plt.subplots(figsize=(18,6))
_ = wmatrix.loc[start_date:end_date, site_id].plot(ax=ax, label=f'Jan 2016 site:{site_id}')

# load test data 2017-2018
weather = pd.read_csv(f'{path}/weather_test.csv', parse_dates=['timestamp'])

# shift 2017 to 2016
weather.timestamp = weather.timestamp - datetime.timedelta(365)
wmatrix = weather.pivot(index='timestamp', columns='site_id', values='air_temperature')
_ = wmatrix.loc[start_date:end_date, site_id].plot(ax=ax, label=f'Jan 2017 site:{site_id}', alpha=0.5)

# shift 2018 to 2016
weather.timestamp = weather.timestamp - datetime.timedelta(365)
wmatrix = weather.pivot(index='timestamp', columns='site_id', values='air_temperature')
_ = wmatrix.loc[start_date:end_date, site_id].plot(ax=ax, label=f'Jan 2018 site:{site_id}', alpha=0.5)

_ = plt.legend()


# ## Some different interpolation strategies
# Cubic interpolation is generally the best option but the data is a bit too noisy for it, so 50/50% mix with linear interpolation seems better

# In[ ]:


#load training data 2016
weather = pd.read_csv(f'{path}/weather_train.csv', parse_dates=['timestamp'])
# pivot to plot
wmatrix = weather.pivot(index='timestamp', columns='site_id', values='air_temperature')
# site with largest amount of missing data points
site_id = wmatrix.count().idxmin()
# plot perid
start_date, end_date = datetime.date(2016, 1, 1), datetime.date(2016, 1, 9)
f,ax = plt.subplots(figsize=(18,6))

# load test data 2017-2018
weather_test = pd.read_csv(f'{path}/weather_test.csv', parse_dates=['timestamp'])

# shift 2017 to 2016
weather_test.timestamp = weather_test.timestamp - datetime.timedelta(365)
wtmatrix = weather_test.pivot(index='timestamp', columns='site_id', values='air_temperature')
_ = wtmatrix.loc[start_date:end_date, site_id].plot(ax=ax, label=f'Jan 2017 site:{site_id}', alpha=0.4)

# shift 2018 to 2016
weather_test.timestamp = weather_test.timestamp - datetime.timedelta(365)
wtmatrix = weather_test.pivot(index='timestamp', columns='site_id', values='air_temperature')
_ = wtmatrix.loc[start_date:end_date, site_id].plot(ax=ax, label=f'Jan 2018 site:{site_id}', alpha=0.4)


# def fill_with_avg(wmatrix, w=12):
#     return wmatrix.fillna(wmatrix.rolling(window=w, win_type='gaussian', center=True, min_periods=1).mean(std=2))

def fill_with_po3(wmatrix):
    return wmatrix.fillna(wmatrix.interpolate(method='polynomial', order=3))

def fill_with_lin(wmatrix):
    return wmatrix.fillna(wmatrix.interpolate(method='linear'))

def fill_with_mix(wmatrix):
    wmatrix = (wmatrix.fillna(wmatrix.interpolate(method='linear', limit_direction='both')) +
               wmatrix.fillna(wmatrix.interpolate(method='polynomial', order=3, limit_direction='both'))
              ) * 0.5
    # workaround: fill last NANs with neighbour
    assert wmatrix.count().min() >= len(wmatrix)-1 # only the first item is missing
    return wmatrix.fillna(wmatrix.iloc[1])         # fill with second item


_ = fill_with_lin(wmatrix).loc[start_date:end_date, site_id].plot(ax=ax, label=f'linear Jan 2016 site:{site_id}', alpha=0.5)
_ = fill_with_po3(wmatrix).loc[start_date:end_date, site_id].plot(ax=ax, label=f'cubic Jan 2016 site:{site_id}', alpha=0.5)
_ = fill_with_mix(wmatrix).loc[start_date:end_date, site_id].plot(ax=ax, label=f'mix Jan 2016 site:{site_id}', alpha=0.5)
_ = wmatrix.loc[start_date:end_date, site_id].plot(ax=ax, label=f'Jan 2016 site:{site_id}')

_ = plt.legend()


# ### same for dew_temperature

# In[ ]:


# pivot to plot
col = 'dew_temperature'
wmatrix = weather.pivot(index='timestamp', columns='site_id', values=col)
# site with largest amount of missing data points
site_id = wmatrix.count().idxmin()
# plot perid
start_date, end_date = datetime.date(2016, 1, 1), datetime.date(2016, 1, 12)
f,ax = plt.subplots(figsize=(18,6))

_ = fill_with_mix(wmatrix).loc[start_date:end_date, site_id].plot(ax=ax, label=f'mix Jan 2016 site:{site_id}', alpha=0.5)
_ = wmatrix.loc[start_date:end_date, site_id].plot(ax=ax, label=f'Jan 2016 site:{site_id}')

_ = plt.legend()


# # Save the data for later use

# In[ ]:


def fill_temps(weather):
    df = None
    for col in ['air_temperature', 'dew_temperature']:
        filled = fill_with_mix(weather.pivot(index='timestamp', columns='site_id', values=col))
        filled = filled.sort_index().unstack().to_frame(col)
        if df is None:
            df = filled
        else:
            df[col] = filled[col]
    return df


# In[ ]:


for src in ['train', 'test']:
    weather = pd.read_csv(f'{path}/weather_{src}.csv', parse_dates=['timestamp'])
    wf = fill_temps(weather)
    wf = wf.reset_index().merge(weather[['site_id', 'timestamp', 'cloud_coverage', 'precip_depth_1_hr', 'wind_direction', 'wind_speed']],
                           how='left', on=['site_id', 'timestamp']).set_index(['site_id', 'timestamp'])
    for col in ['cloud_coverage', 'precip_depth_1_hr', 'wind_direction', 'wind_speed']:
        wf.loc[wf[col] < 0, col] = 0
        wf.fillna(0, inplace=True)
    wf.to_csv(f'weather_{src}.csv.gz', compression='gzip', float_format='%g')
get_ipython().system('ls -lah *.gz')
wf.describe()


# ### Double check the results

# In[ ]:


test = pd.read_csv(f'weather_{src}.csv.gz', parse_dates=['timestamp'])
ax = test[test.site_id == 7].set_index('timestamp').groupby('site_id').air_temperature.plot()
_ = plt.legend()


# In[ ]:


ax = weather[weather.site_id == 7].set_index('timestamp').groupby('site_id').air_temperature.plot()


# In[ ]:


# buildings
from sklearn.preprocessing import LabelEncoder
building = pd.read_csv(f'{path}/building_metadata.csv')
building.primary_use = LabelEncoder().fit_transform(building.primary_use)
cols =  ['square_feet', 'year_built', 'floor_count']
building.fillna(building[cols].mean(), inplace=True)
building.to_csv(f'building_metadata.csv.gz', index=False, compression='gzip', float_format='%g')
get_ipython().system('ls -lh *.gz')
building.describe()


# # Visualise all meter readings

# ### First a bad case

# In[ ]:


train = pd.read_csv(f'{path}/train.csv', parse_dates=['timestamp'])


# In[ ]:


# pivot to plot
ematrix = train[(train.meter == 0) & (train.building_id <= 104)].                 pivot(index='timestamp', columns='building_id', values='meter_reading')
# site with largest amount of missing data points
building_id = 0 #ematrix.count().idxmin()
# plot perid
start_date, end_date = datetime.date(2016, 1, 1), datetime.date(2016, 12, 15)
f,ax = plt.subplots(figsize=(18,6))

_ = ematrix.loc[start_date:end_date, building_id].plot(ax=ax, label=f'Jan 2016 site:{building_id}')
_ = plt.legend()


# ### All meters

# In[ ]:


# Load data
train = pd.read_csv(f'{path}/train.csv')
# Plot missing values per building/meter
fig, ax = plt.subplots(1, 4, figsize=(20,30))
for meter in range(4):
    df = train[train.meter == meter]
    missmap = pd.DataFrame(index=[i for i in range(train.building_id.nunique())])
    missmap = missmap.merge(df.pivot(index='building_id', columns='timestamp', values='meter_reading'),
                            how='left', left_index=True, right_index=True)
    missmap = np.sign(missmap)  # -1, 0 or +1
    ax[meter].set_title(f'Meter {meter}')
    sns.heatmap(missmap, cmap='Paired', ax=ax[meter], cbar=False)


# ### Remove zeros from meter 0 and save data

# In[ ]:


for src in ['train', 'test', 'sample_submission']:
    df = pd.read_csv(f'{path}/{src}.csv')
    if src is 'train':
        df.drop(index=df[(df.meter_reading <= 0) &
#                          (df.timestamp <= '2016-06') &
                         (df.meter == 0)].index,
                inplace=True)
    df.to_csv(f'{src}.csv.gz', index=False, compression='gzip', float_format='%g')
get_ipython().system('ls -hl *.gz')
wf.describe()


# # PCA weather data

# In[ ]:


import matplotlib.patheffects as PathEffects

# Utility function to visualize the outputs of PCA and t-SNE
def scatter(x, colors=np.arange(0,16)):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], s=80, c=palette[colors.astype(np.int)])
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []
    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(x[colors == i], axis=0)[:2]
        txt = ax.text(xtext, ytext, str(i), fontsize=14, alpha=0.4)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


# In[ ]:


# load and norm
wmatrix = fill_with_mix(weather.pivot(index='timestamp', columns='site_id', values='air_temperature'))
wmatrix = wmatrix - wmatrix.values.mean()

# # dew temps
# dewmatrix = fill_with_mix(weather.pivot(index='timestamp', columns='site_id', values='dew_temperature'))
# dewmatrix = dewmatrix - dewmatrix.values.mean()
# # concat
# wmatrix = pd.concat([wmatrix, dewmatrix], axis=0)

X = wmatrix.values.T
X = X / np.linalg.norm(X)


# In[ ]:


# from sklearn.manifold import TSNE

# tsne_result = TSNE(random_state=42).fit_transform(X)
# _ = scatter(tsne_result)
# # useless :/


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
pd.DataFrame(pca_result, columns=['pca1', 'pca2'])


# In[ ]:


# rotate and plot (cold-up, hot-down)
_ = scatter(-pca_result[:,::-1])


# In[ ]:


# Sites sorted by mean temperature in 2016
weather[['site_id', 'air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'wind_direction', 'wind_speed']] .groupby('site_id').agg([np.mean, np.max, np.min]).sort_values(by=('air_temperature','mean'), ascending=False)


# ### Available output files

# In[ ]:


get_ipython().system('ls -hl *.gz')

