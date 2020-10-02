#!/usr/bin/env python
# coding: utf-8

# # NYC Taxi data exploration with seaborn and datashader
# 
# For my first exploration of the NYC Taxi trip data, I focused on visualizations with `seaborn` and `datashader`. This notebook is not a full-featured exploratory data analysis but more a collection of plots I made to better understand the data.
# 
# * **Part I: Train vs test sets comparison**
#   Comparison of test and train feature distribution to understand the task at hand. 
# * **Part II: Taxi trip visualisation and analysis**
#   Quick exploration of spatial and temporal features and their relationship with taxi trip durations.
# * **Part III: Path visualization using datashader**
#   Graph visualization experiments with datashader.

# In[ ]:


import os.path

import datashader as ds
import datashader.transfer_functions as dtf
from datashader.colors import colormap_select as cm, inferno, viridis
from datashader.utils import lnglat_to_meters

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from IPython.display import display, HTML
from matplotlib.colors import LinearSegmentedColormap

# Datashader helper function:
def bg(img): return dtf.set_background(img, "black")

get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams['figure.dpi'] = 130


# ## Data preparation

# Loading train and test data into a single dataframe:

# In[ ]:


def load_dataset(train_path, test_path):
    
    train = pd.read_csv(train_path)    
    test = pd.read_csv(test_path)
    
    df = pd.concat({'train': train, 'test': test}, ignore_index=False, names=['set']).reset_index(0)
        
    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
    df['dropoff_datetime'] = pd.to_datetime(df.dropoff_datetime)
    
    assert len(df) == len(test) + len(train)
    
    return df


# In[ ]:


dataset_path = '/kaggle/input/' if os.path.exists('/kaggle/input/') else './data/'

df = load_dataset(train_path=os.path.join(dataset_path, 'train.csv'), test_path=os.path.join(dataset_path, 'test.csv'))


# First we add some simple features (DoW, geodesic distance, etc.):

# In[ ]:


# Pickup and dropoff hour of day:
df['pickup_hour'] = df.pickup_datetime.dt.hour
df['dropoff_hour'] = df.dropoff_datetime.dt.hour

# Pickup and dropoff day of week:
df['pickup_DoW'] = df.pickup_datetime.dt.dayofweek
df['dropoff_DoW'] = df.dropoff_datetime.dt.dayofweek

# numerical trip ID:
df['id_num'] = df.id.str[2:].astype('int')

# Log trip duration (see part II):
df['log_duration'] = np.log(1 + df.trip_duration)

# convert the 'set' (train vs set) column into a categorical column (required for datashader plots):
df['set_cat'] = pd.Categorical(df.set)

# Compute trip haversine distances (geodesic distances):

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Source: https://gis.stackexchange.com/a/56589/15183
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    meters = 1000 * 6367 * c
    return meters

df['hdist'] = haversine(df.pickup_longitude, df.pickup_latitude,
                        df.dropoff_longitude, df.dropoff_latitude)

# Add the log haversine distance:

df['log_hdist'] = np.log(1 + df['hdist'])

# Projects longitude-latitude coordinates into Web Mercator(https://en.wikipedia.org/wiki/Web_Mercator) 
# coordinates (for visulization)

df['pickup_x'], df['pickup_y'] = lnglat_to_meters(df['pickup_longitude'], df['pickup_latitude'])
df['dropoff_x'], df['dropoff_y'] = lnglat_to_meters(df['dropoff_longitude'], df['dropoff_latitude'])


# ## Part I - Train vs test sets comparison
# 
# Usually, it is not a good practice to peek at targets and features of the test set before modelling.
# However here I though it was important to understand how the test data was sampled to identify the task at hand.

# In[ ]:


df.set_cat.value_counts()


# ### Pickup hour of day

# In[ ]:


hour_prop = df.groupby('set').pickup_hour.value_counts(dropna=False, normalize=True).reset_index(name='proportion')
sns.factorplot(x="pickup_hour", y='proportion', data=hour_prop, hue="set", size=4, aspect=2);


# ### Pickup day of week

# In[ ]:


dow_prop = df.groupby('set').pickup_DoW.value_counts(dropna=False, normalize=True).reset_index(name='proportion')
sns.factorplot(x="pickup_DoW", y='proportion', data=dow_prop, hue="set", size=4, aspect=2);


# ### Pickup date

# In[ ]:


daily_trip_counts = df.set_index('pickup_datetime').groupby('set').resample('1D').size().transpose()
daily_trip_proportion = daily_trip_counts.div(daily_trip_counts.sum())

ax = daily_trip_proportion.plot.line(figsize=(12,6));
ax.set_ylabel("daily trip proportion")


# ### Passenger count

# In[ ]:


plt.figure(figsize=(12, 3))
sns.heatmap(df.groupby('set').passenger_count.value_counts(dropna=False, normalize=True).unstack(), 
            square=True, annot=True);


# ### Store and forward flag
# > This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip [[kaggle.com]](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)

# In[ ]:


store_and_fwd_flag_prop = df.groupby('set').store_and_fwd_flag.value_counts(dropna=False, normalize=True).reset_index(name='proportion')
grid = sns.factorplot(x="store_and_fwd_flag", y='proportion', data=store_and_fwd_flag_prop, hue="set", size=3, aspect=0.8, kind='bar');
grid.fig.get_axes()[0].set_yscale('log')


# ### Vendor ID
# > A code indicating the provider associated with the trip record [[kaggle.com]](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)

# In[ ]:


vendor_id_prop = df.groupby('set').vendor_id.value_counts(dropna=False, normalize=True).reset_index(name='proportion')
grid = sns.factorplot(x="vendor_id", y='proportion', data=vendor_id_prop, hue="set", size=3, aspect=0.8, kind='bar');


# ### Pickup coordinates

# In[ ]:


# Bounding box where most of the data is:
nyc = {'x_range': (40.635, 40.86), 'y_range': (-74.03,-73.77)}

# Bounding box converted to Web mercator coordinates
bottom_left = lnglat_to_meters(nyc['y_range'][0], nyc['x_range'][0])
top_right = lnglat_to_meters(nyc['y_range'][1], nyc['x_range'][1])
nyc_m = {'x_range': (bottom_left[0], top_right[0]), 'y_range': (bottom_left[1], top_right[1])}


# In[ ]:


color_map = {'train':'gold', 'test':'aqua'}

# Plot train vs test set heatmap:
cvs = ds.Canvas(plot_width=1400, plot_height=1400, **nyc_m)
agg = cvs.points(df, 'pickup_x', 'pickup_y', ds.count_cat('set_cat'))
img = bg(dtf.shade(agg, color_key=color_map, how='eq_hist'))
display(img)

# Display colorbar:
fig = plt.figure(figsize=(10, 3))
fig.add_axes([0.05, 0.80, 0.9, 0.15])
cb = mpl.colorbar.ColorbarBase(ax=fig.axes[0], cmap=mpl.cm.cool.from_list('cus01', list(color_map.values())),
                               orientation='horizontal');
cb.set_ticks([0,1])
cb.set_ticklabels(list(color_map.keys()))


# ## Part II - Taxi trip visualisation and analysis

# Extract train data:

# In[ ]:


train = df[lambda x: x.set == 'train'].copy()


# Create a random sambled subset of the data (15%) for slow operations (seaborn factorplots):

# In[ ]:


train_sample = train.sample(frac=0.15, random_state=1234)


# In[ ]:


len(train_sample)


# ### Hourly rides by per day of week

# In[ ]:


hourly_counts = train.set_index('pickup_datetime').resample('1h').size().reset_index(name='pickups')
hourly_counts['date'] = hourly_counts.pickup_datetime.dt.strftime("%b %d %Y")
hourly_counts['hour'] = hourly_counts.pickup_datetime.dt.hour
hourly_counts['DoW'] = hourly_counts.pickup_datetime.dt.dayofweek


# In[ ]:


plt.figure(figsize=(11, 5))
sns.tsplot(time="hour", value="pickups", unit="date", condition='DoW', data=hourly_counts, err_style="unit_traces", )
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);


# ### Pickup and dropoff locations distribution
# 
# * blue : pickup locations
# * yellow : dropoff locations

# In[ ]:


src = train[['pickup_x', 'pickup_y']].rename(columns={'pickup_x':'x', 'pickup_y':'y'})
dst = train[['dropoff_x', 'dropoff_y']].rename(columns={'dropoff_x':'x', 'dropoff_y':'y'})

all_pts = pd.concat({'pickup': src, 'dropoff': dst}, ignore_index=False, names=['type']).reset_index(0).reset_index(drop=True)
all_pts['type'] = all_pts['type'].astype('category')

cvs = ds.Canvas(plot_width=2000, plot_height=2000, **nyc_m)
agg = cvs.points(all_pts, 'x', 'y', ds.count_cat('type'))
bg(dtf.shade(agg, color_key={'dropoff':'#FF9E25', 'pickup':'#3FBFFF'}, how='eq_hist', min_alpha=100))


# ### Trip duration distribution

# In[ ]:


train.trip_duration[lambda x: x < 3600 * 3].plot.hist(figsize=(12, 6), bins=40)


# In[ ]:


train.log_duration.plot.hist(figsize=(12, 6), bins=40)


# ### Trip id vs log duration

# In[ ]:


grid = sns.jointplot(x="id_num", y="log_duration", data=train, kind='hex');
grid.fig.set_figwidth(12)
grid.fig.set_figheight(6)


# ### Pickup hour of day vs log trip duration

# In[ ]:


grid = sns.factorplot(x="pickup_hour", y="log_duration",  hue="pickup_DoW", data=train_sample, aspect=1.5, size=8);


# ### Passager count versus average trip duration

# In[ ]:


grid = sns.factorplot(x="passenger_count", y="trip_duration", data=train_sample, aspect=2, size=4, kind='bar');


# ### Trip duration by pickup location
# 
# * pink : short trips
# * yellow : long trips

# In[ ]:


# create plasma colormap for datashader:
plasma = [tuple(x) for x in mpl.cm.plasma(range(255), bytes=True)[:, 0:3]]


# In[ ]:


cvs = ds.Canvas(plot_width=2000, plot_height=2000, **nyc_m)
agg = cvs.points(train, 'pickup_x', 'pickup_y', ds.mean('log_duration'))
bg(dtf.shade(agg, cmap = cm(plasma, 0.2), how='eq_hist'))


# ### Trip duration by dropoff location
# 
# 
# * pink : short trips
# * yellow : long trips

# In[ ]:


cvs = ds.Canvas(plot_width=2000, plot_height=2000, **nyc_m)
agg = cvs.points(train, 'dropoff_x', 'dropoff_y', ds.mean('log_duration'))
bg(dtf.shade(agg, cmap = cm(plasma, 0.2), how='eq_hist'))


# ### Geodesic distance vs trip duration

# In[ ]:


train_outliers_filtered = train[lambda x: (x.trip_duration < 3600 * 2) & (x.hdist < 30000)]

grid = sns.jointplot(x="hdist", y="trip_duration", data=train_outliers_filtered, kind='hex',
                     gridsize=80, space=0, mincnt=10, cmap='viridis')

grid.fig.set_figwidth(12)
grid.fig.set_figheight(6)


# Using a log-log scale:

# In[ ]:


grid = sns.jointplot(x="log_hdist", y="log_duration", data=train, kind='hex', gridsize=80,
                     space=0, mincnt=10, cmap='viridis')
grid.fig.set_figwidth(12)
grid.fig.set_figheight(6)


# Residual:

# In[ ]:


grid = sns.jointplot(x="log_hdist", y="log_duration", data=train, kind='resid', space=0, scatter_kws={'alpha': 0.1});
grid.fig.set_figwidth(12)
grid.fig.set_figheight(6)


# ### Outlier visualization
# 
# Here, we are interested  in two type of outlier:
# 
# * Trips with null distances
# * Wery long taxi trip (more than 12 hours)

# In[ ]:


train['is_null_distance'] = train.hdist <= 0.05
train['is_12+_trip'] = train.trip_duration > 3600 * 12


# In[ ]:


# Create a single 'anomaly' categorical variable from ''is_null_distance' and 'is_12+_trip' flags:

train['anomaly'] = train['is_null_distance'].map({True:'null_distance', False: ''}).str.cat(
                   train['is_12+_trip'].map({True:'12+_trip', False: ''}))

train['anomaly'] = train['anomaly'].replace('', 'none').astype('category')


# In[ ]:


train['anomaly'].value_counts()


# In[ ]:


anomaly_color_map = {'none':'gray', 'null_distance':'red', '12+_trip':'yellow', 'null_distance12+_trip':'green'}

cvs = ds.Canvas(plot_width=2000, plot_height=2000, **nyc_m)

# Anomaly heatmap:
agg = cvs.points(train[lambda x:x.anomaly != 'none'], 'pickup_x', 'pickup_y', ds.count_cat('anomaly'))
img_outliers = dtf.shade(agg, color_key=anomaly_color_map, how='linear', min_alpha=140)

# Non-anomaly heatmap:
agg = cvs.points(train[lambda x:x.anomaly == 'none'], 'pickup_x', 'pickup_y', ds.count())
img_regular = dtf.shade(agg, cmap='gray', how='eq_hist', min_alpha=120)

# Show legend as HTML:
display(HTML(''.join(["<p><p style='border-left: 1.5em solid {};padding-left: 10pt;margin: 5px;'>{}</p>".format(color, label) 
              for label, color in anomaly_color_map.items()])))

# Superimpose anomaly and non-anomaly heatmaps
bg(dtf.stack(img_regular, dtf.dynspread(img_outliers)))


# ### Outliers distribution over time

# In[ ]:


def anomaly_ratio(data, freq, anomaly):
    ratio = train.set_index('pickup_datetime').resample(freq)[anomaly].mean().reset_index(name='ratio')

    ratio['date'] = ratio.pickup_datetime.dt.strftime("%b %d %Y")
    ratio['hour'] = ratio.pickup_datetime.dt.hour
    ratio['weekofyear'] = ratio.pickup_datetime.dt.weekofyear
    ratio['DoW'] = ratio.pickup_datetime.dt.dayofweek
    
    return ratio


# #### 12+ hours long trips ratio:

# In[ ]:


extra_long_trip_daily_ratio= anomaly_ratio(train, freq='1D', anomaly='is_12+_trip')

week_vs_DoW_extra_long_trip_ratio = extra_long_trip_daily_ratio.set_index(['DoW', 'weekofyear']).ratio.unstack()


# In[ ]:


plt.figure(figsize=(12, 3))
sns.heatmap(week_vs_DoW_extra_long_trip_ratio, square=True, cmap="summer", cbar_kws={'label':'12+ hours trip ratio'})


# #### Null distance trips

# In[ ]:


null_dist_trip_daily_ratio= anomaly_ratio(train, freq='1D', anomaly='is_null_distance')

week_vs_DoW_null_dist_trip_ratio = null_dist_trip_daily_ratio.set_index(['DoW', 'weekofyear']).ratio.unstack()


# In[ ]:


plt.figure(figsize=(12, 3))
sns.heatmap(week_vs_DoW_null_dist_trip_ratio, square=True, cmap="summer", cbar_kws={'label':'12+ hours trip ratio'})


# ### Outliers vs hour of day

# In[ ]:


hourly_anomalies_prop = (train
                         .groupby('anomaly')
                         .pickup_hour.value_counts(dropna=False, normalize=True)
                         .reset_index(name='proportion'))

# Ignore 'null_distance12+_trip' anomaly (very few points)
hourly_anomalies_prop.anomaly = hourly_anomalies_prop.anomaly.astype(str)
hourly_anomalies_prop = hourly_anomalies_prop[lambda x: x.anomaly != 'null_distance12+_trip']

sns.factorplot(x="pickup_hour", y='proportion', data=hourly_anomalies_prop, hue="anomaly", size=4, aspect=2);


# ## Part III - Path visualization using datashader

# ### Plotting short distance trips
# We use the `datashader.line` function to display all trips with a geodesic distances less than 2km.

# In[ ]:


def get_lines(df):
    return pd.DataFrame({
            'x': df[['dropoff_x', 'pickup_x']].assign(dummy=np.NaN).values.flatten(),
            'y': df[['dropoff_y', 'pickup_y']].assign(dummy=np.NaN).values.flatten()})


# In[ ]:


lines = get_lines(train[lambda x: x.hdist < 2000])


# In[ ]:


cvs = ds.Canvas(plot_width=2000, plot_height=2000, **nyc_m)
agg = cvs.line(lines, 'x', 'y', ds.count())
bg(dtf.shade(agg, cmap=cm(inferno, 0.1), how='log'))


# ### Displaying all trip using datashader edge bundling
# 
# We use the `datashader` [edge bundling](https://anaconda.org/jbednar/edge_bundling/notebook) feature.
# 
# This feature is available in `datashader 0.0.6dev+` only. Use `conda install -c bokeh/label/dev datashader`
# to install the latest development version of datashader.

# In[ ]:


import importlib

datashader_version_0_6_plus = importlib.util.find_spec("datashader.bundling") is not None


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif datashader_version_0_6_plus:\n    \n    from datashader.bundling import directly_connect_edges, hammer_bundle # only in datashader 0.6+\n\n    # Filter origin and destination within the region of interest: \n    train_in_nyc = train[lambda x: x.pickup_x.between(*nyc_m[\'x_range\']) & x.pickup_y.between(*nyc_m[\'y_range\'])\n                  & x.dropoff_x.between(*nyc_m[\'x_range\']) & x.dropoff_y.between(*nyc_m[\'y_range\'])]\n\n    # The `hammer bundle` function is quite slow on my machine,\n    # we limit the number of displayed routes by sampling:\n    train_in_nyc = train_in_nyc.sample(100000, random_state=1234)\n\n    # Create nodes and edges dataframes:\n\n    src = train_in_nyc[[\'pickup_x\', \'pickup_y\']].rename(columns={\'pickup_x\':\'x\', \'pickup_y\':\'y\'})\n    dst = train_in_nyc[[\'dropoff_x\', \'dropoff_y\']].rename(columns={\'dropoff_x\':\'x\', \'dropoff_y\':\'y\'})\n\n    nodes = pd.concat([src, dst], ignore_index=True).copy(deep=True)\n    edges = pd.DataFrame({\'source\': list(range(len(src))), \n                          \'target\':list(range(len(src), len(src) + len(dst)))}).sample(frac=1).copy(deep=True)\n\n    # Compute trip paths using the "hammer bundle" algorithm (slow!). \n    # The chosen parameter set may not be optimal, it is difficult to find\n    # a good trade-off between computational performance and visual pleasantness:\n    lines = hammer_bundle(nodes, edges, initial_bandwidth=0.3, decay=0.1, batch_size=20000, accuracy=300,\n                          max_segment_length=0.1, min_segment_length=0.00001)')


# In[ ]:


if datashader_version_0_6_plus:
    cvs = ds.Canvas(plot_width=2000, plot_height=2000)
    agg = cvs.line(lines, 'x', 'y', ds.count())
    img = bg(dtf.shade(agg, cmap=cm(inferno, 0.1), how='log'))
    display(img)
else:
    display(HTML("<img src='http://i.imgur.com/jDFJW9K.jpg'>"))


# ## Comments

# * Features from the test set seems to follow the same distribution as in the train set (same spatial 
#   and temporal ranges, same distribution over weekdays, pickup hours, vendor ids, passenger counts, store 
#   and forward flags).
#   
# * Contrary to what I expected, the test set is not built from a fixed split date, but on what it appears
#   randomly sampled taxi trips. This means that the task is not exactly to predict future taxi trip
#   durations, but more to estimate past taxi trip durations given historical data from other trips
#   in the same time span.
#   
# * There is an obvious correlation between geodesic distances (as the crow flies) and taxi trip durations.
#   Still, a linear model on this feature is far from enough to account for all the factors affecting taxi
#   trip durations (road network topology, weather conditions, traffic jams, etc.).
#   
# * We see a weekend vs work week trend in trip durations. On weekdays, taxi trips are longer, especially on
#   mornings, most likely due to rush hour traffic congestions.
#   
# * The log trip duration seems normally distributed but there are several outliers, same for the haversine
#   trip distance. For instance, there are some very lengthy trips (saturated trip durations?) and trips
#   with null distances. We should look if these anomalies are also present in the test set, and if there
#   are, find a way to handle them. We could do so within the main model, perhaps adding some features to
#   discriminate theses anomalies, or with distinct single purpose models.
#   
# * Plotting lines with `datashader` is cool but I am not sure if its usefulness here. Plots produced by the
#   edge bundling functionality are shiny but also difficult to interpret (chaotic behaviour with respect 
#   to parameters).
#   
# * There are enough taxi trips to infer a large part of the road networks from pick-up and drop-off locations.
#   This is interesting as this could mean that models, given enough capacity, could learn the road network
#   topology and characteristics without having to use external datasets.
