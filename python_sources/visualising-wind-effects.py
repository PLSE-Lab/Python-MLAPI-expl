#!/usr/bin/env python
# coding: utf-8

# # Visualising Wind Effects
# 
# When the wind is a blowin' the emissions be a showin', or at least, that's what I hope. NO2 emissions are a gas, which means they are easily influenced by wind. Prior research shows that wind can significantly effect the dispersion profile of NO2 gas so we should take this into account when trying to estimate the emissions from the power generation facilities.

# In[ ]:


import os

import rasterio as rio
from rasterio.plot import show, show_hist
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

from datetime import datetime

import numpy as np
import pandas as pd


# ## Read Data
# 
# Get all data for NO2 and weather.

# In[ ]:


def load_sp5():
    # Get the data filenames
    no2_path = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/'
    no2_files = [no2_path + f for f in os.listdir(no2_path)]
    
    data = []
    
    print(f'Reading {len(no2_files)} files')
    for f in no2_files:
        raster = rio.open(f)
        data += [{
            'tif': raster,
            'filename': f.split('/')[-1],
            'path': no2_path + f.split('/')[-1],
            'measurement': 'no2',
            'band1_mean': np.nanmean(raster.read(1)),
            **raster.meta
        }]
        raster.close()
        
    # Get dates
    for d in data:
        d.update({'datetime': datetime.strptime(d['filename'][:23], 's5p_no2_%Y%m%dT%H%M%S')})

    for d in data:
        d['date'] = d['datetime'].date()
        d['hour'] = int(datetime.strftime(d['datetime'], '%H'))
        d['weekday'] = d['datetime'].weekday()  # Mon = 0

    return data

def load_gfs():
    # Get the data filenames
    no2_path = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/'
    no2_files = [no2_path + f for f in os.listdir(no2_path)]
    
    data = []
    
    print(f'Reading {len(no2_files)} files')
    for f in no2_files:
        raster = rio.open(f)
        data += [{
            'tif': raster,
            'filename': f.split('/')[-1],
            'path': no2_path + f.split('/')[-1],
            'measurement': 'weather',
            'band1_mean': np.nanmean(raster.read(1)),
            **raster.meta
        }]
        raster.close()
        
    # Get dates
    for d in data:
        d.update({'datetime': datetime.strptime(d['filename'], 'gfs_%Y%m%d%H.tif')})

    for d in data:
        d['date'] = d['datetime'].date()
        d['hour'] = int(datetime.strftime(d['datetime'], '%H'))
        d['weekday'] = d['datetime'].weekday()  # Mon = 0

    return data

data = load_sp5() + load_gfs()
# data.append(load_gfs)


# In[ ]:


data[0]


# In[ ]:


df = pd.DataFrame(data)
df.head()


# ## Weather
# 
# We'll discuss wind, but first a mention on reflectance.
# 
# ### High Reflectance
# 
# When there is high reflectance, such as cloud cover or snow, the satellite can have a hard time measuring a good signal, as such these values should be ignored. The dataset in use here has already been processed to remove cases of significant cloud coverage but we can still take note of the cloud values as they may affect our readings.
# 
# Thankfully, Puerto Rico is a warm state, meaning that it doesn't receive any snow throughout the year.
# 
# Reading:
# 
# - https://traveltips.usatoday.com/weather-climate-puerto-rico-50065.html
# 
# ### Wind
# 
# Strong winds move the air in the atmosphere which means that the spatial accuracy of measurements quickly becomes incorrect. Thankfully, these measurements can be corrected for if we have a good understanding of the wind. In general, it is sensible to look at days with less wind.
# 
# For a more accurate analysis, we can remove from our dataset dates of measurements which correspond to strong wind conditions. This will more accurately allocate the NO2 readings to the spatial source.
# 
# The wind measurements occur every 6 hours, but the NO2 data is once per day. Since we're trying to move any large cases of wind, we can aggregate the wind values to daily, using the mean/max. This can then be joined with the NO2 data.
# 
# Other things to consider:
# 
# - Counter winds could be OK, e.g. strong easterly followed by strong westerly wind.
# - Wind fluctuates a lot in time, so 6 hourly measurements may not be enough to detect the impact on emissions.
# 
# 
# Let's calculate the magnitude's and have a look.

# In[ ]:


# Add total wind magnitude
for d in data:
    if d['measurement'] == 'weather':
        r = rio.open(d['path'])
        d.update({
            'avg_wind_magnitude': 
            np.nanmean(np.sqrt(np.square(r.read(4)) + np.square(r.read(5)))),
            'max_wind_magnitude': 
            np.nanmax(np.sqrt(np.square(r.read(4)) + np.square(r.read(5))))
        })
        r.close()


# In[ ]:


plt.hist([d['max_wind_magnitude'] for d in data if d['measurement'] == 'weather'])
plt.hist([d['avg_wind_magnitude'] for d in data if d['measurement'] == 'weather'])


# We see there are some readings with a much higher mean (6+), but otherwise the data seems to be distributed with some positive skew.
# 
# The units here are m/s. The average magnitude is 4m/s or 14km/h. For reference on the scale, Puerto Rico is approximately 50km from top to bottom and 150km from left to right. So we see that these winds can definitely move the emissions around on this scale.
# 
# To continue, we can filter out the lower quartile and compare it with the values in the upper quartile.
# 
# Let's plot the daily values to see any patterns, with and without the high wind days.

# In[ ]:


def raster_median(rasters, band=1):
    tmp = []
    for r in rasters:
        r = rio.open(r.name)
        tmp += [r.read(band)]
        r.close()
    return [np.nanmedian(np.stack(tmp), axis=(0))]


# In[ ]:


df = pd.DataFrame(data)
df_daily = df.groupby('date').agg({
    'avg_wind_magnitude': 'median',
    'max_wind_magnitude': 'max',
    'tif': raster_median,
    'measurement': 'first'
}).rename(columns={'tif': 'values'})
df_daily.plot(kind='line')


# In[ ]:


quartile25 = df_daily['avg_wind_magnitude'].quantile(0.25)
quartile75 = df_daily['avg_wind_magnitude'].quantile(0.75)

# df_daily.plot(kind='line')
print('Low Wind Days')
df_daily.query('avg_wind_magnitude < @quartile25').plot(kind='line', marker='.', linestyle='')
print("High Wind Days")
df_daily.query('avg_wind_magnitude > @quartile75').plot(kind='line', marker='.', linestyle='')


# ## Visualising
# 
# We can now compare the difference in the emissions data from removing the wind. How does our data compare?
# 
# We'll use the methods described here: https://www.kaggle.com/jyesawtellrickson/averaging-satellite-data.

# In[ ]:


def plot_average_raster(rasters, band=1, output_file='tmp.tif', avg='mean'):
    all_no2s = []
    print(f'Processing {len(rasters)} files')
    for r in rasters:
        if r.closed:
            r = rio.open(r.name)
        all_no2s += [r.read()[band-1, :, :]]
        r.close()
    temporal_no2 = np.stack(all_no2s)
    print("Done")
    
    if avg == 'mean':
        avg_no2 = np.nanmean(temporal_no2, axis=(0))
    else:
        avg_no2 = np.nanmedian(temporal_no2, axis=(0))

    raster = rasters[0]
    
    new_dataset = rio.open(
        output_file,
        'w',
        driver=raster.driver,
        height=raster.height,
        width=raster.width,
        count=1,
        dtype=avg_no2.dtype,
        crs=raster.crs,
        transform=raster.transform,
    )
    
    new_dataset.write(avg_no2*10**5, 1)
    new_dataset.close()
    
    tmp = rio.open(output_file)
    
    min_val = np.nanmin(tmp.read(1))
    max_val = np.nanmax(tmp.read(1))
    print('Ranges from {:.2E} to {:.2E}'.format(min_val, max_val))
    
    # https://rasterio.readthedocs.io/en/latest/topics/plotting.html
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    # Augment the data so that it can be plotted nicely
    # mult_fact = 10**(-round(np.log10(min_val))+1)
    #data = tmp.read(1)*10**mult_fact
    
    show(tmp, transform=tmp.transform, ax=ax1)
    
    show(tmp, cmap='Greys_r', interpolation='none', ax=ax2)
    show(tmp, contour=True, ax=ax2)

    plt.show()
    
    return tmp


# In[ ]:


def plot_diff_raster(rasters_1, rasters_2, band=1, output_file='tmp.tif'):
    all_no2s_1 = []
    all_no2s_2 = []
    # Read raster 1
    print(f'Processing {len(rasters_1)} files')
    for r in rasters_1:
        r = rio.open(r.name)
        all_no2s_1 += [r.read()[band-1, :, :]]
        r.close()
    temporal_no2_1 = np.stack(all_no2s_1)
    # Read raster 2
    print(f'Processing {len(rasters_2)} files')
    for r in rasters_2:
        r = rio.open(r.name)
        all_no2s_2 += [r.read()[band-1, :, :]]
        r.close()
    temporal_no2_2 = np.stack(all_no2s_2)
        
    # Calculate averages
    avg_no2_1 = np.nanmean(temporal_no2_1, axis=(0))
    avg_no2_2 = np.nanmean(temporal_no2_2, axis=(0))

    avg_no2 = avg_no2_2 - avg_no2_1
    
    raster = rasters_1[0]
    
    new_dataset = rio.open(
        output_file,
        'w',
        driver=raster.driver,
        height=raster.height,
        width=raster.width,
        count=1,
        dtype=avg_no2.dtype,
        crs=raster.crs,
        transform=raster.transform,
    )
    
    new_dataset.write(avg_no2*10**5, 1)
    new_dataset.close()
    
    tmp = rio.open(output_file)
    
    print('Ranges from {:.2E} to {:.2E}'.format(np.nanmin(tmp.read(1)),np.nanmax(tmp.read(1))))
    
    # https://rasterio.readthedocs.io/en/latest/topics/plotting.html
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    
    show(tmp, transform=tmp.transform, ax=ax1)
    
    show(tmp, cmap='Greys_r', interpolation='none', ax=ax2)
    show(tmp, contour=True, ax=ax2)

    plt.show()
    
    return tmp


# In[ ]:


print("Low Wind Days")
plot_average_raster(
    df[df.date.isin(
        df_daily.query('avg_wind_magnitude < @quartile25*0.9'
                      ).index.tolist())].query('measurement == "no2"').tif.tolist()
)


# In[ ]:


print("High Wind Days")
avg = plot_average_raster(
    df[df.date.isin(
        df_daily.query('avg_wind_magnitude > @quartile75'
                      ).index.tolist())].query('measurement == "no2"').tif.tolist()
)


# It's also constructive to plot the difference between the days with less wind vs. the days with a lot of wind.

# In[ ]:


plot_diff_raster(
    df[df.date.isin(
        df_daily.query('avg_wind_magnitude < @quartile25*0.9'
                      ).index.tolist())].query('measurement == "no2"').tif.tolist(),
    df[df.date.isin(
        df_daily.query('avg_wind_magnitude > @quartile75'
                      ).index.tolist())].query('measurement == "no2"').tif.tolist()    
)


# So we can see quite well that on the windy days, there is less contrast, and the clump of pollution around San Juan is more spread out to the west.
# 
# We also see this in the values on the contour plot. On the days without wind, the concentration is stronger (inner circle of 6.8) compared to the days with wind (inner circle of 6.0).
# 
# The difference plot shows a fairly even spread outside of the big decrease in San Juan.
# 
# There's a potential that the date / time of our reading could be leading to the difference in levels, so let's check if there's a correlation between any of the variables with our wind.

# In[ ]:


corr_no2 = df.query('measurement == "no2"').corr()
corr_qgs = df.query('measurement == "weather"').corr()


# https://rasterio.readthedocs.io/en/latest/topics/plotting.html
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(
    corr_no2, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    ax=ax1
)
ax1.set_xticklabels(
    ax1.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.title("NO2")

sns.heatmap(
    corr_qgs, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    ax=ax2
)
ax2.set_xticklabels(
    ax2.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.title('Weather')
plt.show()


# We do see some correlations here.
# 
# For the left plot, the band 1 mean (NO2 concentration) is weakly correlated with the hour of the day. This makes sense, since the emissions are largely driven by cooling in the afternoon and appliance usage in the evening. We should check that the avg. hour for the two plots is roughly the same.
# 
# We can also see that the hour is weakly correlated with the avg wind magnitude, however this is not considered to be a problem as the values for the wind measurements are 6-hourly and averaged out for the day.

# In[ ]:


print("Less wind values, avg. hour: {}".format(
df[df.date.isin(
    df_daily.query('avg_wind_magnitude < @quartile25*0.9'
                  ).index.tolist())].query('measurement == "no2"').hour.mean()
))

print("More wind values, avg. hour: {}".format(
df[df.date.isin(
    df_daily.query('avg_wind_magnitude > @quartile75'
                  ).index.tolist())].query('measurement == "no2"').hour.mean()
))


# The values are very close, so it seems we're OK.
# 
# ## Conclusions
# 
# The wind can have a significant effect on the emissions distributions as shown by removing the readings with large amounts of wind.
# 
# In order to get more precise emissions readings it's better to use days without wind.
# 
# Next steps:
# 
# - There's research which shows the windy days can be utilised, through some smart calculations, this could be applied here in order to make more use of the readings and apply the analytical method to locations that have large amounts of wind (more universal methodology).
# - How long does NO2 last in the readings? If there were strong winds the hour / 6 hours / day before, would the NO2 from a source be lingering in the region? Should we exclude a certain window around strong wind events?
# 
# 
# Reading:
# - <https://www.atmos-chem-phys.net/16/5283/2016/acp-16-5283-2016.pdf>
