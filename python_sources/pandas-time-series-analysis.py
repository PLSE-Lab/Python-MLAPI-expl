#!/usr/bin/env python
# coding: utf-8

# On August 24, 2016, at 3.36am, the center of Italy was hit by a 6.0 magnitude earthquake. There have been 299 deaths, entire villages were destroyed and thousands were left homeless. From that day, tens of thousands of minor earthquakes hit the area with a frequency of about 5 minutes.
# The surrounding area has a notorious fame for geological instability, as Italy lies near the fault line that exists between Eurasian and African tectonic plates. Earthquakes happen when one of these plates scrapes, bumps or drags along another plate.
# In this notebook we will carry out some basic analysis and visualizations of this dataset. 
# 
# A little disclaimer before we begin. I know nearly nothing about earthquakes and geology. I'm just having fun doing some very basic data analysis and visualization, so the things I might say and the thoughts I might have about earthquakes should be taken with a (big) grain of salt.
# 
# You can find the full post about this notebook [here][1].
# 
# 
#   [1]: http://www.thedataware.com/post/italys-recent-earthquakes-a-look-at-the-data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting

plt.style.use("ggplot")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read the dataset
df = pd.read_csv('../input/italy_earthquakes_from_2016-08-24_to_2016-11-30.csv').set_index('Time')
df.index = pd.to_datetime(df.index)
df.head()


# In[ ]:


# Compute summary statistics
df.describe()


# ### Magnitude
# The simplest numeric measure of an earthquake strenght is the magnitude.
# The magnitude of an earthquake is usually measured by the [Richter Scale][1]. You can check the wikipedia page for more info.
# 
#   [1]: https://en.wikipedia.org/wiki/Richter_magnitude_scale

# In[ ]:


df["Magnitude"].resample("2D").apply([np.mean]).plot()
plt.title("Magnitude average every two days")
plt.ylabel("Magnitude")


# From this plot, we can clearly see that there was a spike between October and November. In fact, a 6.5 magnitude earthquake hit on October 30. It also seems like the days preceding that event were characterized by a lower activity. We can see see this better if we average the values over one day, instead that 2 days of the previous plot.

# In[ ]:


df["Magnitude"].resample("D").apply([np.mean]).plot()
plt.title("Magnitude average every day")
plt.ylabel("Magnitude")


# I'm not saying that every earthquake is preceeded by a lower activity in the previous days, it may very well be a coincidence, but it kind of highlights a possible pattern which might deserve future analysis.

# ### Magnitude Rolling Mean
# Another possible way of visualizing the magnitude is the rolling mean, or moving average. Here, instead of taking the mean of each day, we take the mean of a fixed window of n elements. Also, we take only earthquakes with magnitude 3+.

# In[ ]:


# Pandas series with magnitudes greater than 3.0
magn3 = df.ix[df["Magnitude"] >= 3.0, "Magnitude"]


# In[ ]:


pd.Series.rolling(magn3, window=25).mean().plot(style="-g")
plt.title("Magnitude Rolling Mean")
plt.ylabel("Magnitude")


# Besides the same spike as before, we can also see that October had a relatively regular mean, while from November the mean started to fluctuate a bit more. This could mean that the 6.5 earthquake of October 30 may have opened new rifts in the tectonic plate.

# ### Depth
# Let's take a look at the depth now. By plotting a histogram of the different values, we can see a nicely shaped bell curve centered at 10km. We can model this with a gaussian distribution of parameters (mu=10, sigma=2.25) and begin to answer some simple questions like: what is the probability of an earthquake having a depth less than 5km?
# Using this rule, this probability is (approximately) 2.3%. Of course, this value is only related to this event, it is not a fact about earthquakes in general.

# In[ ]:


plt.figure()

depth = df[(df["Depth/Km"] < 20) & (df["Depth/Km"] >= 0)]["Depth/Km"]
depth.plot(kind="hist", stacked=True, bins=50)

plt.title("Depth histogram")
plt.xlabel("Depth/Km")


# Another interesting question we might try to answer is if earthquakes are more likely to happen at a specific time of day. The 6.0 happened at 3.36am, the 6.5 happened at around 7.50am, so this might be a reasonable thing to search for. Maybe earthquakes are more likely to hit in the morning, or maybe it was just a coincidence. Let's visualize it.

# In[ ]:


depth_magn = df.where((df["Magnitude"] >= 3.0)).dropna()[["Magnitude", "Depth/Km"]]
dm = depth_magn.groupby(depth_magn.index.hour).mean()

fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_ylim([2.5, 4.0])
ax.set_ylabel("Magnitude")
ax.set_xlabel("Hour of the day")
ax.yaxis.label.set_color("#1f77b4")

ax2 = ax.twinx()
ax2.set_ylim([4.0, 12])
ax2.set_ylabel("Depth/Km")
ax2.set_xlabel("Hour of the day")
ax2.yaxis.label.set_color("#ff7f0e")

width = 0.5

dm["Magnitude"].plot(kind="bar", color='#1f77b4', ax=ax, width=width, position=1)
dm["Depth/Km"].plot(kind="bar", color="#ff7f0e", ax=ax2, width=width, position=0)

plt.grid(False)
plt.title("Magnitude and Depth during the day")


# There does not seem to be a significant difference between hours of the day. Thus, in line with the fact the earthquakes are unpredictable, we can conclude that there isn't a particular time of the day in which they are more likely to happen.There does not seem to be a significant difference between hours of the day. Thus, in line with the fact the earthquakes are unpredictable, we can conclude that there isn't a particular time of the day in which they are more likely to happen.

# In[ ]:


# This function is taken from here:
# https://gist.github.com/tartakynov/83f3cd8f44208a1856ce

def fourierExtrapolation(x, n_predict):
    n = x.size
    n_harm = 10                     # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = np.fft.fft(x_notrend)  # detrended x in frequency domain
    f = np.fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t


# In[ ]:


n_predict = 300
resample_period = "2D"
predict_period = "D"

pred = pd.Series(
    fourierExtrapolation(magn3, n_predict),
    index=magn3.index.append(pd.DatetimeIndex(start="2016-12-01", freq="45T", periods=n_predict))
)

fig = plt.figure()

fitted = pred[:-n_predict].resample(resample_period).mean()
predict = pred[-n_predict:].resample(predict_period).mean()

fitted.plot(linewidth=3, label="extrapolation")
predict.plot(linewidth=3, style="-g", label="prediction")
magn3.resample(resample_period).mean().plot(label="data")

plt.title("Magnitude forecasting")
plt.ylabel("Magnitude")
plt.legend(loc="lower left")


# Obviously, this forecasting technique was very simple and should not be considered reliable. However, it is an example of the importance of forecasting and time series prediction in general.
