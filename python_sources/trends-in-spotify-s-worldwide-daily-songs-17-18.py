#!/usr/bin/env python
# coding: utf-8

# ![](https://media.giphy.com/media/e53HTalHxfY2s/giphy.gif)

# 
# 
# **Finding Trends in Spotify Music Goals:**
# * Find incremental changes in positions on the charts 
# * Model time elapsed changes in position
# * Find features that can model likelihood of being at position 1

# ![](https://media.giphy.com/media/952bbbsLYuuNW/giphy.gif)

# **CONTENT of Notebook:**
# 
# * Loading of datasets
# * Preprocessing
# * Cleaning Ranking data
# * Combining datasets
# * Statistical Analysis
# * Matlab Analysis: Classification Learner Results with **Conclusion on Most Preferred Features of 2017**
# * ARIMA attempt
# 

# Below are standard packages to import

# In[ ]:


import os # accessing directory structure
import itertools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pandas.plotting import andrews_curves
import plotly.plotly as py
import math
import datetime as dt
from datetime import datetime
from dateutil.parser import parse
#df.to_csv('pandas_dataframe_importing_csv/example.csv') #exporting to csv

# Any results you write to the current directory are saved as output.


# Loading Data and Checking Column Labels

# In[ ]:


print(os.listdir('../input'))


# Here is where we load the data into dataframes:
# * df is Spotify's Worldwide Daily Song Ranking
#     This dataset is used for understanding streams and positions of tracks.
# * df2 is Top Spotify Tracks of 2018
# * df3 is Top Spotify Tracks of 2017
#     The two datasets above are for pairing track names with their qualitative features

# In[ ]:


df = pd.read_csv('../input/spotifys-worldwide-daily-song-ranking/data.csv')
df2 = pd.read_csv('../input/top-spotify-tracks-of-2018/top2018.csv')
df3 = pd.read_csv('../input/top-tracks-of-2017/featuresdf.csv')
df.columns = (df.columns.str.lower()
                .str.replace(' ', '_'))
df.columns


# Above we see the column names of the Spotify's Worldwide Daily Song Ranking dataset

# In[ ]:


df.head()


# For this time series data, we want the dates to be of the type datetime for ease of modeling and plotting.
# **Scatterplot of Streams vs Time**

# In[ ]:



df['date'] = df['date'].astype('datetime64[ns]')
plt.rcParams["figure.figsize"] = [16,9]
plt.plot(df['date'], df['streams'],'*')
plt.xticks(rotation='vertical')
plt.ylabel('Streams')
plt.xlabel('Dates')
plt.title('Scatterplot of Streams vs Time')


# In[ ]:


df3.columns


# Above we see the column names of the Top Spotify Tracks of 2017 datasets

# **In the Top Spotify Tracks of 2017, the track name column is called 'name'.
# We need to change this to 'track_name', so we can link this data to the Spotify's Worldwide Daily Song Ranking dataset.
# The 'id' column will be removed as well since it is not need.
# We will then peek at the header of the information.**

# In[ ]:


df3 = df3.rename(columns={'name': 'track_name'})
df3 = df3.drop(columns=['id'])
df3.head()


# We do this again for the Track feature of 2018

# In[ ]:


df2 = df2.rename(columns={'name': 'track_name'})
df2 = df2.drop(columns=['id'])
df3.head()


# Dropping Url as they are also not needed.

# In[ ]:


df.drop('url', axis=1, inplace=True)


# Checking to NaN values and dropping them from the set.

# In[ ]:


# Validate all artists and track names are the same missing
(df['artist'].isna() == df['track_name'].isna()).all()


# In[ ]:



# drop null rows
df = df.dropna(axis=0)


# **Considering only the US**

# In[ ]:


df = df[df['region'].str.contains("us")] #include only a region, US


# See quick description of data of Spotify's Worldwide Daily Song Ranking dataset

# In[ ]:


df.describe(include='all')


# **Below are quick descriptions of the columns to understand the size of our trimmed dataset**

# In[ ]:


df.artist.describe()


# **We have 487 unique artists, with Drake being counted the most often.**

# In[ ]:


df.position.describe()


# **We have ~74,000 positions. Since there is a little more than 365 days that the ranking data was collected for the top 200 positions,
# this makes sense**

# In[ ]:


df.track_name.describe()


# **We also now know there are 1624 unique track names and Drake's Unforgettable is the most frequency position.**

# ![](https://media.giphy.com/media/8zT0D36Myf9C0/giphy.gif)

# In[ ]:


df.streams.describe()


# **Visualizations**

# Checking on the distribution of the Position to make sure they look like what is expected.

# In[ ]:


#histogram
sns.distplot(df['position'])


# Here we take the log of the streams to see that the distrubition is heaviest for the bottom positions.

# In[ ]:


x = np.log10(df.loc[:,'streams'])
sns.distplot(x, hist=False)


# Quick visuals of the streams vs position to help understand the distribution above.

# In[ ]:


ax1 = df.plot.scatter(x='position', y='streams', c='DarkBlue')


# Joint plot of streams and positions with distributions plotted on the secondary axes.

# In[ ]:


sns.jointplot(x='position', y='streams', data=df);


# **Below we can quickly find out the range of the data entries.
# This data was first collected on the first day of 2017 and ended
# on the Jan. 9th 2018**

# In[ ]:


df['date'].min() #First date entry


# In[ ]:


df['date'].max() #Last date entry


# **Let us use the dates in 2018 as the test set.
# We will cut these dates off from the training set and store it
# in dateframe called dftest.**

# In[ ]:


dftest = df[(df['date'] > '2017-12-31')]
df=df[(df['date'] < '2018-01-01')]


# In[ ]:


df.tail()


# In[ ]:


dftest.head()


# In[ ]:


dftest.tail()


# **Here we will combine ranking data with song features.
# This will give us insight on position and streams related to song qualities.
# However, be warned that this will drop all track names that are not stored in the features data.
# We can use this to draw out the list of track names that have reached the top 100 list of 2017.**

# Combining Song Features with Ranking Data

# In[ ]:


dfnew = pd.merge(df, df3, on='track_name')
dfnew = dfnew.drop(columns=['artists'])
dfnew.sort_values(by='date')
dfnew.head()


# Combine features with songs for test data.

# In[ ]:


dftest = pd.merge(dftest, df2, on='track_name')
dftest = dftest.drop(columns=['artists'])
dftest.head()


# In[ ]:


dfnew.describe()


# Let us find the track names that are in the test set to try and model.

# In[ ]:


testtn = dftest.track_name.unique()


# In[ ]:


dftesttn = pd.DataFrame({'track_name': testtn})
print(dftesttn)


# In[ ]:


dftrain = pd.merge(dfnew, dftesttn, on='track_name')

dftrain.head()


# **Now our training data has the same track names as our
# test data and we can model with validation data.
# We want to model predictions regarding streams and position, respectively to time.**

# In[ ]:



fig,ax= plt.subplots()
for n, group in dftrain.groupby('track_name'):
    group.plot(x='date',y='streams', ax=ax,label=n)
    
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=4, mode="expand", borderaxespad=1.)
plt.title('Scatterplot of Streams vs Time for Training Data Track Names')


# In[ ]:



fig,ax= plt.subplots()
for n, group in dftrain.groupby('track_name'):
    group.plot(x='date',y='position', ax=ax,label=n)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=4, mode="expand", borderaxespad=1.)
plt.gca().invert_yaxis()
plt.title('Scatterplot of Position vs Time for Training Data Track Names')


# **Below is an exploration of the song features and their correlation to ranking and streams**

# In[ ]:


x=dftrain['time_signature']
plt.hist(x, bins=10)
plt.gca().set(title='Time Signature in Ranking Data', ylabel='Frequency');


# In[ ]:



x=dftrain['instrumentalness']
plt.hist(x, bins=10)
plt.gca().set(title='Instrumentalness in Ranking Data', ylabel='Frequency');


# In[ ]:



x=dftrain['speechiness']
plt.hist(x, bins=10)
plt.gca().set(title='Speechiness in Ranking Data', ylabel='Frequency');


# Correlation Heatmap of song features

# In[ ]:


corr = dftrain.corr()
corr.style.background_gradient(cmap='cool').set_precision(2)


# In[ ]:


dfin = dfnew.set_index('date')
dfin['streams'].plot(linewidth=0.5);


# In[ ]:


ax = dfin.loc['2017-08':'2017-10', 'streams'].plot(marker='o', linestyle='-')


# **Matlab Results**

# Here is where I export the cleaned data out of python and into Matlab for further analysis. 
# Results from **Matlab** posted below.
# ![](https://media.giphy.com/media/NvMls3V5aID4c/giphy.gif)

# **Categorical Comparisons of Features relative to Stream and Position.
# Position is represented by the rainbow color scale seen below.**

# In[ ]:


from IPython.display import Image
Image("../input/images2/positioncolormap.PNG")


# In[ ]:



Image("../input/images/img/img/artistvsstreams.png")


# **From Artist vs Streams, we can see that Kendrick Lamar had the most streams as well as number 1 positions with Ed Sheeran trailing second in this cleaned data set. **

# In[ ]:



Image("../input/images2/tracknamevsstreams.png")


# **From Track Name vs Streams, we can see that Kendrick Lamar's HUMBLE. had the most streams as well as number 1 positions as well as Ed Sheeran's Shape of You trailing in 2nd in this cleaned data set. **

# In[ ]:



Image("../input/images/img/img/DanceabilityvsStreams.png")


# **From Danceability vs Streams, we see that most hits were very high in their Danceability scale. People like to cut a rug **

# In[ ]:



Image("../input/images/img/img/DurationvsStreams.png")


# **From Duration vs Streams, we see that shorter songs are preferred over longer songs. **

# In[ ]:



Image("../input/images/img/img/EnergyvsStreams.png")


# **From Energy vs Streams, we see that higher energy is more preferred than lower energy. **

# In[ ]:



Image("../input/images/img/img/InstrumentalnessvsStreams.png")


# **From Instrumentalness vs Streams, listeners prefer songs with mostly lyrics. **

# In[ ]:



Image("../input/images/img/img/KeyvsStreams.png")


# **From Key vs Streams, the keys are well distributed and there is no preference.**

# In[ ]:



Image("../input/images/img/img/LivenessvsStreams.png")


# **From Liveness vs Streams, we see liveliness is hot around 0.1 but the highest value on this plot is a little over 0.2. **

# In[ ]:



Image("../input/images/img/img/LoudnessvsStreams.png")


# **From Loudness vs Streams, we see loudness is most preferred around -6. **

# In[ ]:



Image("../input/images/img/img/ModevsStreams.png")


# **From Mode vs Streams, mode 0 is most preferred.**

# In[ ]:



Image("../input/images/img/img/SpeechinessvsStreams.png")


# **From Speechiness vs Streams, speechiness between 0.1 and 0.01 is highly preferred. **

# In[ ]:



Image("../input/images/img/img/TempovsStreams.png")


# **From Tempo vs Streams, tempos are spread across 90 to 190 but it appears our top songs are around 150. **

# In[ ]:



Image("../input/images/img/img/TimeSignaturevsStreams.png")


# **From Time signature vs Streams, 4 is the hottest choice.**

# In[ ]:



Image("../input/images/img/img/ValencevsStreams.png")


# **From Valence vs Streams, there is a spread in the values but around 0.4 seems to have the most expected value. **

# In[ ]:



Image("../input/images/img/img/AcousticnessvsStreams.png")


# **From Acousticness vs Streams, There is a high preference for little to no acousticness. **

# > Categorical Comparisons of Features relative to Stream and Position.
# 
# **In 2017, listeners preferred songs with:**
# * High Danceability
# * Short Durations (length of song)
# * Energy == 0.6 or higher
# * Low Instrumentalness
# * Liveliness == 0.1
# * Loudness around -6
# * Mode == 0
# * Low Speechiness
# * Tempos between 150 and 90
# * Time Signatures == 4
# * Average Valence
# * Low Acoustiness

# **Below is an attempt at ARIMA modeling**

# In[ ]:


dftrainin = dftrain.set_index('date')
y = dftrainin['streams'].resample('D').mean()
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[ ]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 1, 0),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[ ]:


results.plot_diagnostics(figsize=(16, 8))
plt.show()


# In[ ]:


pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2017':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('streams')
plt.legend()
plt.show()

