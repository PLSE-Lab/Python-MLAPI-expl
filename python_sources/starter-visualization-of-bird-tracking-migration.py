#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings from the Krishna! This is an starter kernel with starter code demonstrating how to read in the data and begin exploring. If you're inspired to dig deeper, click the blue "Fork Notebook" button at the top of this kernel to begin editing by creating your own copy.
# 
# PS: Need Suggestions for improvements as this is my first notebook
# Thankyou all.

# ## Exploratory Analysis
# To begin this exploratory analysis, first import libraries and define functions for plotting the data using `matplotlib`. Depending on the data, not all plots will be made. (Hey, I'm just a simple starter, not a Kaggle Competitions Grandmaster!)

# # Basic Imports

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# There is 1 csv file in the current version of the dataset:
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# In[ ]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# # Correlation Matrix

# In[ ]:


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# # Scatter & Density plots

# In[ ]:


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# Now you're ready to read in the data and use the plotting functions to visualize the data.

# ### Let's check 1st file: /kaggle/input/bird_tracking.csv
# 
# nRowsRead = None which means all the rows of dataset will be imported.

# In[ ]:


nRowsRead = None  # specify 'None' if want to read whole file
# bird_tracking.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/bird_tracking.csv',index_col= 0, delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'bird_tracking.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df1.head(5)


# # Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plotPerColumnDistribution(df1, 10, 5)


# # Correlation matrix:

# In[ ]:


plotCorrelationMatrix(df1, 8)


# # Scatter and density plots:

# In[ ]:


plotScatterMatrix(df1, 18, 10)


# So far what we have seen is the KaggleBot i.e
# 
# it loads data and does some visualisation and processing to make it easy to work here
# 
# let's start from here!
# 
# df1 is the name of our dataframe.
# 
# For more DataFrames related to bird tracking and related aspects u can visit: 
# 
# https://www.gbif.org/dataset/83e20573-f7dd-4852-9159-21566e1e691e

# 

# In[ ]:


df1.info()


# info gives us the col names of df1 & how many nonnull ie 'not nan'
# 
# if the row conts and non-nullcount are same which means there is no missing data, but here are some missing values in the speed_2d and direction.
# 
# it also tells about the datatypeof the columns

# # Data preprocessing

# In[ ]:


df1.describe()


# describe() - gives us the basic statistics of all the columns which are of Data type int and float ie numeric data only.
# we can get min, 1,2, 3 quartile and max.
# 
# here u can understand that std of altitue is 136 feet. which means there is a change of height aorung 136 feet per timeperiod/ frequency of data logging in device, in here we are taking approximately around  1 hour to collect the data. 1 hr is the inerval. 
# 
# we can observe the device_info_seriel whose basic stats are of no use as it's of no value and is only used for identification of devices from mixing up with other device data. so we can convert it to object or delete it.

# In[ ]:


df1.bird_name.value_counts()


# each bird data is collected approx 20k times.
# 
# let's do some basic imports of seaborn and matplotlib

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# Now let's have a basic plot of latitude and long of each bird
# 
# To do that we need to collect data of each bird seperately and give those as points for plotting or u can go with seaborn for quick visualisation and with less hassle. seaborn is a visualisation package which is built on top of matplotlib or u can say it's an advaned version of matplotlib
# 
# plot longitude on x axis and latitude on y axis

# # line plot & scatter plot

# In[ ]:


plt.figure(figsize= (15,15))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Latitude vs Longitude')
sns.lineplot(x='longitude' , y= 'latitude', hue = 'bird_name', data= df1, legend= 'full', alpha =  0.7)
plt.show()


# In[ ]:


plt.figure(figsize= (10,10))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Latitude vs Longitude')
sns.scatterplot(x='longitude' , y= 'latitude', hue = 'bird_name', data= df1, legend= 'full', alpha = 0.3 )
plt.show()


# As we can see from the graph there is a lot of scatterdness and all the three birds have simillar flight pattern u can play with several visualisation and can understand data in a different ways
# 
# Now lets deal with understanding Speed_2d it gives us the average speed of the bird in a 2D plane which is an local approximation of the earth curved surface
# 
# let's start dealing with null or nan valus in this column

# # Dealing with null values

# In[ ]:


df1.speed_2d.isna().sum()


# There are 443 null values or mising data from the df1
# 
# Now it will be a problem to plot missing values so we consider the values and we ignore the missing values
# 
# let's define x which gives us the speed component of bird Eric for only of it's non null values.
# 
# and repeat the process for the other two birds and plot and histogram because it's a continous data.
# 

# # Distribution plot

# In[ ]:


x = df1.speed_2d[df1.bird_name == 'Eric']
x1 = df1.speed_2d[df1.bird_name == 'Nico']
x2 = df1.speed_2d[df1.bird_name == 'Sanne']
#print(x,x1,x2)
plt.figure(figsize=(15,9))
sns.distplot(x,bins =30,rug = True )
plt.show();
sns.distplot(x1,bins =20,rug = True )
plt.show();
sns.distplot(x2,bins =10,rug = True )
plt.show();


# ** 1. Here we can see all the 3 graphs, and bins give u the light blue square one in the graph which means that if N no. of bins are given then the range of x is taken and it does make that range into N-1 parts and bars are drawn with breath of the cal value,more bins means less widthh of the bar which means the each partition value decreases
# 
# 
# suppose we have range 0-10x  and want to make into 10 parts then we need 11 bins each bin corresponds to change of 1x
# 
# same range and if we give 21 bins each bin corresponds to change of 0.5x
# 
# which inturn results in decreasing the breath of the rectangle in graph.
# 
# rug = true : which gives a small blue line for all the values
# 
# the dark blue lines are dense/more at starting of graph and scattered at the right side meaning more observations on left side and less on the right side and even the length of light blue rectangle means the same ie if length\ height of rect is more which means more obs of data are there
# 
# length dir proportional to no. of observations.
# 

# In[ ]:


df1.head()


# let's import datetime so that we can convert our date_time clumn which is an object will be converted to datetime so we can subtract two rows value to get the time elapsed in between the observations 

# # Working with Date Time

# In[ ]:


import datetime
#convert column to Date time format
df1.date_time = pd.to_datetime(df1.date_time)


# In[ ]:


df1.info()


# 

# In[ ]:


df1.head()


# 2013     -08     -15     00      :18     :08             +00:00
# 
# year    month    day    hours    mins   second           Timezone it's in UTC so +00: 00 if it's supposed u want to convert it to IST(indian standard time u need to +05:30 i.e 5 hrs and 30 mins.
# 
# Now let's subtract the first and last value of date_time col to know the complete elapsed time.
# 
# as both the values are in the same time zone we have no problem seperating the values and getting the time elapsed in btwn the observed data.
# 
# suppose if they are in different time zones then we need to either convert them to same timezone or we need to delete the timezone to do the subtraction and get the difference btwn them.
# 

# In[ ]:


df1.date_time.iloc[-1] - df1.date_time.iloc[0]


# so that's roughly around 259 days from first data recording to the last in our dataset.

# In[ ]:


time_Eric = (df1.date_time[df1.bird_name == 'Eric']).astype('datetime64[ns]')
time_Eric.iloc[-1] - time_Eric.iloc[0]


# In[ ]:


time_Nico = (df1.date_time[df1.bird_name == 'Nico']).astype('datetime64[ns]')

time_Nico.iloc[-1] - time_Nico.iloc[0]


# now let's get the time_date of each individual bird and save it in a variable as a **"datetime64[ns]"** so that the variable will still be in the datime format and later we can use that to plot for better understanding.
# 
# After getting the Variable now we calcualte the time taken by each bird from starting of their jourey to the end of their journey.
# 

# In[ ]:


time_Sanne = (df1.date_time[df1.bird_name == 'Sanne']).astype('datetime64[ns]')
time_Sanne.head()
#time_Sanne.iloc[-1] - time_Sanne.iloc[0]


# # Visualisation of Time period and number of observations.

# In[ ]:


plt.title('Eric')
plt.xlabel('Time taken in DateTime format')
plt.ylabel('Number of observations')
plt.plot(time_Eric)
plt.show()

plt.title('Nico')
plt.xlabel('Time taken in DateTime format')
plt.ylabel('Number of observations')
plt.plot(time_Nico)
plt.show()


plt.title('Sanne')
plt.xlabel('Time taken in DateTime format')
plt.ylabel('Number of observations')
plt.plot(time_Sanne)
plt.show()


# From the graph we can infer that all the three birds data has been started collecting from the same time, date and there are no deviations in the data which means the data has been collected over periodic time and it looks like there is little iregularity (ie some data might be missed or the device has sompe problem storing the data or any othr reasons) in the bird named Eric of observations btwn 6500-7500 which can be neglected considering the majority of data following an upward trend or we could say that there are observations that are further apart from one other than the remaining obsrvations.
# 
# 

# # Visualization of Avg speed for every 30 mins and Datetime

# In[ ]:


plt.figure(figsize=(30,12))
plt.plot( time_Eric, x, linestyle= '-', marker = 'o')
plt.show()


# In[ ]:


df1.describe()


# **Difference of time between two Data points**

# In[ ]:


#time_Eric.iloc[1] - time_Eric.iloc[0]
#time_Eric.iloc[2] - time_Eric.iloc[1]
time_Eric.iloc[3] - time_Eric.iloc[2]


# Well as we have seen the graph we can't get much understanding out of it.so let's look at why we can't get inference out of it and where did we go wrong.
# 
# if we see at the data there is a less standard deviation in the speed_2d but why did the graph shows us that there is more scatterdness/ noise.
# 
# when we look at the datetime of each bird we notice that each observation is made on an average of 30 minutes, which means that the speed_2d is getting noted down for each bird over a period of 30 mins or half an hour, let's think of stock market for a sec, when we see the stock price of certain company over a time interval of let's say 30 mins then we get a more noisier graph but when we increase the timeperiod say 4 hrs we get less noise than before and as we increase the timeperiod let's say 1 day we will get less noise because it only shows us the day's starting price and end of day closing price.
# 
# so as we plot graphs of various timeperiod we get less and less noisy grahs and after certain timeperiod the stocks generally follow a trend an some call it as seasonality which occurs due to weather seasons and festivals associated with and less sales at financial year end and so on...
# 
# In this bird data we will try to find the seasonal migration period of the birds and preict the same in the future.
# 
# So now let's try to take the average of speed_2d along each day and plot them along no. of days for better understanding.
# 

# # Converting elapsed time i.e. 30 mins into whole day to calculate Avg Speed of each day.
# * *All observations with in 24 hrs i.e of same day are taken and their speed_2d values are taken and summed up and their Average spped of particular day is calculated.*

# In[ ]:


elapsed_time = [time - time_Eric[0] for time in time_Eric]
elapsed_time[:5]


# In[ ]:


elapsed_days= np.array(elapsed_time) / datetime.timedelta(days=1)
elapsed_days[:5]
# it will convert the datetime series into number of days as we need to iterate we converted it to a array as we can't iterate over list


# In[ ]:


'''for (i,t) in enumerate(elapsed_days):
    print(i,t)
'''


# In[ ]:


next_day = 1
indeces=[]
daily_mean_speed = []
for (i,t) in enumerate(elapsed_days):
    # Here i is the count or index we use for later gtting the spped_2d and 
    # t is the datetime in the time delta RATIO OF timedata so it says 
    if t< next_day:
        indeces.append(i)
        # we get a list of indeces and those speed_2d are with in the same day
        #we get multiple list of each day indeces as a list
        #print(indeces)
    else:
        daily_mean_speed.append(np.mean(df1.speed_2d[indeces]))
        #using the list of indeces of day 1 and getting their values from Speed_2d and
        #cal mean of that day and storing in daiy_mean_speed by append
        # so now we get mean speed of day 1 as one value and the ssame continues for all days
        #print(indeces)
        next_day += 1
        indeces = []
daily_mean_speed[:10]


# As we have seen now we have obtained the mean speed of Eric over it's entire journey so let's plot it.

# # Plot of Avg Speed of each day vs Days

# In[ ]:


plt.figure(figsize=(15,9))
plt.plot(daily_mean_speed)
plt.xlabel('Days')
plt.ylabel('Mean Speed per Day')
plt.title('Mean speed per Day of Eric')
plt.show()


# As we can now get a less noise graph from before which helps us to inference that the mean avg speed per day is around 2 to 3 and max speed of 9 has been observed.
# 
# The anormaly i.e the max speed per days are the days where migration are taking place from one location to another.
# 
# so now let's look at to where does the migraton happens.
# 
# u can similarly get the graphs for other2 birds using the same code and replacing their components.
# 
# https://scitools.org.uk/cartopy/docs/latest/
# 
# cartopy - Cartopy is a Python package designed for geospatial data processing in order to produce maps and other geospatial data analyses.
# 
# 

# # pip install cartopy 
# # For visualisation of flight of bird over a MAP

# In[ ]:



#!pip install cartopy
import cartopy.crs as ccrs
import cartopy.feature as cft


# In[ ]:


proj = ccrs.Mercator()
plt.figure(figsize=(12,12))
ax = plt.axes(projection = proj)
ax.set_extent((-25.0, 20.0, 52.0, 10.0))
#to set the long and lat take the min and max of lat and long and add some degree to max 
#and subtract some degrees to min and give the range by trial and error method.
ax.add_feature(cft.LAND)
ax.add_feature(cft.OCEAN)
ax.add_feature(cft.COASTLINE)
ax.add_feature(cft.BORDERS)
# adding features to show on the map like land, ocean, coastline, borders
for name in df1.bird_name:
    ix = df1.bird_name == name
    x, y = df1.longitude.loc[ix], df1.latitude.loc[ix] 
ax.plot(x,y, '.', transform = ccrs.Geodetic(), label = name )
plt.legend(loc= 'upper left' )
plt.show()


# # Instead of for loops and all basic functions u can use groupby and other pandas functions to get the data more easily.
# 
# if u can understand the content give it a like if anything need to be updated type in the comment section below.

# ![](http://)

# 

# ## Conclusion
# This concludes your starter Visualisation! To go forward from here, click the blue "Fork Notebook" button at the top of this kernel. This will create a copy of the code and environment for you to edit. Delete, modify, and add code as you please. Happy Kaggling!
