#!/usr/bin/env python
# coding: utf-8

# # Executive summary
# 
# Try interactive mode if this fails to load. This kernel is a lenghty exploration into outliers in the market data. It ends with a conclusion that I don't quite understand what the different values in the market dataset represent. It is mainly long because of all the figures and partial dataframes it prints when looking for the outliers. 
# 
# # Exploring Outliers and Data features/values in the Market Data
# 
# I have no experience in trading, so I am not familiar with many of the terms used in the 2sigma dataset. In this kernel I do some basic data exploration by looking for outlier values in the market dataset, while also trying to get a bit better understanding what the values are and what they represent. 
# 
# This kernel only looks at the market data, not the news dataset.
# 
# The description of the market dataset is given as:

# The marketdata contains a variety of returns calculated over different timespans. All of the returns in this set of marketdata have these properties:
# 
# * Returns are always calculated either open-to-open (from the opening time of one trading day to the open of another) or close-to-close (from the closing time of one trading day to the open of another).
# * Returns are either raw, meaning that the data is not adjusted against any benchmark, or market-residualized (Mktres), meaning that the movement of the market as a whole has been accounted for, leaving only movements inherent to the instrument.
# * Returns can be calculated over any arbitrary interval. Provided here are 1 day and 10 day horizons.
# * Returns are tagged with 'Prev' if they are backwards looking in time, or 'Next' if forwards looking.
# 

# Within the marketdata, you will find the following columns:
# 
# * __time(datetime64[ns, UTC])__ - the current time (in marketdata, all rows are taken at 22:00 UTC)
# * __assetCode(object)__ - a unique id of an asset
# * __assetName(category)__ - the name that corresponds to a group of assetCodes. These may be "Unknown" if the corresponding assetCode does not have any rows in the news data.
# * __universe(float64)__ - a boolean indicating whether or not the instrument on that day will be included in scoring. This value is not provided outside of the training data time period. The trading universe on a given date is the set of instruments that are avilable for trading (the scoring function will not consider instruments that are not in the trading universe). The trading universe changes daily.
# * __volume(float64)__ - trading volume in shares for the day
# * __close(float64)__ - the close price for the day (not adjusted for splits or dividends)
# * __open(float64)__ - the open price for the day (not adjusted for splits or dividends)
# * __returnsClosePrevRaw1(float64)__ - see returns explanation above
# * __returnsOpenPrevRaw1(float64)__ - see returns explanation above
# * __returnsClosePrevMktres1(float64)__ - see returns explanation above
# * __returnsOpenPrevMktres1(float64)__ - see returns explanation above
# * __returnsClosePrevRaw10(float64)__ - see returns explanation above
# * __returnsOpenPrevRaw10(float64)__ - see returns explanation above
# * __returnsClosePrevMktres10(float64)__ - see returns explanation above
# * __returnsOpenPrevMktres10(float64)__ - see returns explanation above
# * __returnsOpenNextMktres10(float64)__ - 10 day, market-residualized return. This is the target variable used in competition scoring. The market data has been filtered such that returnsOpenNextMktres10 is always not null.

# ## Basic Setup
# 
# First, load the data and and start exploring:

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


from kaggle.competitions import twosigmanews
#   You  can  only    call    make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')


# In[ ]:


mt_df, news_df = env.get_training_data()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn  as sns


# ## Helper Functions
# 
# Define a function to plot a given asset for a given date range. Borrowed initial code from kernel: https://www.kaggle.com/bielrv/two-sigma-extensive-eda. Thanks :)

# In[ ]:


# plotAsset plots assetCode1 from date1 to date2
def plotAsset(assetCode1, date1, date2):
    asset_df = mt_df[(mt_df['assetCode'] == assetCode1) 
                      & (mt_df['time'] > date1) 
                      & (mt_df['time'] < date2)]

    x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = asset_df['close'].values

#    plt.figure(figsize=(10,6))
    plt.figure(figsize=(8,5))
    plt.title(assetCode1+": Opening and closing price")
    plt.plot(asset_df.time, asset_df.open, label='Open price')
    plt.plot(asset_df.time, asset_df.close, label='Close price')
    plt.legend()
    plt.show()


# The above function draws a graph for a single asset at a time. This is very useful for exploring the data in more detail. But using it for all the data items I was exploring for outliers resulted in 100+ charts drawn. This worked fine in interactive mode, but the committed kernels seemed to never render. So I defined the following version to draw several charts as subplots in a single plot. Not sure if this works any better, but anyway. As an extra benefit, it also gives a useful overview at a glance, all-in-one. The multi-chart version:

# In[ ]:


import matplotlib.dates as mdates

# plotAssetGrid plots assetCode1 from date1 to date2 as a subplot
def plotAssetGrid(assetCode1, date1, date2, ax):
    asset_df = mt_df[(mt_df['assetCode'] == assetCode1) 
                      & (mt_df['time'] > date1) 
                      & (mt_df['time'] < date2)]

    center = int(asset_df.shape[0]/2)
    ts = asset_df.iloc[center]["time"]
    ts_str = pd.to_datetime(str(ts)) 
    d_str = ts_str.strftime('%Y.%b.%d')
    #date_str = str(ts.year)+"/"+str(ts.month)+"/"+str(ts.day)
    #y_str = str(asset_df.iloc[center]["time"].year)
    #m1_str = str(asset_df.iloc[center]["time"].month)
    #d1_str = str(asset_df.iloc[center]["time"].day)
    ax.set_title(assetCode1+": "+d_str)
    ax.plot(asset_df.time, asset_df.open, label='Open price')
    ax.plot(asset_df.time, asset_df.close, label='Close price')
    myFmt = mdates.DateFormatter('%d/%b')
    ax.xaxis.set_major_formatter(myFmt)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    #ax.tick_params(labelrotation=45)
    #ax.get_xticklabels().set_rotation(45)


# Define a function to display the raw numbers for a given asset for a given date range:

# In[ ]:


from IPython.display import display

def printAssetDf(assetCode1, date1, date2):
    asset_df = mt_df[(mt_df['assetCode'] == assetCode1) 
                      & (mt_df['time'] > date1) 
                      & (mt_df['time'] < date2)]

    display(asset_df)


# Next, define a function to build both the graphs and show the raw values for outliers. This expects the global variable "outliers" to contain the set of outliers to visualize. In this list each row should be a single row from the market data and thus contains a single point in time. For each point in this list, a date range is defined to include 7 days before and 7 days after the outlier event, as well as the event date itself. A graph for this time range is plotted, and the raw data for this time range is printed as a dataframe.
# 
# The print order is:
# 
# * The time of the outlier event, this should be the chart center point (assuming data available before and after)
# * The time range of the chart/data printed.
# * The raw data in a dataframe format for the time-range.
# * The graphic plot for the time-range.
# 
# The parameters allow to limit whether to always draw the charts or print the data values.
# 

# In[ ]:


def graph_outliers(show_data=False, show_plots=True):
    offset = pd.Timedelta(7, unit='d')
    for index, row in outliers.iterrows():
        print (row["time"], row['assetCode'])
        start = row["time"]-offset
        end = row["time"]+offset
        print(str(start) + " -> " + str(end))

        if show_data:
            printAssetDf(row['assetCode'], row["time"]-offset, row["time"]+offset)
        if show_plots:
            plotAsset(row['assetCode'], row["time"]-offset, row["time"]+offset)


# Besides combining the plots, I also limited the number of top assets this prints he data for to 5 first assets, to make the committed kernel draw the results better. Just change the *print_assets* default value to larger to  get them all printed. The version using the combined plots next:

# In[ ]:


import math

def graph_outliers_grid(show_data=True, show_plots=True, print_assets=5):
    count = len(outliers)
    cols = 5
    rows = int(math.ceil(count/cols))
    
    if show_plots:
        offset = pd.Timedelta(7, unit='d')
        fig, axes = plt.subplots(ncols=cols, nrows=rows,figsize=(15,15))
        #fig.autofmt_xdate(rotation=25)
        axes = [col for row in axes for col in row]

        idx = 0
        for index, row in outliers.iterrows():
            plotAssetGrid(row['assetCode'], row["time"]-offset, row["time"]+offset, axes[idx])
            idx += 1
        fig.tight_layout()
        plt.show()

    if show_data:
        offset = pd.Timedelta(5, unit='d')
        count = 0
        for index, row in outliers.iterrows():
            print (row["time"], row['assetCode'])
            start = row["time"]-offset
            end = row["time"]+offset
            print(str(start) + " -> " + str(end))
            printAssetDf(row['assetCode'], row["time"]-offset, row["time"]+offset)
            count += 1
            if count >= print_assets:
                break
        


# ## Looking at the Actual Data
# 
# 
# The following was my initial attempt at finding outliers in the dataset. I first tried my regular approach for values that are 3 * STD from mean as outliers. In this dataset, it was still quite a few items, so I tried to limit the number of assets I had to look at by using higher thresholds. After playing with the data for a while, I figured the raw **open** and **close** values are (or seem to be) percentages in relation to previous day, so 1 should mean there was 100% gain in value vs previous day value. Using this as a threshold for an outlier gave me a reasonable starting point to look at the data, so I went with that.

# In[ ]:


#mask = np.abs(market_df['returnsClosePrevRaw1']-market_df['returnsClosePrevRaw1'].mean()) > (20*market_df['returnsClosePrevRaw1'].std())
mask = np.abs(mt_df['returnsClosePrevRaw1']-mt_df['returnsClosePrevRaw1'].mean()) > 1
outliers = mt_df.loc[mask]
outliers


# The following visualizes the above "outliers" list, showing already interesting assets on the chart rows 4-6 (those sharp spikes). Later on, in the following sections I just printed the top price/value changers and looked at those. But this seems to have been a good start nevertheless:

# In[ ]:


graph_outliers_grid(True, True)


# Since I limited the number of assets to print to 5 to get the committed kernel to show, the above raw data sets do not include the spikes. Just change the print_assets for the plot function to a bigger value to see them all. But now a look at the data that was printed above, which has big changes that are permanent (no up/down spikes).

# ### Are the large permanent changes valid?
# 
# I tried to look up some of the stocks for the dates that they seemed to have a large permanent change on a day. This seems to confirm my hypothesis:
# 
# * MFRM: https://www.nasdaq.com/article/mattress-firm-holding-corp-mfrm-stock-skyrockets-on-steinhoff-deal-cm662148
# * EVHC: https://www.iclub.com/faq/Home/Article?id=561&category=26&parent=0
# * SDLR: https://www.businessinsider.com/seadrill-share-price-rally-march-4-2016-3?international=true&r=US&IR=T
# 
# These articles all confirm some related event leading to large stock price changes for the respective outlier days in the above lists. So these permanent changes seem valid. I did not go through all of them but this seemed to make enough sense that I just focused on the one-day spikes for further analysis. 
# 
# ### One-day spikes
# 
# The one-day spikes for BBBY, DISH, FLEX, MAT, PCAR, SHLD, ZNGA seem much more likely to be outliers - errors in the data input or something similar. Collecting these one-day spikes from above graphs to a new dataframe:

# In[ ]:


outlier_collection = mt_df.head(0)
outlier_collection = outlier_collection.append(mt_df.iloc[3845015])
outlier_collection = outlier_collection.append(mt_df.iloc[3845309])
outlier_collection = outlier_collection.append(mt_df.iloc[3845467])
outlier_collection = outlier_collection.append(mt_df.iloc[3845835])
outlier_collection = outlier_collection.append(mt_df.iloc[3846067])
outlier_collection = outlier_collection.append(mt_df.iloc[3846276])
outlier_collection = outlier_collection.append(mt_df.iloc[3846636])
outlier_collection


# From the above outlier list, it seems someone liked to mess with the data for July 6th on 2016 and input 123.45 in many places. In some places they just typoed it to 123.47? Might be worth looking at that date to see if other items have similar values. But I was looking for the biggest spikes today, so did not go further that rabbit hole.

# ### Look at price change during a day
# 
# Add one more column for absolute gain during the day. Percentage would likely be useful as well but lets try this for now:

# In[ ]:


mt_df['price_diff'] = mt_df['close'] - mt_df['open']


# After looking at different number of values a few times, I settled on picking the top 30 highest change values for **price_diff** (**open** and **close** difference). It seemed to capture more of the values with single-day spikes:

# In[ ]:


outliers = mt_df.sort_values('price_diff', ascending=False)[:30]


# In[ ]:


graph_outliers_grid(True, True)


# There would likely be more to be found, but this is good enough for me to start looking at these and see what might be the issue, etc. There are many data points in there that were not in the previous list. So adding them all:

# In[ ]:


outlier_collection = outlier_collection.append(mt_df.iloc[50031])
outlier_collection = outlier_collection.append(mt_df.iloc[92477])
outlier_collection = outlier_collection.append(mt_df.iloc[206676])
outlier_collection = outlier_collection.append(mt_df.iloc[459234])
outlier_collection = outlier_collection.append(mt_df.iloc[132779])
outlier_collection = outlier_collection.append(mt_df.iloc[50374])
outlier_collection = outlier_collection.append(mt_df.iloc[276388])
outlier_collection = outlier_collection.append(mt_df.iloc[3845946])
outlier_collection = outlier_collection.append(mt_df.iloc[616236])
outlier_collection = outlier_collection.append(mt_df.iloc[3846151])
outlier_collection = outlier_collection.append(mt_df.iloc[49062])


# In[ ]:


#outlier_collection


# ### Highest peaks for returnsOpenPrevRaw1
# 
# Now to see similarly the top 30 with highest value for raw opening value for 1-day time range:

# In[ ]:


outliers = mt_df.sort_values('returnsOpenPrevRaw1', ascending=False)[:30]


# In[ ]:


graph_outliers_grid(True, True)


# This still uncovers a few more spikes that were not shown by the previous visualizations. To add those:

# In[ ]:


outlier_collection = outlier_collection.append(mt_df.iloc[588960])
outlier_collection = outlier_collection.append(mt_df.iloc[165718])
outlier_collection = outlier_collection.append(mt_df.iloc[25574])
outlier_collection = outlier_collection.append(mt_df.iloc[555738])
outlier_collection = outlier_collection.append(mt_df.iloc[56387])
#TW.N seems to have started trading on this day since there is no previous day
#so maybe the beginning spike is a high initial listing price?
#https://www.sec.gov/Archives/edgar/data/1470215/000119312510212218/d424b1.htm:
#"Towers Watson was formed on January 1, 2010, from the merger of Towers Perrin and Watson Wyatt"
outlier_collection = outlier_collection.append(mt_df.iloc[1127598])


# In[ ]:


#outlier_collection


# ### returnsClosePrevRaw1 and returnsClosePrevMktres1
# 
# Above graphed the opening prices, the same for closing prices (to make the kernel smaller and render better, I disabled this print but removing the comments and re-running should show it):

# In[ ]:


#outliers = mt_df.sort_values('returnsClosePrevRaw1', ascending=False)[:30]


# In[ ]:


#graph_outliers_grid(True, True)


# This showed plenty of spikes but they were all covered by the previous visualizations already.

# In[ ]:


#outlier_collection


# The previous values were for raw 1-day open and close values. How about the market residualized 1-day values? Lets see:

# In[ ]:


outliers = mt_df.sort_values('returnsClosePrevMktres1', ascending=False)[:30]


# In[ ]:


graph_outliers_grid(True, True)


# DNDO is a new asset that shows a slightly interesting shape, with **close** going down and **open** up for one day around 2009-04-23. But overall the scale is not so huge and the shape is not quite like the other biggest spikes. So I skipped it for this investigation.
# 
# ### What relation do the Mktres value represent?
# 
# Looking at this data also helped me understand a bit more about what the values in this dataset are. In this case, a DNDO rise from 11.81 to 22.94 from 2009-04-28 to 2009-04-29 gives **returnsClosePrevMktres1** as 1.00, so seems this metric value is showing the % rise. So 1.00 would refer to 100% rise. While 11.81 * 2 is not exactly 22.94, the market adjustment could make up for the small difference. The previous day drop from 21.55 to 11.81 gives **returnsClosePrevRaw1** as -0.46, which could indicate a similar drop of about 46%. It seems so.
# 
# However, looking at the other data items, it does not seem quite so simple. For example, EVHC on 2016-12-01 has an open value of 22.72 and on the following day an open value of 66.25. But the **returnsOpenPrevRaw1** is only 1.9. And 22.72 * 1.9 = 42.3, which is not quite the value of 66.25. And this is the "*raw*" value, so I thought it should not be modified by any market related multiplier? I guess I still don't quite understand the return values.
# 
# No need to add any new values to the outliers collection based on my limited understanding here.

# In[ ]:


#outlier_collection


# Now the similar residualized 1-day open values:

# In[ ]:


outliers = mt_df.sort_values('returnsOpenPrevMktres1', ascending=False)[:30]


# In[ ]:


graph_outliers_grid(True, True, 5)
#graph_outliers_grid(True, True, 30)


# Again, there are no obvious new spikes here. But when I used *graph_outliers_grid* function to print  the asset data for all the 30 assets visualized above, I found some more data that looks strange to me. ABV has the previously noted outlier on 2008-09-18. The opening price drops from 53.33 to 0.02 and then goes back up to 54.54 the following day. Looking for the market dataset highest values for **returnsOpenPrevMktres1**, ABV actually comes up many times, showing up as multiple charts above. Looking at this for ABV, the single large outlier actually continues to produce high **returnsOpenPrevMktres1** values for several days into the future as a series of 2088, -396, 405, 170, and so on. Yet the daily **open**/**close** changes are small after the single spike. I cannot quite figure out how the daily change is calculated for **returnsOpenPrevMktres1** since a 1-day outlier impact seems to go on for many days and I though 1 referred to single day impact?
# 
# On October 2nd, the **returnsOpenPrevMktres10** for ABV goes up to 2734, indicating the 10-day value is for exactly 10 days in the past. Not something that sums up the past 10 days as I first thought it might be.
# 
# ABV is not the only one with this behaviour but similarly at least in TEO.N and TX.N show such behaviour. In fact, all these come up multiple times here. For example, TX has an outlier on 2008-08-22, which shows up as value of 3498 for **returnsOpenPrevRaw1**, which is huuuge. The day before that shows the same value as -0.999, indicating a 99.9% drop. However, besides this single day impact, the TX values for **returnsOpenPrevMktres1** still go on at very high levels up to 2008-09-22 and likely further, which is a full month after the one-day spike for a "1-day return" value.
# 
# The timeframe for the **returnsOpenPrevRaw10** and **returnsOpenPrevMktres10** for TX shows a high value for 2008-09-08, which seems more than 10 days apart from from 2008-08-22. This is similar to the ABV relation of spike at 2008-09-18 and observing a 10-day effect on Oct. 2nd. So after a whlie of looking at this, I realize the 10 days refers to business/trading days and not calendar days.
# 
# I guess all this just summarizes that I don't quite understand the terminology or what the values represent here. Makes it a bit harder to figure what to do with these outliers, and how much data should I drop or replace, and what should I replace it with. Or what good are the value for, huh? :)
# 
# To collect the high ABV values from above. Look especially the development of the **returnsOpenPrevMktres1** and **returnsOpenPrevMktres10** columns vs the **open**/**close** prices:
# 

# In[ ]:


#ABV strange values:
abv_collection = outlier_collection.head(0)
abv_collection = abv_collection.append(mt_df.iloc[611442])
abv_collection = abv_collection.append(mt_df.iloc[613039])
abv_collection = abv_collection.append(mt_df.iloc[614639])
abv_collection = abv_collection.append(mt_df.iloc[616236])
abv_collection = abv_collection.append(mt_df.iloc[617832])
abv_collection = abv_collection.append(mt_df.iloc[619423])
abv_collection = abv_collection.append(mt_df.iloc[621008])
abv_collection = abv_collection.append(mt_df.iloc[622591])
abv_collection = abv_collection.append(mt_df.iloc[624178])
abv_collection = abv_collection.append(mt_df.iloc[625765])
abv_collection = abv_collection.append(mt_df.iloc[627355])
abv_collection = abv_collection.append(mt_df.iloc[628947])
abv_collection = abv_collection.append(mt_df.iloc[630544])
abv_collection = abv_collection.append(mt_df.iloc[632142])
abv_collection = abv_collection.append(mt_df.iloc[633738])
abv_collection = abv_collection.append(mt_df.iloc[635332])
abv_collection = abv_collection.append(mt_df.iloc[636927])
abv_collection = abv_collection.append(mt_df.iloc[638521])
abv_collection = abv_collection.append(mt_df.iloc[640115])
abv_collection = abv_collection.append(mt_df.iloc[641710])
abv_collection = abv_collection.append(mt_df.iloc[643305])
abv_collection = abv_collection.append(mt_df.iloc[644902])
abv_collection = abv_collection.append(mt_df.iloc[646503])
abv_collection = abv_collection.append(mt_df.iloc[648105])
abv_collection = abv_collection.append(mt_df.iloc[649712])
abv_collection = abv_collection.append(mt_df.iloc[651319])
abv_collection = abv_collection.append(mt_df.iloc[652923])
abv_collection = abv_collection.append(mt_df.iloc[654527])
abv_collection[['time', 'assetCode', 'open', 'close', 'returnsOpenPrevRaw1', 'returnsOpenPrevMktres1', 'returnsOpenPrevRaw10', 'returnsOpenPrevMktres10']]


# The above summarizes a bit more clearly how the outlier effect goes on for also the 1-day columns far into the future (as compared to my initial 1-day effect assumption). We can view this a bit better graphically:

# In[ ]:


#outlier_collection


# In[ ]:


import matplotlib.dates as mdates

asset_df = abv_collection
fig, ax = plt.subplots(ncols=2, nrows=1,figsize=(10,4))

ts = asset_df.iloc[0]["time"]
ts_str = pd.to_datetime(str(ts)) 
d_str = ts_str.strftime('%Y.%b.%d')

ax[0].set_title("ABV open/close: "+d_str)
ax[0].plot(asset_df.time, asset_df.open, label='Open price')
ax[0].plot(asset_df.time, asset_df.close, label='Close price')
ax[0].legend(loc="lower right")
myFmt = mdates.DateFormatter('%d/%b')
ax[0].xaxis.set_major_formatter(myFmt)
for tick in ax[0].get_xticklabels():
    tick.set_rotation(45)
    
ax[1].set_title("ABV Mktres1/10: "+d_str)
ax[1].plot(asset_df.time, asset_df.returnsOpenPrevMktres1, label='returnsOpenPrevMktres1')
ax[1].plot(asset_df.time, asset_df.returnsOpenPrevMktres10, label='returnsOpenPrevMktres10')
ax[1].legend(loc="upper right")
myFmt = mdates.DateFormatter('%d/%b')
ax[1].xaxis.set_major_formatter(myFmt)
for tick in ax[1].get_xticklabels():
    tick.set_rotation(45)

fig.tight_layout()
plt.show()


# The two charts above show the **open**/**close** prices and the **returnOpenPrevMktres1** / **returnOpenPrevMktres10** values for the same dates for ABV. The actual price has a a single outlier drop on a single day for the open value, on September 18th. 
# 
# **returnOpenPrevMktres1** shows a 2088x spike for the day following the outlier. The **returnOpenPrevMktres10** shows a 2762x spike for the 10th business day after the spike. After these, the 1-day Mktres value wildly swings above and below zero, showing gains and losses in multiplies of several hundred each day. Well, it would if this is what the Mktres would show, I guess it must be something else. 
# 
# The **returnOpenPrevMktres10** spikes much higher for some reason on the 10th day, and slowly declines after for a long period (over 10 days anyway..).

# ## Negative Values
# 
# The above visualizations mostly looked at positive values for price change. The following is one look at biggest negatives (disabled the printing for now to save space).
# 
# Looking at some of the data points presented above, for example, a negative return of 0.97 is 97% loss. I guess maximum would be 1.00 for 100% loss, or the asset going to zero value. So the scale to look for here is quite different from the positive values, which ranged to several thousand (due to the outliers/errors in data):

# In[ ]:


#outliers = mt_df.sort_values('returnsClosePrevRaw1', ascending=True)[:30]


# In[ ]:


#graph_outliers_grid(True, True)


# Nothing new here again, so I guess I caught most of the biggest one already. Which makes sense, since for spikes, the highest rise is likely followed by the highest low, and similarly deepest low is likely followed by a highest high. When it "spikes" like this I guess one of the metrics captures the outliers fine.

# In[ ]:


#outlier_collection


# One final sort to show how there were a few dates with several instances of similar outliers:

# In[ ]:


outlier_collection.sort_values(by='time')


# # Some final points
# 
# After this look for outliers, and using it to try to understand the data, I still have some questions:
# 
# * What kind of assumptions are made about people on Kaggle? Reading the discussions, many seem to have questions about what the data is about. Yet the variable explanations leave room for many questions. Is this some general approach every time?
# * The data seems to have some pretty basic issues. How is such basic cleaning not done as to fix these obvious outliers? Is this also on purpose?
# * Where does the data come from? does someone pay for this messy data? How would they not have an automated process in place since long ago to highlight such issues? :)
# * What are the market adjusted values actually? My 1-day and 10-day assumptions did not quite hold. Especially the 1-day ones. What are they adjusted on, and what is the relation to past data?
# * What happened in June 6th, 2016? Somebody had a bad day with the data?
# * Are those outliers all errors and if not, what happened? If they are, what happened.. I would've though market data is collected automatically and not so prone to input errors :)
# 
# I initially though it would be useful to clean this data by just dropping the data for an asset for the two days when the spike occurs (up/down). I thought this would get rid of the days with biggest change, and with the biggest change for 1-day metrics that follow. And then for 10-day metrics, maybe it would be smoothed out. All these proved false assumption obviously. I cannot quite understand how are all the values calculated and why they persist so long after the outliers, and with such strong numbers.
# 
# Overall, I should probably automated this type of search if I used it more. 
# 
# So. Any domain experts (or anyone) with good ideas and explanations? :)
# 

# In[ ]:





# In[ ]:





# In[ ]:




