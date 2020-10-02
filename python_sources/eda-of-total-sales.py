#!/usr/bin/env python
# coding: utf-8

# # Data Analysis Approach
# 
# This notebook describes the main steps in my analysis of the "Store Item Demand Forecasting Challenge".  The task is about forecasting sales of 50 different items in a list of 10 stores over a three month period.  My objective was to get hands-on with a data forecast challenge and understand the structure of the data. I wanted to use this challenge as an opportunity to improve my skill with plotting and analyzing data  in Python.
# 
# The main phases of my analysis were as follows:
# 
# 1.  High Level look at the data
# 
# 2. Look for structure trends and seasonality in the data (I chose to do it myself rather than use available tools)
# 
# 3. Compare my effort with what a seasonal decomposition would have done out of the box.
# 
# The main Findings were:
#     *  This data is clean. It feels like synthetic data.
#     *  The sales pattern is similar for every item and store.
#     *  There is  year over year growth in sales volume for which I did not find a clear pattern.
#     *  There is a repeated seasonal pattern every year.
#     *  There is a repeated weekly pattern.
#     *  Within a month, variations are mostly linked to Day-Of-Week.
#     * The level of sales for a specific day is linked to number of days in a month.
# 
# The material is organized as a diary. It reads sequentially. It sometimes runs down a path and backtracks when the data shows something that I had not foreseen.
# 
# It concludes with an analysis of the training that reduces it to a random noise (though not quite white).

# ## 1. Data input and high level description 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np                 # linear algebra
import pandas as pd                # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime as dt
from sklearn import linear_model
from pandas.tseries.holiday import USFederalHolidayCalendar as cal1
import calendar


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# let's remove some of the warnings we get from the libraries


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# In[ ]:


MyTrain = pd.read_csv('../input/train.csv',parse_dates=['date'], index_col=['date'])
MyTest  = pd.read_csv('../input/test.csv',parse_dates=['date'], index_col=['date'])

print("Number of training samples read : ", len(MyTrain) )
print("        Shape is (rows, columns): ", MyTrain.shape)
print("Number of testing samples read  : ", len(MyTest)  )
print("        Shape is (rows, columns): ", MyTest.shape )


# In[ ]:


cal = cal1()
holidays = cal.holidays(start=MyTrain.index.min(), end=MyTrain.index.max())


# # 1. High Level Look at the data
# First Look at the data:

# In[ ]:


MyTrain.describe()


# The above seems very regular. Another question I wanted to validate was weather there was an entry for each item for each store for each day. Perhaps simple math can tell us that:
# 
# If  $n_s$ is number of Stores, $n_i$ is number of items and $n_d$ is number of days, we can calculate the expected number of entries $n_r$ in the table:
# 
#    \begin{align} n_r &= n_s * n_i * n_d        \\
#                      &= 10 * 50 * (365 *5 +1 ) \\     
#                      &= 913,000 \end{align} 
#    
# Note the +1 for 2016 which has 366 days!   
# This is exactly the number of rows we have read from the training file which pretty much confirms that our data table has exactly one entry per item per store per day.
# 
# 
# **Our Training Data is Clean!** No missing value and an entry for each row. Just as was announced in the description.
# 
# Note that some rows in the data have a zero value. Let's have a quick look at how many they are and at the story, items and dates involved.
# 

# In[ ]:


ZeroSales = MyTrain[MyTrain.sales == 0]
ZeroSales


# There is only one zero entry in the whole training set. Something to keep in mind but probably not statistically significant.

# # Let's visualize the data

# In[ ]:


g = sns.FacetGrid(data=MyTrain,row='item',col='store')
g = g.map(plt.plot,'sales') 


# 500 very similar plots up there. I gather the following
# 
# * All store and items follow a very similar pattern: Year over year growth (looks linear) and a seasonal pattern.
# * There are variation in volume between different stores.
# * There are significant variations in volume between differen items.
# 
# Some additional questions:
# 
# * Are the seasonality and the trend related to the store and item... or can I just derive them from the full dataset? I wil tackle this by working on total values first (over all items and stores) to build a first baseline.
# * The plots have a pretty broad brush... is that randomness or is that related to a shorter period in the signal (e.g. DOW)

# 
# # 2. Structure in the dataset: looking at Total Sales
# 
# Our approach here will be to look at the data, identify some structure, fit data to the structure and compute the residual.
# Then we can iterate on the residual.
# 
# ## 2.1 View of Total Sales.
# 
# Since all the sales graph for item, store have a similar shape, we would suspect that if we look to total sales for each day, we would see the same structure.
# 
# 

# In[ ]:


MyTrain.groupby(MyTrain.index).sum()['sales'].plot(style='.',figsize=(20,5))


# This graph is strange... We had already mentionned the broad brush.  It looks like there might be a structure to it... Let's zoom in.

# In[ ]:


MyTrain.groupby(MyTrain.index).sum()['sales'][0:160].plot(style='.',figsize=(20,5))


# > ## 2.2 Detailed structure analysis of Total Sales.
# We see a clear repeated structure for seven points.... A  weekly "seasonality". Let's dig a little deeper in that. We wil add a few feature and look at weekly trends

# In[ ]:


MyTrain['year']  = MyTrain.index.year
MyTrain['month'] = MyTrain.index.month
MyTrain['DoM']   = MyTrain.index.day
MyTrain['DoW']   = MyTrain.index.dayofweek # Mondays are 0
MyTrain['DoY']   = MyTrain.index.dayofyear

MyTrain['Holiday'] = MyTrain.index.isin(holidays)
MyTrainHolidays = MyTrain[MyTrain.Holiday==True]

plt.figure(figsize=(20,14))

plt.subplot(221)
plt.title('Sales per individual store/item for each day-of-week')
sns.boxplot(x=MyTrain.DoW,y=MyTrain.sales)

y = MyTrain.groupby(MyTrain.index).sum().sales
x = MyTrain.groupby(MyTrain.index).min().DoW

plt.subplot(222)
plt.title('Aggregated sales over all stores/items for each day-of-week')
sns.boxplot(x=x,y=y)



plt.subplot(223)
plt.title('Holiday Sales per individual store/item for each day-of-week')
x=MyTrainHolidays.DoW
y=MyTrainHolidays.sales
sns.boxplot(x=x,y=y)


plt.subplot(224)
plt.title('Holiday Aggregates Sales over all store/item for each day-of-week')
x=MyTrainHolidays.groupby(MyTrainHolidays.index).min().DoW
y=MyTrainHolidays.groupby(MyTrainHolidays.index).sum().sales
sns.boxplot(x=x,y=y)


# The broad brush that we have observed in the data  is indeed due to  a weekly pattern. This answers the question we asked in cell 18 when we first visualize the data. This conclusion and the above drawings  suggest that:
# 1. When we try to fit a model, perhaps we want to include a feature representing the day of week
# 2. Maybe we want to try to remove the day-of-week variation from the data before we do seasonal analysis.
# 3. Maybe we want to work on weekly totals rather than daily sales. (But doing that might cause issues at month boundaries or year boundaries which do not all start on a monday).
# 4. Holidays have a mixed impact: They depress sales on Wednesday and they improve sales on friday. Their impact on other days is not as clear.
# 
# It looks like option 2 above might be a good choice, we will come back to it later.

# 
# 
# ## 2.3 Trend analysis on total sales
# 
# In this section, I would like to remove the linear growth trend and look at growth year over year.
# 
# To calculate the linear trend, I will average sales each day and fit a straight line through it. ( a variant could be to have different trends for each store/item.... I could test to see which is best.
# 
# 
# 
# 

# In[ ]:


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets

# Take total sale for each day and index days with ordinal values.
GroupedMyTrain = MyTrain.groupby(MyTrain.index).sum()['sales'].reset_index()
GroupedMyTrain['OrdinalDate'] = GroupedMyTrain['date'].map(dt.datetime.toordinal)

# prepare x and y data frames
x = GroupedMyTrain[['OrdinalDate']] 
y = GroupedMyTrain[['sales']]

fit = regr.fit(x,y)
P   = regr.predict(x) 

plt.figure(figsize=(20,8))
plt.annotate('Coef: {}'.format(regr.coef_) , xy=(x.iloc[750].OrdinalDate,P[[750]]),  xycoords='data',
            xytext=(0.35, 0.85), textcoords='axes fraction',
            arrowprops=dict(facecolor='green', shrink=1),
            horizontalalignment='right', verticalalignment='top',
            )

plt.plot(GroupedMyTrain[['date']], y,  color='black', marker='.'  , ls=' ' , markersize=2 )
plt.plot(GroupedMyTrain[['date']], P,  color='green' , linewidth=4 )


# Let's correct the growth by substracting the trend:

# In[ ]:


yCor = y - P
yCor.plot(figsize=(20,5),marker='.',markersize=2,ls=' ')


# This did NOT quite work as the fifth year seems to have a wider range. Let's try a multiplicative correction instead.
# 

# In[ ]:


yCor = y * P[0]/P 
yCor.plot(figsize=(20,5),marker='.',markersize=2,ls=' ')


# This is much improved. It looks like we have achieved some stationarity.  Let's look at how this transformation affected the weekly structure. I will show the corrected sales and the original sales and zoom in at the first few days and the last few days.

# In[ ]:


plt.figure(figsize=(20,5))

plt.subplot(121)
plt.plot(
    x.OrdinalDate[0:140],
    MyTrain.reset_index().groupby(MyTrain.index).sum()['sales'][0:140],
    marker='.',ls=' ',color='blue')
plt.plot(
    x.OrdinalDate[0:140],
    yCor[0:140],
    color='red',marker='.',ls=' ')

plt.subplot(122)
plt.plot(
    x.OrdinalDate[-140:],
    MyTrain.groupby(MyTrain.index).sum()['sales'][-140:],
    marker='.',ls=' ',color='blue')
plt.plot(
    x.OrdinalDate[-140:],
    yCor[-140:],
    color='red',marker='.',ls=' ')


# It seems that the weekly structure was pretty much preserved by the correction.  So let's keep working on it.
# 
# ### 2.2.1 Seasonal Pattern
# 
# Let's sum sales over the five years to get a seasonal pattern.

# In[ ]:


yS = yCor
yS['date']  = GroupedMyTrain.date
yS['DoY']   = yS.date.dt.dayofyear
yS['year']  = yS.date.dt.year
yS['DoW']   = yS.date.dt.dayofweek
yS['month'] = yS.date.dt.month


pattern = yS.groupby(yS.DoY).sum()['sales'] / 10 /50 / 5


# In[ ]:


pattern.plot(marker='.',ls=' ',markersize=2,figsize=(20,5))


# This did not quite work as well as I expected. The brush is still wide but it seemed it affected the weekly pattern. Of course not all years start on the same day of the week.
# 
# I need to refine the model here. Perhaps I need to remove the day of week pattern before I can put in the month of year pattern.
# 
# So let's backtrack, we will first remove the weekly pattern and then compute the seasonal pattern.
# 
# ## 2.2.2 Correcting for Day of Week
# 
# We will calculate a multiplicative factor for each DoW. This factor applies to the average of the week depending on what day we are looking at.
# 

# In[ ]:


DoWFactor = yS.groupby(yS.DoW).sum().sales
DoWFactor = DoWFactor / DoWFactor.mean()
plt.bar(x=DoWFactor.index, height=DoWFactor)


# Now let's apply the DoW factor to the data set before we compute a seasonal pattern.

# In[ ]:


yS2 = MyTrain.groupby([MyTrain.index,MyTrain.DoW]).sum()['sales'].reset_index()
yS2['DoY']   = yS2.date.dt.dayofyear
yS2['month'] = yS2.date.dt.month
yS2['year'] = yS.date.dt.year
yS2['DiM']   = yS2.date.dt.day
yS2['DoW']   = yS2.date.dt.dayofweek
yS2['DoWFactor']   = yS2.groupby(yS2.index).min()['DoW'].map(lambda x: DoWFactor[x])
yS2['GFactor'] = P[0] / P
yS2['CorrSales'] = yS2.sales * yS2.GFactor /yS2.DoWFactor

yS2 = yS2.set_index('date')

pattern = yS2.groupby('DoY').sum()['CorrSales'] / 10 /50 / 5
pattern.plot(marker='.',ls=' ',markersize=2,figsize=(20,5))


# This is much improved and it shows we are making progress in analyzing the data: 
# 
# It looks like the weekly factor helped improve our view of total sales.
# 1.  We clearly see the seasonal pattern 
# 2.  we clearly see the differences every month.
# 3.  We see that year 2014 seems to have very high values.
# 
# We can now make a second attempt at looking at a seasonal pattern.
# 
# ## 2.2.3 Seasonal Pattern Revisited.
# 
# Let's try to average up all five years and compute the seasonal pattern
# 
# 

# This last graph  is much more clear than our previous attempt at line 42.
# We clearly see pretty stable total sales prediction throughout each month.
# But a glitch immediately jumps at us: starting in the third month, there is an outlier at the beginning of every month.
# It does not show in January and in February, but it shows in march.
# 
# Let's zoom in and look at one month. Say April: 
# 
# 
# 

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(121)
pattern[90:120].plot(marker='.',ls=' ',markersize=10)
plt.subplot(122)
sns.distplot(pattern[90:120])
plt.show()


# 
# This could be an effect of the leap year in 2016. Since we have done our computation based on DoY.... and not on month, the February 29th 2016 date, is sometimes added with march data.
# In other words,  day 91 of 2016 is really March 31st.... so adding it to April distorted the April numbers.
# The relevant variables should have been month and Day of Month.... not  Day of Year. 

# In[ ]:


pattern = yS2.groupby(['month','DiM']).sum()['CorrSales'] / 10 /50 / 5
pattern.plot(marker='.',ls=' ',markersize=3,figsize=(20,5))


# We have a clear pattern. Note the lonely point for Feb 29th 2016. 
# let's compute the associated factor for each month.

# In[ ]:


MonthFactor = yS2.groupby(yS2.month).sum().sales
MonthFactor = MonthFactor / MonthFactor.mean()
plt.bar(x=MonthFactor.index, height=MonthFactor)


# ## 2.4 Validation
# 
# In this section we will take another look at the our findings, compute the residual and try to see if we can learn something additional about the data.
# 
# So far we have three factors:
# * A growth trend which we correct with a P[i]/P[0] factor
# * A weekly pattern which we correct with a multiplicative DoWFactor
# * A monthly pattern which we correct with a multiplicative MonthFactor
# 
# Let's apply all three to get a prediction over the training set and look at residual!

# In[ ]:


GroupedMyTrain['DoWFactor']   = MyTrain.groupby(MyTrain.index).min().reset_index()['DoW'].map(lambda x: DoWFactor[x])
GroupedMyTrain['MonthFactor'] = MyTrain.groupby(MyTrain.index).min().reset_index().date.dt.month.map(lambda x: MonthFactor[x])
GroupedMyTrain['GeomFactor']  = P /P[0] # make sure that those P values still come from the Geom calculation


# In[ ]:


RefVal = GroupedMyTrain.sales[0] / GroupedMyTrain['DoWFactor'][0] /GroupedMyTrain['MonthFactor'][0]
GroupedMyTrain['Predicted'] =           RefVal                        *     GroupedMyTrain['DoWFactor']   *     GroupedMyTrain['MonthFactor'] *     GroupedMyTrain['GeomFactor']


# In[ ]:


GroupedMyTrain[['Predicted','sales']].plot(figsize=(20,5),marker='.',markersize=2,ls=' ')


# The Blue dots, showing predicted sales seem to do a relatively good job tracking total sales over the training set.What does the residual look like?
# 

# In[ ]:


GroupedMyTrain['Delta'] = GroupedMyTrain['Predicted'] - GroupedMyTrain['sales']
GroupedMyTrain.plot(x='date',y='Delta',figsize=(20,5),marker='.',markersize=2,ls=' ',grid=True)


# This is interesting. We get two Surprises.
# 
# ***Surprise 1:
# We see some additional structure that we did not expect. It looks like there is a monthly pattern.
# Every other month (except month 7 and 8)  are alternatingly high and low. And month 2 is pretty low. 
# ![](http://)***The immediate thought is that this correlates to the number of days in a month !!!!
# 
# I can't think of a rational reason for this, but we will now adjust the data for the number of days in a month and look at the residual again.
# 
# ***Surprise 2:
# the difference grows over the year.... it looks like we were too ambitious thinking growth would be a day to day trend. It looks like growth might really just happens at the end of the year, and we are looking for a single factor every year.
# 
# 

# In[ ]:


import calendar

def DiM(MyDate) :
    MyYear = MyDate.year 
    MyMonth = MyDate.month
    return calendar.monthrange(MyYear,MyMonth)[1]

GroupedMyTrain['year'] = GroupedMyTrain.date.dt.year
GroupedMyTrain['month'] = GroupedMyTrain.date.dt.month
GroupedMyTrain['DayInMonth'] = GroupedMyTrain.date.map(lambda x: DiM(x))
GroupedMyTrain['DiMFactor']  = 31 / GroupedMyTrain['DayInMonth'] 


GeomFactor2 = GroupedMyTrain.groupby('year').sum()['sales']
GeomFactor2 =  GeomFactor2/GeomFactor2[2013]

GroupedMyTrain['GeomFactor2'] = GroupedMyTrain.year.map(lambda x: GeomFactor2[x])


# In[ ]:


RefVal = GroupedMyTrain.sales[0] / GroupedMyTrain['DoWFactor'][0] /GroupedMyTrain['MonthFactor'][0]
GroupedMyTrain['Predicted'] =           RefVal                        *     GroupedMyTrain['DoWFactor']   *     GroupedMyTrain['MonthFactor'] *     GroupedMyTrain['GeomFactor2'] *     GroupedMyTrain['DiMFactor']


# In[ ]:


GroupedMyTrain['Delta'] = GroupedMyTrain['Predicted'] - GroupedMyTrain['sales']
GroupedMyTrain.plot(x='date',y='Delta',figsize=(20,5),marker='.',markersize=2,ls=' ')


# So DayInMonth Factor was pretty good at removing variations from month to month.
# 
# This is really surprising. The graph above shows daily sales. And somehow the sale amount each day of February depended on the number of days in February. Each day had sales higher than expected because it was a short month!
# 
# Our approach was to multiply all februaries by the same factor.... we do see one more glitch in the graph for February 2016, which was 29 days long instead of 28.  A next iteration would be to fix that. 
# 
# The only explanation I could find was that this is a synthetic data set, and the authors calculated monthly totals and distributed it within the month according to some rule.
# 
# Let's fix February 2016!
# 
# 
# 

# In[ ]:


GroupedMyTrain['DiMFactor'][(GroupedMyTrain.year==2016) & (GroupedMyTrain.month==2)] = 31.0 / 28
GroupedMyTrain['Predicted'] =          RefVal                        *    GroupedMyTrain['DoWFactor']   *    GroupedMyTrain['MonthFactor'] *    GroupedMyTrain['GeomFactor2'] *    GroupedMyTrain['DiMFactor']
GroupedMyTrain['Delta'] = GroupedMyTrain['Predicted'] - GroupedMyTrain['sales']
GroupedMyTrain.plot(x='date',y='Delta',figsize=(20,5),marker='.',markersize=2,ls=' ')



# In[ ]:



from scipy import stats
z,p = stats.normaltest(GroupedMyTrain['Delta'])
sns.distplot(GroupedMyTrain['Delta'])
plt.title("z={:4.2e}    p={:4.2e}".format(z,p))


# **** It looks like we have reduced our training data set to white noise and we are in a good position to use the learnings to make a predictive model.
# 
# In the next section we will try to see if we could have gotten the same results from a canned seasonal analysis.
# 

# # 3 . Seasonal Analysis based on packages.
# 
# In this section, we will use a sklearn to do a seasonal analysis. 

# We have learned that there are a weekly and a yearly seasonal pattern. Let's try to apply both for seasonal analysis.

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
Seasonal = MyTrain.groupby(MyTrain.index).sum()['sales']
SDW = seasonal_decompose(Seasonal, model='additive',freq=7)
SDA = seasonal_decompose(SDW.trend.dropna(), model='additive',freq=365)
fig = plt.figure()  
fig = SDW.plot()
fig = SDA.plot()


# This is pretty good... but not quite as effective as our detailed analysis above. 

# # 4. Next Steps
# 
# I have learned a lot about the dataset and total sales. 
# My next steps will be to take these learnings and use them to make predictions at the store and item level.
# The list of new features (columns) we have added to the data set should be useful for this purpose.
# 
# 

# 
