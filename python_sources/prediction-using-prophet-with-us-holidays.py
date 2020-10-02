#!/usr/bin/env python
# coding: utf-8

# **Web Traffic Time Series Forecasting: Forecast future traffic to Wikipedia pages**
# 
# **Problem: ** Using time series training data from 1st July 2015 to 31st December 2016, predict the number of page views for ~145 000 Wikipedia articles between 1st January 2017 to 10th November 2017 (in various stages).
# 
# **Approach Taken: **Given that this was a time series forecasting problem, it was a good opportunity to utilise Facebook's Prophet package (https://facebookincubator.github.io/prophet/), which was designed to simplify this type of problem. 
# 
# **Next Steps: ** 
# 1. Fit Prophet models and make predictions for all ~145 000 articles using Google Cloud.
# 2. Find out if sorting pages by language, then accounting for local holiday effects rather than fitting all with US holiday effects,  produces better results.
# 
# **Evaluation: ** Entries are evaluated using SMAPE between forecast and actual values (https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error). 
#         
#                            

# **Data Preperation: **

# In[ ]:


#import relevant Python libraries
import numpy as np 
import pandas as pd 
from fbprophet import Prophet
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime 
from numba import jit
import math


# In[ ]:


#import required data
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

key = pd.read_csv('../input/key_1.csv')
train = pd.read_csv('../input/train_1.csv')

print(train.shape)


# In[ ]:


#generate list of dates
dates = list(train.columns.values)[1:]


# In[ ]:


#convert page views to integers to save memory
for col in train.columns[1:]:
    train[col] = pd.to_numeric(train[col], downcast = 'integer')


# Prophet has no problem working with missing values so I will leave them in.

# In[ ]:


#generate 3 psuedo-random between 0 and 145063 (number of articles) using numpy. 
r1 = 19738 
r2 = 7003 
r3 = 41291 


# In[ ]:


#get randomly chosen time series
r1_ts = train.iloc[r1,:]
r2_ts = train.iloc[r2,:]
r3_ts = train.iloc[r3,:]

print(r1_ts[0])
print(r2_ts[0])
print(r3_ts[0])


# Time series to analyse are relevant to the following articles: <br>
# ts1: https://www.mediawiki.org/wiki/Extension:Maps <br>
# ts2: https://fr.wikipedia.org/wiki/TRAPPIST-1 <br>
# ts3: https://en.wikipedia.org/wiki/Miss_Peregrine%27s_Home_for_Peculiar_Children_(film) <br>
# 
# The first and third articles are accessed by all types of devices, whereas the second article is only accessed by readers using desktop. All articles can be accessed by all types of agents.  

# In[ ]:


#get IDs and page names from the key file
pages = key.Page.values
ids = key.Id.values


# In[ ]:


#concatenate chosen time series into iterable list 
ts_list = [r1_ts, r2_ts, r3_ts]
days = list(range(550)) 


# **Time Series Visualisation: **

# In[ ]:


#plot number of article views against day number for the three time series
for r, n in zip(ts_list, list(range(0,3))):
    plt.plot(days, r[1:])
    plt.ylabel('Number of Views')
    plt.xlabel('Day Number')
    plt.title(r[0])
    plt.show()
   


# Note that data for article 2 did not exist before ~day 300, so training data for this article is more limited.

# In[ ]:


#drop page column from our time series, not necessary for use with Prophet
for t in ts_list:
    t.drop('Page', inplace=True)
    


# In[ ]:


#function to create a DataFrame in the format required by Prophet
def create_df(ts):    
    df = pd.DataFrame(columns=['ds','y'])
    df['ds'] = dates
    df = df.set_index('ds')
    df['y'] = ts.values
    df.reset_index(drop=False,inplace=True)
    return df


# In[ ]:


#get DataFrames suitable for use with Prophet for chosen articles
r1_pro, r2_pro, r3_pro = create_df(r1_ts), create_df(r2_ts), create_df(r3_ts)


# In[ ]:


#check these are in correct format
print(r1_pro.head())
print(r2_pro.head())
print(r3_pro.head())


# In[ ]:


#function to remove outliers
def outliers_to_na(ts, devs):
    median= ts['y'].median()
    #print(median)
    std = np.std(ts['y'])
    #print(std)
    for x in range(len(ts)):
        val = ts['y'][x]
        #print(ts['y'][x])
        if (val < median - devs * std or val > median + devs * std):
            ts['y'][x] = None 
    return ts
        


# In[ ]:


#check number of nan values in each DataFrame prior to outlier removal
print(r1_pro.info())
print('-------------')
print(r2_pro.info())
print('-------------')
print(r3_pro.info())


# In[ ]:


#remove outliers more than 2 standard deviations from the median number of views during the training period
r1_pro, r2_pro, r3_pro = outliers_to_na(r1_pro, 2), outliers_to_na(r2_pro, 2), outliers_to_na(r3_pro, 2)


# In[ ]:


print(r1_pro.info())
print('-------------')
print(r2_pro.info())
print('-------------')
print(r3_pro.info())


# Multiple outliers were removed from all three time series.

# **Prophet Models and Analysis**

# In[ ]:


#dataframe of annual US Public Holidays + 2017 Presidential Inauguration over training and forecasting periods 

ny = pd.DataFrame({'holiday': "New Year's Day", 'ds' : pd.to_datetime(['2016-01-01', '2017-01-01'])})  
mlk = pd.DataFrame({'holiday': 'Birthday of Martin Luther King, Jr.', 'ds' : pd.to_datetime(['2016-01-18', '2017-01-16'])}) 
wash = pd.DataFrame({'holiday': "Washington's Birthday", 'ds' : pd.to_datetime(['2016-02-15', '2017-02-20'])})
mem = pd.DataFrame({'holiday': 'Memorial Day', 'ds' : pd.to_datetime(['2016-05-30', '2017-05-29'])})
ind = pd.DataFrame({'holiday': 'Independence Day', 'ds' : pd.to_datetime(['2015-07-04', '2016-07-04', '2017-07-04'])})
lab = pd.DataFrame({'holiday': 'Labor Day', 'ds' : pd.to_datetime(['2015-09-07', '2016-09-05', '2017-09-04'])})
col = pd.DataFrame({'holiday': 'Columbus Day', 'ds' : pd.to_datetime(['2015-10-12', '2016-10-10', '2017-10-09'])})
vet = pd.DataFrame({'holiday': "Veteran's Day", 'ds' : pd.to_datetime(['2015-11-11', '2016-11-11', '2017-11-11'])})
thanks = pd.DataFrame({'holiday': 'Thanksgiving Day', 'ds' : pd.to_datetime(['2015-11-26', '2016-11-24'])})
christ = pd.DataFrame({'holiday': 'Christmas', 'ds' : pd.to_datetime(['2015-12-25', '2016-12-25'])})
inaug = pd.DataFrame({'holiday': 'Inauguration Day', 'ds' : pd.to_datetime(['2017-01-20'])})

us_public_holidays = pd.concat([ny, mlk, wash, mem, ind, lab, col, vet, thanks, christ, inaug])


# In[ ]:


#function to calculate in sample SMAPE scores
def smape_fast(y_true, y_pred): #adapted from link to discussion 
    out = 0
    for i in range(y_true.shape[0]):
        if (y_true[i] != None and np.isnan(y_true[i]) ==  False):
            a = y_true[i]
            b = y_pred[i]
            c = a+b
            if c == 0:
                continue
            out += math.fabs(a - b) / c
    out *= (200.0 / y_true.shape[0])
    return out


# In[ ]:


#function to remove any negative forecasted values.
def remove_negs(ts):
    ts['yhat'] = ts['yhat'].clip_lower(0)
    ts['yhat_lower'] = ts['yhat_lower'].clip_lower(0)
    ts['yhat_upper'] = ts['yhat_upper'].clip_lower(0)


# ts 1:

# In[ ]:


#fit Prophet model and create forecast 
m = Prophet(yearly_seasonality=True, holidays=us_public_holidays)
m.fit(r1_pro)
future = m.make_future_dataframe(periods=31+28, freq='D', include_history=True)
forecast_1 = m.predict(future)


# In[ ]:


remove_negs(forecast_1)


# In[ ]:


#plot forecasted values and components 
m.plot(forecast_1)
m.plot_components(forecast_1)


# **Analysis: **
# 1. Higher number of views during week days compared to weekends, peaking on Tuesdays.
# 2. Declines in views over Christmas - New Year Holidays reflected through seasonality and holiday effects.
# 
# These trends are consistent with what would be expected of an article related to web page development, which is likely to be done during business hours.

# In[ ]:


#get in sample SMAPE score
print(smape_fast(r1_pro['y'].values, forecast_1['yhat'].values))


# ts 2:

# In[ ]:


m = Prophet(yearly_seasonality=True, holidays=us_public_holidays)
m.fit(r2_pro)
future = m.make_future_dataframe(periods=31+28, freq='D', include_history=True)
forecast_2 = m.predict(future)


# In[ ]:


remove_negs(forecast_2)


# In[ ]:


#plot forecasted values and components 
m.plot(forecast_2)
m.plot_components(forecast_2)


# **Analysis: ** 
# 1. This was a very noisy dataset with only limited data (~6 months), so the conclusions able to be drawn are less substantial than from the other two time series.
# 2. Seasonal and holiday effects appear weak. There is however a spike around Columbus Day 2016. Given that there is no obvious connection between Christopher Columbus and French readers researching the solar system, it is more likely that this spike is due to news regarding the observations of the Spitzer Space Telescope around this time (http://www.trappist.one/#timeline). Important news releases can also be seen around the same time as other spikes in the data. It follows that a model that scrapes scientific news pages may be more appropriate for anticipating future spikes in page views. 

# In[ ]:


print(smape_fast(r2_pro['y'].values, forecast_2['yhat'].values))


# ts 3:

# In[ ]:


m = Prophet(yearly_seasonality=True, holidays=us_public_holidays)
m.fit(r3_pro)
future = m.make_future_dataframe(periods=31+28, freq='D', include_history=True)
forecast_3 = m.predict(future)


# In[ ]:


remove_negs(forecast_3)


# In[ ]:


#plot forecasted values and components 
m.plot(forecast_3)
m.plot_components(forecast_3)


# **Analysis: **
# 1. Increasing trend as the film's release period date approaches.
# 2. Spikes in page views around days of trailer releases (13/03/2016 and 20/06/2016). These could be included in a Prophet model as holidays to stop them adversely affecting future predictions. 
# 3. Weekly seasonality shows rising numbers of views at weekends and declines during the week.
# 4. Holiday effects are ambigious, some have positive effects while others have negative effects. 
# consistent with what would be expected of an article covering a recreational topic like a film. 
# 

# In[ ]:


print(smape_fast(r3_pro['y'].values, forecast_3['yhat'].values))

