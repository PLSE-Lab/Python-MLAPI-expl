#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm

import warnings 
warnings.filterwarnings("ignore")


# In[ ]:


dir_data = "/kaggle/input/temperature-timeseries-for-some-brazilian-cities"
input_data = {}
for filename in os.listdir(dir_data):
    if filename.endswith(".csv"):
        variable_name = filename.split('.')[0]
        input_data[variable_name] = pd.read_csv(os.path.join(dir_data,filename))


# In[ ]:


input_data.keys()


# Can not apply the reduce memory usuage as this changes the values a little bit for me , like it changed 999.90 to 1000.000

# In[ ]:


input_data['station_vitoria'][:30]


# In[ ]:


input_data['station_vitoria'].replace(999.90, np.NaN).fillna(method='ffill')[:10]


# We can go for forward fill also, I have chosen to go with mean of last 12 data points. 

# In[ ]:


for i in input_data.keys():
    for j in input_data[i].columns:
        input_data[i][j] = input_data[i][j].replace(999.90, np.NaN)
        input_data[i][j] = input_data[i][j].fillna(input_data[i][j].rolling(12,1).mean())


# Pattern over the months for given year 

# In[ ]:


for i in input_data.keys():
    input_data[i].drop(['D-J-F','M-A-M','J-J-A','S-O-N'], axis=1, inplace=True)
    df = input_data[i].T
    df.columns = df.iloc[0]
    df.drop(['YEAR'], axis=0, inplace=True)
    
    #plt.figure(figsize=(18,12))
    df.iloc[:-1,-11:].plot(figsize=(12,5), title=i)


# In[ ]:


for i in input_data.keys():
    input_data[i] = pd.melt(input_data[i], id_vars=['YEAR','metANN'], value_vars=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC',], 
                            var_name='month',value_name='Temp')
    input_data[i]['Date'] = pd.to_datetime(input_data[i]['YEAR'].astype(str)+'/'+input_data[i]['month'].astype(str)+'/01')
    input_data[i].drop(['YEAR','month'],axis=1,inplace=True)
    input_data[i].sort_values(by='Date',inplace=True)


# In[ ]:


temp_data = {}
metANN_data = {}

for i in input_data.keys():
    temp_data[i] = input_data[i][['Date','Temp']]
    temp_data[i] = temp_data[i].set_index('Date')
    metANN_data[i] = input_data[i][['Date','metANN']]
    metANN_data[i] = metANN_data[i].groupby(pd.Grouper(key='Date', freq='Y')).mean()


# Imp: Another way is to merge these station's dataframe into one keeping the column as temp_fortaleza, temp_belem and so on. In this way processing time and space can be saved. 
# You can do this way too :)

# In[ ]:


fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(20,12), constrained_layout=True)
fig.suptitle("temp of stations over the years", fontsize=22)

stations = [list(temp_data.keys())[:2], list(temp_data.keys())[2:4],list(temp_data.keys())[4:6],list(temp_data.keys())[6:8],list(temp_data.keys())[8:10],
           list(temp_data.keys())[10:12]]

for row, s in zip(ax,stations):
    for col,i in zip(row, s):
        col.plot(temp_data[i])
        col.set_title(i)

plt.show()


# Observations:
#     1.  Min recorded temperature is in station "station_curitiba" i.e 12.5 in so many years.
#     2.  Max recorded temp is in station "station_manaus" i.e. 32 around in late 20's decade.
#     3. "station_belem", "station_fortaleza" and "station_manaus" has increasing trend somewhat over the years.
#     4. "station_macapa" has trend also variarble trend. 
#     5. "station_recife" has temp increasing trend over the years 1970 to 1990. 

# Now will check for the metANN of stations over the years: 

# In[ ]:


fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(20,12),constrained_layout=True)
fig.suptitle("metANN over the years", fontsize=22)

stations = [list(metANN_data.keys())[:2], list(metANN_data.keys())[2:4],list(metANN_data.keys())[4:6],list(metANN_data.keys())[6:8],
            list(metANN_data.keys())[8:10],list(metANN_data.keys())[10:12]]

for row, s in zip(ax,stations):
    for col,i in zip(row, s):
        col.plot(metANN_data[i])
        col.set_title(i)

#fig.tight_layout()
plt.show()


# Observations:
#     1. Global warming and other factors seems to be powerful as all has increasing trend majorly after 1990. 

# Deompose the data to have look at seasonlity, trend and residulals

# Merge all temp dataframe into one for easy processing:

# In[ ]:


zero_index = list(temp_data.keys())[0]
one_index = list(temp_data.keys())[1]
temp_df = temp_data[zero_index].merge(temp_data[one_index], left_on="Date", right_on='Date', suffixes=('_'+zero_index,'_'+one_index))

for i in list(temp_data.keys())[2:]:
    temp_df = temp_df.merge(temp_data[i], left_on='Date', right_on='Date').rename(columns={'Temp':'Temp_'+i+''})


# In[ ]:


temp_df.head()


# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
for i in temp_df.columns:
    #print(i)
    try:
        decomposition = seasonal_decompose(temp_df[i], model="additive")
    except Exception as e:
        #print(e)
        pass
        
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,3), constrained_layout=True)
    fig.subplots_adjust(wspace=0.15)

    ax1= plt.subplot(121)
    ax1.plot(decomposition.trend)
    ax1.set_title("Trend--> "+i+"")

    ax2 = plt.subplot(122)
    ax2.plot(decomposition.seasonal)
    ax2.set_title("Seasonality--> "+i+"")
    

plt.tight_layout()
plt.show()    


# A pattern of seasonlality is clearly visible in all stations 

# In[ ]:


#Now lets analyze the stationarity of time series :
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller


# In[ ]:


#Rolling Mean and Standard Deviation 
def TestStationaryPlot(ts):
    rol_mean = ts.rolling(window=12, center=False).mean()
    rol_std = ts.rolling(window=12, center=False).std()
    
    plt.figure(figsize=(18,6))
    
    plt.plot(ts, color="red",label="Time Series")
    plt.plot(rol_mean, color="blue", label="Rolling mean")
    plt.plot(rol_std, color="yellow", label="Rolling standard deviation")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Consumption", fontsize=15)
    plt.legend(loc="best", fontsize=15)
    
    plt.title("Rolling Mean and Standard Deviation of Series Data", fontsize=15)
    plt.show(block=True)
    


# In[ ]:


#Adfuller test
#null hypothesis : Series has a unit root 
#True ---> Stationary
#False ---> Non-stationary 

def TestStationaryAdfuller(ts, cutoff=0.05):
    ts_test = adfuller(ts, autolag='AIC')
    ts_test_output = pd.Series(ts_test[0:4], index = ['Test Stats', 'p-value', '#Lags Used', 'Number of observation used'])
    
    for k, v in ts_test[4].items():
        ts_test_output['Critical Value (%s)'%k] = v
        
    
    if ts_test[1] <= cutoff:
        #print("ADF TEST :  Weak evidence against the null hypothesis, reject the null hypothesis. Data has no unit root, hence it is stationary")
        return True
    else:
        #print("ADF TEST :  Strong evidence against null hypothesis, time series has a unit root, indicating it is non-stationary")
        return False


# In[ ]:


#KPSS test 
#null hyposthesis : Series is trend stationary 

from statsmodels.tsa.stattools import kpss

def testKPSStationary(timeseries, cutoff=0.05):
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
 
    if kpsstest[1] <= cutoff:
        #print("KPSS TEST : Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary")
        return False
    else:
        #print("KPSS TEST : Strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root, hence it is stationary")
        return True
    


# In[ ]:


def isStationary(timeseries, station):
    
    Station = station
    adf_result = TestStationaryAdfuller(timeseries)
    kpss_result = testKPSStationary(timeseries)
    
    if (adf_result==False and kpss_result==True):
        Type = 'Trend'
        Stationary = 'No'
    elif(adf_result==True and kpss_result==True):
        Type = np.NaN
        Stationary = 'Yes'
    elif(adf_result==True and kpss_result==False):
        Type = 'Difference'
        Stationary = 'No'
    else:
        Type = np.NaN
        Stationary = 'No'
    
    return pd.DataFrame([{'Station':station, 'Adfuller_result':adf_result, 'KPSS_result':kpss_result, 'Type':Type, 'Stationary':Stationary}])


# **Temperature Data**

# In[ ]:


#Split the data into train and test 
temp_df_train = temp_df[:430]
temp_df_test = temp_df[430:]


# Check if the training temperature data is stationary or not:

# In[ ]:


warnings.filterwarnings("ignore")
stationary_df = pd.DataFrame(columns=['Station','Adfuller_result', 'KPSS_result', 'Type','Stationary'])
for i in temp_df_train.columns:
    try:
        stationary_df = stationary_df.append(isStationary(temp_df_train[i], i))
    except Exception as e:
        print("This series of __"+i+"__ contains nans, will see it later(NaN at the start of the series we can safely drop it)")
print(stationary_df)


# It is clear from the above data that only two station has stationary series i.e. Curitiba and Rio. 
# Rest all are non-stationary, few are diff stationary. We can make them stationary by differencing. 

# In[ ]:


for i in temp_df_train.columns:
    for diff_order in range(1,13):
        d = isStationary((temp_df_train[i] - temp_df_train[i].shift(diff_order)).dropna(), i)
        if d.Stationary.values == 'Yes':
            print(i, diff_order)
            break


# It is clearly visible that all the station series are stationary after 1 diff only. while Curitiba and Rio were stationary without diff also 

# In[ ]:


#This is mean absolute error:
from sklearn.metrics import mean_squared_error
from math import sqrt 

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Applying holt winters model in data of every city of Brazil and checking... 

# In[ ]:


from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
for i in temp_df_train.columns:
    print('\n\n')
    print("********************************************"+i+"*************************************************************")
    y_hat_avg = temp_df_test[i].copy()
    fit1 = ExponentialSmoothing(np.asarray(temp_df_train[i].dropna()) ,seasonal_periods=12 ,trend='add', seasonal='add',).fit()
    y_hat_avg['Holt_Winter'] = fit1.forecast(len(temp_df_test[i]))

    plt.figure(figsize=(16,8))
    plt.plot(temp_df_train[i][252:], label='Train')
    plt.plot(temp_df_test[i], label='Test')
    plt.plot(temp_df_test.index, y_hat_avg['Holt_Winter'], label='Holt_Winter')
    plt.legend(loc='best')
    plt.show()
    print("---------------------------------Mean Absolute Percentage Error------------------------------------------------------")
    print(mean_absolute_percentage_error(temp_df_test[i], y_hat_avg.Holt_Winter))
    print("---------------------------------------RMS----------------------------------------------------------------")
    print(sqrt(mean_squared_error(temp_df_test[i], y_hat_avg.Holt_Winter)))


# Note : Maximum percentage error in station "Temp_station_sao_luiz" because it contains NaN and "Temp_station_curitiba" 

# **metANN Data**

# The same analysis can be applied to the metANN too

# 
# *If you like this kernal, please upvote :) 
#     Thank you*

# In[ ]:




