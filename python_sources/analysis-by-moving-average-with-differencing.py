#!/usr/bin/env python
# coding: utf-8

# **I have tried to find many data bases which can help in this case. I feel if we can get SARS ourbreak time data or other such outbreak in which the virus can be transmitted through person to person as corona does it can be helpful in predicting the future casualties**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# Importing the Data set
# 

# In[ ]:


data = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv")


# In[ ]:


data.tail()


# In[ ]:


data.info()


# In[ ]:


data.set_index('Date',inplace=True)


# In[ ]:


data.index = pd.to_datetime(data.index)


# In[ ]:


data.asfreq = "D"


# In[ ]:


PositiveCase =  data[data['ConfirmedCases']>0]


# In[ ]:



PositiveCase['ConfirmedCases'].plot()


# In[ ]:



PositiveCase['Fatalities'].plot()


# Adding Feature of Average addition of cases on daily basis

# In[ ]:


PositiveCase["ConfirmedCases"][0]


# In[ ]:


PositiveCase['FirstDerivative']= 0
for i in range(len(PositiveCase)):
    if i==0:
        PositiveCase["FirstDerivative"][i]= 0
    else:
        PositiveCase['FirstDerivative'][i]= (PositiveCase['ConfirmedCases'][i]-PositiveCase['ConfirmedCases'][0])/i
    


# In[ ]:


PositiveCase.head()


# In[ ]:


PositiveCase['FirstDerivative'].plot()


# DF test to check for stationarity

# In[ ]:


def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# In[ ]:


Positiveseries = PositiveCase[['ConfirmedCases','Fatalities']]


# In[ ]:


Positiveseries.head()


# In[ ]:


get_ipython().system('pip install pmdarima')


# In[ ]:


from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from statsmodels.tsa.stattools import adfuller
# from pmdarima import auto_arima
from statsmodels.tools.eval_measures import rmse


# In[ ]:


adf_test(Positiveseries['ConfirmedCases'],title="Confirmed Cases")


# In[ ]:


Positiveseries_tr = Positiveseries.diff()


# In[ ]:


Positiveseries_tred = Positiveseries_tr.dropna()
adf_test(Positiveseries_tred['ConfirmedCases'],title="Confirmed Cases")


# In[ ]:


Positive = Positiveseries_tred.diff()


# In[ ]:


Positived =  Positive.dropna()
adf_test(Positived['ConfirmedCases'],title="Confirmed Cases")


# In[ ]:


Positived1 = Positived.diff()


# In[ ]:


Positived1 =  Positived1.dropna()
adf_test(Positived1['ConfirmedCases'],title="Confirmed Cases")


# In[ ]:


Positived1.head()


# In[ ]:


from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults


# In[ ]:


# Trying varius ARIMA model for 3 differencing level


# In[ ]:


ARIMA(1,3,1)


# In[ ]:


model = ARIMA(Positiveseries['ConfirmedCases'],order=(1,0,1))
results = model.fit()
results.summary()


# **As we can see the ARIMA model is failing to provide any results due to lack of data points also ARIMA could not work beyond 2 differencing So No we will try to do in manually using Moving Average Concept**

# In[ ]:


Positiveseries['ConfirmedCasesDiff1']= Positiveseries['ConfirmedCases'].diff()


# In[ ]:


Positiveseries.head()


# In[ ]:


Positiveseries['Fatalitiesdiff1']= Positiveseries['Fatalities'].diff()
Positiveseries['ConfirmedCasesDiff2'] = Positiveseries['ConfirmedCasesDiff1'].diff()
Positiveseries['Fatalitiesdiff2']= Positiveseries['Fatalitiesdiff1'].diff()
Positiveseries['ConfirmedCasesDiff3'] = Positiveseries['ConfirmedCasesDiff2'].diff()
Positiveseries['Fatalitiesdiff3']= Positiveseries['Fatalitiesdiff2'].diff()


# In[ ]:


Positiveseries


# In[ ]:


Positiveseries.plot()


# In[ ]:


Positiveseries[['ConfirmedCasesDiff3','Fatalitiesdiff3']].plot()


# As we notice we could not find any trend in this data Lets try to apply Moving Average

# In[ ]:


#Adding rows for testing set
test = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv",index_col='Date',parse_dates=True)


# In[ ]:


test.head()


# In[ ]:


len(test)


# In[ ]:


test = test.loc['25-03-2020':]


# In[ ]:





# In[ ]:


for i in list(Positiveseries.columns):
    test[i] = 0


# In[ ]:


test.drop(['ForecastId','Province/State','Lat','Long'],inplace=True,axis=1)


# In[ ]:


test.drop('Country/Region',axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:


Positiveseries = Positiveseries.append(test)


# In[ ]:


Positiveseries.tail()


# Taking 3 days Moving Average

# In[ ]:


Positiveseries


# 
# By tring many combinations it seems the current one is working where we take 4 point moving averge for Cases and 2 point MA for fatalities. It loosk around 2nd April the cases should end. 
# 

# In[ ]:


# Positiveseries.asfreq ="D"
for i in range(15,len(Positiveseries)):
    Positiveseries.iloc[i]['ConfirmedCasesDiff3'] = (Positiveseries.iloc[i-1]['ConfirmedCasesDiff3'] + Positiveseries.iloc[i-2]['ConfirmedCasesDiff3']+Positiveseries.iloc[i-3]['ConfirmedCasesDiff3']+Positiveseries.iloc[i-4]['ConfirmedCasesDiff3'])/4
    Positiveseries.iloc[i]['Fatalitiesdiff3'] = (Positiveseries.iloc[i-1]['Fatalitiesdiff3'] + Positiveseries.iloc[i-2]['Fatalitiesdiff3'])/2
    Positiveseries.iloc[i]['ConfirmedCasesDiff2'] = Positiveseries.iloc[i-1]['ConfirmedCasesDiff2']+ Positiveseries.iloc[i]['ConfirmedCasesDiff3']
# #     Positiveseries.iloc[i]['ConfirmedCasesDiff2'] = Positiveseries.iloc[i-1]['ConfirmedCasesDiff2']+ Positiveseries.iloc[i]['ConfirmedCasesDiff3']
    Positiveseries.iloc[i]['ConfirmedCasesDiff1'] = Positiveseries.iloc[i-1]['ConfirmedCasesDiff1']+ Positiveseries.iloc[i]['ConfirmedCasesDiff2']
    Positiveseries.iloc[i]['ConfirmedCases'] = Positiveseries.iloc[i-1]['ConfirmedCases']+ Positiveseries.iloc[i]['ConfirmedCasesDiff1']
    Positiveseries.iloc[i]['Fatalitiesdiff2']= Positiveseries.iloc[i-1]['Fatalitiesdiff2'] + Positiveseries.iloc[i]['Fatalitiesdiff3']
    Positiveseries.iloc[i]['Fatalitiesdiff1']= Positiveseries.iloc[i-1]['Fatalitiesdiff1'] + Positiveseries.iloc[i]['Fatalitiesdiff2']
    Positiveseries.iloc[i]['Fatalities']= Positiveseries.iloc[i-1]['Fatalities'] + Positiveseries.iloc[i]['Fatalitiesdiff1']
    
# Positiveseries['ConfirmedCasesDiff3']['24-03-2020']


# In[ ]:


Output = Positiveseries[['ConfirmedCases','Fatalities']]


# In[ ]:


Output.loc['2020-04-03' :] =0


# In[ ]:


Output.head()


# In[ ]:


submission = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv")


# In[ ]:


submission.head()


# In[ ]:


submission['ConfirmedCases']= list(Output.loc['2020-03-12':]['ConfirmedCases'])


# In[ ]:


submission['Fatalities']= list(Output.loc['2020-03-12':]['Fatalities'])


# In[ ]:


submission.to_csv("submission.csv",index=False)

