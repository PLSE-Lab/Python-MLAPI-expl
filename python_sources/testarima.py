# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Python
# Python
import pandas as pd
import itertools
import numpy as np
import sklearn
import statsmodels
from matplotlib import pyplot
import matplotlib.pyplot as plt
from fbprophet import Prophet
from pandas import Series
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
df = pd.read_excel('../input/test.xls')
# summarize first few rows
print(df.head())
df.describe()
#extraction of coulunms
ener1=df[['Energie']]
Tm=df[['Tmax']]
date= df[['Date']]
y=np.array(ener1)
x=np.array(date)
#plot grphics
plt.figure(figsize=(15, 5))
plt.xlabel("Date")
plt.ylabel("Energie(MWh)")
plt.plot(ener1)
plt.show()
#decomposition
df['Date'] = df['Date'].astype('datetime64[ns]')
df1=pd.DataFrame(y,index=df['Date'])
decomposition = sm.tsa.seasonal_decompose(df1)
fig = decomposition.plot()
plt.show()
#plotting autocorrelation
autocorrelation_plot(df1)
pyplot.show()
lag_acf = acf(df1, nlags=20)
lag_pacf = pacf(df1, nlags=20, method='ols')
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df1)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df1)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df1)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df1)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()
#fit an auto arima model
p = d = q = range(0, 5)
pdq = list(itertools.product(p, d, q))
o=(1,0,0)
mod1=ARIMA(y, order=o)
res1=mod1.fit()
aic1=res1.aic
for i in pdq:
    try:
        mod=ARIMA(y, order=i)
        results = mod.fit()
        if results.aic<=aic1:
            o=i
                
    except:
        continue
print(o)
model=ARIMA(y,order=o)
res=model.fit()
print(res.summary())