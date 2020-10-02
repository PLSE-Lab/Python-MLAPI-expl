import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline
#libraries

df=pd.read_excel('excel_data',index_col=[0],parse_dates=[0])
# DATASET
series_value=df.values
type(series_value)
#To know about the type od dataset
df.describe()
#Describe the dataset
df.plot()
#plot the graph in respect of x and y
df_mean=df.rolling(window=5).mean()
#moving average(10 day rolliing)
df.plot()
df_mean.plot() #moving_average
#plot the above code
value=pd.DataFrame(series_value)
#baseline model  ( using previous data as next day value(current day value))
df1=pd.concat([value,value.shift(1)],axis=1)
df1.columns=['Actual','forecast'] ######NAIVE MODEL 

from sklearn.metrics import mean_squared_error
import numpy as np

df_test=df1[1:35]

df_test.head()

df_error= mean_squared_error(df_test.Actual,df_test.forecast)

df_error# error 
np.sqrt(df_error)# sqrt error
##ARIMA-MODEL####AUTOREGRESSIVE(P)####INTEGRATED(D)###MOVING AVERAGE(Q)
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

#plot_acf is to identify parameter Q
#ARIMA(p,d,q)

plot_acf(df) #plot_acf is to identify parameter Q
plot_pacf(df) # to identify of value p
# p = 2 d = 0,1 & q = 2

df.size

df_train = df[0:35] #70% of the dataset
df_test = df[35:54]

df_test.shape

from statsmodels.tsa.arima_model import ARIMA

df_model = ARIMA(df_train,order=(2,0,2))

df_model_fit = df_model.fit()

df_model_fit.aic

df_forecast = df_model_fit.forecast(steps = 19)[0]
df_forecast#forecast the value
df_test.shape

np.sqrt(mean_squared_error(df_test,df_forecast))#######error value 

df_forecast = df_model_fit.forecast(steps = 30)[0]
df_forecast

type(df_forecast)#numpy.ndarray

df3= pd.DataFrame(df_forecast)
df4 = df_test.assign(QUANTITY_F = ['8.52692557',  '8.13121023',  '8.68986284',  '9.60577961', '10.15976367',
       '10.04116777',  '9.47785933',  '8.96056252',  '8.84216136',  '9.1198747' ,
        '9.51372272',  '9.72251327',  '9.63963289',  '9.38595464',  '9.17612891',
        '9.14787327',  '9.28194144',  '9.44938922',  '9.52560357'])

df5=df4.drop(['QUANTITY'], axis=1)
df6=df4.drop(['QUANTITY_F'], axis=1)

df7=df5.astype(float)
df7.plot()

plt.plot(df7)#forcasted data
plt.plot(df6,color='red')# test data

####next week forecast(value in numbers) --------######