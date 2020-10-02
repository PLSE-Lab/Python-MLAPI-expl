#!/usr/bin/env python
# coding: utf-8

# ## ARIMA vs. KALMAN filtering for modelling stock prices. 

# We have seen many kernals using the autoregressive ARIMA model to model the Goldman Sachs Stock price prediction. Although this model work very well, the library access and training of the models take a considerable processing time.
# 
# We have tried two things here:
# 1) enhance the accuracy of the model 
# 2) reduce the processing time
# 
# Towards both these objectives, we choose to implement a Kalman filtering algorithm. The reason to implement this lies in the non-stationary nature of the stock prices. The ARIMA algorithm first changes the data to a stationary process by taking differences in consecutive samples as input data. In this process, we expect the ARIMA model to loose accuracy.
# In addition, the Kalman filter is known to a simpler algorithm and we expect to get some benefits in processing time. 
# 
# We shall see here that the Kalman filter performs similar to an ARIMA model with processing time of just $0.08$ seconds!! As the time latency is very important in stock market dealings, this improvement in processing time is a significant contribution.

# Lets load the libraries and packages now.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
import seaborn as sns

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


data = pd.read_csv('../input/Data/Stocks/gs.us.txt', sep=',', header=0).fillna(0)
data.head()


# Lets split the stock opening price for Goldman Sachs data in test and training set. We shall see the correlation properties of the data with its previous data. 

# In[ ]:


train_data, test_data = data[0:int(len(data)*0.9)], data[int(len(data)*0.9):]
plt.figure(figsize=(16,8))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Open Prices')
plt.plot(data['Open'], 'red', label='Train data')
plt.plot(test_data['Open'], 'blue', label='Test data')
plt.legend()
plt.title('Opening price for Goldman Sachs')
plt.show()

def shiftLbyn(arr, n=0):
    return arr[n::] + arr[:n:]

def shiftRbyn(arr, n=0):
    return arr[n:len(arr):] + arr[0:n:]
#print(test_data['Open'].values.tolist())

#print(test_data['Open'].corr(test_data['Open'],method='pearson'))

test_lag1=shiftLbyn(test_data['Open'].values.tolist(), 1)
test_lag2=shiftLbyn(test_data['Open'].values.tolist(), 2)
test_lag3=shiftLbyn(test_data['Open'].values.tolist(), 3)

df=pd.DataFrame()
df['data']=test_data['Open'].values.tolist()
df['test_lag1']=test_lag1
df['test_lag2']=test_lag2
df['test_lag3']=test_lag3

print("Lag 1 correlation: "+str(df['data'].corr(df['test_lag1'])))
print("Lag 2 correlation: "+str(df['data'].corr(df['test_lag2'])))
print("Lag 3 correlation: "+str(df['data'].corr(df['test_lag3'])))


##plot correlation with lags
tr=train_data['Open'].values.tolist()
dff=pd.DataFrame()
dff['cor']=np.correlate(tr,tr,mode='full')
dff['cor']=dff['cor'].apply(lambda x:x/dff['cor'].max())

plt.stem(dff['cor'][int((len(dff['cor'])+1)/2):])
plt.xlabel('lags')
plt.ylabel('Correlation')
plt.title('Correlation')
plt.show()



sns.pairplot(df)


# We divided the data in the ratio of 90-10 as train:test dataset. The correlation plot shows that the correlation continues till the end of the data set. It intuitively indicated that the first data sample is somehow related to the last data sample as well. 
# We can observe the scatter plots telling the same story as they all are quite linear for all combinations of lags. we do see some outliers in the scatter plot that arise because of the sudden jump in the test data from 160 to 230. The same jump also results in the trench in the histograms above.
# 
# Anyway, barring the outliers, the scatter plots and the correlation behavious indicates towards a typical autoregressive process where every sample is dependant on the previous sample as follows:
# 
# $X(k)=X(k-1)+n_k$
# where $n_k$ represents a normal gaussian process. If the variance of this noise remains constant with time, it is called stationary process and non-stationary otherwise. Generally, the stock price data tends to be non-stationary.
# 
# A commonly used method for time series is ARIMA model. This model converts a non-stationary process to stationary process by taking the differences of consecutive samples and training the co-efficients of these differences.
# 
# $X(k)=c+\phi_1(X(k-1)-X(k-2))+\phi_2(X(k-2)-X(k-3))+..$
# where $c$ is a constant and $\phi_1, \phi_2....$ are coefficients trained by the ARIMA model. Once these parameters are trained, the test data set is feed forwarded through the ARIMA model for predictions.
# 
# On the other hand, a Kalman filter, being able to work with non-stationary process, tries to imitate the $n_k$ parameter and get the next sample's prediction.
# 
# 
# 
# Lets try the ARIMA model first.

# ## ARIMA model

# In[ ]:


import time
start = time.time()

train_arima = train_data['Open']
test_arima = test_data['Open']
#print(test_arima)
history = [x for x in train_arima]
y = test_arima
# make first prediction
predictions = list()
model = ARIMA(history, order=(1,1,0))
model_fit = model.fit(disp=0)
yhat = model_fit.forecast()[0]
predictions.append(yhat)
history.append(y.iloc[0])
start = time.time()
# rolling forecasts
for i in range(1, len(y)):
    # predict
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit(disp=0)
    yhat = model_fit.forecast()[0]
    # invert transformed prediction
    predictions.append(yhat)
    # observation
    obs = y.iloc[i]
    history.append(obs)
# report performance
mse = mean_squared_error(y, predictions)
print('MSE: '+str(mse))
mae = mean_absolute_error(y, predictions)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(y, predictions))
print('RMSE: '+str(rmse))
#import time

end = time.time()

elapsed = end - start
print("\nTime elapsed:" +str(elapsed)+" s")
#print("\nResult Summary: ")
#print(model_fit.summary())

plt.plot(y.values,label='test data')
plt.plot(predictions,label='ARIMA predictions')
plt.title('ARIMA model')
plt.legend()
plt.show()


# ## Try Kalman filtering

# In[ ]:


#Kalman filter

import time
start = time.time()

test_dataa=test_arima.values

A=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
const=0
P_init=[[10**-7,0,0,0],[0,10**-7,0,0],[0,0,10**-7,0],[0,0,0,10**-7]]
R=[[10**-7,0,0,0],[0,10**-7,0,0],[0,0,10**-7,0],[0,0,0,10**-7]]
Q=[[10**-7,0,0,0],[0,10**-7,0,0],[0,0,10**-7,0],[0,0,0,10**-7]]
KF=[]
update=[]
KF.append(test_dataa[0])
KF.append(test_dataa[1])
KF.append(test_dataa[2])
KF.append(test_dataa[4])
for i in range(4,len(test_dataa)-4):
    x_init=[[test_dataa[i-4]],[test_dataa[i-3]],[test_dataa[i-2]],[test_dataa[i-1]]]
    #prediction
    #print(i)
    prediction=np.dot(A,x_init)+const
    #print(x_min[1])
    P_min=np.dot(np.dot(A,P_init),A)+Q
    KF.append(prediction[3].tolist()[0])
    #measurement update
    y_min=prediction[3]
    #print(y_min)
    P_y_min=P_min+R
    K_gain=np.dot(P_min,np.linalg.inv(P_y_min))[3][3]
    #print(K_gain)
    x_init=prediction-K_gain*(y_min-test_dataa[i])
    update.append(x_init)
    #x_init=np.array([])
    #print(x_init)
    P_init=P_min-K_gain*P_min
#print(KF[0:10])   
#print(test_dataa[0:10])
#df['KF']=KF

mse = mean_squared_error(KF, test_dataa[0:len(test_dataa)-4])
print('MSE: '+str(mse))
mae = mean_absolute_error(KF, test_dataa[0:len(test_dataa)-4])
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(KF, test_dataa[0:len(test_dataa)-4]))
print('RMSE: '+str(rmse))

end = time.time()

elapsed = end - start
print("Time elapsed:" +str(elapsed)+" s")

plt.plot(test_dataa,label='test data')
plt.plot(KF,'green',label='Kalman filter prediction')
plt.title('Kalman filter')
plt.legend()
plt.show()

#print(len(KF))
#print(len(train_arima))


# Alright, so lets compare the performances of the two algorithms.
# 
# | Algorithm | RMSE | Processing Time | MSE | MAE |
# | --- | --- | --- |
# | ARIMA | 2.685970779042733| 19.115743398666382 s | 7.214439025871427 | 2.0280254139213065 |
# | Kalman filter | 2.639540096730153| 0.07828688621520996 s | 6.967171922246225 | 1.9818142548596116 |
# ||||||
# 
# 
# 
# Brilliant! Although we did not see a major benefit in RMSE performance, we definitely see a benefit in MSE and MAE. But the best part is the procesisng time. Where the ARIMA algorithm takes about 19 seconds to achieve this RMSE, we can achieve the same or even better in just 0.08 seconds with Kalman filter!! The benfit in processing time comes from the dynamic nature of Kalman filter and the fact that it avoids training of massive models. The simple formulation of the Kalman filter allows it to keep track of even a non-stationary process as demonstrated in this project. The caveat (ofcourse) is that the Kalman filter works for prediction of immediate neighboring samples as it requires the lates observations for updating the predictions. Unlike the ARIMA models, it cannot predict samples further down the line in future.
# 
# Here I have kept the model in Kalman filter very simple. But you can play around with the matrix "$A$" and the constant "const" to see if it gives any benefits.

# In[ ]:




