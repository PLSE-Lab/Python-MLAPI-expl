#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.seasonal import seasonal_decompose


# In[ ]:


#import data
train_data = pd.read_csv('../input/train.csv',parse_dates=['date'],index_col='date')
test_data = pd.read_csv('../input/test.csv',parse_dates=['date'],index_col='date')


# **Data Exploration**

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data.info()


# In[ ]:


train_data.describe()


# We have 10 stores and 50 items. 

# ** For store 1 and item 1**

# In[ ]:


sales = train_data[(train_data['store']==1)&(train_data['item']==1)]['sales']


# In[ ]:


print(sales)


# In[ ]:


plt.plot(sales)
plt.xlabel('date')
plt.ylabel('sales')
plt.title('STORE 1 ITEM 1')
plt.show()


# In[ ]:


plt.plot(sales.groupby(pd.Grouper(freq='Y')).mean())
plt.xlabel('date')
plt.ylabel('sales')
plt.xticks(rotation=45)
plt.title('Yearly average')
plt.show()


# We can see a clear trend here.

# In[ ]:


plt.plot(sales.groupby(pd.Grouper(freq='M')).mean())
plt.xlabel('date')
plt.ylabel('sales')
plt.xticks(rotation=45)
plt.title('Monthly average')
plt.show()


# Sale is higher in mid of a year. Data is Seasonal.

# In[ ]:


plt.plot(sales.groupby(pd.Grouper(freq='W')).mean())
plt.xlabel('date')
plt.ylabel('sales')
plt.xticks(rotation=45)
plt.title('Weekly average')
plt.show()


# In[ ]:


decomposition = seasonal_decompose(sales.groupby(pd.Grouper(freq='M')).mean(),model='multiplicative')
decomposition.plot()
plt.show()


# Here, we can see the trend, seasonality and residual. 

# **ARIMA Model**

# In[ ]:


stores = train_data['store'].unique()
items = train_data['item'].unique()


# In[ ]:


store  = 1
item = 1
sales = train_data[(train_data['store'] == store)&(train_data['item'] == item)]['sales']
sales = sales.astype('float64')


# In[ ]:


plt.plot(sales)
plt.xlabel('date')
plt.ylabel('sales')
plt.show()


# In[ ]:


#train-validation split
train_size = int(len(sales)*0.7)
train, val = sales[:train_size], sales[train_size:]


# In[ ]:


autocorrelation_plot(sales)
plt.show()


# In[ ]:


#model 
model = ARIMA(train,order=(5,1,0))
model_fit = model.fit(disp=0)


# In[ ]:


print(model_fit.summary())


# In[ ]:


#plot residual error
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.title('Residual Error')
plt.show()


# In[ ]:


residuals.plot(kind='kde')
plt.xlabel('Residual')
plt.show()


# In[ ]:


#predict on validation set
history = [x for x in train]
prediction_val = list()
for t in range(len(val)):
    model = ARIMA(history,order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    prediction_val.append(int(output[0]))
    history.append(val[t])
    


# In[ ]:


error = mse(prediction_val,val)
print('Mean Squared Error = {}'.format(error))


# In[ ]:


#plot
plt.plot(prediction_val,color='red')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Prediction on Validation set')
plt.show()


# In[ ]:


plt.plot(val)
plt.xlabel('date')
plt.ylabel('sales')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#predict on test set
history = [x for x in sales]
prediction_test = list()
for t in range(90):
    model = ARIMA(history,order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    prediction_test.append(int(output[0]))
    history.append(val[t])
    


# In[ ]:


#creating a dataframe
dates = pd.date_range('1/1/2018',periods=90,freq='D')
prediction_test_df = pd.DataFrame(prediction_test,index=dates)
prediction_test_df.columns = ['sales']


# In[ ]:


#plot
plt.plot(prediction_test_df,color='green')
plt.xlabel('date')
plt.ylabel('sales')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


ids = list(test_data[(test_data['store'] == store)&(test_data['item'] == item)]['id'])


# In[ ]:


submission = pd.DataFrame({'id':ids,'sales':prediction_test})


# In[ ]:


print(submission)


# **Similarly,  for other store-item combination,**

# In[ ]:


complete_submission = pd.DataFrame({'id':[],'sales':[]})
for store in stores:
    for item in items:
        sales = train_data[(train_data['store'] == store)&(train_data['item'] == item)]['sales']
        sales = sales.astype('float64')
        #predict on test set
        history = [x for x in sales]
        prediction_test = list()
        for t in range(90):
            model = ARIMA(history,order=(5,1,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            prediction_test.append(int(output[0]))
            history.append(val[t])
            
        #creating a dataframe
        dates = pd.date_range('1/1/2018',periods=90,freq='D')
        prediction_test_df = pd.DataFrame(prediction_test,index=dates)
        prediction_test_df.columns = ['sales']
        ids = list(test_data[(test_data['store'] == store)&(test_data['item'] == item)]['id'])
        submission = pd.DataFrame({'id':ids,'sales':prediction_test})
        
        complete_submission = complete_submission.append(submission)

        


# In[ ]:


complete_submission.to_csv('submission.csv',index=False)


# In[ ]:




