#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pandas_datareader import data
import pandas_datareader as pdr
import seaborn as sns; sns.set()
import datetime as dt
from matplotlib.pylab import plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from matplotlib.pylab import rcParams
from datetime import datetime, timedelta
import re

rcParams['figure.figsize'] = 20,10
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('max_columns', 50)


# In[ ]:


# set start and end date for stock data
def stock_dates(num_of_days_before_now, until_when):
    
    global start
    
    start = datetime.now()-timedelta(days=(num_of_days_before_now)) #8/29
    
    global end
    
    if until_when == 'now':
        end = datetime.now()#-timedelta(days=(1)) #must be day prior to predicted date
    else:
        end = until_when
    
    print('start:', start)
    print('end:', end)


# In[ ]:


stock_dates(600, '2019-11-12')


# In[ ]:


def predict_stock_movement(target_stock_symbol, target_column, stock1=None, stock2=None, stock3=None):
    """creates global variables
    target_stock_symbol: the symbol of the stock that we want to make a prediction on
    target_column: High, Low, Close etc.
    stocks1-3: additional stocks to be included in analysis"""
   
    global target_stock 
    
    target_stock = pdr.get_data_yahoo(target_stock_symbol, start, end)
    
    s1 = pdr.get_data_yahoo(stock1, start, end)
    s2 = pdr.get_data_yahoo(stock2, start, end)
    s3 = pdr.get_data_yahoo(stock3, start, end)

    global stocks 
    stocks = target_stock.merge(s1, 
                      how='left', 
                      on='Date', 
                      suffixes=('{}'.format(target_stock_symbol), '{}'.format(stock1))
                     ).merge(s2, 
                             how='left', 
                             on='Date', 
                             suffixes=('{}'.format(stock1), '{}'.format(stock2))
                            ).merge(s3, 
                                    how='left', 
                                    on='Date', 
                                    suffixes=('{}'.format(stock2),'{}'.format(stock3)))
    global X  
    
    #drop target column
    #exclude last row which is reserved for prediction
    X = stocks.drop(columns=[target_column+'{}'.format(target_stock_symbol)]).iloc[:-1]  ####remove last row
    
    global y 
    
    #our target variable 
    #shifts index negative by one: the date on the target colum is pushed back one day. 
    #the values in the target column belong to the next day, i.e. Jan 1 is from Jan 2
    y = stocks['{}'.format(target_column) + '{}'.format(target_stock_symbol)].shift(-1).iloc[:-1]
    
    global pred_values 
    
    #gets only last row
    pred_values = stocks.drop(columns=[target_column + '{}'.format(target_stock_symbol)]).iloc[[-1]]
    
    return stocks.tail()
    


# In[ ]:


#instantiate function
predict_stock_movement('NTLA', 'High', '^VIX', '^GSPC', '^DJI')

#function returns the last 5 rows of our full stocks dataframe


# In[ ]:


#pred_values


# In[ ]:


#X.tail()


# In[ ]:


#y.tail()


# # Models

# In[ ]:


from xgboost import XGBRegressor
model = XGBRegressor(gamma=1, n_estimators=5000, n_jobs=-1, max_depth=6, objective='reg:squarederror', early_stopping_rounds=50, learning_rate=.005)

#from sklearn.svm import SVR
#model = SVR(kernel='poly', gamma='scale', C=1.0, epsilon=0.1)

#from sklearn.linear_model import LinearRegression
#model = LinearRegression()

#from sklearn.gaussian_process import GaussianProcessRegressor
#model = GaussianProcessRegressor()


# # Test

# In[ ]:


#from sklearn.model_selection import train_test_split, GridSearchCV

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model.fit(X_train, y_train)

preds = model.predict(X_test)

model.score(X_test, y_test)


# In[ ]:


X.tail(1)


# In[ ]:


y.tail(1)


# In[ ]:


#make future prediction
model.fit(X, y)


# In[ ]:


#run a test prediction
#X[X.index] returns a df

def date_to_predict(YYYY, MM, DD):
    
    date_to_predict = datetime(YYYY,MM,DD)
    
    date_to_make_prediction_from = date_to_predict - timedelta(days=(1))
    
    prediction = model.predict(X[X.index == date_to_make_prediction_from])
    
    print(prediction)

    print('actual high:', y.loc[date_to_make_prediction_from])


# In[ ]:


date_to_predict(2019,11,12)


# In[ ]:


pred_values


# In[ ]:


print('predicted high for 10-13-19', model.predict(pred_values))


# In[ ]:


linear test:
predicted high [19.20171624]
actual high: 19.079999923706055

linear forecast: 19.04427673

xbg test:
predicted high [19.477018]
actual high: 19.079999923706055

xbg forecast: 19.529238


# # Make Stock Prediction

# In[ ]:


#fit full data
model.fit(X, y)


# In[ ]:


latest_values


# In[ ]:


model.predict(latest_values)


# # Additional Code

# In[ ]:


plt.figure(figsize=(20,15))

tvix_close = tvix['Close']
tvix_open = tvix['Open']
sp_close = sp['Close'] - 2600
dj_close = dj['Close'] - 25000 
spy_close = spy['Close'] - 281
spy_open = spy['Open'] - 281
tvix_move = tvix['movement']


#spy_open.plot(label='so', style='-')
#spy_close.plot(label='sc', style='-')

#tvix_open.plot(label='to', style='o')
#tvix_close.plot(label='close')
#tvix_move.plot(label='move', style='o')

#sp500_close.plot()
#dj_close.plot()
#plt.legend(loc='lower left')
#plt.errorbar(tvix_move.index, tvix_move, yerr=.5, marker='s', mfc='red', mec='green', ms=20, mew=4);
#plt.plot(tvix_move.index, tvix_close);
plt.boxplot(tvix_move)


# In[ ]:


stock.plot(alpha=.5, style='.')
stock.resample('BA').mean().plot(style=':')
stock.asfreq('BA').plot(style='--');

