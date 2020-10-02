# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import warnings
warnings.simplefilter('ignore')


import seaborn
seaborn.set_style('darkgrid')
# ARIMA Model vs Regression Model
# This is a working document

seaborn.mpl.rcParams['figure.figsize'] = (10.0, 6.0)
seaborn.mpl.rcParams['savefig.dpi'] = 90
seaborn.mpl.rcParams['font.family'] = 'sans-serif'
seaborn.mpl.rcParams['font.size'] = 14



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from pandas import datetime
from matplotlib import pyplot
from statistics import mean
from pandas.plotting import autocorrelation_plot
import datetime
import dateutil.parser as dparser
from statsmodels.tsa.arima_model import ARIMA
import datetime as dt
from sklearn.metrics import mean_squared_error
import pyflux as pf
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

#Total Return: 1.520005000000026
#Average Return: 0.11692346153846354
#Trading Days: 13


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
def parser(x):
    return dparser.parse(x)

# Any results you write to the current directory are saved as output.
file_path = '/kaggle/input/fbstock/FB.csv'
X_org = pd.read_csv(file_path)
time_adj_X = pd.read_csv(file_path, header=0, parse_dates=['Date'], index_col=0, squeeze=True, date_parser=parser)
#print(time_adj_X.head())
X = time_adj_X
time_adj_X.plot()
pyplot.show()

# Add var for next day close (y variable) and set index to datetime obj
X['next_day_close']=X.High.shift(-1)
X['prior_day_close']=X.High.shift(+1)
X['next_day_baseline']=X.Close.shift(-1)

X = X.dropna(subset=['next_day_close'])
X = (X.fillna(0)).reset_index()
X = X[X.next_day_close != 0]
X = X[X.prior_day_close != 0]
X_reg = X
X = (X.iloc[1:]).set_index('Date')
# Find autocorrelations
autocorrelation_plot(X)
pyplot.show()

output_variable = 'next_day_close'

# Split train/test batches by index (as index correlates to datetime obj's)
X = X.loc[:, (X != X.iloc[0]).any()]
size = int(len(X) * 0.75)
legh = len(X)
cur_day = X[size:len(X)].Close.values
next_close = X[size:len(X)].next_day_baseline.values
X_ar = X.next_day_close.values
toast = X[size:len(X)]
train, test = X_ar[0:size], X_ar[size:len(X_ar)]

history = [x for x in train]
predictions = list()

perct = 10
# ARIMA Model for time series flattening
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    today = cur_day[t]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('last=%f, predicted=%f, real=%f' % (today, yhat, obs))
    # To show how far you have gone on a percent basis, helpful for longer model runs
    if t % (int(len(test)/10)) == 0 and t != 0:
        print(str(perct)+'% Complete')
        perct+=10
        
# Create list of % change of entire value (for Error Total)
numo=0    
average_error = []
for o in predictions:
    if ((test[numo]-o)/test[numo]) > .05:
        print('Prediction: ' + str(o)[1:-1] + ' Real: ' + str(test[numo]))
    elif ((o-test[numo])/test[numo]) > .05:
        print('Prediction: ' + str(o)[1:-1] + ' Real: ' + str(test[numo]))
    average_error.append(((float(test[numo])-float((o)))/float(test[numo])))
    numo+=1

# Error benchmarks ; Total refers to error % relative to total values ; Change refers to error % relative to change in values
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
print('Mean Absolute Error:', mean_absolute_error(test, predictions))
print('Median Absolute Error:', median_absolute_error(test, predictions))
print('Mean Absolute Percentage Error Total:', mean(map(abs, average_error))*100)
print('Median Absolute Percentage Error Total:', (np.median(list(map(abs, average_error))))*100)
print('Mean Absolute Percentage Error Change:', np.mean(np.abs((test-predictions)/test))*100)
print('Median Absolute Percentage Error Change:', np.median(np.abs((test-predictions)/test))*100)

output = pd.DataFrame({'Date' : toast.index, 'Current' : cur_day, 'Next' : next_close, 'Prediction' : predictions, 'Actual' : test})
output = output.set_index('Date')
print(output.index.min(), output.index.max())
output.loc[(output['Prediction']-output['Current'])/output['Current'] > 0.008, 'Buy'] = 1
output = output.fillna(0)
buy_dates = output.loc[output.Buy==1]
#print(output)
total_returns = []
good_days = buy_dates.loc[(buy_dates['Current']+((buy_dates['Prediction']-buy_dates['Current'])*0.65))<= buy_dates['Actual']]
good_days['Gain_Loss'] = (buy_dates['Prediction'] - buy_dates['Current'])*0.65
good_days =  good_days.apply(lambda x: x.replace('[','').replace(']','')) 

#convert the string columns to int
good_days = good_days.astype(float)

good = good_days['Gain_Loss'].values.tolist()

bad_days = buy_dates.loc[(buy_dates['Current']+((buy_dates['Prediction']-buy_dates['Current'])*0.65))> buy_dates['Actual']]
bad_days['Gain_Loss'] = buy_dates['Next'] - buy_dates['Current']
bad = bad_days['Gain_Loss'].values.tolist()
for trade in bad:
    print(trade)
print(sum(bad))
print(len(bad))
#print('JUGUYFUYGJKGIYJGHFGUYKGUVYKJDGCVUYKUTGUYKGBCVJ,BKJLUJIOGJKYJTHGCGUKLIUHIYJGDUCYJJKJGH')
#for trades in good:
    #print(trades)
print(sum(good))
print(len(good))

values = good+bad

#buy_dates['Change'] = buy_dates['Actual']-buy_dates['Current']
total_return = sum(values)
avg_return = sum(values)/len(values)
#print(buy_dates)
print('Total Return: $' + str(total_return))
print('Average Return: $' + str(avg_return))
print('Money Used: $' + str(max(buy_dates['Current'])))
print('Final Balance: $' + str(total_return+max(buy_dates['Current'])))
print('Percent Return: ' + str(((total_return+max(buy_dates['Current']))/max(buy_dates['Current'])*100)) + '%')
print('Trading Period Length: ' + str(len(output)) + ' Days')
print('Trading Days: '+ str(len(values)) + ' Days')
#output.to_csv('/kaggle/input/fb_stock_predictions')
'''
# I kept the below code as a comparison tool so as to show how ARIMA models flatten the time series into a static dataset
# Regression
X = X_reg
cal = calendar()
holidays = cal.holidays(start=X['Date'].min(), end=X['Date'].max())
X['holiday'] = X['Date'].isin(holidays)
X['holiday'] = X['holiday'].astype(int)

X['Date'] = pd.to_datetime(X['Date'], infer_datetime_format=True)
dates = X['Date']

X = X.assign(year=dates.dt.year,
                           month=dates.dt.month.astype('uint8'),
                           day=dates.dt.day.astype('uint8'),
                           day_of_week=dates.dt.dayofweek,
                           )

X.drop(['Date'], axis=1, inplace=True)

X.loc[X['day'] > 24, 'end_of_month'] = 1
X.loc[X['day'] <= 24, 'end_of_month'] = 0
X.loc[X['day'] < 7, 'beginning_of_month'] = 1
X.loc[X['day'] >= 7, 'beginning_of_month'] = 0

train_day = pd.get_dummies(X['day'], prefix_sep='_', drop_first=True)
train_day_week = pd.get_dummies(X['day_of_week'], prefix_sep='_', drop_first=True, columns=['sun','mon','tues','wed','thurs','fri','sat'])

X.drop(['day', 'day_of_week'], axis=1, inplace=True)
X = pd.concat([X,train_day, train_day_week], axis=1)

X = X.reset_index()

train_, test_ = X[0:size], X[size:len(X)]
train_ = pd.DataFrame(train_)
test_ = pd.DataFrame(test_)
X_train = train_.drop('next_day_close', axis=1)
y_train = train_.next_day_close
X_valid = test_.drop('next_day_close', axis=1)
y_valid = test_.next_day_close

my_model = RandomForestRegressor(n_estimators=100, random_state=0)
my_model.fit(X_train, y_train)
# Predict outcomes from validation data
preds = my_model.predict(X_valid)
numb = 0
valid = y_valid.values
for d in preds:
    if ((valid[numb]-d)/valid[numb]) > .05:
        print('Prediction: ' + str(d) + ' Real: ' + str(valid[numb]))
    numb+=1
# Print mean absolute error for validation data
error_ = mean_squared_error(y_valid, preds)
print('Test MSE: %.3f' % error_)
print('Mean Absolute Error:', mean_absolute_error(y_valid, preds))
print('Median Absolute Error:', median_absolute_error(y_valid, preds))
print('Mean Absolute Percentage Error:', np.mean(np.abs((y_valid-preds)/y_valid))*100)
print('Median Absolute Percentage Error:', np.median(np.abs((y_valid-preds)/y_valid))*100)
'''
