# This kernel is designed to be straight to the point of our model in the competition
# This is a synthetic reproduction of our full codes, so it is likely that RMLSE would differ a bit
# For detailed parts, we explain it in our report

__author__ = '4185_EVA'

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# import necessities
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# initialize random states
np.random.seed(4185)
random.seed(4185)

# define date parser
date_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# load data
BikeShare  = pd.read_csv('/kaggle/input/bike-share-demand/train.csv', parse_dates = ['datetime'], date_parser = date_parser)
To_Predict = pd.read_csv('/kaggle/input/bike-share-demand/test.csv', parse_dates = ['datetime'], date_parser = date_parser)

# copy the data, so we don't lose the original data
BS = deepcopy(BikeShare)
TP = deepcopy(To_Predict)

# manipulate the data, obtained after analysis
BS['temp']       = (47*BS['temp']) - 8
BS['atemp']      = (66*BS['atemp']) - 16
BS['humidity']   = 100*BS['humidity']
BS['windspeed']  =  67*BS['windspeed']
TP['temp']       = (47*BS['temp']) - 8
TP['atemp']      = (66*BS['atemp']) - 16
TP['humidity']   = 100*BS['humidity']
TP['windspeed']  =  67*BS['windspeed']

# set categorical data
BS['season']     = BS['season'].astype('category')
BS['holiday']    = BS['holiday'].astype('category')
BS['weekday']    = BS['weekday'].astype('category')
BS['workingday'] = BS['workingday'].astype('category')
BS['weather']    = BS['weather'].astype('category')
TP['season']     = TP['season'].astype('category')
TP['holiday']    = TP['holiday'].astype('category')
TP['weekday']    = TP['weekday'].astype('category')
TP['workingday'] = TP['workingday'].astype('category')
TP['weather']    = TP['weather'].astype('category')

# separate season per values, for feature enhancement
train_season     = pd.get_dummies(BS['season'], prefix = 'season')
BS               = pd.concat([BS, train_season], axis = 1)
test_season      = pd.get_dummies(TP['season'], prefix = 'season')
TP               = pd.concat([TP, test_season], axis = 1)

# do the same for weather
train_weather    = pd.get_dummies(BS['weather'], prefix = 'weather')
BS               = pd.concat([BS, train_weather], axis = 1)
test_weather     = pd.get_dummies(TP['weather'], prefix = 'weather')
TP               = pd.concat([TP, test_weather], axis = 1)

# extract datetime as own features
BS['year']       = [el.year for el in BS['datetime']]
BS['year']       = BS['year'].map({2011: 0, 2012: 1})
BS['month']      = [el.month for el in BS['datetime']]
BS['day']        = [el.dayofweek for el in BS['datetime']]
BS['hour']       = [el.hour for el in BS['datetime']]
TP['year']       = [el.year for el in TP['datetime']]
TP['year']       = TP['year'].map({2011: 0, 2012: 1})
TP['month']      = [el.month for el in TP['datetime']]
TP['day']        = [el.dayofweek for el in TP['datetime']]
TP['hour']       = [el.hour for el in TP['datetime']]

# define extraction function
def extractData(data, to_drop):
    y = data[to_drop]
    x = data.drop(to_drop, axis = 1)
    return x, y

# extract the data
BS, _            = extractData(BS, ['datetime', 'season', 'weather', 'cnt'])
TP, _            = extractData(TP, ['datetime', 'season', 'weather'])
x, y             = extractData(BS, ['casual','registered'])

# predict test data
data, target     = x, y
model            = MultiOutputRegressor(GradientBoostingRegressor())
model.fit(data, target)
Preds            = model.predict(TP)
Preds            = pd.DataFrame(Preds)
Preds.columns    = ['casual', 'registered']
for col in Preds.columns: # rectified linear unit
    Preds[col]  = [max(0, el) for el in Preds[col]]

# prepare submission
example          = pd.read_csv('/kaggle/input/bike-share-demand/sampleSubmission.csv')
Out              = deepcopy(example)
Out['count']     = Preds['casual'] + Preds['registered']
Out.to_csv('4185_EVA Submission.csv', index = False)