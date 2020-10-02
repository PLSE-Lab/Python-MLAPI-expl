# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
TrainData = pd.read_excel('../input/combinedfinaldatanew/FinalTrain.xlsx',sheet_name='combined')

# def printBestParameters(inputData):
#     p = d = q = range(0, 4)
#     lowestrmse=0
#     lowestparam=[]
#     lowestparam_seasonal=[]
#     pdq = list(itertools.product(p, d, q))
#     seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
#     print('Checking parameters...')
#     for param in pdq:
#         for param_seasonal in seasonal_pdq:
#             try:
#                 mod = sm.tsa.statespace.SARIMAX(inputData,
#                                                 order=param,
#                                                 seasonal_order=param_seasonal,
#                                                 enforce_stationarity=False,
#                                                 enforce_invertibility=False)
#                 results = mod.fit()
#                 pred = results.get_prediction(start=pd.to_datetime('2012-01-01'), dynamic=False)
#                 pred_ci = pred.conf_int()
#                 y_forecasted = pred.predicted_mean
#                 y_truth = inputData['2012-01-01':]
#                 mse = ((y_forecasted - y_truth) ** 2).mean()
#                 rmse=np.sqrt(mse)
#                 print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 2)))

#                 if lowestrmse==0:
#                     lowestrmse=rmse
#                 elif lowestrmse>rmse:
#                     lowestrmse = rmse
#                     lowestparam=param
#                     lowestparam_seasonal=param_seasonal
#             except:
#                 continue
#     return lowestparam,lowestparam_seasonal,lowestrmse

# files = rd.loadDataFiles()
# TrainData = files.readingFiles()
ProductFilteredData=TrainData.set_index('MonthYear')

y = ProductFilteredData['Sales(In ThousandDollars)'].resample('MS').mean()
param=(9, 3, 9)
param_seasonal=(2, 0, 1, 12)

mod = sm.tsa.statespace.SARIMAX(y,
                                order=param,
                                seasonal_order=param_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()

pred = results.get_prediction(start=pd.to_datetime('01-2014'),end=pd.to_datetime('12-2015'), dynamic=False)
# print(cols)
pred_ci = pred.conf_int()
# print(pred_ci)
ax = y['2012':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Sales(In ThousandDollars')
plt.legend()
plt.show()
y_forecasted = pred.predicted_mean
y_truth = y['01-2014':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print(y_forecasted)
# y_forecasted.to_csv('../input/combinedfinaldatanew/Sumbission.csv', sep='\t')



print('The Root Mean Squared Error of our forecasts using One-Step is {}'.format(round(np.sqrt(mse), 2)))

# Any results you write to the current directory are saved as output.