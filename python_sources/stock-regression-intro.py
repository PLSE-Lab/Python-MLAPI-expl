# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from statistics import mean
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# path = '../input/prices-split-adjusted.csv'

# df = pd.read_csv(path,header=0)
# df = df[['date','open','high','low','close','volume']]
# df['HL_PCT'] = (df['high'] - df['low']) / df['low'] * 100.0
# df['PCT_change'] = (df['close'] - df['open']) / df['open'] * 100.0
# df = df[['close','HL_PCT','PCT_change','volume']]
# forecast_col = 'close'
# df.fillna(value=-99999, inplace=True)
# forecast_out = int(math.ceil(0.01 * len(df)))
# df['label'] = df[forecast_col].shift(-forecast_out)
# X = np.array(df.drop(['label'], 1))
# X = preprocessing.scale(X)
# X = X[:-forecast_out]
# df.dropna(inplace=True)
# y = np.array(df['label'])
# test_sizing = np.linspace(0.1, 0.9, num=30,endpoint=False)
# sampling = np.linspace(1,100,num=30,endpoint=False,dtype=int)
# results = []
# for test in test_sizing:
#     for sample in sampling:        
#         X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test,random_state=sample)
#         clf = LinearRegression(n_jobs=-1)
#         clf.fit(X_train,y_train)
#         accurancy = clf.score(X_test, y_test)
#         res = [accurancy,test,sample]
#         results.append(res)
xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m = (mean(xs) * mean(ys) - mean(xs*ys))/((mean(xs)**2)-(mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m,b
    
def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

m,b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x)+b for x in xs]

r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)

