# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

datacsv = "../input/Kaggle_Training_Data.csv"
trainingdata = pd.read_csv(datacsv)

print(" ")
print(trainingdata.columns)

# naming columns

x_i = trainingdata['x'].values
y_i = trainingdata['y'].values

# calculating covariance and variance to find a

data_cov = np.cov(x_i,y_i)[0][1]
data_var = np.var(x_i)


a = data_cov / data_var

# calculating mean of x_i and y_i to find b

xmean = np.mean(x_i)
ymean = np.mean(y_i)

b = ymean - (a * xmean)

print("predicted a value = " + str(a))
print("predicted b value = " + str(b))

# make new y data, make index

numPts   = np.size(x_i,0)
newy_i = a*x_i + b
index = np.arange(0,numPts,dtype = np.int32);

# spreadsheet

spreadsheet = pd.DataFrame({'id': index, 'x' : x_i, 'y' : newy_i})
print(spreadsheet)

submissionName = "KruseSubmission.csv"
spreadsheet.to_csv(submissionName,index=False)