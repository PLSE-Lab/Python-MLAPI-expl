#!/usr/bin/env python
# coding: utf-8

# This notebook explains how to make a simple prediction algorithm based on the mean of the item sales per item and stores. The code shown here has been written to understand how to organize the data. The code is not efficient in terms of computing time but is fast enough. 

# **Organizing the data in a matrix**. Matrix format is easy to understand and manipulate (even more if you come from matlab, which is my case). The training data will be organized in a three-dimensional matrix: date - items - stores. You can understand this as a book where each page contains the dates and the sells of all the items. Each page is an store (makes sense, no?). 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data (use only data from 2017)
train = pd.read_csv('../input/train.csv',skiprows=range(1, 101688780))

#Remove information about returned items
ind = train['unit_sales'] >= 0
train = train[ind]
train.reset_index(inplace=True) #reset the indexes to account for the removed rows
train['date'] = pd.to_datetime(train['date']) # make the date column a valid datetime object

stores = pd.read_csv('../input/stores.csv') # load stores information
items = pd.read_csv('../input/items.csv') #load information about items for sale



grouped = train.groupby(['item_nbr', 'store_nbr']) #group data by items and stores
grouped = grouped.groups # get the groups
groupedKeys = list(grouped.keys()) # save the indexes that forma the groups in a list


dates = pd.date_range(train['date'].iloc[0], train['date'].iloc[-1]) # generate all dates from january 01 of 2017
itemSales = np.zeros((len(dates),len(items),len(stores))) # matrix to save the item sales

#here we go throgh all the groups and organize the data in the three-dimensional matrix (Future kernels will use this)
for k in groupedKeys:
    indItem = np.where(items['item_nbr']==k[0]) # map the item_nbr to the index in the items.csv data
    indStore = k[1] # index for the store
    
    index = grouped[k] #indexes that form te group 'k'
    ind = dates.searchsorted(np.array(train['date'].iloc[index])) #every index is associated with a date, here we map the dates to our new format in the 3D matrix
    
    itemSales[ind,indItem,indStore-1] = train['unit_sales'].iloc[index] #assign the value of the items.
    


# Having the data organized as a matrix we can do several analysis (more later). The simpliest thing to do is use the average sales as a predictor. However, note that in the evaluation, the cost function is the Normalized Weighted Root Mean Squared Logarithmic Error. A look to the equation show that the error is calculated as the difference of the logarithms of the predicted versus the actual values. For this reason, our predictor should use the logarithm of the values to calculate the value of any stimator we calculate. In the case of the calculation of the mean:

# In[ ]:


meanSales = np.expm1(np.mean(np.log1p(itemSales),axis=0))


# Note that we calculate the mean of the log1p() of the sales and then we undo the logarithm using exponential expm1(). Now we can use the calculated mean as prediction for the test data:

# In[ ]:


test = pd.read_csv('../input/test.csv') #load test data

grouped_test = test.groupby(['item_nbr','store_nbr']) # group the data by items and stores
grouped_test = grouped_test.groups
testkeys = list(grouped_test.keys())

predicted = np.zeros((len(test),1)) # initialize the predicted output vector
for k in testkeys:
    # for cada group look in the meanSales the prediction. note that meanSales is 4100X54 (items,stores)
    indItem = np.where(items['item_nbr']==k[0])
    indStore = k[1]
    
    index = grouped_test[k]
    predicted[index,0] = meanSales[indItem,indStore-1]
    
    
# now save the data

submit = pd.DataFrame(np.random.randn(len(test),2), columns=['id', 'unit_sales'])    
submit['id'] = test['id']
submit['unit_sales'] = predicted # undo the log transform
 
submit.to_csv('prediction_01.csv', index = False)


# Once this is submitted, a performance of **0.615** is obtained. Not state of the art, but pretty good for such a simple approach.

# The key points exposed here are as follow: 1. The train data does not contain the days when a particular item is not sold, therefore not organizing the data leads to over-stimation of the mean. 2 working with the log of the data leads to an estimator that is more suited to the proposed task.

# I will be updating this with new methods to estimate the sales and including the information about the oil price and holidays to obtain better stimators.
