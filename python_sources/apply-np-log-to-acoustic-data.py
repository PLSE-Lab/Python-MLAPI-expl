#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
STEPSIZE=150000
#print(os.listdir("../input"))
# due to huge volume of data, only select 33%
acoustic=pd.read_csv("../input/train.csv",nrows =210000000)
# select partial of the data to plot
filter1=range(0,len(acoustic),100)
plt.plot(acoustic.iloc[filter1,0])
# Any results you write to the current directory are saved as output.


# As we can see from this image, most of acoustic value is betwen (-100,100), while maximun is over 3000 or below -3000. No matter analysis by human being or machine code, we tend to focus on the extreme value, and overlook data which vary in a smaller range. This will impact final performance significantly. The longer 'time_to_failure', the bigger impact it will have, because at every cycle beginning, the acoustic value is in a small range.
# 

# By applying function log (np.log10), it help us to evenly analysis acoustice in every stage. Below is the python code to smoothen acoustic data

# In[ ]:


def logX(data):
    # Handle positive value
    positive = np.where(data.iloc[:,0]>0,data.iloc[:,0],0)+1
    logp=np.log10(positive)
    # Handle negative value, convert to ABS
    absneg = np.where(data.iloc[:,0]<0,data.iloc[:,0]*-1,0)+1
    logn=np.log10(absneg)*-1
    logall = logn+logp
    data.iloc[:,0]=logall
    return data
acoustic=logX(acoustic)
plt.plot(acoustic.iloc[filter1,0])


# From above result we can see the value fluctuating much less.
# 
# 
# Remaining code continue to build a sklearn MLPRegressor model and validate the result.
# The function dataReFeaturing() is referenced from [Earthquakes FE. More features and samples](http://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples) by Andrew Lukyanenko

# In[ ]:


def dataReFeaturing(dataSource,shiftStep=STEPSIZE,includeY=True):
    # shiftStep will enable overlap of test data.
    # when dataSource is test data, includeY will be False
    
    rows = STEPSIZE
    segments = int(np.ceil(dataSource.shape[0] / shiftStep))
    
    X_tr = pd.DataFrame(index=range(segments), dtype=np.float64,
                           columns=['ave', 'std', 'max', 'min',
                                   'av_change_abs', 'av_change_rate', 'abs_max', 'abs_min',
                                   'std_first_50000', 'std_last_50000', 'std_first_10000', 'std_last_10000',
                                   'avg_first_50000', 'avg_last_50000', 'avg_first_10000', 'avg_last_10000',
                                   'min_first_50000', 'min_last_50000', 'min_first_10000', 'min_last_10000',
                                   'max_first_50000', 'max_last_50000', 'max_first_10000', 'max_last_10000'])
    y_tr = pd.DataFrame(index=range(segments), dtype=np.float64,
                           columns=['time_to_failure'])
    
    for segment in range(segments):
        if (segment*rows+rows > len(dataSource)):
            segTo=len(dataSource)
            segFrom=segTo-rows
        else:
            segFrom=segment*rows
            segTo=segment*rows+rows
        
        seg = dataSource.iloc[segFrom:segTo]
        x = seg['acoustic_data'].values
        if(includeY):
            y = seg['time_to_failure'].values[-1]
            y_tr.loc[segment, 'time_to_failure'] = y
        else:
            y_tr.loc[segment, 'time_to_failure'] = 0  # for test data
        X_tr.loc[segment, 'ave'] = x.mean()
        X_tr.loc[segment, 'std'] = x.std()
        X_tr.loc[segment, 'max'] = x.max()
        X_tr.loc[segment, 'min'] = x.min()
        
        
        X_tr.loc[segment, 'av_change_abs'] = np.mean(np.diff(x))
        X_tr.loc[segment, 'av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
        X_tr.loc[segment, 'abs_max'] = np.abs(x).max()
        X_tr.loc[segment, 'abs_min'] = np.abs(x).min()
        
        X_tr.loc[segment, 'std_first_50000'] = x[:50000].std()
        X_tr.loc[segment, 'std_last_50000'] = x[-50000:].std()
        X_tr.loc[segment, 'std_first_10000'] = x[:10000].std()
        X_tr.loc[segment, 'std_last_10000'] = x[-10000:].std()
        
        X_tr.loc[segment, 'avg_first_50000'] = x[:50000].mean()
        X_tr.loc[segment, 'avg_last_50000'] = x[-50000:].mean()
        X_tr.loc[segment, 'avg_first_10000'] = x[:10000].mean()
        X_tr.loc[segment, 'avg_last_10000'] = x[-10000:].mean()
        
        X_tr.loc[segment, 'min_first_50000'] = x[:50000].min()
        X_tr.loc[segment, 'min_last_50000'] = x[-50000:].min()
        X_tr.loc[segment, 'min_first_10000'] = x[:10000].min()
        X_tr.loc[segment, 'min_last_10000'] = x[-10000:].min()
        
        X_tr.loc[segment, 'max_first_50000'] = x[:50000].max()
        X_tr.loc[segment, 'max_last_50000'] = x[-50000:].max()
        X_tr.loc[segment, 'max_first_10000'] = x[:10000].max()
        X_tr.loc[segment, 'max_last_10000'] = x[-10000:].max()
    
    return pd.concat([X_tr, y_tr], axis=1, sort=False)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
TRAINRATIO=0.7
trainSet1=acoustic.iloc[0:int(len(acoustic)*TRAINRATIO),:]
valSet1=acoustic.iloc[int(len(acoustic)*TRAINRATIO):-1,:]
trainSet2=dataReFeaturing(trainSet1,int(STEPSIZE/3))
scaler = MinMaxScaler()
trainSetNorm=scaler.fit_transform(trainSet2)
np.random.shuffle(trainSetNorm)

# build MLPRegressor model
from sklearn.neural_network import MLPRegressor
x_train=trainSetNorm[:,0:-1]
y_train=trainSetNorm[:,-1]
model = MLPRegressor(solver='lbfgs', 
                    alpha=0.0001,
                    hidden_layer_sizes=(800,400,200), 
                    random_state=1,
                    activation='relu',
                    verbose=False)
model.fit(x_train, y_train)

# validate model
valSet2=dataReFeaturing(valSet1,STEPSIZE)
valSetNorm=scaler.transform(valSet2)
x_val=valSetNorm[:,0:-1]
y_val=valSetNorm[:,-1]
y_pred_val=model.predict(x_val)
#------------------------------------------------------
# plotting result
#------------------------------------------------------
plt.plot(y_val, "b.")
plt.plot(y_pred_val,"r-")
plt.xlabel("Time")
plt.ylabel("Value")


# Above is model validate result
# By comparison, if not call function logX(), the result will be much worse

# In[ ]:


acoustic=pd.read_csv("../input/train.csv",nrows =210000000)
trainSet1=acoustic.iloc[0:int(len(acoustic)*TRAINRATIO),:]
valSet1=acoustic.iloc[int(len(acoustic)*TRAINRATIO):-1,:]
trainSet2=dataReFeaturing(trainSet1,int(STEPSIZE/3))
scaler = MinMaxScaler()
trainSetNorm=scaler.fit_transform(trainSet2)
np.random.shuffle(trainSetNorm)

# build MLPRegressor model
from sklearn.neural_network import MLPRegressor
x_train=trainSetNorm[:,0:-1]
y_train=trainSetNorm[:,-1]
model = MLPRegressor(solver='lbfgs', 
                    alpha=0.0001,
                    hidden_layer_sizes=(800,400,200), 
                    random_state=1,
                    activation='relu',
                    verbose=False)
model.fit(x_train, y_train)

# validate model
valSet2=dataReFeaturing(valSet1,STEPSIZE)
valSetNorm=scaler.transform(valSet2)
x_val=valSetNorm[:,0:-1]
y_val=valSetNorm[:,-1]
y_pred_val=model.predict(x_val)
#------------------------------------------------------
# plotting result
#------------------------------------------------------
plt.plot(y_val, "b.")
plt.plot(y_pred_val,"r-")
plt.xlabel("Time")
plt.ylabel("Value")

