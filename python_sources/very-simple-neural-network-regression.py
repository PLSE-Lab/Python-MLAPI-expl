#!/usr/bin/env python
# coding: utf-8

# This is a minimalist script to perfom regression of the 'House Prices' data set using the keras deep learning library. As an example, for feature selection I have used the top eight features obtained from my scikit-learn [recursive feature elimination script](https://www.kaggle.com/carlmcbrideellis/recursive-feature-elimination-hp-v1).
# The purpose of this script is to serve as a basic starting baseline from which you can launch your own feature engineering, model/parameter tuning with a grid search, stratified k-fold cross validation, different activation functions, net topology, etc etc...

# In[ ]:


#!/usr/bin/python3
# coding=utf-8
#===========================================================================
# This is a minimal script to perform a regression on the kaggle 
# 'House Prices' data set using the keras deep learning library 
# Carl McBride Ellis (15.IV.2020)
#===========================================================================
#===========================================================================
# load up the libraries
#===========================================================================
import pandas  as pd
import numpy   as np
from   keras.models import Sequential
from   keras.layers import Dense             # i.e.fully connected

#===========================================================================
# read in the data from your local directory
#===========================================================================
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

#===========================================================================
# select some features of interest ("ay, there's the rub", Shakespeare)
#===========================================================================
features = ['OverallQual' , 'GrLivArea' , 'TotalBsmtSF' , 'BsmtFinSF1' ,
            '2ndFlrSF'    , 'GarageArea', '1stFlrSF'    , 'YearBuilt'  ]

#===========================================================================
#===========================================================================
X_train       = train_data[features]
y_train       = train_data["SalePrice"]
final_X_test  = test_data[features]

#===========================================================================
# essential preprocessing: imputation; substitute any 'NaN' with mean value 
#===========================================================================
X_train      = X_train.fillna(X_train.mean())
final_X_test = final_X_test.fillna(final_X_test.mean())


#===========================================================================
# parameters for keras
#===========================================================================
input_dim        = X_train.shape[1] # number of neurons in the input layer
n_neurons        =  25       # number of neurons in the first hidden layer
epochs           = 150       # number of training cycles

#===========================================================================
# keras model
#===========================================================================
model = Sequential()        # a model consisting of successive layers
# input layer
model.add(Dense(n_neurons, input_dim=input_dim, 
                kernel_initializer='normal', 
                activation='relu'))
# output layer, with one neuron
model.add(Dense(1, kernel_initializer='normal'))
# compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

#===========================================================================
# train the model
#===========================================================================
model.fit(X_train, y_train, epochs=epochs, verbose=0)

#===========================================================================
# use the model to predict the prices for the test data
#===========================================================================
predictions = model.predict(final_X_test)

#===========================================================================
# write out CSV submission file
#===========================================================================
output = pd.DataFrame({"Id":test_data.Id, "SalePrice":predictions.flatten()})
output.to_csv('submission.csv', index=False)

