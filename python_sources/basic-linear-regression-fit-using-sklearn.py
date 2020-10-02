# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:38:57 2017

@author: Mac Laptop
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in training and test csv's
df=pd.read_csv('train.csv')
dftst=pd.read_csv('test.csv')
# Create dataframes of the input features we want to use
df1=df[['OverallQual','TotalBsmtSF','GrLivArea','FullBath','GarageArea']]
dftst1=dftst[['OverallQual','TotalBsmtSF','GrLivArea','FullBath','GarageArea']]

# Checking test features for nan values (should do this for training set also for completeness)
a=dftst1.as_matrix()
b=np.argwhere(np.isnan(a))
for i in range(0,5):
    for j in range(0,1459):
        if np.isnan(a[j,i]) == True:
            a[j,i]=0
            

from sklearn.linear_model import LinearRegression

# Log-transforms training output for more linearized data sets
y=np.log1p(df['SalePrice'])

# Fitting the training data
lr=LinearRegression()
lr.fit(df1,y)

# out is the predicted outputs from test input features
out=lr.predict(a)
out=np.expm1(out)  # Un-transforms the log output features
plt.scatter(dftst1[['TotalBsmtSF']],out)  # Small chek for correctness

# creates output dictionary dataset to be used in dataframe later.
d={'SalePrice' : out}
dfout=pd.DataFrame(d)
dfout['Id']=dftst['Id']
dfout=dfout[['Id','SalePrice']]  # Rearranges columns in correct order
dfout.to_csv('LinRegDiscreteCont.csv',index=False)