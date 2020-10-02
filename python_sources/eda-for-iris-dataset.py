# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 08:55:37 2018

@author: OMKAR KULKARNI
"""
import numpy as np
import pandas as pd
import os
import gc, sys
gc.enable()

import warnings 
warnings.filterwarnings('ignore',category= DeprecationWarning)

#from sklearn import datasets

iris=pd.read_csv('../input')
print(iris.head())

import matplotlib.pyplot as plt

print(iris.shape)
print(iris.columns)
print(iris.describe())

print (iris['Species'].value_counts())

# plot histogram
plt.figure(figsize=(15,10))
iris.hist()
plt.suptitle('Histogram',fontsize=12)
plt.show()


# plot Boxplot
iris.boxplot()
plt.title('Boxplot',fontsize=12,)
plt.show()

# plot Boxplot by species

iris.boxplot(by='Species',figsize=(15,10))

# create correlation matrix
#print(iris.corr())

iris.groupby(by='Species').mean()



# plot for mean of each feature for each label class
iris.groupby(by='Species').mean().plot(kind='bar')
plt.title('Class vs Measeurements',fontsize=12)
plt.ylabel('mean measurement(cm)')
plt.xticks(rotation=0)
plt.grid(True)

# Use bbox_to_anchor option to place the legend outside plot area to be tidy 
plt.legend(loc="upper left", bbox_to_anchor=(1,1))


#correlation matrix

correlation=iris.corr()
print(correlation)

import statsmodels.api as sm
sm.graphics.plot_corr(correlation, xnames=list(corr.columns))
plt.show()


#Scatter plot
from pandas.tools.plotting import scatter_matrix
scatter_matrix(iris,figsize=(15,15))

plt.suptitle("Pair Plot", fontsize=20) # use suptitle to add title to all sublots


























