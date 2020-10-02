# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

## 1.0 Call libraries

import time                   # To time processes
import warnings               # To suppress warnings

import numpy as np            # Data manipulation
import pandas as pd           # Dataframe manipulatio 
import matplotlib.pyplot as plt                   # For graphics

from sklearn import cluster, mixture              # For clustering
from sklearn.preprocessing import StandardScaler  # For scaling dataset

import os                     # For os related operations
import sys                    # For data 

import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
warnings.filterwarnings('ignore')



X= pd.read_csv("../input/2017.csv", header = 0)
X.head(2)

# Make a copy of the original data 
X_tr = X

# 3. Explore and scale
X.columns.values
X.shape                 # 155 X 12
X = X.iloc[:, 2: ]      # Ignore Country and Happiness_Rank columns
X.head(2)
X.columns.values  # Check the column values 
X.dtypes
X.info

ss = StandardScaler()
# 3.1.3 Use ot now to 'fit' &  'transform'
ss.fit_transform(X)



#### 4. Begin Clustering   
                                  
# 5.1 How many clusters
#     NOT all algorithms require this parameter
n_clusters = 3      
                                  
## 5 KMeans
# Ref: http://scikit-learn.org/stable/modules/clustering.html#k-means                                  
# KMeans algorithm clusters data by trying to separate samples in n groups
#  of equal variance, minimizing a criterion known as the within-cluster
#   sum-of-squares.                         

# 5.1 Instantiate object
km = cluster.KMeans(n_clusters =n_clusters )

# 5.2.1 Fit the object to perform clustering
km_result = km.fit_predict(X)

# 5.3 Draw scatter plot of two features, coloyued by clusters
plt.subplot(4, 2, 1)
plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=km_result)

labels=km_result

#labels = km.labels_  # Getting cluster label for each record 

#Glue back to originaal data
#X['clusters'] = labels
#
#X.dtypes
#Add the column into our list
X_tr['clusters']=labels

X_tr = X_tr.iloc[:,[0,12]]

X_tr.dtypes

## K means clustering visualization 

data = [dict(type = 'choropleth',
                  locations =   X_tr['Country'],
                      z = X_tr['clusters'],
                    text =X_tr['Country']
                        ) 
        ]
            
layout = dict(
                title = 'Happiness Report',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )))
        
choromap3 = go.Figure(data = data, layout=layout)
       
iplot(choromap3)
plt.show()