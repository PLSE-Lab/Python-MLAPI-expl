# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#read csv using id, latitude and longitude and city columns
df = pd.read_csv('../input/300k.csv', delimiter=',',usecols=["pokemonId", "latitude", "longitude","city", "appearedMinute"])

#restrict dataset to america
chicago = df[(df['city'] == 'Chicago')]
y = chicago['pokemonId'].tolist()
X = np.column_stack((chicago['latitude'].tolist(), chicago['longitude'].tolist(), chicago['appearedMinute'].tolist()))

#custom distance metric
#source: http://jonisalonen.com/2014/computing-distance-between-coordinates-can-be-simple-and-fast/
import math
def dist(a,b):
    deglen = 110.25
    x = a[0] - b[0]
    y = (a[1] - b[1])*math.cos(b[0])
    km = deglen*math.sqrt(x*x + y*y)
    return math.sqrt(km**2 + (a[2]-b[2])**2)
    
#set NN with k=10
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10,metric=dist)

#use k-fold cross validation with k = 5
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(neigh, X, y, cv=5)
from sklearn import metrics
print(metrics.accuracy_score(y, predicted)) 
