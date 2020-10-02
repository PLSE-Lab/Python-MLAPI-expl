# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:59:52 2020

@script_author: Lavin
@last_edited: 13 January 2020
"""

from sklearn import datasets
from sklearn import model_selection as ms
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
iris=datasets.load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=ms.train_test_split(x,y,test_size=.3,random_state=1)
k=range(1,105)
scores=[]
for i in k:
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
scores

           
    