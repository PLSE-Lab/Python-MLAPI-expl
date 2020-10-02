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
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 15:20:27 2017

@author: Petar
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

print("This program calculates the accuracy of a regression algolrithm guessing the year of landing based on the coordinates of the meteor...")

#reading the data
df = pd.read_csv('../input//meteorite-landings.csv')

#preparing the data
df1 = df[['reclat','reclong','year']]
df1_clean = df1.dropna()
X= np.array(df1_clean.as_matrix(columns=['reclat','reclong']))
y= np.array(df1_clean.as_matrix(columns=['year']).ravel())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#Setting up the KNeighborsClassifier
#It is used here for its spead (as it compared the data only within a certain range)
reg = KNeighborsClassifier()
reg.fit(X_train,y_train)
print("The accuracy is: ",reg.score(X_test,y_test))
pred=reg.predict(X_test)
print("Some examples of the guessed years, and the correct ones:")
print(pred[10:])
print(y_test[10:])
