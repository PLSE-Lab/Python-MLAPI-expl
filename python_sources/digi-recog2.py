#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 23:58:56 2017

@author: debajyoti
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('../input/train.csv')

X=dataset.iloc[:,1:785].values
Y=dataset.iloc[:,0].values
#img=dataset.iloc[1,1:785].values
#img=np.reshape(img,(28,28))
#plt.imshow(img,cmap='gray')


dataset_test=pd.read_csv('../input/test.csv')
X_test=dataset_test.iloc[:,0:784].values

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)         
X_test=sc_X.transform(X_test)   
#Y_test=dataset_test.iloc[:,0].values
#from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X,Y)
y_pred=classifier.predict(X_test)
#writing to csv
np.savetxt('result.csv', np.c_[range(1,len(X_test)+1),y_pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')


