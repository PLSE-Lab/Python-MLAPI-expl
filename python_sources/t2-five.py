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
Created on Sat Nov 18 17:35:08 2017

@author: ArianPrabowo
"""

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

alpha = 0.01;
batchSize=200
nHL1=200
nHL2=50
nIter=20000

nSample = 42000

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Read data
train = pd.read_csv('../input/train.csv')
labels = train.iloc[:,0].values.astype('int32')
X_train = (train.iloc[:,1:].values).astype('float32')
X_train /= X_train.max()
X_train -= X_train.mean()


def liat(x):
    plt.imshow(X_train[x,:].reshape(28,28))

ally=np.zeros((nSample,10),dtype=np.bool_)
ally[np.arange(0,nSample),labels]=1

syn0 = 2*np.random.random((784,nHL1)) - 1
syn1 = 2*np.random.random((nHL1,nHL2)) - 1
syn2 = 2*np.random.random((nHL2,10)) - 1

ze=np.zeros(nIter)
for iter in range(nIter):
    
    iSample=(np.random.random((batchSize))*nSample).astype('int64')
    x=X_train[iSample];
    y=ally[iSample];
    
    # forward propagation
    l0 = x
    l1 = sigmoid(np.dot(l0,syn0))
    l2 = sigmoid(np.dot(l1,syn1))
    l3 = sigmoid(np.dot(l2,syn2))
    
    e=y - l3
    ze[iter]=sum(sum(abs(e)))
    
    print(repr(iter)+' '+repr(ze[iter]))
    
    l3_delta = e*(l3*(1-l3))
    l2_delta = l3_delta.dot(syn2.T) * (l2 * (1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    
    syn0 += alpha*l0.T.dot(l1_delta)
    syn1 += alpha*l1.T.dot(l2_delta)
    syn2 += alpha*l2.T.dot(l3_delta)
    
#plt.imshow(np.concatenate([l3[:30].T,y[:30].astype('float64').T]))
#plt.imshow(syn0)

test = pd.read_csv('../input/test.csv')
X_test = (test.iloc[:,:].values).astype('float32')
X_test /= X_test.max()
X_test -= X_test.mean()

l1 = sigmoid(np.dot(X_test,syn0))
l2 = sigmoid(np.dot(l1,syn1))
y_test = sigmoid(np.dot(l2,syn2))
y = y_test.argmax(axis=1)

pd.DataFrame({"ImageId": list(range(1,len(y)+1)), "Label": y}).to_csv('dr_sbmsn_t1.csv', index=False, header=True)
































