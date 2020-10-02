import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randint
import math as mth
import seaborn as sns

data = pd.read_csv("../input/housingdata.csv", header=None)

train = data.sample(frac=0.6)

test  = data.loc[~data.index.isin(train.index)]
train.reset_index(drop=True, inplace = True)
test.reset_index(drop=True, inplace = True)

alpha= 0.00001

trainY = train.iloc[:,13]
trainX = train.iloc[:,0:13] 


testY = test.iloc[:,13]
testX = test.iloc[:,0:13]



A0 = np.ones(trainX[0].count()) 
A0t = np.ones(testX[0].count())
colindices = range(0,len(trainX.columns)+1)
colindicest = range(0,len(testX.columns)+1)
trainX.insert(loc = 0 ,column = 0 , value = A0, allow_duplicates=True)
testX.insert(loc=0, column = 0, value = A0t, allow_duplicates=True)

trainX.columns = colindices
testX.columns = colindicest

trainY = trainY.transpose()
trainX = trainX.transpose()

testY = testY.transpose()
testX = testX.transpose()


A = 0.1*np.random.rand(trainX[0].count())

Er = 10.0

count = 0
while (Er>0.5):
    count = count+1
    j = randint(0,len(trainX.columns)-1)
    
    for i in range(0, len(trainX)):
     
        Er = (trainY[j] - trainX[0].dot(A))* trainX.iloc[i,j]
     
        A[i] = A[i] + (alpha/len(trainX)) * Er 

    Er = mth.sqrt((trainY[0] - trainX[0].dot(A))**2)
    if (count%1000==0):
        print ("Error =", Er)

print ("no of iterations: ", count)
yPred = []
er = []
xdata = []
Err = []
for i in range(0, len(testX.columns)):
    yPred.append(testX[i].dot(A))
    xdata.append(i)

    SQE = (yPred[i]-testY[i])**2

RMS=mth.sqrt(SQE/len(xdata))
plt.plot(xdata,yPred,'ro',xdata,testY,'g^')

print (RMS)

plt.show()