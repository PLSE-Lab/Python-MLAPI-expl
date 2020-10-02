# -*- coding: utf-8 -*-

# Imports

import pandas as pd
import numpy as np
import time
import datetime

MAXTIME = 3500
CLEAR='\x1b[J'

def geneToAlpha(g):
    return 2**(-int(g,2))

def geneToTopology(g):
    nHiddenLayer = sum(list(map(int,g)))
    topology = np.zeros((nHiddenLayer+1),dtype=np.int64)
    iLayer=0;
    for i in range(len(g)):
        if g[i] == '1':
            iLayer += 1
        topology[iLayer] += 1
    return topology[1:]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidDerivative(x):
    return x*(1-x)

def trainNN(geneAlpha,geneTopology):
    
    dfTraining = pd.read_csv('../input/train.csv')
    #dfTraining = pd.read_csv('train.csv')
    nTraining = len(dfTraining.index)
    xTraining = (dfTraining.iloc[:,1:].values).astype('float32')
    xTraining /= xTraining.max()
    xTraining -= xTraining.mean()
    nx = xTraining.shape[1]
    yLabels = dfTraining.iloc[:,0].values.astype('int32')
    ny = np.unique(yLabels).size
    yTraining=np.zeros((nTraining,10),dtype=np.bool_)
    yTraining[np.arange(0,nTraining),yLabels]=1
    
    alpha = geneToAlpha(geneAlpha)
    
    topology = geneToTopology(geneTopology)
    synapse = []
    hiddenLayers = []
    synapse.append(2*np.random.random((nx,topology[0]))-1)
    for iLayer in range(topology.size-1):
        synapse.append(2*np.random.random((topology[iLayer],topology[iLayer+1]))-1)
        hiddenLayers.append(np.zeros((topology[iLayer])))
    synapse.append(2*np.random.random((topology[-1],ny))-1)
    hiddenLayers.append(np.zeros((topology[-1])))
    
    error = 0
    timeStart = time.time()
    iIteration = 1
    while time.time()-timeStart < MAXTIME:
        # forward propagation
        hiddenLayers[0] = sigmoid(np.dot(xTraining,synapse[0]))
        for iLayer in range(topology.size-1):
            hiddenLayers[iLayer+1] = sigmoid(np.dot(hiddenLayers[iLayer],synapse[iLayer+1]))
        y = sigmoid(np.dot(hiddenLayers[-1],synapse[-1]))
            
        error = yTraining - y
        
        print(CLEAR)
        print('iteration:     '+repr(iIteration))
        print('max time:      '+repr(MAXTIME))
        print('time elapsed:  '+repr(time.time()-timeStart))
        print('time remainig: '+repr(MAXTIME-time.time()+timeStart))
        print('time per itr:  '+repr((time.time()-timeStart)/iIteration))
        print('last error:    '+repr(sum(sum(abs(error)))))
        
        layerDelta = list(range(topology.size+1))
        layerDelta[-1] = error*sigmoidDerivative(y)
        for iLayer in range(2,topology.size+2):
            layerDelta[-iLayer] = layerDelta[-iLayer+1].dot(synapse[-iLayer+1].T)*sigmoidDerivative(hiddenLayers[-iLayer+1])
        
        synapse[0] += alpha * xTraining.T.dot(layerDelta[0])
        for iLayer in range(1,topology.size):
            synapse[iLayer] += alpha * hiddenLayers[iLayer-1].T.dot(layerDelta[iLayer])
            
        iIteration +=1
    
    return synapse

def testNN(aSynapse):
    dfTesting = pd.read_csv('../input/test.csv')
    #dfTesting = pd.read_csv('test.csv')
    xTesting = (dfTesting.iloc[:,:].values).astype('float32')
    xTesting /= xTesting.max()
    currentLayer = xTesting - xTesting.mean()
    
    for iSynapse in aSynapse:
        currentLayer = sigmoid(np.dot(currentLayer,iSynapse))
    y = currentLayer.argmax(axis=1)
    
    pd.DataFrame({"ImageId": list(range(1,len(y)+1)), "Label": y})\
    .to_csv('drGANN_sbmsn_t1.csv', index=False, header=True)

GA='001101'
GT='0001000111001101010000101100001101111101110111011001011110111110110001101'\
'0001010010100111101010110101011011000010110011010100100'

ts=time.time()
testNN(trainNN(GA,GT))
print('\n\nTotal time elapsed:     '
      +str(datetime.timedelta(seconds=time.time()-ts)))




