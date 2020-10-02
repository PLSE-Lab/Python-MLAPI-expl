# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:56:22 2015

@author: lifeng
"""

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

import numpy
import pandas as pd

print('Read training data...')
df = pd.read_csv('../input/train.csv',header=0).head(1000)
data = df.values
train_label = data[:,0]
train_data = data[:,1:]
ds = SupervisedDataSet(784, 10)
for i in range(0, len(train_label)) :
    vec = [0] * 10
    vec[train_label[i]] = 1
    ds.addSample(train_data[i], vec)
#assert(train_label.shape[0]==train_data.shape[0])
#ds.setField('input',train_data)
#ds.setField('target',train_label)
print('Samples Added') 

###############################################################################
## Build FeedForward Network
###############################################################################
print('Build Network')
n = FeedForwardNetwork()
print('Build Layers')
inLayer = LinearLayer(784)
hiddenLayer = SigmoidLayer(50)
outLayer = LinearLayer(10)
print('Add layers into network')
n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)
print('Build Connections of the layer')
in2hidden = FullConnection(inLayer,hiddenLayer)
hidden2out = FullConnection(hiddenLayer,outLayer)
print('Add connection into network')
n.addConnection(in2hidden)
n.addConnection(hidden2out)

n.sortModules()

trainer = BackpropTrainer(n, ds)
#Maxiter = 20
#for i in range(Maxiter):
#    trainer.trainEpochs(1)

print('Read testing data...')
df_test = pd.read_csv("../input/test.csv",header=0)
test_data = df_test.values
print('Predicting...')
predict = []
for i in range(0, len(test_data)) :
  predict.append(numpy.argmax(n.activate(test_data[i,:])))

print('Predict Done')

print('Submission')
output = predict
imageId = numpy.arange(1,len(predict)+1)
submission = pd.DataFrame({"imageId":imageId,"label":output})
submission.to_csv("submission.csv",index=False)

