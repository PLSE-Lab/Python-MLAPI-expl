#!/usr/bin/python

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

TrainX = pd.read_csv("../input/train.csv").iloc[:,1:]
TrainY = pd.read_csv("../input/train.csv")[[0]].values#.ravel()
Test = pd.read_csv("../input/test.csv").values

#TrainingSet = np.array(TrainX).reshape((-1, 1, 28, 28)).astype(np.uint8)
TestSet = np.array(Test).reshape((-1, 1, 28, 28)).astype(np.uint8)
#print(TrainingSet[0][0])
#print(TrainY)

#plt.imshow(TrainingSet[1729][0], cmap=cm.binary)
#plt.show()

NN = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1, activation='logistic', max_iter=1000 )
NN.fit(TrainX, TrainY)
#print(NN.score(TrainX, TrainY))

prediction = NN.predict(Test)
np.savetxt('submission.csv', np.c_[range(1,len(Test)+1),prediction], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
