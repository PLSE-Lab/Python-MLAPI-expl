import numpy as np
from plotdecboundaries import plotDecBoundaries
import sys

a = np.loadtxt('../input/mdtcm-dataset/Train_csv.csv', dtype=int, delimiter=',')
a = np.loadtxt('../input/mdtcm-dataset/Test_csv.csv', dtype=int, delimiter=',')

(sampleNum, featureNum) = a.shape
classNum = len(np.unique(a[:, 2]))
classMean = np.zeros((classNum, featureNum))
label_train_done = np.zeros(sampleNum)

for x in range(0, classNum):
    classMean[x, :] = np.mean((a[(a[:, 2] == x + 1), :]), axis=0)

training_data = a[:, 0:2]
minError = sys.maxsize
selectedFeature = (0, 1)
for s in range(0, featureNum - 2):
    for t in range(s + 1, featureNum - 1):
        for x in range(0, sampleNum):
            distance_Between_Class1_and_DataPoints = np.square(training_data[x, s] - classMean[0, s]) + np.square(
                training_data[x, t] - classMean[0, t])
            distance_Between_Class2_and_DataPoints = np.square(training_data[x, s] - classMean[1, s]) + np.square(
                training_data[x, t] - classMean[1, t])

            if min(distance_Between_Class1_and_DataPoints, distance_Between_Class2_and_DataPoints) == distance_Between_Class1_and_DataPoints:
                label_train_done[x] = 1
            else:
                label_train_done[x] = 2

        errorNum = (label_train_done != a[:, 2]).sum()
        errorRate = (errorNum / sampleNum)
        accuracy = (1-errorRate)*100

        #print(errorNum)
        #print(sampleNum)
#print(errorRate)
print("Accuracy:", accuracy)

plotDecBoundaries(a[:, 0:2], a[:, 2], classMean[:, 0:2])

