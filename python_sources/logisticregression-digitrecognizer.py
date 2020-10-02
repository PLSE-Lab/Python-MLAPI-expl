import pandas as pd
from os import listdir
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

print(train)

def sigmoid(inX):
    return 1.0 / (1 + ma.exp(-inX))
    
def gradAscent(trainData, trainLabel, alpha, step):
    trianDataMat = mat(trainData)
    trainLabelMat = mat(trainLabel)
    m, n = shape(trianDataMat)
    theta = ones((n, 1))    #Vectorization
    for i in range(step):
        vecMat = trianDataMat * theta
        error = sigmoid(vecMat) - trainLabelMat
        theta = theta - alpha * trianDataMat.T * error
    return theta
    
def classify(testPath, theta):
    testData, testLabel = loadData(testPath)
    testDataMat = mat(testData)
    testLabel = mat(testLabel)
    h = sigmoid(testDataMat * theta)
    m = len(h)
    error = 0.0
    for i in range(m):
        if int(h[i]) > 0.5:
            print(int(testLabel[i]), ' is classify as: 1')
            if int(testLabel[i]) != 1:
                error += 1
                print('error')
        else:
            print(int(testLabel[i]), ' is classify as: 0')
            if int(testLabel[i]) != 0:
                error += 1
                print('error')
    print('error rate is: %.4f' % (error / m))