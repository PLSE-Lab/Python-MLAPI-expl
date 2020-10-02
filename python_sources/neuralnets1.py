import pandas as pd
from numpy import *
#from scipy import io
#import matplotlib.pyplot as plt
#import pybrain
from pybrain.structure import *
from pybrain.datasets import SupervisedDataSet
#from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
#from PIL import Image



def convertToOneOfMany(Y):
    '''converts supervised dataset to softmax classifier'''
    rows, cols = shape(Y)
    numLabels = len(unique(Y))

    Y2 = zeros((rows, numLabels))
    #for i in range(0, rows):
    #    Y2[i, Y[i]] = 1
    
    for i in range(0, numLabels):
        Y2[:, i] = (Y[:,0] == i).astype(int)

    return Y2



def write_preds(preds, fname):
    pd.DataFrame({"ImageId": range(1,len(preds)+1), "Label": preds}).to_csv(fname, index=False, header=True)



# load data    
data = pd.read_csv('../input/train.csv')

image_height, image_width = (20, 20)
pixel_brightness_scaling_factor = data.max().max()


X_train = (data.ix[:,1:].values/pixel_brightness_scaling_factor).astype('float32')
#X_train = X_train.resize(X_train.shape[0], 1, image_height, image_width)


X_test = (pd.read_csv('../input/test.csv').values/pixel_brightness_scaling_factor).astype('float32')
#X_test = X_test.resize(X_test.shape[0], 1, image_height, image_width)


[m, n] = shape(X_train)
#print('n = ' + str(n) + '   m = ' + str(m))


Y = data.ix[:,0].values.astype('int32')
Y = reshape(Y, (len(Y), -1))


numLabels = len(unique(Y))


#threshold the images 
X_train[X_train < 0.5] = 0
X_train[X_train >= .5] = 1

X_test[X_test < 0.5] = 0
X_test[X_test >= .5] = 1


# set sizes of layers
nInput = n
nHidden0 = int(n/10)
nOutput = numLabels


# define layer structures
inLayer = LinearLayer(nInput)
hiddenLayer = SigmoidLayer(nHidden0)
outLayer = SoftmaxLayer(nOutput)


# add layers to network
net = FeedForwardNetwork()
net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)

# define conncections for network
theta1 = FullConnection(inLayer, hiddenLayer)
theta2 = FullConnection(hiddenLayer, outLayer)

# add connections to network
net.addConnection(theta1)
net.addConnection(theta2)

# sort module
net.sortModules()



# create a dataset object, make output Y a softmax matrix
allData = SupervisedDataSet(n, numLabels)
Y2 = convertToOneOfMany(Y)

# add data samples to dataset object, both ways are correct
allData.setField('input', X_train)
allData.setField('target', Y2)


#separate training and testing data
dataTrain= allData


# create object for training
train = BackpropTrainer(net, dataset=dataTrain, learningrate=0.03, momentum=0.3)


# evaluate correct output for trainer
#trueTrain = dataTrain['target'].argmax(axis=1)



# train step by step
EPOCHS = 2
#size = EPOCHS
#accTrain = zeros(size)


train.trainEpochs(EPOCHS)

'''
i = 0
for i in range(EPOCHS):
    train.trainEpochs(1)

  
    # accuracy on training dataset
    outTrain = net.activateOnDataset(dataTrain)
    outTrain = outTrain.argmax(axis=1)
    accTrain[i-1] = 100 - percentError(outTrain, trueTrain)

    print("epoch: %4d " % train.totalepochs,"\ttrain acc: %5.2f%% " % accTrain[i-1])
'''


# create a dataset object for testing
testData = SupervisedDataSet(n, numLabels)


# add data samples to dataset object, both ways are correct
testData.setField('input', X_test)
testData.setField('target', Y2)


outTest = net.activateOnDataset(testData)
outTest = outTest.argmax(axis=1)


write_preds(outTest, "convolutional_nn.csv")