import theano.tensor as T
import theano
import numpy as np
import time 
import sys
from theano.tensor.signal.downsample import max_pool_2d as maxPool2d
from matplotlib import pyplot as plt

trainPercentage = 0.8
testPercentage = 0.2


def loadData(filename):
    x, y = [], []
    tests = []
    f = open(filename, 'r')
    line = f.readline()
    for line in f:
        tests.append(np.array(list(map(int, line.split(',')))))
    f.close()

    np.random.seed(42)
    np.random.shuffle(tests)
    for test in tests:
        x.append((np.array(test[1:]) ).reshape(1, 28, 28) / 256.0)
        tmp = [0] * 10
        tmp[int(test[0])] = 1                                                                                                                                                           
        y.append(np.array(tmp))
        
    return np.array(x), np.array(y).T

class Visualizer:
    def __init__(self, imagePrefix):
        self.accuracy = []
        self.cost = []
        self.imagePrefix = imagePrefix

        self.figure = plt.figure()
        self.accuracyPlot = self.figure.add_subplot(121)
        self.costPlot = self.figure.add_subplot(122)

    def addData(self, epoch=None, cost=None, accuracy=None):
        if accuracy:
            self.accuracy.append((epoch, accuracy))
        if cost:
            self.cost.append((epoch, cost))

        if accuracy and len(self.accuracy) > 1:
            self.accuracyPlot.plot([self.accuracy[-2][0], self.accuracy[-1][0]], [self.accuracy[-2][1], self.accuracy[-1][1]], 'r-')

        if cost and len(self.cost) > 1:
            self.costPlot.plot([self.cost[-2][0], self.cost[-1][0]], [self.cost[-2][1], self.cost[-1][1]], 'r-')

        self.figure.savefig(self.imagePrefix+'.png')


class InputImageLayer:
    def __init__(self, batchSize, imageShape):
        self.batchSize = batchSize
        self.output = self.trainOutput = T.ftensor4("input")
        self.trainOutputShape = (batchSize, 1) + imageShape
        self.outputShape = (1, 1) + imageShape
        self.params = []

class FullLayer:    
    def __init__(self, prevLayer, neuronCount, numpyRng):
        self.batchSize = prevLayer.batchSize
        self.w = theano.shared(numpyRng.normal(scale=0.1, size=(neuronCount, prevLayer.outputShape)).astype(np.float32))
        self.bias = theano.shared(numpyRng.normal(loc=0.5, scale=0.5, size=(neuronCount,)).astype(np.float32))

        self.outputShape = self.trainOutputShape = neuronCount
        self.trainOutput = T.nnet.softplus(T.dot(self.w, prevLayer.trainOutput) + self.bias.dimshuffle(0, 'x'))
        self.output = T.nnet.softplus(T.dot(self.w, prevLayer.output) + self.bias.dimshuffle(0, 'x'))
        self.params = prevLayer.params + [self.w, self.bias]


class OutputLayer:
    def __init__(self, prevLayer, outputSize, numpyRng):
        self.batchSize = prevLayer.batchSize
        self.w = theano.shared(numpyRng.normal(scale=0.1, size=(outputSize, prevLayer.outputShape)).astype(np.float32))
        self.bias = theano.shared(numpyRng.normal(scale=1.0, size=(outputSize,)).astype(np.float32))

        self.outputShape = self.trainOutputShape = outputSize 
        #self.trainOutput = T.clip(T.nnet.sigmoid(T.dot(self.w, prevLayer.trainOutput) + self.bias.dimshuffle(0, 'x')), 1e-6, 1 - 1e-6)
        self.trainOutput = T.nnet.sigmoid(T.dot(self.w, prevLayer.trainOutput) + self.bias.dimshuffle(0, 'x'))
        self.output = T.nnet.sigmoid(T.dot(self.w, prevLayer.output) + self.bias.dimshuffle(0, 'x'))
        self.params = prevLayer.params + [self.w, self.bias]

class DropoutLayer:
    #dropoutRate shows how many neurons remains
    def __init__(self, prevLayer, theanoRng, dropoutRate):
        self.batchSize = prevLayer.batchSize
        self.dropout = theanoRng.binomial(size=(prevLayer.outputShape,), p=dropoutRate).dimshuffle(0, 'x').astype('float32')
        self.trainOutput = prevLayer.trainOutput * self.dropout
        self.output = prevLayer.output * dropoutRate
        self.outputShape = prevLayer.outputShape
        self.trainOutputShape = prevLayer.trainOutputShape
        self.params = prevLayer.params

class ConvolutionLayer:
    def __init__(self, prevLayer, filterShape, numpyRng):
        self.batchSize = prevLayer.batchSize
        
        imgCount = prevLayer.outputShape[1]

        self.w = theano.shared(numpyRng.normal(scale=0.1, size=(filterShape[0], imgCount) + filterShape[1:]).astype(np.float32))
        self.bias = theano.shared(numpyRng.normal(loc=0.5, scale=0.5, size=(filterShape[0],)).astype(np.float32))

        trainImg = T.nnet.conv2d(prevLayer.trainOutput, self.w)
        self.trainOutput = T.nnet.softplus(trainImg + self.bias.dimshuffle('x', 0, 'x', 'x'))
        
        outputImg = T.nnet.conv2d(prevLayer.output, self.w)
        self.output = T.nnet.softplus(outputImg + self.bias.dimshuffle('x', 0, 'x', 'x'))

        newImageShape = tuple(prevLayer.outputShape[i + 2] - filterShape[i + 1] + 1 for i in range(2))
        self.trainOutputShape = (self.batchSize, filterShape[0]) + newImageShape
        self.outputShape = (1, filterShape[0]) + newImageShape

        self.params = prevLayer.params + [self.w, self.bias]

class MaxPoolLayer:
    def __init__(self, prevLayer, downsampleFactor):
        self.batchSize = prevLayer.batchSize
        self.output = maxPool2d(prevLayer.output, downsampleFactor) 
        self.trainOutput = maxPool2d(prevLayer.trainOutput, downsampleFactor) 

        newImageShape = tuple(prevLayer.outputShape[i + 2] // downsampleFactor[i] for i in range(2))
        self.outputShape = prevLayer.outputShape[:2] + newImageShape
        self.trainOutputShape = prevLayer.trainOutputShape[:2] + newImageShape

        self.params = prevLayer.params

class UnrollLayer:
    def __init__(self, prevLayer):
        self.batchSize = prevLayer.batchSize
        self.outputShape =  np.prod(prevLayer.outputShape)
        self.trainOutputShape =  (np.prod(prevLayer.trainOutputShape) // self.batchSize, self.batchSize)
        self.output = prevLayer.output.reshape((self.outputShape, 1))
        self.trainOutput = prevLayer.trainOutput.reshape((self.trainOutputShape[1], self.trainOutputShape[0])).T
        self.params = prevLayer.params


class NeuralNetwork:
    def __init__(self, inputLayer, outputLayer):
        self.batchSize = outputLayer.batchSize
        self.inp = inputLayer.output
        self.answer = T.fmatrix("y")
        self.cost = T.sum(T.nnet.binary_crossentropy(outputLayer.trainOutput, self.answer)) / self.batchSize
        print('Compiling prediction function', file=sys.stderr)
        self.prediction = theano.function(inputs=[inputLayer.output], outputs=T.argmax(outputLayer.output))
        print('Finished', file=sys.stderr)
        ans = T.fscalar('ans')
        print('Compiling error function', file=sys.stderr)
        self.error = theano.function(inputs=[inputLayer.output, ans], outputs=(T.eq(T.argmax(outputLayer.output), ans)))
        print('Finished', file=sys.stderr)
 
        print('Compiling batchCost function', file=sys.stderr)
        self.batchCost = theano.function(inputs=[inputLayer.trainOutput, self.answer], outputs=self.cost)
        print('Finished', file=sys.stderr)

        self.inputLayer = inputLayer
        self.outputLayer = outputLayer
         
    def train(self, trainX, trainY, learningRate, learningRateDecay, frictionRate, epochCount, visualizer = None):
        learningRate = theano.shared(np.cast['float32'](learningRate), "eta")
        frictionRate = np.cast['float32'](frictionRate)
        params = self.outputLayer.params
        paramsGradient = [T.grad(self.cost, param) for param in params]
        paramsVelocity = [theano.shared(np.zeros(param.get_value().shape).astype(np.float32)) for param in params]
        
        updates = [(param, param - learningRate * paramVelocity) for param, paramVelocity in zip(params, paramsVelocity)] + \
                  [(paramVelocity, paramVelocity * (np.cast['float32'](1.0) - frictionRate) + paramGradient) 
                  for param, paramVelocity, paramGradient in zip(params, paramsVelocity, paramsGradient)]

        index = T.lscalar('index')
            
        print('Compiling train function', file=sys.stderr)
        trainFunction = theano.function(inputs=[index], outputs=self.cost, updates=updates,
                                        givens=((self.inp, trainX[index * self.batchSize: (index + 1) * self.batchSize]), 
                                                (self.answer, trainY[:, index * self.batchSize: (index + 1) * self.batchSize])))
        print('Finished', file=sys.stderr)

        for epoch in range(1, epochCount + 1):
            print('Epoch#', epoch, file=sys.stderr)
            cost = 0.0
            for i in range(trainX.get_value().shape[0] // self.batchSize):
                cost += trainFunction(i)
            print('Cost:', cost, file=sys.stderr)
            if visualizer:
                cost = self.getBatchCost(trainX.get_value(), trainY.get_value())
                if epoch % 5 == 0:
                    visualizer.addData(epoch=epoch, cost=cost, accuracy=self.getAccuracy(trainX.get_value(), trainY.get_value()))
                else:
                    visualizer.addData(epoch=epoch, cost=cost)

            if epoch % 10 == 0:
                learningRate.set_value(np.cast['float32'](learningRate.get_value() / learningRateDecay))

            sys.stderr.flush()
     
        print('Training finished', file=sys.stderr)


        sys.stderr.flush()

    def getBatchCost(self, testX, testY):
        cost = 0.0
        for i in range(testX.shape[0] // self.batchSize):
            cost += self.batchCost(testX[self.batchSize * i: self.batchSize * (i + 1)], 
                                   testY[:, i * self.batchSize: (i + 1) * self.batchSize])
        return cost

    def getAccuracy(self, testX, testY):
        cnt = 0.0
        for i in range(testX.shape[0]):
            if self.error(testX[i: i+1, :], np.argmax(testY[:, i])):
                cnt += 1
        
        return cnt / testX.shape[0]

    def getOutput(self, x):                
        return self.prediction(x)        

def main():
    theano.device = 'gpu'
    theano.config.warn_float64="warn"
    theano.config.floatX = 'float32'

    x, y = loadData('../input/train.csv')    
    #x, y = np.empty((2, 2)), np.empty((2, 2))

    print(x.shape, y.shape)

    N = x.shape[0]
    print(N)
    trainCount = int(N * trainPercentage)
    testCount = int(N * testPercentage)    
    xTrain, yTrain = theano.shared(x[: trainCount, :].astype(np.float32), name="xTrain", borrow=True), theano.shared(y[:, : trainCount].astype(np.float32), name="yTrain", borrow=True)
    xTest, yTest = theano.shared(x[trainCount: trainCount + testCount, :].astype(np.float32), name="xTest", borrow=True), theano.shared(y[:, trainCount: trainCount + testCount].astype(np.float32), name="yTest", borrow=True)
    del x
    del y
    cnt = 0
    batchSize = 128 

    bestAccuracy = 0
    bestNetwork = None
    for eta in [0.06]:
        for friction in [0.9]:
            for etaDecrease in [1.0]:
                timer = time.clock()
                numpyRng = np.random.RandomState(123)
                theanoRng = T.shared_randomstreams.RandomStreams(123)
                inp = InputImageLayer(batchSize, (28, 28))
                l1 = ConvolutionLayer(inp, (10, 5, 5), numpyRng)
                l2 = MaxPoolLayer(l1, (2, 2))
                l3 = ConvolutionLayer(l2, (20, 5, 5), numpyRng)
                l4 = MaxPoolLayer(l3, (2, 2))
                l5 = UnrollLayer(l4)
                l6 = FullLayer(l5, 250, numpyRng)
                l7 = DropoutLayer(l6, theanoRng, 0.5)
                l8 = OutputLayer(l7, 10, numpyRng)
                netw = NeuralNetwork(inp, l8)
                netw.train(xTrain, yTrain, eta, etaDecrease, friction, 7, Visualizer(str(cnt)))
                print('Cnt:', cnt)
                print('Eta:', eta, 'etaDecrease:', etaDecrease, 'friction:', friction)
                print('Time spent:', time.clock() - timer)
                print('Train accuracy:', netw.getAccuracy(xTrain.get_value(), yTrain.get_value()))
                accuracy = netw.getAccuracy(xTest.get_value(), yTest.get_value())
                print('Accuracy:', accuracy)
                print('')
                cnt += 1
                if accuracy > bestAccuracy:
                    bestNetwork = netw
                    bestAccuracy = accuracy

    del xTest
    del yTest
    del xTrain
    del yTrain
    f = open('../input/test.csv', 'r')  
    tests = []
    line = f.readline()
    for line in f:
        tests.append(np.array(list(map(int, line.split(',')))))
    f.close()
    x = []                                                 
    for test in tests:
        x.append((np.array(test) - 127).reshape(1, 28, 28) / 128.0)
    tests.clear()
    

    out = open('out.csv', 'w')
    out.write('ImageId,Label\n')
    for i in range(len(x)):
        out.write(str(i + 1) + ',' + str(bestNetwork.getOutput(x[i: i+1])) + '\n')

if __name__ == '__main__':
    main()