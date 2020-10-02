from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer
import numpy


print('Read training data...')
with open('../input/train.csv', 'r') as reader:
    reader.readline()
    train_label = []
    train_data = []
    for line in reader.readlines():
        data = list(map(int, line.rstrip().split(',')))
        train_label.append(data[0])
        train_data.append(data[1:])
print('Loaded ' + str(len(train_label)))

train_label = numpy.array(train_label)
train_data = numpy.array(train_data)
ds = SupervisedDataSet(784, 10)
for i in range(0, len(train_label)) :
    vec = [0] * 10
    vec[train_label[i]] = 1
    ds.addSample(train_data[i], vec)
print('Samples Added') 

net = buildNetwork(784, 100,100, 10, bias=True)

print ("Training the neural network")
trainer = BackpropTrainer(net, dataset=ds, momentum = 0.1, learningrate= 0.01,
                                              verbose = True, weightdecay = 0.01)

print('Read testing data...')
with open('../input/test.csv', 'r') as reader:
    reader.readline()
    test_data = []
    for line in reader.readlines():
        pixels = list(map(int, line.rstrip().split(',')))
        test_data.append(pixels)
print('Loaded ' + str(len(test_data)))
test_data = numpy.array(test_data)

print('Predicting...')
predict = []
for i in range(0, len(test_data)) :
  predict.append(numpy.argmax(net.activate(test_data[i,:])))

print('Predict Done')



with open('neural_predict.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')

