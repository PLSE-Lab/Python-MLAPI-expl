# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd

#leavesData = pd.read_csv('/home/kemsys/PycharmProjects/Kaggle/leaves/leavesData/leaf-classification/train.csv')
leavesData = pd.read_csv('../input/train.csv')
print(leavesData.head())

# COUNT UNIQUE SPECIES
leavesNameColumn = leavesData['species']
print(len(leavesNameColumn))
print(leavesNameColumn.unique().tolist())
uinqueSpecies = leavesNameColumn.unique().tolist()
print(len(uinqueSpecies))

leavesIndexColumn = leavesData['id']
print(len(leavesIndexColumn))
print(len(leavesIndexColumn.unique().tolist()))

# id column is waste

# CREATE FEATURES
featureDataFrame = leavesData.drop(["id", "species"], axis=1)
print(featureDataFrame)

# CREATE LABELS
tempIndexLabelDictionary = dict()
tempLabelIndexDictionary = dict()
index = -1
for species in uinqueSpecies:
    index = index + 1
    print(species + "--" + index.__str__())
    tempIndexLabelDictionary[index] = species
    tempLabelIndexDictionary[species] = index

print(tempLabelIndexDictionary)
print(tempIndexLabelDictionary)

labelDataFrame = []
for name in leavesNameColumn:
    print(tempLabelIndexDictionary[name])
    labelDataFrame.append(tempLabelIndexDictionary[name])

print(labelDataFrame)

labelToIdDict = dict()
idToLabelDict = dict()

# for labelt, idt in labelDataFrame, leavesIndexColumn:
for i in range(len(leavesIndexColumn)):
    labelToIdDict[labelDataFrame[i]] = leavesIndexColumn[i]
    idToLabelDict[leavesIndexColumn[i]] = labelDataFrame[i]

import numpy as np

featureNumpyArray = np.array(featureDataFrame)
labelNumpyArray = np.array(labelDataFrame)
labelNumpyArray = labelNumpyArray.reshape([990, 1])
print(featureNumpyArray.shape, labelNumpyArray.shape)

import tensorflow as tf

y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

m1 = tf.Variable(tf.random_normal(shape=[192, 306], dtype=tf.float32))
m2 = tf.Variable(tf.random_normal(shape=[306, 108], dtype=tf.float32))
m3 = tf.Variable(tf.random_normal(shape=[108, 1], ), dtype=tf.float32)

x = tf.placeholder(shape=[None, 192], dtype=tf.float32)

b1 = tf.Variable(tf.random_normal(shape=[306], dtype=tf.float32))
b2 = tf.Variable(tf.random_normal(shape=[108], dtype=tf.float32))
b3 = tf.Variable(tf.random_normal(shape=[1], dtype=tf.float32))

layer1 = tf.add(tf.matmul(x, m1), b1)
layer1 = tf.nn.relu(layer1)
layer2 = tf.add(tf.matmul(layer1, m2), b2)
layer3 = tf.add(tf.matmul(layer2, m3), b3)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(layer3, feed_dict={x: featureNumpyArray}))

loss = tf.reduce_mean(tf.sqrt(tf.square(layer3 - y) + 1e-10))
lr = tf.placeholder(dtype=tf.float32)
trainingStep = tf.train.GradientDescentOptimizer(lr).minimize(loss)

lrFactor20max = 50
while lrFactor20max > 2.5:
    lossNp = np.array(sess.run(loss, feed_dict={x: featureNumpyArray, y: labelNumpyArray}))
    lrFactor = np.sum(lossNp)
    print(lrFactor)
    lrFactor20max = min(lrFactor, 20.0)
    print(sess.run([trainingStep, loss, lr],
                   feed_dict={x: featureNumpyArray, y: labelNumpyArray, lr: (lrFactor20max * 0.005)}))

print(np.array(sess.run(layer1, feed_dict={x: featureNumpyArray})).shape)
print(np.array(sess.run(layer2, feed_dict={x: featureNumpyArray})).shape)
print(np.array(sess.run(layer3, feed_dict={x: featureNumpyArray})).shape)

#testDataframe = pd.read_csv('/home/kemsys/PycharmProjects/Kaggle/leaves/leavesData/leaf-classification/test.csv')
testDataframe = pd.read_csv('../input/test.csv')
testFeatures = testDataframe.drop(['id'], axis=1)
testFeaturesNP = np.array(testFeatures)
testFeaturesNP = np.reshape(testFeaturesNP, [-1, 192])
testIds = testDataframe['id']
testIds = np.array(testIds)
predictionNP = np.array(sess.run(tf.nn.relu(layer3), feed_dict={x: testFeaturesNP}))
print(predictionNP)
for i in range(len(predictionNP)):
    print(labelToIdDict[int(round(float(predictionNP[i])))])
    print("=")
    print(testIds[i])
    print("\n------------------\n")
