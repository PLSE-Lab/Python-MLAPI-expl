# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd

trainingData = pd.read_csv('../input/train.csv')

print(trainingData)

# trainingData.to_html('trainingData.html')

labelDF = trainingData['label']
labelHotArray = []
for eachLabel in labelDF:
    print(eachLabel)
    if eachLabel == 0:
        labelHotArray.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif eachLabel == 1:
        labelHotArray.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif eachLabel == 2:
        labelHotArray.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif eachLabel == 3:
        labelHotArray.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif eachLabel == 4:
        labelHotArray.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif eachLabel == 5:
        labelHotArray.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif eachLabel == 6:
        labelHotArray.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif eachLabel == 7:
        labelHotArray.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif eachLabel == 8:
        labelHotArray.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif eachLabel == 9:
        labelHotArray.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

featureDF = trainingData.drop('label', axis=1)

import numpy as np

labelNP = np.array(labelHotArray)
# labelNP = np.reshape(labelNP, newshape=[-1, 1])
featureNP = np.array(featureDF)

print(featureNP.shape, labelNP.shape)

import tensorflow as tf

y = tf.placeholder(tf.float64, shape=[None, 10])

m = tf.Variable(tf.random_normal(shape=[784, 800], dtype=tf.float64))
m1 = tf.Variable(tf.zeros(shape=[800, 400], dtype=tf.float64))
m2 = tf.Variable(tf.zeros(shape=[400, 200], dtype=tf.float64))
m3 = tf.Variable(tf.zeros(shape=[200, 10], dtype=tf.float64))

x = tf.placeholder(tf.float64, shape=[None, 784])
b = tf.Variable(tf.zeros(shape=[800], dtype=tf.float64))
b1 = tf.Variable(tf.zeros(shape=[400], dtype=tf.float64))
b2 = tf.Variable(tf.zeros(shape=[200], dtype=tf.float64))
b3 = tf.Variable(tf.zeros(shape=[10], dtype=tf.float64))

mx_b = tf.add(tf.matmul(x, m), b)

mx_b1 = tf.add(tf.matmul(mx_b, m1), b1)
mx_b1 = tf.nn.relu(mx_b1)

mx_b2 = tf.add(tf.matmul(mx_b1, m2), b2)
mx_b2 = tf.nn.relu(mx_b2)

mx_b3 = tf.add(tf.matmul(mx_b2, m3), b3)

sess = tf.Session()

# print(sess.run(mx_b, feed_dict={x: featureNP}))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mx_b3, labels=y))
trainingStep = tf.train.AdamOptimizer(0.0001).minimize(loss)
sess.run(tf.global_variables_initializer())

for i in range(2700):
    print(sess.run([trainingStep, loss], feed_dict={x: featureNP, y: labelNP}))
    print(i)

testDataCSV = pd.read_csv('../input/test.csv')
testFeatureNP = np.array(testDataCSV)
output = sess.run(mx_b3, feed_dict={x: testFeatureNP})
print(output)

outputLabel = []
for eachOutput in output:
    print(np.argmax(eachOutput))
    outputLabel.append(np.argmax(eachOutput))

labels = []
for i in range(len(outputLabel)):
    labels.append(i + 1)

outputDF = pd.DataFrame()
outputDF['ImageId'] = labels
outputDF['Label'] = outputLabel

print(outputDF.head())

outputDF.to_csv('output.csv', index=False)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.