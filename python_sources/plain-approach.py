import pandas as pd

titanicTrainData = pd.read_csv('../input/train.csv')
print(titanicTrainData.head())

usefulData = titanicTrainData[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
print(usefulData.head())

repalaceSex = usefulData['Sex']
toBeReplaced = []
for eachSex in repalaceSex:
    if eachSex == 'male':
        toBeReplaced.append(1)
    elif eachSex == 'female':
        toBeReplaced.append(0)
    else:
        toBeReplaced.append(2)

print(toBeReplaced.__len__())

usefulData = usefulData.drop('Sex', axis=1)
usefulData['sexBin'] = toBeReplaced
print(usefulData)
usefulData = usefulData.fillna(0)
print(usefulData)

labelS = titanicTrainData['Survived']

lebelHotArray = []
for eahLabel in labelS:
    print(eahLabel)
    if eahLabel == 0:
        lebelHotArray.append([1, 0])
    elif eahLabel == 1:
        lebelHotArray.append([0, 1])

import numpy as np

featureDataNP = np.array(usefulData)
labelDataNP = np.array(lebelHotArray)
# labelDataNP = np.reshape(labelDataNP, [-1, 1])
print(featureDataNP.shape, labelDataNP.shape)

import tensorflow as tf

y = tf.placeholder(tf.float32, shape=[None, 2])

m1 = tf.Variable(tf.random_normal(shape=[6, 12]))
m2 = tf.Variable(tf.random_normal(shape=[12, 6]))
m3 = tf.Variable(tf.random_normal(shape=[6, 3]))
m4 = tf.Variable(tf.random_normal(shape=[3, 2]))

x = tf.placeholder(tf.float32, shape=[None, 6])

b1 = tf.Variable(tf.random_normal(shape=[12]))
b2 = tf.Variable(tf.random_normal(shape=[6]))
b3 = tf.Variable(tf.random_normal(shape=[3]))
b4 = tf.Variable(tf.random_normal(shape=[2]))

mx_b1 = tf.add(tf.matmul(x, m1), b1)
mx_b2 = tf.add(tf.matmul(mx_b1, m2), b2)
mx_b2Acivated = tf.nn.relu(mx_b2)
mx_b3 = tf.add(tf.matmul(mx_b2, m3), b3)
mx_b4 = tf.add(tf.matmul(mx_b3, m4), b4)

lr = tf.placeholder(tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(mx_b4, feed_dict={x: featureDataNP}))

loss = tf.reduce_mean(tf.sqrt(tf.square(mx_b4 - y) + 1e-10))
trainingStep = tf.train.GradientDescentOptimizer(lr).minimize(loss)

lrFactor = 50
while lrFactor > 0.234:
    lrFactor = np.sum(np.array(sess.run(loss, feed_dict={x: featureDataNP, y: labelDataNP})))
    lrFactor = min(lrFactor, 20.0)
    print(sess.run([trainingStep, loss], feed_dict={x: featureDataNP, y: labelDataNP, lr: lrFactor * 0.00005}))

titanicTestData = pd.read_csv('../input/test.csv')
passengerId = titanicTestData['PassengerId']
titanicTestData = titanicTestData[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

toChangeSex = titanicTestData['Sex']
putSex = []

for eahSex in toChangeSex:
    if eahSex == 'male':
        putSex.append(1)
    elif eahSex == 'female':
        putSex.append(0)
    else:
        putSex.append(2)

titanicTestData = titanicTestData.drop('Sex', axis=1)
titanicTestData['sexInBin'] = putSex

predictNp = np.array(sess.run(tf.nn.relu(mx_b4), feed_dict={x: titanicTestData}))

predictNp0or1 = []
for eahPre in predictNp:
    print(eahPre, "21421")
    print(np.argmax(eahPre))
    predictNp0or1.append(np.argmax(eahPre))

outputCSV = pd.DataFrame()
outputCSV['PassengerId'] = passengerId
outputCSV['Survived'] = predictNp0or1
outputCSV.to_csv('output.csv', index=False)
