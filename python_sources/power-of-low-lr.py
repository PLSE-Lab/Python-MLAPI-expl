import pandas as pd

trainigHouseCSV = pd.read_csv('../input/train.csv')
print(trainigHouseCSV.head())

trainigHouseCSV.fillna('5')

trainigHouseCSV = trainigHouseCSV.drop('Id', axis=1)
print(trainigHouseCSV)


def getUniqueValuesFor(param, trainigHouseCSV):
    columnReadData = trainigHouseCSV[param]
    columnReadDataUniques = columnReadData.unique().tolist()
    print(columnReadDataUniques)

    index = 15.000001
    itemToIndexDict = dict()
    for eachColumnReadDataUniques in columnReadDataUniques:
        index = index + 115.000001
        itemToIndexDict[eachColumnReadDataUniques] = index

    newColumnToReturn = []
    for eachColumnReadData in columnReadData:
        newColumnToReturn.append(itemToIndexDict[eachColumnReadData])

    trainigHouseCSV = trainigHouseCSV.drop(param, axis=1)
    trainigHouseCSV[param + 'uniquefied'] = newColumnToReturn
    print(trainigHouseCSV)
    return trainigHouseCSV


trainigHouseCSV = getUniqueValuesFor('MSZoning', trainigHouseCSV)


def removoNAs(param, trainigHouseCSV):
    colToRemoveNA = trainigHouseCSV[param].fillna(0)
    trainigHouseCSV = trainigHouseCSV.drop(param, axis=1)
    trainigHouseCSV[param + 'naremoved'] = colToRemoveNA
    return trainigHouseCSV


trainigHouseCSV = removoNAs('LotFrontage', trainigHouseCSV)
print(trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('Street', trainigHouseCSV)

trainigHouseCSV = removoNAs('Alley', trainigHouseCSV)
trainigHouseCSV = getUniqueValuesFor('Alley' + 'naremoved', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('LotShape', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('LandContour', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('Utilities', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('LotConfig', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('LandSlope', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('Neighborhood', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('Condition1', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('Condition2', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('BldgType', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('HouseStyle', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('RoofStyle', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('RoofMatl', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('Exterior1st', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('Exterior2nd', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('MasVnrType', trainigHouseCSV)

trainigHouseCSV = removoNAs('MasVnrArea', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('ExterQual', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('ExterCond', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('Foundation', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('BsmtQual', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('BsmtCond', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('BsmtExposure', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('BsmtFinType1', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('BsmtFinType2', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('Heating', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('HeatingQC', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('CentralAir', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('Electrical', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('KitchenQual', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('Functional', trainigHouseCSV)

trainigHouseCSV = removoNAs('FireplaceQu', trainigHouseCSV)
trainigHouseCSV = getUniqueValuesFor('FireplaceQu' + 'naremoved', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('GarageType', trainigHouseCSV)

trainigHouseCSV = removoNAs('GarageFinish', trainigHouseCSV)
trainigHouseCSV = getUniqueValuesFor('GarageFinish' + 'naremoved', trainigHouseCSV)

trainigHouseCSV = removoNAs('GarageQual', trainigHouseCSV)
trainigHouseCSV = getUniqueValuesFor('GarageQual' + 'naremoved', trainigHouseCSV)

trainigHouseCSV = removoNAs('GarageCond', trainigHouseCSV)
trainigHouseCSV = getUniqueValuesFor('GarageCond' + 'naremoved', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('PavedDrive', trainigHouseCSV)

trainigHouseCSV = removoNAs('PoolQC', trainigHouseCSV)
trainigHouseCSV = getUniqueValuesFor('PoolQC' + 'naremoved', trainigHouseCSV)

trainigHouseCSV = removoNAs('Fence', trainigHouseCSV)
trainigHouseCSV = getUniqueValuesFor('Fence' + 'naremoved', trainigHouseCSV)

trainigHouseCSV = removoNAs('MiscFeature', trainigHouseCSV)
trainigHouseCSV = getUniqueValuesFor('MiscFeature' + 'naremoved', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('SaleType', trainigHouseCSV)

trainigHouseCSV = getUniqueValuesFor('SaleCondition', trainigHouseCSV)

print(trainigHouseCSV)

featureDF = trainigHouseCSV.drop('SalePrice', axis=1)
labelDF = trainigHouseCSV['SalePrice']

import numpy as np

featureNP = np.array(featureDF)
labelNP = np.array(labelDF)
labelNP = np.reshape(labelNP, [1460, 1])
labelNPreduced = []
for eachLabelNP in labelNP:
    labelNPreduced.append(eachLabelNP / 100000)
labelNPreducedNP = np.array(labelNPreduced)

tempfeat = pd.DataFrame(featureNP).fillna(5)
tempfeat.to_html('feature.html')
templabel = pd.DataFrame(labelNPreducedNP).fillna(5)
templabel.to_html('labelNPreducedNP.html')

featureToXNP = np.array(tempfeat)
labelToYNP = np.array(templabel)

print(featureToXNP.shape, labelToYNP.shape)

import tensorflow as tf

y = tf.placeholder(tf.float64, shape=[None, 1])

m = tf.Variable(tf.zeros(shape=[79, 1], dtype=tf.float64))
x = tf.placeholder(tf.float64, shape=[None, 79])
b = tf.Variable(tf.zeros(shape=[1], dtype=tf.float64))

mx_b = tf.add(tf.matmul(x, m), b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(mx_b, feed_dict={x: featureToXNP}))

loss = tf.reduce_mean(tf.sqrt(tf.square(mx_b - y) + 0.0000000001))
trainingStep = tf.train.GradientDescentOptimizer(0.000000000001).minimize(loss)

for i in range(27000000000):
    print(sess.run([trainingStep, loss], feed_dict={x: featureToXNP, y: labelToYNP}))
    print(i)

testingDataCSV = pd.read_csv('../input/test.csv')

testingDataCSV = testingDataCSV.fillna('5')
testingDataCSV = testingDataCSV.drop('Id', axis=1)

testingDataCSV = removoNAs('LotFrontage', testingDataCSV)
print(testingDataCSV)
testingDataCSV = getUniqueValuesFor('MSZoning', testingDataCSV)

testingDataCSV = getUniqueValuesFor('Street', testingDataCSV)

testingDataCSV = removoNAs('Alley', testingDataCSV)
testingDataCSV = getUniqueValuesFor('Alley' + 'naremoved', testingDataCSV)

testingDataCSV = getUniqueValuesFor('LotShape', testingDataCSV)

testingDataCSV = getUniqueValuesFor('LandContour', testingDataCSV)

testingDataCSV = getUniqueValuesFor('Utilities', testingDataCSV)

testingDataCSV = getUniqueValuesFor('LotConfig', testingDataCSV)

testingDataCSV = getUniqueValuesFor('LandSlope', testingDataCSV)

testingDataCSV = getUniqueValuesFor('Neighborhood', testingDataCSV)

testingDataCSV = getUniqueValuesFor('Condition1', testingDataCSV)

testingDataCSV = getUniqueValuesFor('Condition2', testingDataCSV)

testingDataCSV = getUniqueValuesFor('BldgType', testingDataCSV)

testingDataCSV = getUniqueValuesFor('HouseStyle', testingDataCSV)

testingDataCSV = getUniqueValuesFor('RoofStyle', testingDataCSV)

testingDataCSV = getUniqueValuesFor('RoofMatl', testingDataCSV)

testingDataCSV = getUniqueValuesFor('Exterior1st', testingDataCSV)

testingDataCSV = getUniqueValuesFor('Exterior2nd', testingDataCSV)

testingDataCSV = getUniqueValuesFor('MasVnrType', testingDataCSV)

testingDataCSV = removoNAs('MasVnrArea', testingDataCSV)

testingDataCSV = getUniqueValuesFor('ExterQual', testingDataCSV)

testingDataCSV = getUniqueValuesFor('ExterCond', testingDataCSV)

testingDataCSV = getUniqueValuesFor('Foundation', testingDataCSV)

testingDataCSV = getUniqueValuesFor('BsmtQual', testingDataCSV)

testingDataCSV = getUniqueValuesFor('BsmtCond', testingDataCSV)

testingDataCSV = getUniqueValuesFor('BsmtExposure', testingDataCSV)

testingDataCSV = getUniqueValuesFor('BsmtFinType1', testingDataCSV)

testingDataCSV = getUniqueValuesFor('BsmtFinType2', testingDataCSV)

testingDataCSV = getUniqueValuesFor('Heating', testingDataCSV)

testingDataCSV = getUniqueValuesFor('HeatingQC', testingDataCSV)

testingDataCSV = getUniqueValuesFor('CentralAir', testingDataCSV)

testingDataCSV = getUniqueValuesFor('Electrical', testingDataCSV)

testingDataCSV = getUniqueValuesFor('KitchenQual', testingDataCSV)

testingDataCSV = getUniqueValuesFor('Functional', testingDataCSV)

testingDataCSV = removoNAs('FireplaceQu', testingDataCSV)
testingDataCSV = getUniqueValuesFor('FireplaceQu' + 'naremoved', testingDataCSV)

testingDataCSV = getUniqueValuesFor('GarageType', testingDataCSV)

testingDataCSV = removoNAs('GarageFinish', testingDataCSV)
testingDataCSV = getUniqueValuesFor('GarageFinish' + 'naremoved', testingDataCSV)

testingDataCSV = removoNAs('GarageQual', testingDataCSV)
testingDataCSV = getUniqueValuesFor('GarageQual' + 'naremoved', testingDataCSV)

testingDataCSV = removoNAs('GarageCond', testingDataCSV)
testingDataCSV = getUniqueValuesFor('GarageCond' + 'naremoved', testingDataCSV)

testingDataCSV = getUniqueValuesFor('PavedDrive', testingDataCSV)

testingDataCSV = removoNAs('PoolQC', testingDataCSV)
testingDataCSV = getUniqueValuesFor('PoolQC' + 'naremoved', testingDataCSV)

testingDataCSV = removoNAs('Fence', testingDataCSV)
testingDataCSV = getUniqueValuesFor('Fence' + 'naremoved', testingDataCSV)

testingDataCSV = removoNAs('MiscFeature', testingDataCSV)
testingDataCSV = getUniqueValuesFor('MiscFeature' + 'naremoved', testingDataCSV)

testingDataCSV = getUniqueValuesFor('SaleType', testingDataCSV)

testingDataCSV = getUniqueValuesFor('SaleCondition', testingDataCSV)

print(testingDataCSV)
testingDataCSV.to_html('testfeature.html')

featureToXTestNP = np.array(testingDataCSV)

prediction = np.array(sess.run(mx_b, feed_dict={x: featureToXTestNP}))

predList = []
for eachPrediction in prediction:
    predList.append(eachPrediction*100000)
predList = np.array(predList).flatten().tolist()
outPutDF = pd.DataFrame()
outPutDF['Id'] = pd.read_csv('../input/test.csv')['Id']
outPutDF['SalePrice'] = predList

outPutDF.to_csv('output.csv', index=False)