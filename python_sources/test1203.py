#coding=gbk


import numpy
import math
import heapq
from numpy import *
import timeit
from sklearn.decomposition import PCA

pca = PCA(n_components=11)


def dist(a, b):
    return numpy.linalg.norm(a - b)

def find_k_neigh(x, trainX, trainY, k):
    Q = []
    for i in range(trainX.shape[0]):
        d = dist(x, trainX[i][1:len(trainX[0])])

        heapq.heappush(Q, (-d, i))
        if len(Q) > k:
            heapq.heappop(Q)

    Q.sort()
    Q.reverse()
    return map(lambda x: x[1], Q)

def k_neigh_vote(idxs, trainY, k):
    M = {}
    dixs=list(map(int,idxs))
    for i in range(10): M[i] = 0
    maxLabel = 0
    for i in range(k):
        # idx = idxs[i]
        idx= dixs[i]
        label = trainY[idx]
        weight = 1. / (1 + i)
        M[label] +=  weight
        maxLabel = maxLabel if M[maxLabel] > M[label] else label
    return maxLabel

def knn(testX, trainX, trainY, k):
    testY = []
    for i in range(testX.shape[0]):
        neighs = find_k_neigh(testX[i], trainX, trainY, k)
        testY.append(int(k_neigh_vote(neighs, trainY, k)))
    return testY

def file2Mat(fileName):
        f = open(fileName)

        #skip first line
        f.readline()

        lines = f.readlines()
        matrix = []
        for line in lines:
            mat = [int(x) for x in line.strip().split(',')[0:]]
            matrix.append(mat)
        f.close()
        # print 'Read file ' + str(fileName) + ' to array done! Matrix shape:' + str(shape(matrix))
        return matrix

def getFirstCol(dataSet):
    res = [x[0] for x in dataSet]
    return res

def getColsExceptFirst(dataSet):
    res = [x[1:len(dataSet[0])] for x in dataSet]
    return res

train = file2Mat("../input/train.csv")
test = file2Mat("../input/test.csv")

train = train[0:200]
test = test[0:20]
labels = getFirstCol(train[0:len(train)])
train = getColsExceptFirst(train)
pca.fit(train)
train =pca.fit_transform(train)

test = pca.fit_transform(test)
k=7
result =[]

for i in range(0, len(test)):
    queryPoints = [test[i][0:len(train[0])]]
    result.append(knn(numpy.array(queryPoints), numpy.array(train), numpy.array(labels), k))



with open('predict.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in result:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')