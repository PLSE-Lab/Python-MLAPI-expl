import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
import operator
from numpy import *
import csv

def  toInt(array):
    array =mat(array)
    m,n = shape(array)
    newArray = zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i,j] = int(newArray[i,j])
    return newArray

def nomalizing(array):
    array = mat(array)
    m,n = shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[m,n]!=0:
                array[m,n]=1
    return array

def loadTrainData():
    l=[]
    with open('train.csv')as file:
        lines = csv.reader(file)
        for line in lines():
            l.append(line)

    l.remove(l[0])
    l=array(l)
    lable = l[:,0]
    data = l[:,1:]
    return nomalizing(toInt(data)),toInt(lable)

def loadTestData():
    l=[]
    with open('test.csv')as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)

    l.remove(l[0])
  #  l=array(l)
  #  lable = l[:,0]
    data = l[:,1:]
    return nomalizing(toInt(data))

def loadsample_submission():
    l=[]
    with open('sample_submission.csv')as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    #l=array(l)
    lable = array(l)
    return toInt(lable[:,1])



def classify(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = array(diffMat)**2
    sqDistances =sqDiffMat.sum(axis=1)
    distances =sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlable = labels[sortedDistIndicies[i],0]
        classCount[voteIlable]=classCount.get(voteIlable,0)+1
    sortedClassCount =sorted(classCount.iteritems(),key=operator.itemgetter,reverse =True)
    return sortedClassCount[0][0]

def saveResult(result):
    with open('result.csv','wb')as myFile:
        myWriter =csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)

def handwritingClassTest():
    trainData,trainLable = loadTrainData()
    testData = loadTestData()
    testLable = loadsample_submission()
    m,n = shape(testData)
    errorCount = 0
    resultList=[]
    for i in range(m):
        classifyResult =classify(testData[i],trainData,trainLable.transpose(),5)
        resultList.append(classifyResult)
       # print "the classifier came back with: %d,the real answer is :"% (classifyResult,testLable[0,i])
        if (classifyResult !=testLable[0,i]): errorCount+=1.0
    #print "\n the total number of errors is: %d" %errorCount
    #print "\n the total error rate is: %f  "% (errorCount/float(m))
    saveResult(resultList)
