#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import csv
import random
import math


# ### 1. Load data

# The first thing we need to do is load our data file

# In[ ]:


get_ipython().system('ls ../input/pimaindian')
path = "../input/pimaindian/pima-indians-diabetes.data.csv"


# In[ ]:


def loadCsv(path):
    lines = csv.reader(open(path))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


# In[ ]:


dataset = loadCsv(path)


# In[ ]:


dataset


# In[ ]:


pdfData = pd.read_csv(path)
print(pdfData.shape)


# In[ ]:


len(dataset)


# ### 2. Summarize data

# The first task is to separate the training dataset instances by class value so that we can calculate statistics for each class.

# In[ ]:


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


# In[ ]:


dataset = [[1], [2], [3], [4], [5]]
splitRatio = 0.67 # (2/1)
train, test = splitDataset(dataset, splitRatio)
print('Split {0} rows into train with {1} and test with {2}'.format(len(dataset), train, test))


# In[ ]:


# Label is the las value in the vector
# Separated by label
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


# In[ ]:


dataset = [[1,20,1], [2,21,0], [3,22,1]]
separated = separateByClass(dataset)
print('Separated instances: {0}'.format(separated))


# #### Data Distribution for the Dataset

# We need to calculate the mean of each attribute for a class value. The mean is the central middle or central tendency of the data, and we will use it as the middle of our gaussian distribution when calculating probabilities.

# In[ ]:


def mean(numbers):
    # TODO:
    sum = 0
    for num in numbers:
        sum += num
    
    mean = sum / len(numbers)
    return mean

def stdev(numbers):
    # TODO:
    meann = mean(numbers)
    sum = 0
    for num in numbers:
        sum += (num - meann) ** 2
    
    stdev = math.sqrt(sum / len(numbers))    
    return stdev


# In[ ]:


def summarize(dataset):
    data = zip(*dataset)
    
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*(dataset))]
    
    del summaries[-1]
    return summaries


# In[ ]:


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


# In[ ]:


## Print out (mean, std) value for each class 
dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
# summary = summarizeByClass(dataset)
print('Summary by class value: (class:\n{0}'.format(summary))


# ### 3. Make prediction

# In[ ]:


## Gaussian Probability Density Function
# --- #
# We can use a Gaussian function to estimate the probability of a given attribute value, 
# given the known mean and standard deviation for the attribute estimated from the training data
# https://en.wikipedia.org/wiki/Gaussian_function
def calculateProbability(x, mean, stdev):
    """
    Calculate the probability of instance x belong to the class with (mean, stdev) value
    """
    # TODO
    Probability = (1 / (math.sqrt( 2 * math.pi ) * stdev)) * math.exp( - ((x - mean) ** 2) / (2 * (stdev**2)))
    return Probability


# In[ ]:


# Test the above function
x = 71.5
mea = 73
stde = 6.2
probability = calculateProbability(x, mea, stde)
print('Probability of belonging to this class: {0}'.format(probability))
# Expected value: 0.06248965759370005


# In[ ]:


## Calculate class probability
# --- #
def calculateClassProbabilities(summaries, inputVector):
    """
    Given the input vector, calculate the probability of the instance belong to each class
    """
    probabilities = {}
    sum = 0
    # TODO:
    for dis in summaries:
        InstanceProbability = calculateProbability(inputVector[0], summaries[dis][0][0], summaries[dis][0][1])
        probabilities[dis] = InstanceProbability
        sum += InstanceProbability
    
    for key in probabilities:
        probabilities[key] /= sum
    
    return probabilities


# In[ ]:


# Test the above function
summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
inputVector = [1.1, "?"]
probabilities = calculateClassProbabilities(summaries, inputVector)
print("Probabilities for each class: {0}".format(probabilities))


# In[ ]:





# In[ ]:


## Make the prediction
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


# In[ ]:


# Test the above function
summaries = {"A":[(1, 0.5)], "B":[(20, 5.0)]}
inputVector = [1.1, "?"]
result = predict(summaries, inputVector)
print("Prediction: {0}".format(result))


# In[ ]:





# In[ ]:


## Run the preduction for the set
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


# In[ ]:


summaries = {"A":[(1, 0.5)], "B":[(20, 5.0)]}
testSet = [[1.1, "?"], [19.1, "?"]]
predictions = getPredictions(summaries, testSet)
print("Predictions: {0}".format(predictions))


# In[ ]:


## Check the performance
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


# In[ ]:


testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: {0}'.format(accuracy))


# ### 4. Test on the data

# In[ ]:


def main():
    filename = "../input/pimaindian/pima-indians-diabetes.data.csv"
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print("Split {0} rows into train={1} and test={2} rows"
          .format(len(dataset), len(trainingSet), len(testSet)))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print("Accuracy: {0}%".format(accuracy))


# In[ ]:


main()

