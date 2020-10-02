#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection using k-Nearest Neighbours
# ***

# ## 1. Function to load dataset
# ***

# In[3]:


import csv
import random

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rt', encoding='utf-8') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(1,len(dataset)-1):
	        for y in range(31):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])


# ## 2. Function for calculating Euclidean distance
# ***

# In[4]:


import math

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)


# ## 3. Function which returns k most similar neighbours
# ***

# In[5]:


import operator

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


# ## 4. Function to generate a response from a set of data instances.
# ***

# In[6]:


def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


# ## 5. Function to get accuracy of the model in %
# ***

# In[7]:


def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


# ## 6. main()
# ***

# In[12]:


def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.9999
	loadDataset('../input/creditcard.csv', split, trainingSet, testSet)
	print('Train set: ' + repr(len(trainingSet)))
	print('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	k = 1
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print(('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1])))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
    
    

main()


# In[ ]:




