#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# NBC from Scratch without using any in-built Function

# In[ ]:


import numpy as np 
from sklearn.datasets import load_iris


# In[ ]:


iris = load_iris()


# In[ ]:


data = iris.data
data


# Now we will slice the data as we need it and remove the last columns which are of no use in our prediction for defined model

# In[ ]:


data = data[:,:2]
data


# In[ ]:


classes = iris.target
classes


# Now We will combine our data with Class data which will help us determine the Class of the data, of which that belong

# In[ ]:


iris_data = np.insert(data,2,classes, axis=1)
iris_data


# In[ ]:


si = np.random.permutation(iris_data.shape[0])
si


# The data was in ordered manner which can make our algorithm Biased so we will randomize it with permutation function

# In[ ]:


iris_data = iris_data[si]
iris_data


# In[ ]:


dataset = iris_data[:15,:]
dataset


# Define function Separate by Classes

# In[ ]:


def Separate_by_classes(dataset):
    separated = {}
    for i in range(len(dataset)):
        row = dataset[i]
        Class = row[-1]
        if Class not in separated:
            separated[Class] = []
        separated[Class].append(row)
    return separated

separated = Separate_by_classes(dataset)
for i in separated:
    print("Class:",i)
    for row in separated[i]:
        print(row)


# Above we divided the data according to Classes with our function

# In[ ]:


def Mean(numbers):
    return sum(numbers)/len(numbers)


# In[ ]:


def std(numbers):
    mean = Mean(numbers)
    std = 0
    Sum = 0
    for i in numbers:
        Sum += (i-mean)**2
    std = (Sum/len(numbers))**0.5
    return std


# In[ ]:


def Manage_Dataset(dataset):
    Summaries = []
    for i in zip(*dataset):
        Summ = [Mean(i),std(i),len(i)]
        Summaries.append(Summ)
    return Summaries[:-1]


# In[ ]:


def Manage_Dataset_by_Classes(dataset):
    separated = Separate_by_classes(dataset)
    Manage = {}
    Keys = separated.keys()
    for i in Keys:
        Manage[i] = Manage_Dataset(separated[i])
    return Manage
    


# We defined our own functions for Mean, Standard Deviation and Variance and to manage dataset by classes 
# and in the end we are calling our function and gettng the output as managed final dataset

# In[ ]:


dataClass = Manage_Dataset_by_Classes(dataset)
for i in dataClass:
    print(i,dataClass[i])


# ![image.png](attachment:image.png)

# In[ ]:


def Calc_Prob(x, mean, std):
    part2_exp = np.exp(-((x-mean)**2) / (2*(std**2)))
    part1 = 1/((np.sqrt(2*np.pi))*std)
    return part1*part2_exp


# Now we will find out **Class Porbability**

# In[ ]:


def Accuracy(act,pred):
    if len(act) == len(pred):
        total_correct = 0
        for i in range(len(act)):
            if act[i] == pred[i]:
                total_correct += 1
            return total_correct/len(act)
    else:
        print("Length of both datasets should be same")
        return None
    
    
def Find_Prob(dataClass,newData):
    total_rows = 0
    for i in dataClass:
        total_rows = total_rows + dataClass[i][0][-1]
    prob = {}
    for i in dataClass:
        prob[i] = dataClass[i][0][-1]/total_rows
        class_summary = dataClass[i]
        for j in range(len(class_summary)):
            mean, std, Unuse = class_summary[j]
            prob1= Calc_Prob(newData[j], mean, std)
            prob[i]  = prob[i]*prob1
    return prob


# In[ ]:


def Pred(data,newData):
    Probs = Find_Prob(data,newData)
    max_prob = [0,0]
    for i in Probs:
        if Probs[i] > max_prob[1]:
            max_prob = [i,Probs[i]]
    return max_prob[0]


# Now we will define our last function to train Naive Bayes Algorithm with all the functions we defined above then we call it and will get the probability.

# In[ ]:


def Naive_Bayes_Algo(train,test):
    summarize = Manage_Dataset_by_Classes(train)
    predictions = []
    for row in test:
        output = Pred(summarize,row)
        predictions.append(output)
    return(predictions)


# In[ ]:


data = Naive_Bayes_Algo(iris_data,iris_data[:40])
Accuracy(iris_data[:40,-1],data)


# In[ ]:




