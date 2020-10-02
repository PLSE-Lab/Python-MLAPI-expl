#!/usr/bin/env python
# coding: utf-8

# # Checking Grid Sizes in the Abstraction and Reasoning Challenge
# 
# In the interactive version of the ARC, the grid size can be changed.  This indicates that there may not be a one-to-one correspondence between grid sizes of input and output pairs.  This notebook looks into that possibility and we find that, indeed, only 63% of training examples have such a one-to-one correspondence.  However, of those that did not have one-to-one correspondence, 55% had identical grid sizes across the training outputs.
# 
# The consequence of this is that the ARC problem space can be broken into three parts of (most likely) increasing difficulty:
# 1. Tasks where the grid size of the inputs can just be copied to the grid size for the outputs (63% of the training tasks).
# 2. Tasks where the grid sizes of the inputs do not match the grid sizes of the outputs, but where all the outputs have the same grid size (20% of the training data).
# 3. Tasks where the grid sizes of the inputs do not match the grid sizes of the outputs, and the input grid somehow informs the grid size of the output (17% of the training data)
# 
# This notebook does not create a submission file.

# In[ ]:


import json
import re # Regular expressions
import os # To walk through the data files provided
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from matplotlib import colors


# In[ ]:


# Specify the different directories of data files
testDirectory = "/kaggle/input/abstraction-and-reasoning-challenge/test/"
trainingDirectory = "/kaggle/input/abstraction-and-reasoning-challenge/training/"
evaluationDirectory = "/kaggle/input/abstraction-and-reasoning-challenge/training/"


# In[ ]:


# Function to read a task file and return the parsed data
def readTaskFile(filename):
    #print("Reading file: "+filename)
    
    # Open the file
    f = open(filename, "r")
    
    # Parse the JSON
    data = json.loads(f.read())
    
    # Add in an 'id' that's extracted from the filename
    data["id"] = re.sub("(.*/)|(\.json)", "", filename)
    
    # Close the file
    f.close()
    
    # Return the parsed data
    return data


# In[ ]:


# Quick test to see that our readTaskFile function is working.
# Note the addition of the 'id' field.
filename = testDirectory+"19bb5feb.json"
readTaskFile(filename)


# In[ ]:


# Function to compare the grid sizes of the input and output fields.
# Returns True if the sizes are the same, and False otherwise.
def getGridSizeComparison(filename):
    data = readTaskFile(filename)
    #print(data)
    trainSection = data["train"]
    ident = data["id"]
    
    numTrain = len(trainSection)
    result = {}
    for i in range(numTrain):
        trainCase = trainSection[i]
        trainCaseInput = trainCase["input"]
        trainCaseOutput = trainCase["output"]
        sameY = len(trainCaseInput) == len(trainCaseOutput)
        sameX = len(trainCaseInput[0]) == len(trainCaseOutput[0])
        result[ident + "_train_" + str(i)] = sameX and sameY
        
    return result


# In[ ]:


# Quick test with the first file "19bb5feb.json"
# "3b4c2228.json" has two tests
filename = testDirectory+"19bb5feb.json"
getGridSizeComparison(filename)


# In[ ]:


# A function to loop through the questions in the given directory
# applying a function 'f' to each question.
def getResults(directory, f):
    results = {}
    for _, _, filenames in os.walk(directory):
        for filename in filenames:
            results.update(f(directory+filename))

    return results


# In[ ]:


# Execute our comparison function on the training directory
results = getResults(trainingDirectory, getGridSizeComparison)

print(str(results)[1:1000]+"[...]")


# In[ ]:


# What proportion of training examples have the same input:output grid sizes?
count = 0
for _, value in results.items():
    if value: count+=1

print("Proportion of training examples with the same grid size: "+str(round(count/len(results), 2)))


# In[ ]:





# In[ ]:


# Visualise the training cases for a task
# Code inspiration from https://www.kaggle.com/inversion/abstraction-and-reasoning-starter-notebook
def plotTaskTraining(task):
    """
    Plots the training pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    # Plot all the training cases
    nTrainingCases = len(task["train"])
    scale = 3
    fig, axs = plt.subplots(nTrainingCases, 2, figsize=(2*scale,nTrainingCases*scale))
    for i in range(nTrainingCases):
        axs[i][0].imshow(task['train'][i]['input'], cmap=cmap, norm=norm)
        axs[i][0].axis('off')
        axs[i][0].set_title('Train Input')
        axs[i][1].imshow(task['train'][i]['output'], cmap=cmap, norm=norm)
        axs[i][1].axis('off')
        axs[i][1].set_title('Train Output')
    plt.tight_layout()
    plt.show()


# In[ ]:


# Task ID 19bb5feb gives an example of a task where the input:output grid sizes are not the same
filename = testDirectory+"19bb5feb.json"

task = readTaskFile(filename)
plotTaskTraining(task)


# The above example shows that the grid sizes for all the **outputs** of the training cases of the task were identical.  Perhaps a significant proportion of tasks that have differing input:output grid sizes have fixed output grid sizes that do not change across the training cases?  The following code considers that possibility.

# In[ ]:


# Function to compare the grid sizes of the input and output fields.
# Returns a dict with the following fields:
#   allCorrespond: True iff the input and output grid sizes are the same for all training cases
#   outputsSame: True iff the output gird sizes are the same for all training cases 
def getGridSizeComparison2(filename):
    data = readTaskFile(filename)
    #print(data)
    trainSection = data["train"]
    ident = data["id"]
    
    numTrain = len(trainSection)
    result = {"allCorrespond": True,
              "outputsSame": True}
    
    # Check for allCorrespond
    for i in range(numTrain):
        trainCase = trainSection[i]
        trainCaseInput = trainCase["input"]
        trainCaseOutput = trainCase["output"]
        sameY = len(trainCaseInput) == len(trainCaseOutput)
        sameX = len(trainCaseInput[0]) == len(trainCaseOutput[0])
        if not (sameX and sameY):
            result["allCorrespond"] = False
            break

    # Check for outputsSame
    outputX = None
    outputY = None
    for i in range(numTrain):
        trainCase = trainSection[i]
        trainCaseOutput = trainCase["output"]
        same = True
        if outputY == None:
            outputY = len(trainCaseOutput)
        else:
            if not outputY == len(trainCaseOutput):
                same = False
            
        if outputX == None:
            outputX = len(trainCaseOutput[0])
        else:
            if not outputX == len(trainCaseOutput[0]):
                same = False

        if not same:
            result["outputsSame"] = False
            break
        
    return {ident: result}


# In[ ]:


# Task ID 19bb5feb gives an example of a task where the 
# input:output grid sizes are not the same, but all the 
# outputs are the same size

filename = testDirectory+"19bb5feb.json"
print(getGridSizeComparison2(filename))


# In[ ]:


# Task ID 0b148d64 gives an example of a task where both the 
# input:output grid sizes are not the same, and all the 
# outputs are not the same size
filename = trainingDirectory+"0b148d64.json"
print(getGridSizeComparison2(filename))


# In[ ]:


# Task ID 0b148d64 gives an example of a task where the input:output grid sizes are not the same
filename = trainingDirectory+"0b148d64.json"

task = readTaskFile(filename)
plotTaskTraining(task)


# In[ ]:


# Execute our comparison function on the training directory
results = getResults(trainingDirectory, getGridSizeComparison2)

print(str(results)[1:1000]+"[...]")


# In[ ]:


# Of the training examples where the input:output grid sizes
# are not all the same, what proportion have all the same size
# training grids?
countAllCorrespondFalse = 0
for _, value in results.items():
    if not value["allCorrespond"]: countAllCorrespondFalse+=1
        
countAllCorrespondFalseOutputsSameTrue = 0
for _, value in results.items():
    if (value["allCorrespond"]==False and 
        value["outputsSame"]==True): countAllCorrespondFalseOutputsSameTrue+=1

print("Of the "+str(countAllCorrespondFalse)+" tasks where the input:output "+
      "grid sizes were not the same,\n"+
      str(countAllCorrespondFalseOutputsSameTrue)+" had identical grid sizes "+
      "for all their outputs, or "+
      str(round(countAllCorrespondFalseOutputsSameTrue/countAllCorrespondFalse*100))+
      "%.")


# In[ ]:




