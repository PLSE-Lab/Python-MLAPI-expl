#!/usr/bin/env python
# coding: utf-8

# import pandas as pd
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# # import cv2
# 
# # http://scikit-image.org/docs/dev/auto_examples/plot_hog.html
# 
# from skimage.feature import hog
# from skimage import data, color, exposure
# from sklearn import preprocessing as prep
# from sklearn.ensemble import RandomForestClassifier as RFC
# 
#     
# 
# train = pd.read_csv("../input/train.csv")
# test  = pd.read_csv("../input/test.csv")
# testAnswers = pd.read_csv("../input/results.csv")
# 

# In[ ]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage

# http://scikit-image.org/docs/dev/auto_examples/plot_hog.html

from skimage.feature import hog
from skimage import data, color, exposure
from sklearn import preprocessing as prep
from sklearn.ensemble import RandomForestClassifier as RFC

def preProcessStd(dataFrame, testFrame):
    #remove standard deviation if less than 0.4 as those columns hold little impact over the data set. Feature Engineer Threshholding
    testFrame = testFrame.loc[:, dataFrame.std()>0.003]    
    print testFrame.shape
    dataFrame = dataFrame.loc[:, dataFrame.std()>0.003]    
#     print dataFrame
    # Change to set to zero
    return dataFrame, testFrame

def preProcessHog(dataFrame, testFrame):
    
    dataValue = dataFrame.values
    testValues = testFrame.values
    populatingData = np.empty((len(dataFrame),81))
    populatingTest = np.empty((len(testFrame),81))
    for index in range(len(dataFrame)):
        rowDataFrame = dataFrame.loc[index]
        rowDataFrame = rowDataFrame.reshape((28, 28))
        hog_image = hog(rowDataFrame)
        hog_image.resize((1,81))
        populatingData[[index]] = hog_image
    hogData = pd.DataFrame(data = populatingData[0:,0:], index=range(len(dataFrame)))
#     print hogData.shape
    for index in range(len(testFrame)):
        rowTestFrame = testFrame.loc[index]
        rowTestFrame = rowTestFrame.reshape((28, 28))
        hog_image = hog(rowTestFrame)
        hog_image.resize((1,81))
        populatingTest[[index]] = hog_image    
    hogTest = pd.DataFrame(data = populatingTest[0:,0:], index=range(len(testFrame)))
#     print hogTest.shape
#     print hogData, hogTest
    return hogData, hogTest

def preProcessEdgeDetection(dataFrame, testFrame):
    
    dataValue = dataFrame.values
    testValues = testFrame.values
    populatingData = np.empty((len(dataFrame),784))
    populatingTest = np.empty((len(testFrame),784))
    for index in range(len(dataFrame)):
        rowDataFrame = dataFrame.loc[index].astype(np.float64)
        rowDataFrame = rowDataFrame.reshape((28, 28))
        rowDataFrame = ndimage.sobel(rowDataFrame)
        rowDataFrame.resize((1,784))
        populatingData[[index]] = rowDataFrame
        
    for index in range(len(testFrame)):
        rowTestFrame = testFrame.loc[index].astype(np.float64)
        rowTestFrame = rowTestFrame.reshape((28, 28))
        rowTestFrame = ndimage.sobel(rowTestFrame)
        rowTestFrame.resize((1,784))
        populatingData[[index]] = rowTestFrame 
#     print populatingTest
    return dataFrame, testFrame


def preProcess(DataSet, TestSet):
#     DataSet, TestSet = preProcessEdgeDetection(DataSet, TestSet)
#     DataSet, TestSet = preProcessHog(DataSet, TestSet)
#     DataSet, TestSet = preProcessStd(DataSet, TestSet)
    return DataSet, TestSet

    

copyTrain = train.copy()
copyTest = test.copy()
# print copyTest
copyAnswers = testAnswers.copy()
trainLabel = copyTrain["label"]
trainData = copyTrain.drop(["label"], axis=1)
processedTrain, processedTest = preProcess(trainData, copyTest)

# VotingClassifier to combine models
# 0.940142857143 With both preprocessing

# 0.939821428571 No preprocessing


# In[ ]:





# In[ ]:




