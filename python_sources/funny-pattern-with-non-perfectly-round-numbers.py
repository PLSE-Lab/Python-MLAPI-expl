#!/usr/bin/env python
# coding: utf-8

# In[14]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os, gc
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Funny pattern in numbers that are not perfectly round
# 
# Looking at the data, something called my attention. 
# It's probably really silly, but well, I'm looking for magic in desperate places (Ok, maybe not the smartest place, I know)    
# 
# The data should have round numbers with 4 decimal places. Nonetheless, some numbers haven't got perfecly rounded, they kept lots of decimal places.  
# 
# This kernel takes a quick look at how these numbers behave.
# 
# --------------------------------------
# 
# (Hidden code, loading data - Takes into accout real and fake test data as found here: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/85125#latest-509548)    

# In[ ]:


#Loading data

nClasses = 200
features = ['var_'+ str(i) for i in range(nClasses)]

if not ('trainFrame' in globals()):
    trainFrame = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
    testFrame = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
    trueIndices = np.load('../input/list-of-fake-samples-and-public-private-lb-split/real_samples_indexes.npy')
    
    trainData = trainFrame[features].values
    testData = testFrame[features].values
    realData = testData[trueIndices]
    totalData = np.concatenate([trainData, testData], axis=0)
    realTotal = np.concatenate([trainData, realData], axis=0)
    targets = trainFrame['target'].values
    posSelector = targets.astype(bool)
    negSelector = (1-targets).astype(bool)
    posData = trainData[posSelector]
    negData = trainData[negSelector]


# ## Plotting the numbers
# 
# Let's plot a few vars and a general graph for the entire data.

# In[ ]:


#base plotting and analysing functions

def strTrunc(x):
    return str(x).split('.')[1]
strTrunc = np.vectorize(strTrunc)

getLen = np.vectorize(len)

def bigIsZero(x):
    return ((x[3] == '0') or (x[3:10] == '9'*len(range(3,10))))
bigIsZero = np.vectorize(bigIsZero)

    
def plotWeirdDecimals(i, data, showNormalized = False):
    truncatedData = strTrunc(data) #decimal part of the numbers as string
    dataLen = getLen(truncatedData) #count of decimals    
    luniq, lcount = np.unique(dataLen, return_counts=True)
        
    bigNums = truncatedData[dataLen > 10] #select only the long numbers
    #print(bigNums)
    print('var', i, 
          '\n\tlong numbers are around zero?:', bigIsZero(bigNums).all(),
          '\n\tunique decimal lengths:', luniq, 
          '\n\tunique counts:', lcount)
    del bigNums

    #generate a histogram for each decimal count for big numbers
    bigLens = luniq[luniq > 10]
    lenRang = range(bigLens.min(), bigLens.max()+1)   

    bigByLen = [data[dataLen == l] for l in lenRang]
    hist, bins = np.histogram(data, bins=300)
    hist = hist / hist.max()
    
    hists = [np.histogram(d, bins=bins) for d in bigByLen]
    hists1 = [h[0]/h[0].max() for h in hists]
    bins = (bins[:-1] + bins[1:])/2
    
    if showNormalized == False:
        plt.figure(figsize=(20,5))
        for h, lab in zip(
            [hist] + hists1, 
            ['original data'] + [str(j) + ' decimals' for j in lenRang]
        ):
            plt.plot(bins, h, label=lab)
            
        plt.legend()
        plt.suptitle('var_'+str(i))
        plt.show()
    
    if showNormalized == True:
        hists2 = [h/hist for h in hists1]
        plt.figure(figsize=(20,5))
        for h, lab in zip(
            [hist] + hists2, 
            ['original data'] + [str(j) + ' decimals' for j in lenRang]
        ):
            plt.plot(bins, h, label=lab)
            
        plt.legend()
        plt.suptitle('var_'+str(i))
        plt.show()
        


# ### Vars 0 to 2

# In[ ]:


for i in range(0,3):    
    data = trainData[:,i]
    plotWeirdDecimals(i, data)


# ### Remaining vars (same behavior)

# In[ ]:


for i in range(3,200):    
    data = trainData[:,i]
    plotWeirdDecimals(i, data)
    
#outputs hidden - click to show


# ### Total data 
# 
# If you take a look at all those graphs, you will see that they all follow exactly the same pattern.   
# 
# - Some sudden bumps near numbers like 4, 5, 8, 10, 16, 20.5, 32, etc.
# 
# Here we plot them both with their original values and normalized.

# In[ ]:


def plotEntire(data, name, normed):
    flatData = data.reshape((-1,))
    if normed == True:
        flatData = flatData[np.logical_and(flatData>-25, flatData < 34)]
    plotWeirdDecimals(name, flatData, normed)
    
plotEntire(trainData, 'entire train data', False)
plotEntire(trainData, 'entire train data normalized', True)
plotEntire(realData, 'entire real test data normalized', True)
plotEntire(posData, 'entire positive targets normalized', True)


# # Some observations
# 
# ## All long numbers are around zero
# 
# We can see that all numbers that are long should result in 0 at the 4th decimal place. (Results for `long numbers around zero?` are `True` for all cases)   
# I don't know how machines work deep in their calculations, but I would expect these "accidents(?)" to happen with a lot of numbers, not only with the ones that end with 0 at the 4th place. (Selections from different machines? Hidden codes in decimals? Conspiracy theory?)    
# 
# ## But not all numbers ended in zero suffer from this
# 
# We can also see, by the counts, that there are numbers with less than 4 decimals, meaning not all roundings of zero got imperfect.   
# So, again, what is special about some numbers that they don't round?    
# 
# ## A very consistent pattern for all vars
# 
# The graphs are very consistent for all vars, and for the entire data.    
# Especially when you look at the normalized graphs, you see clear steady plateaus in the pattern.   
# This really tends to tell me that it's just a result of an underlying machine architecture, considering underflows or something like that. But it's weird.
# 
# It should be natural to expect bumps in multiples of 10 (since this increases/decreases the length of the part before the decimal point).   
# But other places? Well, considering 4, 8, 16 and 32, we could be seeing effects of binary systems?   
# What about 20.5? Magic number?  
# 
# PS, the higher left end of the positive targets graph is very probably due to normalization with very few numbers. (Had to cut the borders of all graphs)    
# 
# ## If zeros are special, how is the distribution of other endings?

# In[ ]:


data = trainData.reshape((-1,))

def getLastDigit(x):
    x = str(x).split('.')[1]
    if len(x) == 4:
        return x[-1] #for regular numbers
    elif len(x) < 4: 
        return '0' #for perfect zeros
    else:
        return 'L' #for long numbers
getLastDigit = np.vectorize(getLastDigit)
    
lastDigits = getLastDigit(data)
unique, counts = np.unique(lastDigits, return_counts=True)

for u,c in zip(unique,counts):
    print('digit',u,":",c,'instances')


# So, the numbers are evenly distributed (you can check individual vars as well).
