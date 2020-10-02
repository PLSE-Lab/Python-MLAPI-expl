# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# ===================================
# helper functions for calculating energy in a vectorized way
# ===================================
# a: the list of energy columns, e: predicted energy
def findAnsVec(a,e):
    if (a >= e):
        return a - e
    else:
        return 0
vfunc = np.vectorize(findAnsVec)  

# calculate the total energy above a given threshold
def calcEnergy(threshold,energies):
    return sum(vfunc(energies,threshold))
    
# calculate all the energies for a vector of thresholds
findEnergies = np.vectorize(calcEnergy)


# ===================================
# use only for loops and integers
# ===================================
def findEnergy1(es,ans):
    if(len(es) > 0):
        estotal = sum(es)
        eslen = len(es)
        total = 0
        for x in range(eslen +1):
            under = list(i for i in es if i < x+1)
            total = total + (eslen - len(under))
            if (estotal - total == ans):
                return x+1
    return 0    

# ===================================
# use some vectorization and integers
# ===================================
def findEnergy2(es,ans):
    if (len(es) > 0):
        for x in range(len(es) + 1):
            if(calcEnergy(x+1,es) == ans):
                return x+1
    return 0
    
# ===================================
# binary search with integers
# ===================================
def findEnergy3(es,ans):
    if len(es) > 0:
        first = 0
        last = len(es) + 1
        while True:
            print()
            m = (first + last) // 2
            if last < first:
                return -1
            elif calcEnergy(m,es) == ans:
                return m
            elif (calcEnergy(m,es) > ans):
                # if the calculated energy is larger than expected, need to increase the threshold
                first = m + 1
            elif (calcEnergy(m,es) < ans):
                # if the calculated energy is less than expected, need to lower the threshold
                last = m - 1
    return 0

# ===================================
# fully vectorized with integers
# ===================================            
def findEnergy4(es,ans):
    if(len(es) > 0): 
        thresholdVec = range(len(es)+2)
        # for vectorization, create a numpy array where the first element is the list of energies
        list_obj_array = np.ndarray((1,), dtype=object)
        list_obj_array[0] = es
        # find the threshold that gives the correct answer
        return np.where(findEnergies(thresholdVec,list_obj_array) == ans)[0][0]
    return 0

# ===================================
# TESTS 
# ===================================
def runTest(energies,ans,exp,testName):
    e1 = findEnergy1(energies,ans)
    e2 = findEnergy2(energies,ans)
    e3 = findEnergy3(energies,ans)
    e4 = findEnergy4(energies,ans)
    print("##### " + testName + " #####")
    print("findEnergy1 expected: " + exp + " , returned: " + str(e1))
    print("findEnergy2 expected: " + exp + " , returned: " + str(e2))
    print("findEnergy3 expected: " + exp + " , returned: " + str(e3))
    print("findEnergy4 expected: " + exp + " , returned: " + str(e4))
    print("")
    
def tests():
    runTest([3,2,4],0,"4","Test 1")
    runTest([3,2,4],9,"0","Test 2")
    runTest([3,2,4],6,"1","Test 3")
    runTest([3,2,4],3,"2","Test 4")
    runTest([3,1,4,1],3,"2","Test 5")
    runTest([],0,"0","Test 6")

    
    
    
    
    
    
    

    
    
    
    
    
    
    
    